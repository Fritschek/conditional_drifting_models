"""Decoder-free, finite-panel channel metrics from existing AWGN/SSPA checkpoints."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.e2e_implants import load_implant_from_checkpoint
from conditional_drifting.feature_metrics import (
    empirical_embedding_gram, embedding_scores, moment_scores, perturbed_inputs,
)
from conditional_drifting.metrics import sliced_wasserstein_distance
from scripts.run_local_gradient_fidelity import checkpoint_paths, physical_channel


def design_anchors(dimension, directions, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(directions, dimension))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return torch.tensor(np.concatenate([v * radius * math.sqrt(dimension) for radius in (.5, 1., 1.5)]), dtype=torch.float32)


@torch.no_grad()
def query_clouds(implant, queries, samples, seed, mode):
    clouds = []
    for index, x in enumerate(queries):
        # CRN couples perturbations within one simulator only, never P and Q.
        torch.manual_seed(seed if mode == "crn" else seed + index * 1009)
        clouds.append(implant(x[None].repeat(samples, 1)).detach())
    return torch.stack(clouds)


def record(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


SCORES = ["swd", "mmd", "embedding_derivative_op", "mean_l2", "covariance_fro", "mean_derivative_op", "covariance_derivative_op"]


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["channel"], row["method"], row["mode"], row["step"])
        groups.setdefault(key, []).append(row)
    output = []
    for (channel, method, mode, step), group in groups.items():
        result = {"channel": channel, "method": method, "mode": mode, "step": step}
        for score in SCORES:
            repeats = sorted({r["repeat"] for r in group})
            means = [np.mean([r[score] for r in group if r["repeat"] == rep]) for rep in repeats]
            maxima = [max(r[score] for r in group if r["repeat"] == rep) for rep in repeats]
            result[score] = {"anchor_mean": float(np.mean(means)),
                             "anchor_mean_mc_se": float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None,
                             "anchor_max_mean": float(np.mean(maxima)), "repeat_anchor_means": means}
        output.append(result)
    return output


def save_outputs(out, manifest, rows):
    (out / "results.json").write_text(json.dumps({"manifest": manifest, "summary": summarize(rows), "rows": rows}, indent=2, allow_nan=False) + "\n")
    if not rows:
        return
    with (out / "per_anchor.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Decoder-free feature metric pilot", "",
        "Development seed only; no downstream codec is loaded to calculate a score.",
        "Full RBF-kernel empirical embedding norms (V-statistics), not random features.",
        "Analytic row is an independent exact-channel sampling floor, not a subtraction.",
        "CRN shares noise across input perturbations within a simulator; independent mode does not.",
        "Bandwidths are fixed from physical noise, not tuned against downstream results.",
        "Table entries average fixed anchors and Monte Carlo repeats; maxima and MC SE are in JSON.",
        "Finite-panel maxima do not certify population suprema. Historical training budgets differ.", "",
        "| Channel | Model | Noise pairing | h | SWD | MMD | Embedding derivative | Mean derivative | Covariance derivative |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summarize(rows):
        values = [f"{row[s]['anchor_mean']:.5g}" for s in ("swd", "mmd", "embedding_derivative_op", "mean_derivative_op", "covariance_derivative_op")]
        lines.append(f"| {row['channel']} | {row['method']} | {row['mode']} | {row['step']} | " + " | ".join(values) + " |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default="AWGN,SSPA")
    parser.add_argument("--methods", default="analytic,kernel_target,joint_sinkhorn,fiber_sinkhorn,wgan,ddim10,ddim100")
    parser.add_argument("--modes", default="crn,independent")
    parser.add_argument("--steps", type=float, nargs="+", default=[.025, .05, .1])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--anchor-directions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if min(args.samples, args.anchor_directions, args.repeats) < 1 or args.samples < 2 or min(args.steps) <= 0:
        parser.error("Positive counts/steps and at least two samples required.")
    modes = args.modes.split(",")
    if not set(modes) <= {"crn", "independent"}:
        parser.error("Modes must be crn and/or independent.")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__).resolve(), ROOT / "conditional_drifting/feature_metrics.py"):
        (out / path.name).write_bytes(path.read_bytes())
    manifest = {"config": vars(args), "torch": torch.__version__,
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "sources": [record(Path(__file__).resolve()), record(ROOT / "conditional_drifting/feature_metrics.py")],
                "channels": {}, "checkpoints": {}, "sampling_cost": {}}
    rows = []
    for channel in args.channels.split(","):
        if channel not in {"AWGN", "SSPA"}:
            parser.error("This pilot supports AWGN and SSPA only.")
        dimension, ebno, rate = (7, 5., 4 / 7) if channel == "AWGN" else (8, 8., .75)
        noise_std = math.sqrt(1 / (2 * rate * 10**(ebno / 10)))
        component_std = noise_std if channel == "AWGN" else noise_std / math.sqrt(2)
        bandwidths = [component_std * math.sqrt(dimension) * scale for scale in (.5, 1., 2.)]
        anchors = design_anchors(dimension, args.anchor_directions, 71001).to(device)
        analytic = physical_channel(channel, noise_std)
        manifest["channels"][channel] = {"dimension": dimension, "ebno_db": ebno, "rate": rate,
            "noise_std": noise_std, "bandwidths": bandwidths, "anchors": anchors.cpu().tolist(),
            "anchor_domain": "Fixed radial panel, radii 0.5/1/1.5 times sqrt(n); coordinate perturbations are not power-renormalized."}
        references = {}
        for mode in modes:
            for step in args.steps:
                for rep in range(args.repeats):
                    for index, x in enumerate(anchors):
                        seed = 10000000 + rep * 1000000 + index * 20000
                        queries = perturbed_inputs(x, step)
                        references[(mode, step, rep, index)] = query_clouds(analytic, queries, args.samples, seed, mode)
        paths = checkpoint_paths(channel, args.seed)
        names = args.methods.split(",")
        if channel == "SSPA" and "fiber_sinkhorn" in names:
            names = list(dict.fromkeys(names + ["fiber_full"]))
        for name in names:
            print(f"[{channel}] {name}, N={args.samples}", flush=True)
            start = time.monotonic()
            if name == "analytic":
                implant = analytic
            else:
                payload = torch.load(paths[name], map_location="cpu", weights_only=False)
                config = payload.get("config", {}) | payload.get("metadata", {}).get("config", {})
                if config["n"] != dimension or not math.isclose(config["noise_std"], noise_std, rel_tol=1e-6):
                    raise ValueError(f"Checkpoint channel contract mismatch: {paths[name]}")
                manifest["checkpoints"][f"{channel}/{name}"] = record(paths[name]) | {"config": config}
                del payload
                implant = load_implant_from_checkpoint(paths[name], device=device, diffusion_sampler="ddim",
                    ddim_steps=int(name[4:]) if name.startswith("ddim") else 100)
            for mode in modes:
                for step in args.steps:
                    for rep in range(args.repeats):
                        for index, x in enumerate(anchors):
                            seed = 100000000 + rep * 1000000 + index * 20000
                            p = references[(mode, step, rep, index)]
                            q = query_clouds(implant, perturbed_inputs(x, step), args.samples, seed, mode)
                            gram = empirical_embedding_gram(torch.cat((p, q)), bandwidths)
                            row = {"channel": channel, "method": name, "mode": mode, "step": step,
                                   "repeat": rep, "anchor": index, "samples_per_query": args.samples}
                            row.update(embedding_scores(gram, dimension, step))
                            row.update(moment_scores(p, q, dimension, step))
                            row["swd"] = sliced_wasserstein_distance(p[0], q[0], num_projections=64, seed=81001 + index)
                            rows.append(row)
                    print(f"  {mode} h={step} done", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            queries = len(modes) * len(args.steps) * args.repeats * len(anchors) * (1 + 2 * dimension)
            manifest["sampling_cost"][f"{channel}/{name}"] = {"surrogate_draws": queries * args.samples,
                "shared_reference_draws": queries * args.samples, "seconds_including_metrics": time.monotonic() - start}
            save_outputs(out, manifest, rows)
            del implant
    print(out / "README.md", flush=True)


if __name__ == "__main__":
    main()
