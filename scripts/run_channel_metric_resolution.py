"""Compare CRN differences with pathwise derivatives, with fixed moment features."""

import argparse
import csv
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
    embedding_matrix_scores, embedding_scores, empirical_embedding_gram,
    pathwise_embedding_matrices, perturbed_inputs, polynomial_embedding_gram,
    sample_input_jacobians,
)
from conditional_drifting.metrics import sliced_wasserstein_distance
from scripts.run_channel_feature_metric_pilot import design_anchors, query_clouds, record
from scripts.run_local_gradient_fidelity import checkpoint_paths, physical_channel


def write_results(out, manifest, rows):
    (out / "results.json").write_text(json.dumps({"manifest": manifest, "rows": rows}, indent=2, allow_nan=False) + "\n")
    if rows:
        with (out / "per_anchor.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Pathwise and finite-difference channel metrics", "",
             "Fixed empirical distributions with common random numbers across perturbations.",
             "Analytic rows are independent sampling floors, not subtracted from scores.",
             "Moment features: y/sqrt(d) and vec(yy^T)/d. Augmented kernel is their direct sum with RBF.",
             "Means below average anchors and MC repeats; neither establishes population suprema.", "",
             "| Channel | Model | Estimator | h | RBF value | RBF derivative | Moment value | Moment derivative | Augmented derivative |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    keys = list(dict.fromkeys((r["channel"], r["method"], r["estimator"], r["step"]) for r in rows))
    for channel, method, estimator, step in keys:
        group = [r for r in rows if (r["channel"], r["method"], r["estimator"], r["step"]) == (channel, method, estimator, step)]
        names = ["rbf_mmd", "rbf_embedding_derivative_op", "moments_mmd", "moments_embedding_derivative_op", "augmented_embedding_derivative_op"]
        values = [f"{np.mean([r[n] for r in group]):.6g}" for n in names]
        lines.append(f"| {channel} | {method} | {estimator} | {step} | " + " | ".join(values) + " |")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default="AWGN,SSPA")
    parser.add_argument("--methods", default="analytic,fiber_sinkhorn,ddim10,ddim100")
    parser.add_argument("--steps", nargs="*", type=float, default=[.05, .0125, .003125, .00078125])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--anchor-directions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.samples < 2 or min(args.anchor_directions, args.repeats) < 1 or any(h <= 0 for h in args.steps):
        parser.error("Positive counts and steps required, samples >= 2.")
    channels = args.channels.split(",")
    if not set(channels) <= {"AWGN", "SSPA"}:
        parser.error("Supported channels: AWGN,SSPA.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT / "conditional_drifting/feature_metrics.py",
               ROOT / "scripts/run_channel_feature_metric_pilot.py", ROOT / "scripts/run_local_gradient_fidelity.py"]
    for source in sources:
        (out / source.name).write_bytes(source.read_bytes())
    manifest = {"config": vars(args), "torch": torch.__version__,
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "sources": [record(p) for p in sources], "checkpoints": {}, "channels": {},
                "seconds": {}, "protocol": "Journal_version/metric_resolution_protocol_20261009.md"}
    rows = []
    for channel in channels:
        dimension, ebno, rate = (7, 5., 4 / 7) if channel == "AWGN" else (8, 8., .75)
        noise_std = math.sqrt(1 / (2 * rate * 10**(ebno / 10)))
        component_std = noise_std if channel == "AWGN" else noise_std / math.sqrt(2)
        bandwidths = [component_std * math.sqrt(dimension) * scale for scale in (.5, 1., 2.)]
        anchors = design_anchors(dimension, args.anchor_directions, 71001).to(device)
        manifest["channels"][channel] = {"anchors": anchors.cpu().tolist(), "bandwidths": bandwidths,
                                          "noise_std": noise_std, "dimension": dimension, "rate": rate, "ebno_db": ebno}
        analytic = physical_channel(channel, noise_std)
        references = {}
        for rep in range(args.repeats):
            for index, x in enumerate(anchors):
                seed = 10000000 + rep * 1000000 + index * 20000
                p, jp = sample_input_jacobians(analytic, x, args.samples, seed)
                differences = {h: query_clouds(analytic, perturbed_inputs(x, h), args.samples, seed, "crn") for h in args.steps}
                references[rep, index] = (p, jp, differences)
        paths = checkpoint_paths(channel, args.seed)
        names = args.methods.split(",")
        if channel == "SSPA" and "fiber_sinkhorn" in names:
            names = list(dict.fromkeys(names + ["fiber_full"]))
        for name in names:
            print(f"[{channel}] seed={args.seed} {name} N={args.samples}", flush=True)
            start = time.monotonic()
            if name == "analytic":
                implant = analytic
            else:
                payload = torch.load(paths[name], map_location="cpu", weights_only=False)
                config = payload.get("config", {}) | payload.get("metadata", {}).get("config", {})
                if config["n"] != dimension or not math.isclose(config["noise_std"], noise_std, rel_tol=1e-6):
                    raise ValueError(f"Channel contract mismatch: {paths[name]}")
                manifest["checkpoints"][f"{channel}/{name}"] = record(paths[name]) | {"config": config}
                del payload
                implant = load_implant_from_checkpoint(paths[name], device=device, diffusion_sampler="ddim",
                                                       ddim_steps=int(name[4:]) if name.startswith("ddim") else 100)
            for rep in range(args.repeats):
                for index, x in enumerate(anchors):
                    seed = 100000000 + rep * 1000000 + index * 20000
                    p, jp, references_fd = references[rep, index]
                    q, jq = sample_input_jacobians(implant, x, args.samples, seed)
                    exact = pathwise_embedding_matrices(p, q, jp, jq, bandwidths)
                    base = {"channel": channel, "method": name, "seed": args.seed, "repeat": rep,
                            "anchor": index, "samples": args.samples}
                    row = base | {"estimator": "pathwise", "step": 0., "q_jacobian_fd_relative_error": 0.}
                    row["swd"] = sliced_wasserstein_distance(p, q, num_projections=64, seed=81001 + index)
                    for key, matrix in exact.items():
                        row.update({f"{key}_{k}": v for k, v in embedding_matrix_scores(matrix).items()})
                    rows.append(row)
                    for h in args.steps:
                        qfd = query_clouds(implant, perturbed_inputs(x, h), args.samples, seed, "crn")
                        torch.testing.assert_close(qfd[0], q, atol=1e-6, rtol=1e-5)
                        pfd = references_fd[h]
                        clouds = torch.cat((pfd, qfd))
                        rbf = empirical_embedding_gram(clouds, bandwidths)
                        moments = polynomial_embedding_gram(clouds)
                        fd_jac = ((qfd[1::2] - qfd[2::2]) / (2 * h)).permute(1, 2, 0)
                        relative = (fd_jac - jq).double().norm() / jq.double().norm().clamp_min(1e-12)
                        row = base | {"estimator": "finite_difference", "step": h,
                                      "q_jacobian_fd_relative_error": relative.item()}
                        row["swd"] = rows[-1]["swd"]
                        for key, gram in {"rbf": rbf, "moments": moments, "augmented": rbf + moments}.items():
                            row.update({f"{key}_{k}": v for k, v in embedding_scores(gram, dimension, h).items()})
                        rows.append(row)
            if device.type == "cuda":
                torch.cuda.synchronize()
            manifest["seconds"][f"{channel}/{name}"] = time.monotonic() - start
            write_results(out, manifest, rows)
            print(f"  done in {time.monotonic() - start:.1f}s", flush=True)
            del implant
    print(out / "README.md", flush=True)


if __name__ == "__main__":
    main()
