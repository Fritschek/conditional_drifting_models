"""Same-input, independently sampled SSPA metric and expected-gradient panel."""

import argparse
import hashlib
import itertools
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
from conditional_drifting.channels import sspa
from conditional_drifting.e2e_implants import load_implant_from_checkpoint
from conditional_drifting.feature_metrics import sample_input_jacobians
from conditional_drifting.metrics import sliced_wasserstein_distance
from scripts.run_channel_metric_controls import (
    expected_output_loss_gradient, sampled_loss_gradient, split_metrics,
)
from scripts.run_local_gradient_fidelity import checkpoint_paths, physical_channel

METHODS = ("analytic", "fiber_sinkhorn", "wgan", "ddim10")
CHEAP = ("swd", "rbf_cross_value_squared", "mean_derivative_cross",
         "covariance_derivative_cross", "moments_cross_trace", "fourth_derivative_cross")
PROTOCOL = ROOT / "Journal_version/learned_metric_gate_protocol_20261009.md"
PANEL = ROOT / "results/channel_metric_pathwise_n512_seed7_20261009/results.json"


def file_record(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def lifted_probes(anchors):
    v1 = anchors[0] / anchors[0].norm()
    v2 = anchors[1] - anchors[1].dot(v1)*v1
    if v2.norm() < 1e-8:
        raise ValueError("Degenerate probe basis")
    v2 = v2/v2.norm()
    probes = []
    for index, (w, b, center) in enumerate((
        (.8*v1+.6*v2, .2, .3*v1-.4*v2),
        (-.6*v1+.8*v2, -.3, -.7*v1+.6*v2),
    )):
        for kind in ("rbf_section", "quadratic", "logistic", "cosine"):
            probes.append({"name": f"direction{index}_{kind}", "kind": kind,
                           "w": w, "offset": b, "center": center})
    return probes


def sspa_reference(x, noise_std, bandwidths, probes):
    mean_fn = lambda u: sspa(u[None], 0., u.device)[0]
    mean = mean_fn(x)
    jac = torch.autograd.functional.jacobian(mean_fn, x)
    gradients = {}
    for probe in probes:
        kwargs = {"sigma": noise_std/math.sqrt(2), "bandwidths": bandwidths}
        grad = jac.T @ expected_output_loss_gradient(mean, probe, **kwargs)
        low = jac.T @ expected_output_loss_gradient(mean, probe, nodes=64, **kwargs)
        if (grad-low).norm() > 1e-10:
            raise ValueError("Reference quadrature failed")
        gradients[probe["name"]] = grad
    return mean, jac, gradients


def moment_vectors(y, jac, directions):
    n = len(y)
    mean, mean_jac = y.mean(0), jac.mean(0)
    centered, centered_jac = y-mean, jac-mean_jac
    cov = centered.T @ centered/(n-1)
    cov_jac = (torch.einsum("noa,np->opa", centered_jac, centered)
               + torch.einsum("no,npa->opa", centered, centered_jac))/(n-1)
    projected = y @ directions.T
    projected_jac = torch.einsum("ko,noa->nka", directions, jac)
    fourth = projected.pow(4).mean(0)
    fourth_jac = (4*projected.pow(3)[..., None]*projected_jac).mean(0)
    return {"mean_value": mean, "mean_derivative": mean_jac,
            "covariance_value": cov, "covariance_derivative": cov_jac,
            "fourth_value": fourth, "fourth_derivative": fourth_jac}


def moment_split_scores(first, second, directions):
    differences = []
    for p, q, jp, jq in (first, second):
        a, b = moment_vectors(p, jp, directions), moment_vectors(q, jq, directions)
        differences.append({k: a[k]-b[k] for k in a})
    row = {}
    for key, a in differences[0].items():
        b = differences[1][key]
        row[key+"_self"] = ((a.square().sum()+b.square().sum())/2).item()
        row[key+"_cross"] = (a*b).sum().item()
        row[key+"_noise"] = ((a-b).square().sum()/2).item()
        row[key+"_split1"] = a.tolist()
        row[key+"_split2"] = b.tolist()
    return row


def stream_seed(n, anchor, repeat, role, model, split):
    # Decimal fields are non-overlapping within the bounded protocol.
    return 100000000 + n*100000 + anchor*1000000 + repeat*10000 + role*1000 + model*10 + split


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "mc_se": float(values.std(ddof=1)/math.sqrt(len(values))) if len(values) > 1 else 0.}


def contrast(values):
    result = summary(values)
    result["ordering"] = (int(np.sign(result["mean"]))
                          if abs(result["mean"]) > 3*result["mc_se"]+1e-12 else 0)
    return result


def contrasts(rows, losses, n):
    metrics = {(r["anchor"], r["method"], r["repeat"]): r for r in rows if r["samples"] == n}
    tasks = {(r["anchor"], r["method"], r["repeat"], r["probe"]): r
             for r in losses if r["samples"] == n}
    output = []
    for anchor in sorted({key[0] for key in metrics}):
        reps = sorted({key[2] for key in metrics if key[0] == anchor})
        probes = sorted({key[3] for key in tasks if key[0] == anchor})
        for a, b in itertools.combinations(METHODS[1:], 2):
            scores = {name: contrast([metrics[anchor, a, r][name]-metrics[anchor, b, r][name]
                                     for r in reps]) for name in CHEAP+("rbf_cross_trace",)}
            for probe in probes:
                target = contrast([tasks[anchor, a, r, probe]["cross_squared_error"]
                                   - tasks[anchor, b, r, probe]["cross_squared_error"] for r in reps])
                order = target["ordering"]
                kernel_agrees = bool(order and scores["rbf_cross_trace"]["ordering"] == order)
                cheap_agrees = [k for k in CHEAP if order and scores[k]["ordering"] == order]
                output.append({"anchor": anchor, "samples": n, "left": a, "right": b,
                               "probe": probe, "target": target, "scores": scores,
                               "kernel_agrees": kernel_agrees, "cheap_agrees": cheap_agrees,
                               "candidate_added_information": kernel_agrees and not cheap_agrees})
    return output


def refinement_decisions(comparisons):
    decisions = []
    for anchor in sorted({r["anchor"] for r in comparisons}):
        group = [r for r in comparisons if r["anchor"] == anchor]
        unresolved = sum(r["target"]["ordering"] == 0 for r in group)
        informative = sum(r["kernel_agrees"] and len(r["cheap_agrees"]) < len(CHEAP) for r in group)
        decisions.append({"anchor": anchor, "refine": bool(unresolved or informative),
                          "unresolved_targets": unresolved, "potentially_informative": informative})
    return decisions


def preflight():
    old = json.loads(PANEL.read_text())["manifest"]
    channel = old["channels"]["SSPA"]
    anchors = torch.tensor(channel["anchors"], dtype=torch.float64)[[3, 4, 5]]
    if anchors.shape != (3, 8) or not torch.allclose(anchors.norm(dim=1), torch.full((3,), math.sqrt(8), dtype=torch.float64)):
        raise ValueError("Stored anchors violate the eight-dimensional power contract")
    paths = checkpoint_paths("SSPA", 7)
    records = {}
    for name in METHODS[1:]:
        rec = file_record(paths[name])
        if rec["sha256"] != old["checkpoints"]["SSPA/"+name]["sha256"]:
            raise ValueError(f"Checkpoint changed: {name}")
        payload = torch.load(paths[name], map_location="cpu", weights_only=False)
        config = payload.get("config", {}) | payload.get("metadata", {}).get("config", {})
        if config["n"] != 8 or not math.isclose(config["noise_std"], channel["noise_std"], rel_tol=1e-6):
            raise ValueError(f"Checkpoint units mismatch: {name}")
        records[name] = rec | {"config": config}
    return channel, anchors, paths, records


def run(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    try:
        channel, anchors, paths, checkpoints = preflight()
    except Exception as error:
        (out/"failure_manifest.json").write_text(json.dumps({"error": str(error), "panel": str(PANEL)}, indent=2))
        raise
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    probes = lifted_probes(anchors)
    directions = torch.stack((probes[0]["w"], probes[4]["w"]))
    bandwidths, noise_std = channel["bandwidths"], channel["noise_std"]
    reference = [sspa_reference(x, noise_std, bandwidths, probes) for x in anchors]
    sources = [Path(__file__).resolve(), PROTOCOL, ROOT/"scripts/run_channel_metric_controls.py",
               ROOT/"conditional_drifting/feature_metrics.py", ROOT/"conditional_drifting/channels.py",
               ROOT/"conditional_drifting/e2e_implants.py", ROOT/"conditional_drifting/metrics.py"]
    for source in sources:
        (out/source.name).write_bytes(source.read_bytes())
    serial_probes = [{k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in p.items()} for p in probes]
    manifest = {"config": vars(args) | {"out_dir": str(out)}, "sources": [file_record(p) for p in sources],
                "input_panel": file_record(PANEL), "checkpoints": checkpoints,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "kernel_device": "CPU float64, four threads", "anchors": anchors.tolist(),
                "original_anchor_indices": [3, 4, 5], "probes": serial_probes,
                "bandwidths": bandwidths, "noise_std": noise_std, "component_std": noise_std/math.sqrt(2),
                "reference": [{"mean": m.tolist(), "jacobian": j.tolist(),
                               "gradients": {k: v.tolist() for k, v in g.items()}} for m, j, g in reference],
                "timing_seconds": {}, "sampled_outputs": {}, "refinement": [],
                "scope": "development only; MC repeats are not trained-model seeds"}
    rows, losses = [], []
    started = time.monotonic()

    def save():
        manifest["elapsed_seconds"] = time.monotonic()-started
        payload = {"manifest": manifest, "rows": rows, "losses": losses}
        temp = out/"results.tmp"
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
        temp.replace(out/"results.json")

    def sample(implant, x, n, seed, label):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        y, jac = sample_input_jacobians(implant, x.to(device=device, dtype=torch.float32), n, seed)
        y, jac = y.cpu().double(), jac.cpu().double()
        if not torch.isfinite(y).all() or not torch.isfinite(jac).all():
            raise ValueError(f"Nonfinite samples: {label}")
        manifest["timing_seconds"][label] = manifest["timing_seconds"].get(label, 0.)+time.monotonic()-start
        manifest["sampled_outputs"][label] = manifest["sampled_outputs"].get(label, 0)+n
        return y, jac

    analytic = physical_channel("SSPA", noise_std)
    anchor_ids = list(range(3)) if not args.preflight_only else [0]
    levels = [512, 2048] if not args.preflight_only else [32]
    for n in levels:
        if not anchor_ids:
            break
        refs = {}
        for a in anchor_ids:
            for rep in range(args.repeats):
                refs[a, rep] = [sample(analytic, anchors[a], n, stream_seed(n, a, rep, 0, 0, s), "reference_score")
                                for s in range(2)]
        torch.save(refs, out/f"reference_score_n{n}.pt")
        for model, name in enumerate(METHODS):
            start = time.monotonic()
            implant = analytic if name == "analytic" else load_implant_from_checkpoint(
                paths[name], device=device, diffusion_sampler="ddim", ddim_steps=10)
            if device.type == "cuda":
                torch.cuda.synchronize()
            manifest["timing_seconds"][f"load_{name}_n{n}"] = time.monotonic()-start
            for a in anchor_ids:
                for rep in range(args.repeats):
                    score_seeds = [stream_seed(n, a, rep, 1, model, s) for s in range(2)]
                    task_seeds = [stream_seed(n, a, rep, 2, model, s) for s in range(2)]
                    scores = [sample(implant, anchors[a], n, seed, name+"_score") for seed in score_seeds]
                    tasks = [sample(implant, anchors[a], n, seed, name+"_task") for seed in task_seeds]
                    torch.save({"scores": scores, "tasks": tasks, "score_seeds": score_seeds, "task_seeds": task_seeds},
                               out/f"samples_{name}_n{n}_a{a}_r{rep}.pt")
                    paired = [(p, q, jp, jq) for (p, jp), (q, jq) in zip(refs[a, rep], scores)]
                    start = time.monotonic()
                    key = {"method": name, "anchor": a, "repeat": rep, "samples": n}
                    row = key | split_metrics(*paired, args.block_size, bandwidths, detailed=True)
                    row.update(moment_split_scores(*paired, directions))
                    row["swd"] = float(np.mean([sliced_wasserstein_distance(p, q, num_projections=64, seed=81001+a)
                                                for p, q, _, _ in paired]))
                    rows.append(row)
                    for probe in probes:
                        g = reference[a][2][probe["name"]]
                        gs = [sampled_loss_gradient(y, j, probe, bandwidths) for y, j in tasks]
                        e1, e2 = gs[0]-g, gs[1]-g
                        losses.append(key | {"probe": probe["name"], "kind": probe["kind"],
                                             "reference_gradient": g.tolist(),
                                             "gradient_split1": gs[0].tolist(), "gradient_split2": gs[1].tolist(),
                                             "absolute_gradient_error": ((e1+e2)/2).norm().item(),
                                             "cross_squared_error": e1.dot(e2).item(),
                                             "self_squared_error": ((e1.square().sum()+e2.square().sum())/2).item(),
                                             "estimated_noise_squared": ((e1-e2).square().sum()/2).item()})
                    label = f"metric_and_probe_algebra_n{n}"
                    manifest["timing_seconds"][label] = manifest["timing_seconds"].get(label, 0.)+time.monotonic()-start
                print(f"[gate] N={n} {name} anchor={a}: {len(rows)} metric rows", flush=True)
                save()
            del implant
        comparisons = contrasts(rows, losses, n)
        (out/f"contrasts_n{n}.json").write_text(json.dumps(comparisons, indent=2)+"\n")
        if n == 512:
            decisions = refinement_decisions(comparisons)
            manifest["refinement"] = decisions
            (out/"refinement_decisions.json").write_text(json.dumps(decisions, indent=2)+"\n")
            save()
            anchor_ids = [r["anchor"] for r in decisions if r["refine"]]
            print(f"[gate] frozen refinement decisions: {decisions}", flush=True)
    manifest["status"] = "complete"
    save()
    print(out/"results.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.repeats = 2 if args.preflight_only else 8
    if args.block_size < 1:
        parser.error("Positive block size required")
    run(args)


if __name__ == "__main__":
    main()
