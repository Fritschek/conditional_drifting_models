"""Exact-law rotation and matched-value, varied-derivative synthetic controls."""

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
from conditional_drifting.feature_metrics import (
    embedding_matrix_scores, pathwise_cross_embedding_matrices, pathwise_embedding_matrices,
)
from conditional_drifting.metrics import sliced_wasserstein_distance


SIGMA = .5
BANDWIDTHS = [.5 * SIGMA * math.sqrt(2), SIGMA * math.sqrt(2), 2 * SIGMA * math.sqrt(2)]
ANCHORS = [(-1., 0.), (0., 1.), (1., 0.)]


def rotation_channel(x, z, omega, anchor):
    angle = omega * (x[..., 0] - anchor[0])
    c, s = angle.cos(), angle.sin()
    rotated = torch.stack((c*z[..., 0] - s*z[..., 1], s*z[..., 0] + c*z[..., 1]), -1)
    return x + SIGMA * rotated


def rotation_samples(anchor, z, omega):
    # At the centered anchor R=I; dR/dx_0 is omega times the planar skew matrix.
    y = anchor + SIGMA*z
    jac = torch.eye(2, device=z.device, dtype=z.dtype).repeat(len(z), 1, 1)
    jac[:, :, 0] += SIGMA*omega*torch.stack((-z[:, 1], z[:, 0]), -1)
    return y, jac


def smooth_bump(x, anchor):
    radius_squared = (x-anchor).square().sum(-1)
    t = (.6**2-radius_squared) / (.6**2-.15**2)
    def positive_exp(v):
        return torch.where(v > 0, torch.exp(-1/v.clamp_min(1e-12)), torch.zeros_like(v))
    a, b = positive_exp(t), positive_exp(1-t)
    return a/(a+b)


def perturbed_mean(x, anchors, eta, slope, u, v):
    shift = torch.zeros_like(x[..., 0])
    for anchor in anchors:
        shift += smooth_bump(x, anchor) * (eta + slope*((x-anchor)*v).sum(-1))
    return x + shift[..., None]*u


def gaussian_population_matrices(p, q, jp, jq):
    result = None
    for length in BANDWIDTHS:
        effective = math.sqrt(length**2+2*SIGMA**2)
        matrices = pathwise_embedding_matrices(p[None], q[None], jp[None], jq[None], [effective])
        if result is None:
            result = {"rbf": torch.zeros_like(matrices["rbf"]), "moments": matrices["moments"]}
        # Equal constant covariances cancel in the polynomial differences.
        result["rbf"] += (length**2/effective**2)**(p.numel()/2)*matrices["rbf"]/len(BANDWIDTHS)
    result["augmented"] = result["rbf"]+result["moments"]
    return result


def split_metrics(first, second, block_size, bandwidths=None, detailed=False):
    bandwidths = BANDWIDTHS if bandwidths is None else bandwidths
    a = pathwise_cross_embedding_matrices(first, first, bandwidths, block_size)
    b = pathwise_cross_embedding_matrices(second, second, bandwidths, block_size)
    cross = pathwise_cross_embedding_matrices(first, second, bandwidths, block_size)
    row = {}
    for name in a:
        ordinary = (a[name]+b[name])/2
        symmetric = (cross[name]+cross[name].T)/2
        scores = embedding_matrix_scores(ordinary)
        row.update({f"{name}_{k}": value for k, value in scores.items()})
        row[f"{name}_self_trace"] = ordinary[1:, 1:].trace().item()
        row[f"{name}_cross_trace"] = symmetric[1:, 1:].trace().item()
        row[f"{name}_cross_value_squared"] = symmetric[0, 0].item()
        row[f"{name}_cross_min_eigenvalue"] = torch.linalg.eigvalsh(symmetric[1:, 1:])[0].item()
        row[f"{name}_estimated_noise_trace"] = (ordinary[1:, 1:]-symmetric[1:, 1:]).trace().item()
        row[f"{name}_self_matrix"] = ordinary.cpu().tolist()
        row[f"{name}_cross_matrix"] = symmetric.cpu().tolist()
        if detailed:
            row[f"{name}_split1_matrix"] = a[name].cpu().tolist()
            row[f"{name}_split2_matrix"] = b[name].cpu().tolist()
            row[f"{name}_noise_matrix"] = (ordinary-symmetric).cpu().tolist()
            if torch.linalg.eigvalsh((ordinary-symmetric)[1:, 1:])[0] < -1e-7:
                raise ValueError("Split variance identity lost positive semidefiniteness")
    return row


def probes(device):
    values = []
    for split, w, offset, center in (
        ("development", [.8, .6], .2, [.3, -.4]),
        ("unused_probe", [-.6, .8], -.3, [-.7, .6]),
    ):
        for kind in ("rbf_section", "quadratic", "logistic", "cosine"):
            values.append({"name": f"{split}_{kind}", "split": split, "kind": kind,
                           "w": torch.tensor(w, device=device, dtype=torch.float64),
                           "offset": offset, "center": torch.tensor(center, device=device, dtype=torch.float64)})
    return values


def expected_output_loss_gradient(mean, probe, nodes=128, sigma=SIGMA, bandwidths=None):
    bandwidths = BANDWIDTHS if bandwidths is None else bandwidths
    kind, w, offset = probe["kind"], probe["w"], probe["offset"]
    t = mean.dot(w)-offset
    if kind == "quadratic":
        return t*w
    if kind == "cosine":
        return torch.exp(-sigma**2*w.square().sum()/2)*t.sin()*w
    if kind == "logistic":
        points, weights = np.polynomial.hermite.hermgauss(nodes)
        points, weights = mean.new_tensor(points), mean.new_tensor(weights)
        logits = t + math.sqrt(2)*sigma*w.norm()*points
        return (weights*torch.sigmoid(logits)).sum()/math.sqrt(math.pi)*w
    if kind == "rbf_section":
        delta = probe["center"]-mean
        result = torch.zeros_like(mean)
        for length in bandwidths:
            total = length**2+sigma**2
            value = (length**2/total)**(mean.numel()/2)*torch.exp(-delta.square().sum()/(2*total))
            result += value*delta/total/len(bandwidths)
        return result
    raise ValueError(kind)


def sampled_loss_gradient(y, jac, probe, bandwidths=None):
    bandwidths = BANDWIDTHS if bandwidths is None else bandwidths
    kind, w = probe["kind"], probe["w"]
    t = y@w-probe["offset"]
    if kind == "quadratic":
        dy = t[:, None]*w
    elif kind == "logistic":
        dy = torch.sigmoid(t)[:, None]*w
    elif kind == "cosine":
        dy = t.sin()[:, None]*w
    elif kind == "rbf_section":
        delta = probe["center"]-y
        dy = sum(torch.exp(-delta.square().sum(-1)/(2*l*l))[:, None]*delta/(l*l) for l in bandwidths)/len(bandwidths)
    else:
        raise ValueError(kind)
    return torch.einsum("no,noa->a", dy, jac)/len(y)


def gradient_comparison(g, q):
    ng, nq = g.norm().item(), q.norm().item()
    dot = g.dot(q).item()
    return {"absolute_gradient_error": (q-g).norm().item(),
            "reference_norm": ng, "surrogate_norm": nq,
            "relative_gradient_error": (q-g).norm().item()/ng if ng > 1e-12 else None,
            "cosine": dot/(ng*nq) if ng*nq > 1e-12 else None,
            "norm_ratio": nq/ng if ng > 1e-12 else None,
            "normalized_step_first_order_change_per_unit_step": -dot/nq if nq > 1e-12 else None,
            "sgd_first_order_change_per_unit_learning_rate": -dot,
            "sgd_progress_ratio": dot/(ng*ng) if ng > 1e-12 else None}


def draws(n, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return [torch.randn(n, 2, generator=generator, device=device, dtype=torch.float64) for _ in range(4)]


def run_rotation(args, save):
    rows = []
    anchor = torch.tensor([.6, -.8], device=args.device, dtype=torch.float64)
    reference_seconds, candidate_seconds, normal_vectors, logical_outputs = 0., 0., 0, 0
    for n in args.rotation_samples:
        for rep in range(args.rotation_repeats):
            start = time.monotonic()
            z = draws(n, 1200000+n*100+rep, args.device)
            p1, jp1 = rotation_samples(anchor, z[0], 0)
            p2, jp2 = rotation_samples(anchor, z[2], 0)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            reference_seconds += time.monotonic()-start
            normal_vectors += 4*n
            logical_outputs += 2*n
            for omega in (0., 1., 4., 16.):
                start = time.monotonic()
                q1, jq1 = rotation_samples(anchor, z[1], omega)
                q2, jq2 = rotation_samples(anchor, z[3], omega)
                row = {"experiment": "rotation", "samples_per_split": n, "repeat": rep, "omega": omega}
                row.update(split_metrics((p1, q1, jp1, jq1), (p2, q2, jp2, jq2), args.block_size))
                d1, d2 = jp1.mean(0)-jq1.mean(0), jp2.mean(0)-jq2.mean(0)
                row["linear_self_trace"] = ((d1.square().sum()+d2.square().sum())/2).item()
                row["linear_cross_trace"] = (d1*d2).sum().item()
                row["linear_expected_self_trace"] = 2*SIGMA**2*omega**2/n
                row["population_discrepancy"] = 0.
                row["swd"] = sliced_wasserstein_distance(p1, q1, num_projections=64, seed=81001)
                rows.append(row)
                logical_outputs += 2*n
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                candidate_seconds += time.monotonic()-start
        print(f"[rotation] N={n}: {len(rows)} rows", flush=True)
        save(rows, [], {"reference_seconds": reference_seconds, "candidate_and_metric_seconds": candidate_seconds,
                       "independent_gaussian_vectors": normal_vectors, "logical_output_evaluations": logical_outputs})


def run_matched(args, save):
    rows, loss_rows = [], []
    anchors = torch.tensor(ANCHORS, device=args.device, dtype=torch.float64)
    eye = torch.eye(2, device=args.device, dtype=torch.float64)
    directions = [("development", [1., 0.], [1., 0.]),
                  ("unused_direction", [2**-.5, 2**-.5], [2**-.5, -2**-.5])]
    reference_seconds, candidate_seconds, normal_vectors, logical_outputs = 0., 0., 0, 0
    n = args.matched_samples
    tasks = probes(args.device)
    for direction_index, (label, u, v) in enumerate(directions):
        u, v = anchors.new_tensor(u), anchors.new_tensor(v)
        for eta_index, eta in enumerate((0., .1)):
            for index, anchor in enumerate(anchors):
                for rep in range(args.matched_repeats):
                    start = time.monotonic()
                    seed = 3000000+direction_index*100000+eta_index*10000+index*1000+rep
                    z = draws(n, seed, args.device)
                    p1, p2 = anchor+SIGMA*z[0], anchor+SIGMA*z[2]
                    jp = eye.expand(n, -1, -1)
                    qmean = anchor+eta*u
                    q1, q2 = qmean+SIGMA*z[1], qmean+SIGMA*z[3]
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    reference_seconds += time.monotonic()-start
                    normal_vectors += 4*n
                    logical_outputs += 2*n
                    for slope in (-4., -1., 0., 1., 4.):
                        start = time.monotonic()
                        jqmean = eye+slope*u[:, None]*v[None]
                        jq = jqmean.expand(n, -1, -1)
                        key = {"experiment": "matched", "direction": label, "eta": eta, "slope": slope,
                               "anchor": index, "repeat": rep, "samples_per_split": n}
                        exact = gaussian_population_matrices(anchor, qmean, eye, jqmean)
                        row = key | split_metrics((p1, q1, jp, jq), (p2, q2, jp, jq), args.block_size)
                        for name, matrix in exact.items():
                            row.update({f"population_{name}_{k}": value for k, value in embedding_matrix_scores(matrix).items()})
                            row[f"population_{name}_matrix"] = matrix.cpu().tolist()
                        row["population_mean_error"] = eta
                        row["population_mean_jacobian_error"] = abs(slope)
                        row["swd"] = sliced_wasserstein_distance(p1, q1, num_projections=64, seed=81001+index)
                        rows.append(row)
                        for probe in tasks:
                            g = eye.T@expected_output_loss_gradient(anchor, probe)
                            qg = jqmean.T@expected_output_loss_gradient(qmean, probe)
                            estimate = sampled_loss_gradient(q1, jq, probe)-sampled_loss_gradient(p1, jp, probe)
                            loss_row = key | {"probe": probe["name"], "probe_split": probe["split"], "loss": probe["kind"]}
                            loss_row.update(gradient_comparison(g, qg))
                            loss_row["sampled_absolute_gradient_error"] = estimate.norm().item()
                            loss_row["gradient_difference_estimation_error"] = (estimate-(qg-g)).norm().item()
                            loss_row["reference_gradient"] = g.cpu().tolist()
                            loss_row["surrogate_gradient"] = qg.cpu().tolist()
                            if probe["kind"] == "logistic":
                                low = jqmean.T@expected_output_loss_gradient(qmean, probe, 64)
                                loss_row["quadrature_difference"] = (low-qg).norm().item()
                                if loss_row["quadrature_difference"] > 1e-10:
                                    raise ValueError("Logistic quadrature did not converge")
                            else:
                                loss_row["quadrature_difference"] = 0.
                            if probe["kind"] == "rbf_section":
                                bound = row["population_rbf_embedding_derivative_op"]
                                if loss_row["absolute_gradient_error"] > bound+1e-7:
                                    raise ValueError("Kernel-section gradient bound violated")
                            loss_rows.append(loss_row)
                        logical_outputs += 2*n
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        candidate_seconds += time.monotonic()-start
            print(f"[matched] {label} eta={eta}: {len(rows)} metric rows", flush=True)
            save(rows, loss_rows, {"reference_seconds": reference_seconds, "candidate_and_metric_seconds": candidate_seconds,
                                 "independent_gaussian_vectors": normal_vectors, "logical_output_evaluations": logical_outputs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["rotation", "matched"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rotation-samples", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--rotation-repeats", type=int, default=16)
    parser.add_argument("--matched-samples", type=int, default=512)
    parser.add_argument("--matched-repeats", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rotation_samples+[args.matched_samples]) < 2 or min(args.rotation_repeats, args.matched_repeats, args.block_size) < 1:
        parser.error("Sample counts >=2 and positive repeat/block counts required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT/"conditional_drifting/feature_metrics.py",
               ROOT/"Journal_version/metric_controls_protocol_20261009.md", ROOT/"conditional_drifting/metrics.py"]
    source_records = []
    for source in sources:
        content = source.read_bytes()
        (args.out_dir/source.name).write_bytes(content)
        source_records.append({"path": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "sources": source_records,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "torch": torch.__version__, "dtype": "float64", "sigma": SIGMA, "bandwidths": BANDWIDTHS,
                "hardware": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
                "anchors": [[.6, -.8]] if args.experiment == "rotation" else ANCHORS,
                "self_statistic": "Average of two self Gram matrices, NOT squared norm of pooled means",
                "cross_statistic": "Symmetrized independent-split product; negative values retained"}
    start = time.monotonic()
    def save(rows, losses, timing):
        manifest["timing_and_queries"] = timing | {"elapsed_seconds_including_saves": time.monotonic()-start}
        for filename, values in (("per_repeat.csv", rows), ("loss_gradients.csv", losses)):
            if values:
                with (args.out_dir/filename).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                    writer.writeheader()
                    writer.writerows(values)
        (args.out_dir/"results.json").write_text(json.dumps({"manifest": manifest, "rows": rows, "losses": losses}, indent=2, allow_nan=False)+"\n")
    if args.experiment == "rotation":
        run_rotation(args, save)
    else:
        run_matched(args, save)
    print(args.out_dir/"results.json", flush=True)


if __name__ == "__main__":
    main()
