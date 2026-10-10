"""Audit production Sinkhorn numerics against small converged log-domain solves."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.channels import sspa
from conditional_drifting.losses import (
    _batched_sinkhorn_barycentric_projection, _sinkhorn_barycentric_projection,
)
from conditional_drifting.transport_reference import (
    envelope_divergence, log_sinkhorn, log_sinkhorn_cost, marginal_residuals,
    quadratic_cost, sinkhorn_divergence,
)

PROTOCOL = ROOT/"Journal_version/transport_reference_protocol_20261010.md"


def tensor_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: tensor_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_json(v) for v in value]
    return value


def cases():
    gen = torch.Generator().manual_seed(20261010)
    normal = lambda n, d: torch.randn(n, d, generator=gen, dtype=torch.float64)
    equal = normal(4, 2)
    return {
        "identical": (equal, equal.clone()),
        "unequal_4x7": (.4*normal(4, 2), .4*normal(7, 2)+.15),
        "near_duplicates": (torch.tensor([[0., 0.], [0., 0.], [1e-6, 0.], [.3, .1]], dtype=torch.float64),
                            torch.tensor([[0., 0.], [1e-6, 0.], [.2, .1], [.2, .1]], dtype=torch.float64)),
        "concentrated": (.01*normal(4, 2), .01*normal(4, 2)+.05),
        "broad": (5*normal(6, 3), 5*normal(5, 3)),
        "separated": (.1*normal(4, 2), .1*normal(4, 2)+4),
        "cluster_occupancy": (torch.tensor([[-2.], [-2.], [-2.], [2.]], dtype=torch.float64),
                              torch.tensor([[-2.], [2.], [2.], [2.]], dtype=torch.float64)),
        "collapsed": (torch.zeros(4, 1, dtype=torch.float64), torch.tensor([[-1.], [1.]], dtype=torch.float64)),
    }


def practical(source, target, epsilon, iterations, batched=False, mode="within"):
    fn = _batched_sinkhorn_barycentric_projection if batched else _sinkhorn_barycentric_projection
    return fn(source, target, target, epsilon=epsilon, min_epsilon=.001,
              iterations=iterations, epsilon_mode=mode, return_diagnostics=True)


def finite_parameter_gradients():
    gen = torch.Generator().manual_seed(20261011)
    latent = .3*torch.randn(5, 2, dtype=torch.float64, generator=gen)
    target = .4*torch.randn(7, 2, dtype=torch.float64, generator=gen)+.15
    parameters = torch.tensor([1., .1, -.15, .8, .1, -.2], dtype=torch.float64, requires_grad=True)
    mapping = lambda theta: latent@theta[:4].reshape(2, 2).T+theta[4:]
    epsilon, scale, h = .4, .3, 1e-5
    kwargs = {"tolerance": 1e-11}
    generated = mapping(parameters)
    surrogate = envelope_divergence(generated, target, epsilon, **kwargs)
    envelope_grad = torch.autograd.grad(surrogate, parameters)[0]
    finite = []
    for direction in torch.eye(len(parameters), dtype=torch.float64):
        plus = sinkhorn_divergence(mapping(parameters+h*direction), target, epsilon, **kwargs)
        minus = sinkhorn_divergence(mapping(parameters-h*direction), target, epsilon, **kwargs)
        finite.append((plus-minus)/(2*h))
    finite = torch.tensor(finite, dtype=torch.float64)
    generated = mapping(parameters)
    cross = log_sinkhorn(generated, target, epsilon, **kwargs)
    self_plan = log_sinkhorn(generated, generated, epsilon, **kwargs)
    drift = cross["barycenter"]-self_plan["barycenter"]
    mse = (generated-(generated+scale*drift).detach()).square().mean()
    detached_grad = torch.autograd.grad(mse, parameters)[0]
    factor = 2*scale/generated.shape[1]
    independent_latent = .3*torch.randn(5, 2, dtype=torch.float64, generator=gen)
    independent = independent_latent@parameters[:4].detach().reshape(2, 2).T+parameters[4:].detach()
    other = log_sinkhorn(generated, independent, epsilon, **kwargs)
    independent_drift = cross["barycenter"]-other["barycenter"]
    numerical = []
    for iterations in (10, 30, 100):
        x = mapping(parameters)
        cp, _ = practical(x, target, epsilon, iterations)
        sp, _ = practical(x, x, epsilon, iterations)
        loss = (x-(x+scale*(cp-sp)).detach()).square().mean()
        gradient = torch.autograd.grad(loss, parameters)[0]
        numerical.append({"iterations": iterations, "gradient": gradient,
                          "max_absolute_error_vs_exact_scaled_gradient": (gradient-factor*envelope_grad).abs().max().item()})
    passed = torch.allclose(finite, envelope_grad, rtol=1e-4, atol=1e-6)
    passed = passed and torch.allclose(detached_grad, factor*envelope_grad, rtol=1e-6, atol=1e-8)
    return tensor_json({"passed": bool(passed), "epsilon": epsilon, "drift_scale": scale,
                       "coordinate_mean_mse_factor": factor, "finite_difference_step": h,
                       "finite_difference": finite, "envelope_gradient": envelope_grad,
                       "detached_mse_gradient": detached_grad,
                       "finite_difference_max_absolute_error": (finite-envelope_grad).abs().max().item(),
                       "mse_identity_max_absolute_error": (detached_grad-factor*envelope_grad).abs().max().item(),
                       "independent_reference_drift_difference": (independent_drift-drift).norm().item(),
                       "production_gradients": numerical,
                       "note": "Independent reference is a different field, not the gradient of this same-batch objective."})


def equilibrium_checks():
    q = torch.zeros(1, 1, dtype=torch.float64)
    p = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    rows = []
    for epsilon in (.001, .05, 1.):
        cross, self_plan = log_sinkhorn(q, p, epsilon), log_sinkhorn(q, q, epsilon)
        actual = sinkhorn_divergence(q, p, epsilon)
        t = 1/epsilon
        expected = .5*epsilon*(t+math.log1p(math.exp(-2*t))-math.log(2))
        row = {"epsilon": epsilon, "divergence": actual, "analytic_divergence": expected,
               "velocity_norm": (cross["barycenter"]-self_plan["barycenter"]).norm().item(),
               "same_law_divergence": sinkhorn_divergence(p, p, epsilon)}
        row["passed"] = abs(actual-expected) < 1e-9 and row["velocity_norm"] < 1e-12 and abs(row["same_law_divergence"]) < 1e-12
        rows.append(row)
    return rows


def projection_panel(save):
    rows, plans = [], []
    for name, (x, y) in cases().items():
        for epsilon in (.001, .05, 1.):
            start = time.monotonic()
            ref = log_sinkhorn(x, y, epsilon, require_convergence=False)
            ref_seconds = time.monotonic()-start
            _, diagnostic = practical(x, y, epsilon, 10)
            modified_cost = -epsilon*diagnostic["kernel"].double().log()
            floor_reference = log_sinkhorn_cost(modified_cost, epsilon, require_convergence=False)
            floor_center = len(x)*floor_reference["coupling"]@y
            plans.append(tensor_json({"case": name, "epsilon": epsilon, "source": x, "target": y,
                                      "reference": ref, "floored_kernel_reference": floor_reference}))
            for iterations in (10, 30, 100):
                center, diag = practical(x, y, epsilon, iterations)
                batched, _ = practical(x[None], y[None], epsilon, iterations, batched=True)
                default = _sinkhorn_barycentric_projection(x, y, y, epsilon=epsilon, min_epsilon=.001, iterations=iterations)
                if not torch.equal(default, center):
                    raise AssertionError("Diagnostic return changed default projection")
                raw_res = marginal_residuals(diag["coupling"])
                row_res = marginal_residuals(diag["row_weights"]/len(x))
                row = {"case": name, "epsilon": epsilon, "iterations": iterations,
                       "reference_converged": ref["converged"], "reference_iterations": ref["iterations"],
                       "reference_seconds": ref_seconds, "reference_row_relative": ref["row_relative"],
                       "reference_column_relative": ref["column_relative"],
                       "raw_row_relative": raw_res["row_relative"], "raw_column_relative": raw_res["column_relative"],
                       "normalized_row_relative": row_res["row_relative"], "normalized_column_relative": row_res["column_relative"],
                       "floor_fraction": diag["kernel_floor_mask"].float().mean().item(),
                       "batched_unbatched_max_difference": (center-batched[0]).abs().max().item(),
                       "barycenter_rms_error": (center.double()-ref["barycenter"]).square().mean().sqrt().item() if ref["converged"] else None,
                       "coupling_l1_error": (diag["coupling"].double()-ref["coupling"]).abs().sum().item() if ref["converged"] else None,
                       "floor_reference_converged": floor_reference["converged"],
                       "floor_reference_row_relative": floor_reference["row_relative"],
                       "floor_reference_column_relative": floor_reference["column_relative"],
                       "floor_only_barycenter_rms_error": (floor_center-ref["barycenter"]).square().mean().sqrt().item()
                           if ref["converged"] and floor_reference["converged"] else None,
                       "practical_vs_floor_reference_rms": (center.double()-floor_center).square().mean().sqrt().item()
                           if floor_reference["converged"] else None}
                rows.append(row)
                plans.append(tensor_json({"case": name, "epsilon": epsilon, "iterations": iterations,
                                          "center": center, "production": diag}))
            print(f"[reference] {name} epsilon={epsilon}: converged={ref['converged']} iterations={ref['iterations']}", flush=True)
            save(rows, plans)
    return rows, plans


def sspa_panel():
    gen = torch.Generator().manual_seed(20261012)
    normal = lambda *shape: torch.randn(*shape, dtype=torch.float64, generator=gen)
    anchors = normal(8, 8)
    mean = sspa(anchors, 0., torch.device("cpu"))
    sigma = .3250531435997416/math.sqrt(2)
    # Independent analytic-only pilot sets the fixed common epsilon.
    pilot1 = mean[:, None]+sigma*normal(8, 4, 8)
    pilot2 = mean[:, None]+sigma*normal(8, 4, 8)
    costs = .5*(pilot1[:, :, None]-pilot2[:, None]).square().sum(-1)
    fixed = costs[costs > 0].median().item()
    perturbation = .2*normal(8, 1, 8)
    x = mean[:, None]+perturbation+1.4*sigma*normal(8, 4, 8)
    y = mean[:, None]+sigma*normal(8, 4, 8)
    r = mean[:, None]+perturbation+1.4*sigma*normal(8, 4, 8)
    cross_cost = .5*(x[:, :, None]-y[:, None]).square().sum(-1)
    self_cost = .5*(x[:, :, None]-r[:, None]).square().sum(-1)
    pooled = torch.cat((cross_cost.flatten(), self_cost.flatten()))
    shared = max(.001, pooled[pooled > 0].median().item())
    rows, details, reference_drifts = [], [], {}
    for policy, epsilon, mode in (("fixed_common", fixed, "within"),
                                  ("shared_adaptive", shared, "within"),
                                  ("legacy_separate_adaptive", None, "within"),
                                  ("historical_global_scale", None, "global")):
        _, cross_diag = practical(x, y, epsilon, 10, True, mode)
        _, self_diag = practical(x, r, epsilon, 10, True, mode)
        refs = [(log_sinkhorn(a, b, cross_diag["epsilon"]), log_sinkhorn(a, c, self_diag["epsilon"]))
                for a, b, c in zip(x, y, r)]
        ref_drift = torch.stack([a["barycenter"]-b["barycenter"] for a, b in refs])
        reference_drifts[policy] = ref_drift
        for iterations in (10, 30, 100):
            pc, pd = practical(x, y, epsilon, iterations, True, mode)
            rc, rd = practical(x, r, epsilon, iterations, True, mode)
            drift = pc-rc
            row = {"policy": policy, "iterations": iterations, "cross_epsilon": pd["epsilon"],
                   "self_epsilon": rd["epsilon"], "reference_drift_rms": ref_drift.square().mean().sqrt().item(),
                   "drift_rms_error": (drift.double()-ref_drift).square().mean().sqrt().item(),
                   "max_raw_row_relative": max(marginal_residuals(d["coupling"])["row_relative"] for d in (pd, rd)),
                   "max_raw_column_relative": max(marginal_residuals(d["coupling"])["column_relative"] for d in (pd, rd)),
                   "max_normalized_column_relative": max(marginal_residuals(d["row_weights"]/4)["column_relative"] for d in (pd, rd)),
                   "floor_fraction": (pd["kernel_floor_mask"].float().mean().item()+rd["kernel_floor_mask"].float().mean().item())/2,
                   "reference_difference_from_fixed_common_rms": (ref_drift-reference_drifts["fixed_common"]).square().mean().sqrt().item()}
            rows.append(row)
            details.append(tensor_json({"policy": policy, "iterations": iterations, "cross": pd, "self": rd, "drift": drift}))
        details.append(tensor_json({"policy": policy, "references": refs, "reference_drift": ref_drift}))
    return {"rows": rows, "inputs": tensor_json({"anchors": anchors, "source": x, "target": y,
             "independent_reference": r, "pilot1": pilot1, "pilot2": pilot2,
             "fixed_epsilon": fixed, "shared_epsilon": shared, "component_std": sigma}), "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    files = [Path(__file__).resolve(), PROTOCOL, ROOT/"conditional_drifting/transport_reference.py",
             ROOT/"conditional_drifting/losses.py", ROOT/"conditional_drifting/channels.py"]
    sources = []
    for path in files:
        raw = path.read_bytes()
        (out/path.name).write_bytes(raw)
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"sources": sources, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "torch": torch.__version__, "device": "CPU", "threads": 1, "reference_dtype": "float64",
                "production_dtype": "float32", "tolerance": 1e-8, "max_iterations": 50000, "status": "running"}
    started = time.monotonic()
    def save(rows, plans):
        manifest["elapsed_seconds"] = time.monotonic()-started
        (out/"projection_records.json").write_text(json.dumps({"manifest": manifest, "rows": rows, "plans": plans}, indent=2, allow_nan=False)+"\n")
    rows, plans = projection_panel(save)
    gradients = finite_parameter_gradients()
    equilibrium = equilibrium_checks()
    sspa_result = sspa_panel()
    if not gradients["passed"] or not all(r["passed"] for r in equilibrium):
        raise AssertionError("Reference mathematical check failed")
    for filename, payload in (("gradient_checks.json", gradients), ("equilibrium_checks.json", equilibrium), ("sspa_policy_records.json", sspa_result)):
        (out/filename).write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    for filename, values in (("projection_summary.csv", rows), ("sspa_policy_summary.csv", sspa_result["rows"])):
        with (out/filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    manifest["status"] = "complete"
    save(rows, plans)
    checks = {"projection_records": len(rows), "sspa_policy_records": len(sspa_result["rows"]),
              "unconverged_original_reference_cases": sorted({(r["case"], r["epsilon"]) for r in rows if not r["reference_converged"]}),
              "unconverged_floored_reference_cases": sorted({(r["case"], r["epsilon"]) for r in rows if not r["floor_reference_converged"]}),
              "gradient_checks_passed": gradients["passed"], "equilibrium_checks_passed": all(r["passed"] for r in equilibrium),
              "default_output_unchanged": True, "elapsed_seconds": manifest["elapsed_seconds"]}
    (out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
