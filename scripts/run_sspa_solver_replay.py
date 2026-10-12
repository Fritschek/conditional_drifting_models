#!/usr/bin/env python3
"""Matched frozen-cloud SSPA solver replay and cloned single-Adam-step checks."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.channels import sspa
from conditional_drifting.losses import _batched_sinkhorn_barycentric_projection
from conditional_drifting.model import ConditionalDriftingGenerator
from conditional_drifting.sspa_solver_replay import (
    barycenter, clipped_field, log_plan, residuals, statistics, variance_terms,
)
from conditional_drifting.sspa_trajectory import (
    POLICIES, TrajectoryConfig, field, isolated_rng, pooled_epsilon, restore_rng, rng_state, source_hashes,
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def model_from(state, device):
    c = state["config"]
    model = ConditionalDriftingGenerator(c["n"], c["n"], c["latent_dim"], c["hidden_dim"]).to(device)
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def panel_values(model, inputs, latent, anchors, samples):
    y = model(inputs, latent).double().reshape(anchors, samples, -1)
    centered = y - y.mean(1, keepdim=True)
    return dict(mean=y.mean(1), variance=centered.square().sum((1, 2)) / ((samples - 1) * y.shape[-1]))


def compare(left, right, mask):
    delta = left.double()[mask] - right.double()[mask]
    return statistics(delta.flatten(1).square().mean(1).sqrt())


def run_case(path, out, device, args):
    start = time.monotonic()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["source_hashes"] != source_hashes() or state["torch_version"] != str(torch.__version__):
        raise ValueError("Parent source or Torch mismatch")
    if state["device"] != str(device):
        raise ValueError("Batch reconstruction needs the original device type")
    cfg = TrajectoryConfig(**state["config"])
    model = model_from(state, device)
    restore_rng(state["rng"])
    before_rng = rng_state()
    batch = 32 if args.smoke else cfg.batch_size
    x = torch.randn(batch, cfg.n, device=device).repeat_interleave(cfg.samples, 0)
    with torch.no_grad():
        positive = sspa(x, cfg.noise_std, device)
        zg = torch.randn(len(x), cfg.latent_dim, device=device)
        generated = model(x, zg)
        zr = torch.randn(len(x), cfg.latent_dim, device=device)
        reference = model(x, zr)
    after_rng = rng_state()
    g, p, r = [v.reshape(batch, cfg.samples, cfg.n) for v in (generated, positive, reference)]
    epsilon = state["fixed_epsilon"] if cfg.policy == "fixed_common" else (
        pooled_epsilon(g, p, r, cfg.min_epsilon) if cfg.policy == "shared_adaptive" else None)
    production, terms = {}, {}
    for name, target in (("cross", p), ("self", r)):
        for steps in (10, 30, 100):
            center, info = _batched_sinkhorn_barycentric_projection(
                g, target, target, epsilon=epsilon, min_epsilon=cfg.min_epsilon,
                iterations=steps, return_diagnostics=True)
            production[name, steps] = dict(plan=info["coupling"], center=center,
                iterations=torch.full((batch,), steps, device=device),
                converged=torch.zeros(batch, dtype=torch.bool, device=device),
                residuals=residuals(info["coupling"]))
            if steps == 10:
                terms[name] = dict(epsilon=float(info["epsilon"]), cost32=info["cost"],
                                   kernel32=info["kernel"], floor_mask=info["kernel_floor_mask"])
    out.mkdir(parents=True, exist_ok=False)
    torch.save(cpu(dict(parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=state["config"],
        update=state["update"], batch_label="smoke_reduced_batch" if args.smoke else "post_checkpoint_reconstructed_batch",
        x=x, positive=p, generated=g, reference=r, source_latent=zg, reference_latent=zr,
        rng_before=before_rng, rng_after=after_rng, terms=terms)), out / "clouds.pt")

    # Every optimizer comparison uses the same fresh evaluation panel and latent draws.
    with isolated_rng(800301):
        eval_x = torch.randn(128, cfg.n, device=device).repeat_interleave(128, 0)
        eval_z = torch.randn(len(eval_x), cfg.latent_dim, device=device)
    before = panel_values(model, eval_x, eval_z, 128, 128)
    initial_vector = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    results, saved = {}, {}
    base_gradient, base_step, base_field = None, None, None

    def evaluate(name, cross, own, reference_solve=False, readout="production"):
        nonlocal base_gradient, base_step, base_field
        cp = cross.get("center") if readout == "production" else None
        sp = own.get("center") if readout == "production" else None
        cp = barycenter(cross["plan"], p, readout) if cp is None else cp
        sp = barycenter(own["plan"], r, readout) if sp is None else sp
        raw, drift = clipped_field(g, cp, sp, cfg.max_drift_norm)
        if name == "production10":
            expected, _ = field(generated, positive, reference, cfg, state["fixed_epsilon"])
            if not torch.equal(drift.flatten(0, 1), expected):
                raise AssertionError("Frozen baseline differs from trainer field")
            base_field = drift
        valid = cross["converged"] & own["converged"] if reference_solve else torch.ones(batch, dtype=torch.bool, device=device)
        if not all(torch.isfinite(v).all() for v in (raw, drift, cross["plan"], own["plan"])):
            raise ValueError(f"Nonfinite field/plan: {name}")
        record = dict(reference_solve=reference_solve, readout=readout, valid_anchors=int(valid.sum()),
                      anchors=batch, raw_norm=statistics(raw.norm(dim=-1)[valid]),
                      drift_norm=statistics(drift.norm(dim=-1)[valid]),
                      clipped_fraction=statistics((raw.norm(dim=-1)[valid] > cfg.max_drift_norm).double()),
                      field_rms_vs_production=compare(drift, base_field, valid),
                      cross_residuals={str(k): statistics(cross["residuals"][:, k]) for k in range(2)},
                      self_residuals={str(k): statistics(own["residuals"][:, k]) for k in range(2)},
                      cross_iterations=statistics(cross["iterations"]), self_iterations=statistics(own["iterations"]))
        for label, velocity in (("raw", raw), ("clipped", drift)):
            variance = variance_terms(g, velocity, cfg.drift_scale)
            record[label + "_particle_variance"] = {k: statistics(v[valid]) for k, v in variance.items()}
            torch.testing.assert_close(variance["actual"], variance["total"], atol=1e-10, rtol=1e-8)
        tensors = dict(cross=cross, self=own, raw_field=raw, clipped_field=drift, valid=valid)
        if valid.all():
            model.load_state_dict(state["model"])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
            model.zero_grad(set_to_none=True)
            prediction = model(x, zg)
            torch.testing.assert_close(prediction, generated, rtol=0., atol=0.)
            loss = F.mse_loss(prediction, (prediction + cfg.drift_scale * drift.flatten(0, 1)).detach())
            loss.backward()
            gradient = torch.cat([v.grad.flatten() for v in model.parameters()]).detach().clone()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step = torch.nn.utils.parameters_to_vector(model.parameters()).detach() - initial_vector
            after = panel_values(model, eval_x, eval_z, 128, 128)
            if base_gradient is None:
                base_gradient, base_step = gradient.clone(), step.clone()
            record["neural_update"] = dict(loss=loss.item(), gradient_norm=grad_norm.item(),
                gradient_relative_difference=((gradient - base_gradient).norm() / base_gradient.norm().clamp_min(1e-12)).item(),
                step_norm=step.norm().item(),
                step_relative_difference=((step - base_step).norm() / base_step.norm().clamp_min(1e-12)).item(),
                before_variance=before["variance"].mean().item(), after_variance=after["variance"].mean().item(),
                variance_change=statistics(after["variance"] - before["variance"]),
                mean_change=statistics((after["mean"] - before["mean"]).norm(dim=-1)))
            tensors.update(gradient=gradient, parameter_step=step, eval_before=before, eval_after=after)
        else:
            record["neural_update"] = None
            record["skip_reason"] = "Unresolved reference anchors; no partial-batch optimizer comparison"
        results[name] = record
        saved[name] = cpu(tensors)
        write_json(out / "partial.json", results)
        print(json.dumps(dict(case=out.name, method=name, valid_anchors=int(valid.sum()),
                             gradient_difference=(record["neural_update"] or {}).get("gradient_relative_difference"))), flush=True)

    for steps in (10, 30, 100):
        evaluate(f"production{steps}", production["cross", steps], production["self", steps])
    evaluate("production10_exact_row_readout", production["cross", 10], production["self", 10], readout="row")
    # Keep the final production readout unchanged while comparing internal scalings.
    for dtype, label in ((torch.float32, "f32"), (torch.float64, "f64")):
        for steps in (10, 30, 100):
            solves = [log_plan(terms[t]["kernel32"].to(dtype).log(), steps) for t in ("cross", "self")]
            evaluate(f"log_floor_{label}_{steps}", *solves)
        solves = [log_plan(terms[t]["kernel32"].to(dtype).log(), 10, denominator=1e-8) for t in ("cross", "self")]
        evaluate(f"log_floor_stabilized_{label}_10", *solves)
    references = {}
    for kind in ("floor_quantized", "floor_cost32", "unfloored_cost32", "unfloored_cost64"):
        solves = []
        for term, target in (("cross", p), ("self", r)):
            info = terms[term]
            scaled32 = info["cost32"] / info["epsilon"]
            scaled32 = scaled32 - scaled32.amin(-1, keepdim=True)
            if kind == "floor_quantized":
                kernel = info["kernel32"].double().log()
            elif kind == "floor_cost32":
                kernel = (-scaled32.double()).clamp_min(float(torch.tensor(1e-8).double().log()))
            elif kind == "unfloored_cost32":
                kernel = -scaled32.double()
            else:
                cost = .5 * (g.double()[:, :, None] - target.double()[:, None]).square().sum(-1)
                kernel = -(cost - cost.amin(-1, keepdim=True)) / info["epsilon"]
            solved = log_plan(kernel, args.reference_cap, tolerance=1e-8)
            solves.append(solved)
        references[kind] = solves
        evaluate("reference_" + kind, *solves, reference_solve=True)
    # Compare only on the intersection of converged reference anchors.
    comparisons = {}
    for left, right in (("production10", "reference_floor_quantized"),
                        ("production100", "reference_floor_quantized"),
                        ("reference_floor_quantized", "reference_floor_cost32"),
                        ("reference_floor_cost32", "reference_unfloored_cost32"),
                        ("reference_unfloored_cost32", "reference_unfloored_cost64")):
        a, b = saved[left], saved[right]
        mask = a["valid"] & b["valid"]
        item = dict(valid_anchors=int(mask.sum()), anchors=batch,
                    raw_field_rms=compare(a["raw_field"], b["raw_field"], mask),
                    clipped_field_rms=compare(a["clipped_field"], b["clipped_field"], mask))
        for term, target in (("cross", p.cpu()), ("self", r.cpu())):
            item[term + "_plan_rms"] = compare(a[term]["plan"], b[term]["plan"], mask)
            for readout in ("production", "row", "uniform"):
                item[term + "_" + readout + "_barycenter_rms"] = compare(
                    barycenter(a[term]["plan"], target, readout), barycenter(b[term]["plan"], target, readout), mask)
        comparisons[left + "__vs__" + right] = item
    torch.save(saved, out / "solves.pt")
    write_json(out / "result.json", dict(parent=str(path), parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        config=state["config"], update=state["update"], methods=results, comparisons=comparisons,
        elapsed_seconds=time.monotonic() - start,
        cross_epsilon=terms["cross"]["epsilon"], self_epsilon=terms["self"]["epsilon"],
        floor_fraction={t: statistics(terms[t]["floor_mask"].double().mean((1, 2))) for t in terms}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=Path("results/sspa_epsilon_trajectories_full_20261010"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-cap", type=int, default=50000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    selections = {"legacy_separate_adaptive": [30000, 80000, 90000, 390720],
                  "fixed_common": [30000, 100000, 110000, 390720],
                  "shared_adaptive": [30000, 90000, 100000, 390720]}
    if args.smoke:
        selections = {"fixed_common": [30000]}
    files = [Path(__file__), ROOT / "conditional_drifting/sspa_solver_replay.py",
             ROOT / "Journal_version/sspa_solver_replay_protocol_20261010.md"]
    write_json(args.out_dir / "manifest.json", dict(selections=selections, seed=9001, smoke=args.smoke,
        reference_cap=args.reference_cap, reference_tolerance=1e-8, device=str(device),
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        source_hashes=source_hashes(), added_source_hashes={str(p.relative_to(ROOT) if p.is_absolute() else p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
    for policy, updates in selections.items():
        for update in updates:
            path = args.suite_dir / policy / "seed9001" / f"update{update}.pt"
            out = args.out_dir / f"{policy}_u{update}"
            try:
                run_case(path, out, device, args)
            except BaseException as exc:
                write_json(args.out_dir / "failure.json", dict(case=out.name, type=type(exc).__name__, message=str(exc)))
                raise


if __name__ == "__main__":
    main()
