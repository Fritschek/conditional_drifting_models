#!/usr/bin/env python3
"""Repeated, paired single-update comparisons at healthy SSPA checkpoints."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.channels import sspa
from conditional_drifting.sspa_solver_replay import variance_terms
from conditional_drifting.sspa_trajectory import TrajectoryConfig, field, isolated_rng, source_hashes
from conditional_drifting.sspa_update_screen import (
    METHODS, OFFICIAL_COMMIT, clone_adam, coordinate_scale, load_official, official_field, panel_moments,
)
from run_sspa_solver_replay import model_from, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    t = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(t).all():
        raise ValueError("Nonfinite summary")
    return dict(n=len(t), mean=t.mean().item(), sd=t.std().item(), se=(t.std() / len(t) ** .5).item())


def run_case(parent, out, device, official, repeats):
    start = time.monotonic()
    parent_hash = sha(parent)
    state = torch.load(parent, map_location="cpu", weights_only=False)
    if state["source_hashes"] != source_hashes() or state["torch_version"] != str(torch.__version__):
        raise ValueError("Checkpoint source/Torch mismatch")
    cfg = TrajectoryConfig(**state["config"])
    if cfg.policy != "fixed_common" or state["update"] != 100000:
        raise ValueError("Expected fixed-common 100k checkpoint")
    model = model_from(state, device)
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    epsilon = state["fixed_epsilon"]
    out.mkdir()
    with isolated_rng(800401):
        eval_x = torch.randn(128, cfg.n, device=device)
        truth_mean = sspa(eval_x, 0., device)
        eval_z = torch.randn(128 * 256, cfg.latent_dim, device=device)
    eval_inputs = eval_x.repeat_interleave(256, 0)
    before = panel_moments(model(eval_inputs, eval_z).reshape(128, 256, cfg.n), truth_mean)
    variance_gradient = torch.cat([v.flatten() for v in torch.autograd.grad(before["variance"], model.parameters())]).detach().double()
    before = {key: value.detach() for key, value in before.items()}
    true_variance = cfg.noise_std ** 2 / 2
    records, tensors = [], {name: dict(gradients=[], steps=[]) for name, *_ in METHODS}
    for repeat in range(repeats):
        model.load_state_dict(state["model"])
        with isolated_rng(810000 + repeat), torch.no_grad():
            anchors = torch.randn(4096, cfg.n, device=device)
            inputs = anchors.repeat_interleave(64, 0)
            positive = sspa(inputs, cfg.noise_std, device).reshape(4096, 64, cfg.n)
            zg = torch.randn(4096, 64, cfg.latent_dim, device=device)
            zr = torch.randn_like(zg)
            generated = model(inputs, zg.flatten(0, 1)).reshape(4096, 64, cfg.n)
            reference = model(inputs, zr.flatten(0, 1)).reshape_as(generated)
        trial = []
        for name, batch, particles, mode in METHODS:
            model.load_state_dict(state["model"])
            model.zero_grad(set_to_none=True)
            optimizer = clone_adam(model, state["optimizer"], mode == "zero_first_moment")
            x = anchors[:batch].repeat_interleave(particles, 0)
            prediction = model(x, zg[:batch, :particles].reshape(-1, cfg.latent_dim)).reshape(batch, particles, cfg.n)
            torch.testing.assert_close(prediction.detach(), generated[:batch, :particles], atol=2e-6, rtol=2e-6)
            g, p, r = prediction.detach(), positive[:batch, :particles], reference[:batch, :particles]
            scale, physical_epsilon = g.new_tensor(1.), epsilon
            if mode == "production":
                drift, info = field(prediction.flatten(0, 1), p.reshape(-1, cfg.n), r.reshape(-1, cfg.n),
                                    replace(cfg, batch_size=batch, samples=particles), epsilon, diagnostics=True)
                drift = drift.reshape_as(prediction)
                max_residual = max(info[k] for k in ("cross_row_relative", "cross_column_relative",
                                                      "self_row_relative", "self_column_relative"))
                loss = F.mse_loss(prediction, (prediction + drift).detach())
                field_clip_fraction = info["drift_clip_fraction"]
            elif mode.startswith("normalized"):
                scale = coordinate_scale(g, p, r)
                R = epsilon / (scale.square().item() * cfg.n) if mode == "normalized_fixed" else .05
                physical_epsilon = R * cfg.n * scale.square().item()
                losses, info = official.drift_loss_ot(prediction, p, r, R_list=(R,), sinkhorn_num_iter=10,
                    disable_diag_mask=True, use_quadratic_cost=True, batch_sinkhorn=False)
                torch.testing.assert_close((info["scale"] / cfg.n ** .5).clamp_min(1e-3), scale)
                loss = losses.mean()
                output_gradient, = torch.autograd.grad(loss, prediction, retain_graph=True)
                drift = (-.5 * prediction.numel() * scale.square() * output_gradient).detach()
                _, _, errors = official_field(official, g / scale, p / scale, r / scale, R * cfg.n)
                max_residual = errors.max().item()
                field_clip_fraction = 0.
            else:
                raw, drift, errors = official_field(official, g, p, r, epsilon)
                max_residual = errors.max().item()
                field_clip_fraction = (raw.norm(dim=-1) > 2.).float().mean().item()
                loss = F.mse_loss(prediction, (prediction + drift).detach())
            loss.backward()
            gradient = torch.cat([v.grad.flatten() for v in model.parameters()]).detach().clone()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step = torch.nn.utils.parameters_to_vector(model.parameters()).detach() - initial
            with torch.no_grad():
                after = panel_moments(model(eval_inputs, eval_z).reshape(128, 256, cfg.n), truth_mean)
            linear = torch.dot(variance_gradient, step.double()).item()
            delta = (after["variance"] - before["variance"]).item()
            particle = variance_terms(g, drift)
            torch.testing.assert_close(particle["total"], particle["actual"], atol=1e-10, rtol=1e-8)
            record = dict(method=name, repeat=repeat, batch_seed=810000 + repeat, anchors=batch,
                samples=particles, outputs_per_cloud=batch * particles, physical_epsilon=physical_epsilon,
                coordinate_scale=scale.item(), loss=loss.item(), max_marginal_relative_error=max_residual,
                field_clip_fraction=field_clip_fraction, gradient_norm=norm.item(), gradient_clipped=bool(norm > 1),
                step_norm=step.norm().item(), variance_change=delta, variance_linear_step=linear,
                variance_remainder=delta - linear,
                variance_negative_gradient_dot=-torch.dot(variance_gradient, gradient.double()).item(),
                step_negative_gradient_cosine=F.cosine_similarity(step.double(), -gradient.double(), dim=0).item(),
                mean_mse_change=(after["mean_mse"] - before["mean_mse"]).item(),
                after_variance=after["variance"].item(),
                squared_variance_bias_change=((after["variance"] - true_variance).square()
                                              - (before["variance"] - true_variance).square()).item(),
                particle_variance={key: value.mean().item() for key, value in particle.items()})
            trial.append(record)
            tensors[name]["gradients"].append(gradient.cpu())
            tensors[name]["steps"].append(step.cpu())
        records.extend(trial)
        write_json(out / f"repeat{repeat:02d}.json", trial)
        print(json.dumps(dict(seed=cfg.seed, repeat=repeat, variance_changes={r["method"]: r["variance_change"] for r in trial})), flush=True)
    aggregate = {}
    metrics = ("variance_change", "variance_linear_step", "variance_remainder", "mean_mse_change",
               "squared_variance_bias_change", "gradient_norm", "step_norm", "max_marginal_relative_error",
               "variance_negative_gradient_dot", "step_negative_gradient_cosine")
    baseline = [r for r in records if r["method"] == "log4"]
    for name, *_ in METHODS:
        selected = [r for r in records if r["method"] == name]
        aggregate[name] = {metric: summary([r[metric] for r in selected]) for metric in metrics}
        aggregate[name]["paired_variance_change_vs_log4"] = summary([
            r["variance_change"] - base["variance_change"] for r, base in zip(selected, baseline)])
        for kind in ("gradients", "steps"):
            values = torch.stack(tensors[name][kind])
            tensors[name][kind] = values
            average = values.double().mean(0)
            aggregate[name][kind] = dict(mean_vector_norm=average.norm().item(),
                noise_rms=(values.double() - average).square().sum(-1).mean().sqrt().item())
    torch.save(dict(methods=tensors, variance_gradient=variance_gradient.cpu(), eval_inputs=eval_x.cpu(),
        eval_latents=eval_z.cpu(), before={k: v.cpu() for k, v in before.items()}), out / "updates.pt")
    if sha(parent) != parent_hash:
        raise ValueError("Parent checkpoint changed")
    write_json(out / "result.json", dict(parent=str(parent), parent_sha256=parent_hash,
        config=state["config"], update=100000, repeats=repeats, true_variance=true_variance,
        before_variance=before["variance"].item(), before_mean_mse=before["mean_mse"].item(),
        master_outputs_per_cloud=4096 * 64 * repeats, methods=aggregate, elapsed_seconds=time.monotonic() - start,
        updates_sha256=sha(out / "updates.pt")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--official-repo", type=Path, default=Path("results/official_sources_20261010/W-Flow"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    official = load_official(args.official_repo)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    seeds, repeats = ([9001], 2) if args.smoke else ([9001, 9002, 9003], 32)
    files = [Path(__file__), ROOT / "conditional_drifting/sspa_update_screen.py",
             ROOT / "conditional_drifting/sspa_solver_replay.py", ROOT / "scripts/run_sspa_solver_replay.py",
             ROOT / "Journal_version/sspa_update_screen_protocol_20261010.md", args.official_repo / "drift_loss_ot.py"]
    write_json(args.out_dir / "manifest.json", dict(seeds=seeds, repeats=repeats, smoke=args.smoke,
        methods=METHODS, official_commit=OFFICIAL_COMMIT, source_hashes=source_hashes(),
        additional_source_hashes={str(p.relative_to(ROOT) if p.is_absolute() else p): sha(p) for p in files},
        torch_version=str(torch.__version__), device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    for seed in seeds:
        parent = ROOT / f"results/sspa_epsilon_trajectories_full_20261010/fixed_common/seed{seed}/update100000.pt"
        try:
            run_case(parent, args.out_dir / f"seed{seed}", device, official, repeats)
        except BaseException as exc:
            write_json(args.out_dir / "failure.json", dict(seed=seed, type=type(exc).__name__, message=str(exc)))
            raise


if __name__ == "__main__":
    main()
