#!/usr/bin/env python3
"""Audit paired checkpoint provenance and the shared-epsilon initial solver error."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from conditional_drifting.channels import sspa
from conditional_drifting.losses import _batched_sinkhorn_barycentric_projection
from conditional_drifting.sspa_trajectory import (SSPATrajectory, TrajectoryConfig,
    isolated_rng, pooled_epsilon, source_hashes)
from conditional_drifting.transport_reference import log_sinkhorn, log_sinkhorn_cost


def digest(tensors):
    sha = hashlib.sha256()
    for t in tensors:
        sha.update(t.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("Live training source differs from archived experiment")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks, hashes, probe_rows, clouds = [], {}, [], {}
    for seed in manifest["seeds"]:
        initial_hashes, final_rng_hashes = [], []
        for policy in manifest["policies"]:
            folder = args.suite_dir / policy / f"seed{seed}"
            initial = torch.load(folder / "update0.pt", weights_only=False, map_location="cpu")
            final = torch.load(folder / f"update{manifest['updates']}.pt", weights_only=False, map_location="cpu")
            if initial["source_hashes"] != source_hashes() or final["source_hashes"] != source_hashes():
                raise ValueError("Checkpoint source mismatch")
            initial_hashes.append(digest(initial["model"].values()))
            final_rng_hashes.append(digest([final["rng"]["cpu"], *final["rng"]["cuda"]]))
            for path in folder.glob("update*.pt"):
                hashes[str(path.relative_to(args.suite_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(set(initial_hashes)) != 1 or len(set(final_rng_hashes)) != 1:
            raise ValueError("Policies do not have matched initialization / RNG consumption")
        checks.append(dict(seed=seed, initial_model_sha256=initial_hashes[0], final_torch_rng_sha256=final_rng_hashes[0]))
        # Post-hoc numerical probe, never used to alter the frozen trajectories.
        cfg = TrajectoryConfig(seed=seed, policy="shared_adaptive")
        trainer = SSPATrajectory(cfg, "cuda")
        trainer.load(args.suite_dir / "shared_adaptive" / f"seed{seed}" / "update0.pt")
        with torch.no_grad(), isolated_rng(cfg.validation_seed):
            x = torch.randn(cfg.validation_anchors, cfg.n, device="cuda")
        with torch.no_grad(), isolated_rng(cfg.validation_seed + 7):
            x = x.repeat_interleave(cfg.samples, dim=0)
            p = sspa(x, cfg.noise_std, trainer.device).reshape(-1, cfg.samples, cfg.n)
            g = trainer.model(x).reshape_as(p)
            r = trainer.model(x).reshape_as(p)
            epsilon = pooled_epsilon(g, p, r, cfg.min_epsilon)
            clouds[str(seed)] = dict(generated=g.cpu(), positive=p.cpu(), reference=r.cpu(), epsilon=epsilon)
            for iterations in (10, 30, 100):
                center, info = _batched_sinkhorn_barycentric_projection(g[:8], p[:8], p[:8],
                    epsilon=epsilon, min_epsilon=cfg.min_epsilon, iterations=iterations, return_diagnostics=True)
                for anchor in range(8):
                    target = p[anchor].cpu().double()
                    ref = log_sinkhorn(g[anchor].cpu(), target, epsilon, require_convergence=False)
                    modified_cost = -epsilon * info["kernel"][anchor].cpu().double().log()
                    floored = log_sinkhorn_cost(modified_cost, epsilon, require_convergence=False)
                    exact_floored_center = cfg.samples * floored["coupling"] @ target
                    row = dict(seed=seed, anchor=anchor, iterations=iterations, epsilon=epsilon,
                        reference_converged=ref["converged"], floored_reference_converged=floored["converged"],
                        floor_fraction=info["kernel_floor_mask"][anchor].float().mean().item(),
                        practical_rms=None, converged_floored_rms=None)
                    if ref["converged"]:
                        row["practical_rms"] = (center[anchor].cpu() - ref["barycenter"]).square().mean().sqrt().item()
                        if floored["converged"]:
                            row["converged_floored_rms"] = (exact_floored_center - ref["barycenter"]).square().mean().sqrt().item()
                    probe_rows.append(row)
    torch.save(clouds, args.out_dir / "shared_initial_clouds.pt")
    result = dict(paired_checkpoints=checks, checkpoint_sha256=hashes, shared_initial_probe=probe_rows)
    (args.out_dir / "checkpoint_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(matched_seeds=len(checks), checkpoints=len(hashes), numerical_probe_rows=len(probe_rows))))


if __name__ == "__main__":
    main()
