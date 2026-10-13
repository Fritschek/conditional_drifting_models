"""Fixed-checkpoint Flow/DDIM step-count study; no generator or AE training."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer
from conditional_drifting.baselines.fast_evaluation import distribution_metrics, timings, save, write_json
from conditional_drifting.baselines.official_sources import sha256
from conditional_drifting.sspa_trajectory import isolated_rng
from scripts.coverage_followup_common import (read, validate_parent, runtime, file_hashes,
    cached, record, CommonNoiseImplant)
from scripts.run_fast_channel_comparison import ensure_manifest
from scripts.run_fast_channel_application import moments, compare_gradients, reference_gradients
from scripts.run_sspa_input_coverage import load_codec

PROTOCOL = ROOT / "Journal_version/coverage_followup_protocol_20261012.md"
GRIDS = dict(flow=(8, 32, 128), diffusion=(20, 50, 100))


def settings_for(profile, parent):
    settings = dict(parent["settings"])
    if profile == "smoke":
        settings.update(eval_size=128, anchors=4, anchor_samples=8, projections=4, eval_batch=128,
                        gradient_samples=8, gradient_repeats=2, surrogate_gradient_repeats=2, timing_repeats=2)
    return settings


def run(args):
    torch.set_num_threads(2)
    parent, official, wflow = validate_parent(args.parent, args.source_dir, args.wflow_source)
    settings = settings_for(args.profile, parent)
    paths = [args.parent / name for name in ("manifest.json", "results.json", "flow/generator.pt",
                                             "diffusion/generator.pt", "analytic/early.pt")]
    manifest = dict(schema=1, study="saved_sampler_panel", profile=args.profile, settings=settings,
        grids={k: list(v) for k, v in GRIDS.items()}, parent_files={str(p.resolve()): sha256(p) for p in paths},
        sources=file_hashes([Path(__file__), ROOT / "scripts/coverage_followup_common.py", PROTOCOL]),
        parent=parent, runtime=runtime(args.device),
        downstream_policy="No new AE training or NFE selection. Prior Flow-8/DDIM-20 remain historical reference settings.",
        noise_policy="Dedicated shared initial-latent stream, independent of internal DDIM RNG draws")
    # Validate saved checkpoints before creating an output directory or running evaluation.
    expected = {r["method"]: r for r in read(args.parent / "results.json")["rows"]}
    trainers = {}
    for family in GRIDS:
        state = torch.load(args.parent / family / "generator.pt", map_location="cpu", weights_only=False)["trainer"]
        cfg = FastConfig(**state["config"])
        if cfg.family != family or cfg.channel != "SSPA" or cfg.inference_weights != "ema":
            raise ValueError("Wrong saved generator identity or weights")
        if asdict(cfg) != expected[family]["cost"]["config"] or state["update"] != cfg.updates:
            raise ValueError("Checkpoint is not the fixed completed parent generator")
        trainer = FastTrainer(cfg, args.device, official, wflow)
        # Evaluation uses its own paired RNG streams, not device-specific training state.
        trainer.model.load_state_dict(state["model"])
        trainer.ema.load_state_dict(state["ema"])
        trainers[family] = trainer
    if args.prepare_only:
        print(json.dumps(dict(validated=True, grids=manifest["grids"], profile=args.profile,
                              training_tasks=0, output=str(args.out_dir)), indent=2))
        return
    ensure_manifest(args.out_dir / "manifest.json", manifest, args.resume)
    manifest_sha = sha256(args.out_dir / "manifest.json")
    eval_args = SimpleNamespace(**dict(settings, device=args.device))
    codec = load_codec(args.parent / "analytic/early.pt", args.device)
    ref_path = args.out_dir / "early_reference.pt"
    if ref_path.exists() and args.resume:
        references = torch.load(ref_path, map_location=args.device, weights_only=False)
    else:
        references = reference_gradients(*codec[:3], eval_args, 989101)
        save(ref_path, references)
    with isolated_rng(989102):
        x = torch.randn(settings["anchors"], 8, device=args.device)
        latent = torch.randn_like(x)
    rows, panel = [], {}
    for family, steps in GRIDS.items():
        trainer = trainers[family]
        for nfe in steps:
            path = args.out_dir / f"{family}_{nfe}.json"
            tensor_path = path.with_suffix(".pt")
            stamp = dict(manifest=manifest_sha, reference=sha256(ref_path))
            row = cached(path, stamp, args.resume)
            if row is not None:
                if sha256(tensor_path) != row["panel_sha256"]:
                    raise ValueError("Saved paired output/derivative panel changed")
                panel[f"{family}_{nfe}"] = torch.load(tensor_path, map_location="cpu", weights_only=True)
                rows.append(row)
                continue
            implant = CommonNoiseImplant(trainer, nfe)
            row = dict(family=family, nfe=nfe,
                distribution=distribution_metrics("SSPA", implant.reset(989103), args.device, 989104,
                    settings["eval_size"], settings["anchors"], settings["anchor_samples"],
                    settings["projections"], settings["eval_batch"]),
                moments=moments("SSPA", implant.reset(989105), args.device, 989106,
                    settings["anchors"], settings["anchor_samples"]),
                timing=timings(implant.reset(989107), 8, args.device, 989108, [1, settings["ae_batch"]],
                    settings["timing_repeats"]),
                early_codec_gradients=compare_gradients(implant.reset(989109), codec, references, eval_args, 989110))
            v = x.detach().requires_grad_(True)
            y = trainer.sample(v, nfe, noise=latent, model=implant.model)
            if not torch.isfinite(y).all():
                raise FloatingPointError("Nonfinite saved sampler outputs")
            # Full per-input Jacobian for the fixed latent, not a task-gradient surrogate.
            jacobian = torch.stack([torch.autograd.grad(y[:, j].sum(), v, retain_graph=j < 7)[0]
                                    for j in range(8)], 1)
            if not torch.isfinite(jacobian).all():
                raise FloatingPointError("Nonfinite pathwise Jacobian")
            tensors = dict(inputs=x.cpu(), latent=latent.cpu(), output=y.detach().cpu(), jacobian=jacobian.cpu())
            panel[f"{family}_{nfe}"] = tensors
            save(tensor_path, tensors)
            row["panel_sha256"] = sha256(tensor_path)
            record(path, stamp, row)
            rows.append(row)
            write_json(args.out_dir / "progress.json", dict(completed=len(rows), expected=6))
            print(f"[complete] {family} NFE={nfe}", flush=True)
    for row in rows:
        name = f"{row['family']}_{row['nfe']}"
        ref = panel[f"{row['family']}_{max(GRIDS[row['family']])}"]
        own = panel[name]
        row["relative_to_finest"] = dict(reference_nfe=max(GRIDS[row["family"]]),
            paired_output_rmse=(own["output"]-ref["output"]).square().mean().sqrt().item(),
            paired_jacobian_rmse=(own["jacobian"]-ref["jacobian"]).square().mean().sqrt().item(),
            qualification="Finite-grid self-consistency only; the finest sampler is not an oracle")
    write_json(args.out_dir / "results.json", dict(manifest_sha256=manifest_sha, rows=rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=ROOT / "results/fast_sspa_application_20261012")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT / "results/official_sources_20261010/W-Flow")
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
