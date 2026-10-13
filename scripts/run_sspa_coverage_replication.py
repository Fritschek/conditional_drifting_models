"""Fresh-generator replication of the frozen SSPA input-coverage intervention."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastImplant, physical_channel
from conditional_drifting.baselines.fast_evaluation import write_json
from conditional_drifting.baselines.official_sources import sha256
from conditional_drifting.sspa_trajectory import isolated_rng
from scripts.coverage_followup_common import (VARIANTS, read, validate_parent, runtime, file_hashes,
    cached, record, MeasuredCoverageTrainer, finish_generator, finish_codec, paired_curves)
from scripts.run_fast_channel_comparison import ensure_manifest
from scripts.run_local_gradient_fidelity import normalized_codebook, expected_loss
from scripts.run_sspa_input_coverage import load_codec, generator_evaluation, moments_at

PROTOCOL = ROOT / "Journal_version/coverage_followup_protocol_20261012.md"


def plan(profile, parent):
    settings = dict(parent["settings"])
    # The child owns these assignments; do not display the pilot seed as a setting.
    settings.pop("seed")
    settings.pop("device")
    seeds = [9201, 9202, 9203]
    if profile == "smoke":
        seeds = [19201]
        settings.update(updates=2, batch_size=128, parameters=1024, ae_updates=2, ae_batch=64,
            ae_eval_samples=8, eval_batch=128, eval_size=128, anchors=4, anchor_samples=8,
            projections=4, save_every=1)
    elif profile != "paper":
        raise ValueError("Unknown profile")
    elif any(settings[k] != v for k, v in dict(updates=40000, batch_size=1024, parameters=60000,
            ae_updates=20000, ae_batch=500, ae_eval_samples=16384).items()):
        raise ValueError("Parent does not match the frozen replication budget")
    return dict(profile=profile, generator_seeds=seeds,
        ae_seeds={str(s): [300000+10*s+i for i in range(3)] for s in seeds}, settings=settings,
        generator_tasks=4*len(seeds), surrogate_ae_tasks=12*len(seeds), analytic_ae_tasks=3*len(seeds),
        training_oracle_outputs=4*len(seeds)*settings["updates"]*settings["batch_size"],
        primary_endpoint="For each family and generator seed: mean over three paired AEs of broad-minus-normal SER at 8 dB",
        selection="Fixed final EMA; no selection/tuning using new outcomes",
        input_laws=dict(normal="N(0,I8)", broad="0.75 N(0,I8) + 0.25 spherical direction * Uniform(0,sqrt(512))"),
        expected_query_energy=dict(normal=8., broad=.75*8+.25*512/3))


def run(args):
    torch.set_num_threads(2)
    parent, official, wflow = validate_parent(args.parent, args.source_dir, args.wflow_source)
    task_plan = plan(args.profile, parent)
    settings = task_plan["settings"]
    parent_results = read(args.parent / "results.json")
    costs = {r["method"]: r["cost"] for r in parent_results["rows"] if r["method"] in ("gaussian", "sinkhorn")}
    parents = [args.parent / name for name in ("manifest.json", "results.json", "gaussian/codec.pt")]
    source_paths = [Path(__file__), ROOT / "scripts/coverage_followup_common.py",
                    ROOT / "scripts/run_sspa_input_coverage.py", PROTOCOL]
    manifest = dict(schema=1, study="sspa_coverage_replication", plan=task_plan, parent=parent,
        parent_files={str(p.resolve()): sha256(p) for p in parents}, sources=file_hashes(source_paths),
        runtime=runtime(args.device))
    if args.prepare_only:
        print(json.dumps(dict(validated=True, plan=task_plan, output=str(args.out_dir)), indent=2))
        return
    ensure_manifest(args.out_dir / "manifest.json", manifest, args.resume)
    manifest_sha = sha256(args.out_dir / "manifest.json")
    old_codec = load_codec(args.parent / "gaussian/codec.pt", args.device)
    historical_codes = normalized_codebook(old_codec[0], old_codec[2]).detach()
    eval_args = SimpleNamespace(device=args.device, anchors=settings["anchors"], anchor_samples=settings["anchor_samples"])
    generators, blocks = [], []
    for seed in task_plan["generator_seeds"]:
        base = args.out_dir / f"seed{seed}"
        implants = {"analytic": lambda x: physical_channel("SSPA", x)}
        generator_hashes = {}
        for variant in VARIANTS[1:]:
            family, law = variant.split("_")
            cfg = replace(FastConfig(**costs[family]["config"]), seed=seed, updates=settings["updates"],
                          batch_size=settings["batch_size"], parameters=settings["parameters"])
            trainer = MeasuredCoverageTrainer(cfg, args.device, official, wflow, law=law)
            checkpoint = base / variant / "generator.pt"
            finish_generator(trainer, checkpoint, settings["save_every"], args.resume)
            if trainer.counts["oracle_outputs"] != cfg.updates*cfg.batch_size:
                raise ValueError("Generator exposure mismatch")
            implant = implants[variant] = FastImplant(trainer)
            digest = generator_hashes[variant] = sha256(checkpoint)
            stamp = dict(manifest=manifest_sha, checkpoint=digest)
            path = base / variant / "generator_evaluation.json"
            result = cached(path, stamp, args.resume)
            if result is None:
                anchors = trainer.counts["input_anchors"]
                result = dict(generator_seed=seed, variant=variant, checkpoint_sha256=digest,
                    cost=dict(config=asdict(cfg), counts=trainer.counts, train_seconds=trainer.train_seconds,
                              broad_anchors=trainer.broad_anchors, input_energy_sum=trainer.input_energy_sum,
                              observed_mean_squared_input_norm=trainer.input_energy_sum/anchors,
                              maximum_training_input_norm=trainer.input_norm_max),
                    evaluation=generator_evaluation(implant, historical_codes, eval_args, settings))
                record(path, stamp, result)
            generators.append(result)
        for ae_seed in task_plan["ae_seeds"][str(seed)]:
            codecs, rows, checkpoint_hashes = {}, {}, {}
            for variant, implant in implants.items():
                path = base / variant / f"ae{ae_seed}" / "codec.pt"
                trained = codecs[variant] = finish_codec(path, implant, args.device, ae_seed, settings, args.resume)
                checkpoint_hashes[variant] = sha256(path)
                stamp = dict(manifest=manifest_sha, codec=checkpoint_hashes[variant],
                             generator=generator_hashes.get(variant, "analytic"))
                result_path = path.with_name("result.json")
                row = cached(result_path, stamp, args.resume)
                if row is None:
                    e, d, messages, seconds = trained
                    codes = normalized_codebook(e, messages).detach()
                    with isolated_rng(ae_seed+300000):
                        _, surrogate = expected_loss(codes, d, implant, settings["ae_eval_samples"],
                            settings["eval_batch"], ae_seed+300000, gradient=False)
                    row = dict(variant=variant, generator_seed=seed, ae_seed=ae_seed,
                        ae_train_seconds=seconds, codebook=codes.tolist(), codebook_norms=codes.norm(dim=1).tolist(),
                        mean_squared_norm=codes.square().sum(1).mean().item(), surrogate_nominal=surrogate,
                        own_moments=moments_at(implant, codes, settings["anchor_samples"], 979105))
                    record(result_path, stamp, row)
                rows[variant] = row
            block_path = base / f"paired_ae{ae_seed}.json"
            stamp = dict(manifest=manifest_sha, codecs=checkpoint_hashes, generators=generator_hashes)
            block = cached(block_path, stamp, args.resume)
            if block is None:
                curves = paired_curves(codecs, settings["ae_eval_samples"], settings["eval_batch"], ae_seed+200000)
                block = dict(generator_seed=seed, ae_seed=ae_seed, rows=rows, curves=curves)
                record(block_path, stamp, block)
            blocks.append(block)
            write_json(args.out_dir / "progress.json", dict(completed_generator_tasks=len(generators),
                completed_ae_tasks=5*len(blocks), expected_generator_tasks=task_plan["generator_tasks"],
                expected_ae_tasks=task_plan["surrogate_ae_tasks"]+task_plan["analytic_ae_tasks"]))
            print(f"[complete] generator_seed={seed} ae_seed={ae_seed} paired_variants=5", flush=True)
    write_json(args.out_dir / "results.json", dict(manifest_sha256=manifest_sha, generators=generators, blocks=blocks))


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
