"""All-family SSPA development under one input law and explicit physical SNR."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.application_channels import SSPACoverageChannel
from conditional_drifting.baselines.application_generators import FastImplant
from conditional_drifting.baselines.fast_evaluation import write_json, distribution_metrics, timings
from conditional_drifting.baselines.official_sources import sha256
from scripts.application_local_common import sources, manifest, build, cached, record
from scripts.coverage_followup_common import finish_codec, paired_curves
from scripts.run_local_gradient_fidelity import normalized_codebook

GRIDS = dict(gaussian=(1,), sinkhorn=(1,), wgan=(1,), flow=(32, 128),
             diffusion=(20, 50), consistency_distilled=(1,))


def settings_for(profile, seed):
    settings = dict(seed=seed, parameters=60000, batch=1024, updates=40000,
        ae_updates=20000, ae_batch=500, ae_seeds=[400000+seed], save_every=1000,
        anchors=64, anchor_samples=256, eval_size=16384, projections=128,
        eval_samples=16384, eval_batch=2048, timing_repeats=10)
    if profile == "smoke":
        settings.update(parameters=2048, batch=128, updates=10, ae_updates=2,
            ae_batch=64, save_every=5, anchors=4, anchor_samples=8, eval_size=128,
            projections=4, eval_samples=8, eval_batch=128, timing_repeats=2)
    return settings


def run(args):
    torch.set_num_threads(2)
    settings = settings_for(args.profile, args.seed)
    channel = SSPACoverageChannel(input_law=args.input_law)
    official, wflow, provenance = sources(args, [Path(__file__).name, "coverage_followup_common.py",
                                                "run_sspa_input_coverage.py", "run_fast_channel_application.py",
                                                "run_consistency_warmstart_development.py", "run_fast_baseline_development.py"])
    settings.update(grids=GRIDS, input_law=args.input_law, profile=args.profile)
    if args.prepare_only:
        print(json.dumps(dict(settings=settings, channel=channel.description(), generator_families=6,
                              downstream_operating_points=9, sources_verified=True), indent=2))
        return
    manifest_sha = manifest(args, channel, settings, provenance)
    result_path = args.out_dir / "results.json"
    if args.resume and result_path.exists():
        data = json.loads(result_path.read_text())
        if data["manifest_sha256"] != manifest_sha:
            raise ValueError("Completed result belongs to a different manifest")
        for name, digest in data["checkpoint_hashes"].items():
            if sha256(args.out_dir / name) != digest:
                raise ValueError(f"Checkpoint changed: {name}")
        print("[complete] verified existing common-coverage results", flush=True)
        return
    generators, implants = [], {}
    for family, steps in GRIDS.items():
        trainer, cost = build(family, channel, settings, args, official, wflow)
        generators.append(dict(family=family, cost=cost))
        for nfe in steps:
            implants[f"{family}_{nfe}"] = FastImplant(trainer, nfe)
    evaluations = {}
    for name, sampler in implants.items():
        path = args.out_dir / name / "distribution.json"
        family = name.rsplit("_", 1)[0]
        stamp = dict(manifest=manifest_sha, generator=sha256(args.out_dir / family / "generator.pt"))
        row = cached(path, stamp, args.resume)
        if row is None:
            # Frozen physical channel equals this contract; retain old metric helper
            # only after independent calibration tests, without changing its labels.
            row = dict(distribution=distribution_metrics("SSPA", sampler, args.device, args.seed+600000,
                settings["eval_size"], settings["anchors"], settings["anchor_samples"],
                settings["projections"], settings["eval_batch"]),
                timing=timings(sampler, 8, args.device, args.seed+610000,
                               [1, settings["ae_batch"]], settings["timing_repeats"]))
            record(path, stamp, row)
        evaluations[name] = row
    blocks = []
    for seed in settings["ae_seeds"]:
        codecs = {}
        for name, sampler in dict(analytic=channel.sample, **implants).items():
            path = args.out_dir / name / f"ae{seed}.pt"
            codecs[name] = finish_codec(path, sampler, args.device, seed, settings, args.resume)
            print(f"[ae] {name} seed={seed} complete", flush=True)
        stamp = dict(manifest=manifest_sha, checkpoints={name: sha256(args.out_dir / name / f"ae{seed}.pt") for name in codecs})
        path = args.out_dir / f"curves_ae{seed}.json"
        curves = cached(path, stamp, args.resume)
        if curves is None:
            curves = paired_curves(codecs, settings["eval_samples"], settings["eval_batch"], seed+700000)
            for row in curves:
                row["legacy_nominal_ebno_db"] = row.pop("ebno_db")
                row["input_ebno_db"] = row["legacy_nominal_ebno_db"]+10*math.log10(2)
                row["real_noise_variance"] = (8/6)/(2*10**(row["input_ebno_db"]/10))
            record(path, stamp, curves)
        blocks.append(dict(ae_seed=seed, curves=curves, models={name: dict(train_seconds=c[3],
            codebook=normalized_codebook(c[0], c[2]).detach().cpu().tolist()) for name, c in codecs.items()}))
    checkpoints = {str(p.relative_to(args.out_dir)): sha256(p) for p in args.out_dir.rglob("*.pt")}
    write_json(result_path, dict(manifest_sha256=manifest_sha, generators=generators,
        evaluations=evaluations, blocks=blocks, checkpoint_hashes=checkpoints,
        qualification="Development comparison; one paired AE per generator seed; no final model selection"))
    print(f"[complete] {result_path}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "development"), default="development")
    p.add_argument("--input-law", choices=("broad", "normal"), default="broad")
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT / "results/official_sources_20261010/W-Flow")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
