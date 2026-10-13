"""Local whole-block OFDM development with clean or intermittent-interference fading."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.application_channels import OFDMChannel
from conditional_drifting.ofdm_application import (Constellation, PilotReceiver, train_receiver,
                                                   evaluate_receiver, block_metrics)
from conditional_drifting.baselines.application_generators import FastImplant
from conditional_drifting.baselines.fast_evaluation import write_json, timings
from conditional_drifting.baselines.official_sources import sha256
from scripts.application_local_common import sources, manifest, build, cached, record

GRIDS = dict(gaussian=(1,), mixture=(1,), sinkhorn=(1,), flow=(32,),
             consistency_distilled=(1,), wgan=(1,), diffusion=(20,))


def run(args):
    torch.set_num_threads(2)
    channel = OFDMChannel(activity=.1 if args.condition == "interference" else 0.)
    settings = dict(profile=args.profile, condition=args.condition, seed=args.seed, parameters=600000,
        batch=256, updates=4000, receiver_updates=5000, receiver_batch=256,
        eval_blocks=16384, eval_batch=512, anchors=16, anchor_samples=256,
        projections=64, save_every=500, timing_repeats=10, esnos=[0., 5., 10., 15., 20.],
        learned_constellation=args.learn_constellation)
    if args.profile == "smoke":
        settings.update(batch=128, updates=10, receiver_updates=2, receiver_batch=16,
            eval_blocks=64, eval_batch=32, anchors=2, anchor_samples=8, projections=4,
            save_every=5, timing_repeats=2, esnos=[10.])
    families = args.families.split(",")
    if len(set(families)) != len(families) or not set(families) <= set(GRIDS):
        raise ValueError("Invalid family panel")
    settings["grids"] = {k: GRIDS[k] for k in families}
    official, wflow, provenance = sources(args, [Path(__file__).name])
    if args.prepare_only:
        print(json.dumps(dict(settings=settings, channel=channel.description(), sources_verified=True), indent=2))
        return
    manifest_sha = manifest(args, channel, settings, provenance)
    result_path = args.out_dir / "results.json"
    if args.resume and result_path.exists():
        value = json.loads(result_path.read_text())
        if value["manifest_sha256"] != manifest_sha:
            raise ValueError("Result manifest changed")
        for name, digest in value["checkpoint_hashes"].items():
            if sha256(args.out_dir/name) != digest:
                raise ValueError(f"Checkpoint changed: {name}")
        print("[complete] verified existing OFDM results", flush=True)
        return
    samplers, generators = dict(analytic=channel.sample), []
    for family in families:
        trainer, cost = build(family, channel, settings, args, official, wflow)
        generators.append(dict(family=family, cost=cost))
        for steps in GRIDS[family]:
            samplers[f"{family}_{steps}"] = FastImplant(trainer, steps)
    rows = []
    for name, sampler in samplers.items():
        path = args.out_dir/name/"receiver.pt"
        receiver, constellation, cost = train_receiver(channel, sampler, path, args.device,
            args.seed+400000, settings["receiver_updates"], settings["receiver_batch"],
            settings["save_every"], args.resume, args.learn_constellation)
        stamp = dict(manifest=manifest_sha, receiver=sha256(path),
                     generator=None if name == "analytic" else sha256(args.out_dir/name.rsplit("_", 1)[0]/"generator.pt"))
        result = cached(path.with_suffix(".json"), stamp, args.resume)
        if result is None:
            result = dict(method=name, receiver=cost, metrics=block_metrics(channel, sampler, args.device,
                args.seed+500000, settings["anchors"], settings["anchor_samples"], settings["projections"]),
                curves=evaluate_receiver(channel, receiver, constellation, args.device, settings["eval_blocks"],
                    settings["eval_batch"], args.seed+600000, settings["esnos"]),
                timings=timings(sampler, channel.n, args.device, args.seed+700000,
                    [1, settings["receiver_batch"]], settings["timing_repeats"]))
            record(path.with_suffix(".json"), stamp, result)
        rows.append(result)
        print(f"[complete receiver] {name}", flush=True)
    conventional = []
    for snr in settings["esnos"]:
        receiver = PilotReceiver(channel, snr, args.device)
        conventional.extend(evaluate_receiver(channel, receiver, Constellation(False).to(args.device),
            args.device, settings["eval_blocks"], settings["eval_batch"], args.seed+600000, [snr]))
    hashes = {str(p.relative_to(args.out_dir)): sha256(p) for p in args.out_dir.rglob("*.pt")}
    write_json(result_path, dict(manifest_sha256=manifest_sha, generators=generators, rows=rows,
        conventional=conventional, checkpoint_hashes=hashes,
        qualification="Development only; fixed-QPSK conventional receiver has statistical covariance knowledge, no activity/realization CSI"))
    print(f"[complete] {result_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    p.add_argument("--condition", choices=("clean", "interference"), default="clean")
    p.add_argument("--seed", type=int, default=9401)
    p.add_argument("--families", default="gaussian,mixture,sinkhorn,flow,consistency_distilled")
    p.add_argument("--learn-constellation", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-dir", type=Path, default=ROOT/"results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT/"results/official_sources_20261010/W-Flow")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    run(p.parse_args())
