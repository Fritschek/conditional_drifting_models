"""Train and evaluate capacity-controlled, differentiable channel generators."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer, FastImplant, FAMILIES, channel_spec, physical_channel, parameter_count
from conditional_drifting.baselines.fast_evaluation import (write_json, save, sync, distribution_metrics,
    timings, train_codec, evaluate_codec, common_codec_gradients)
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from scripts.run_local_gradient_fidelity import normalized_codebook


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT / "results/official_sources_20261010/W-Flow")
    p.add_argument("--device", default="cuda")
    p.add_argument("--channels", default="AWGN,SSPA")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--seeds", default="9101")
    p.add_argument("--parameters", type=int, default=60000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--train-seconds", type=float, default=0., help="Measured training-time cap; updates remains a hard cap")
    p.add_argument("--learning-rate", type=float, default=.0002)
    p.add_argument("--inference-weights", choices=("raw", "ema"), default="ema")
    p.add_argument("--family-config", type=Path, help="JSON per-family overrides for algorithm hyperparameters")
    p.add_argument("--flow-steps", default="1,2,4,8,16")
    p.add_argument("--ddim-steps", default="10,20,50,100")
    p.add_argument("--consistency-steps", default="1,2")
    p.add_argument("--eval-size", type=int, default=10000)
    p.add_argument("--anchors", type=int, default=128)
    p.add_argument("--anchor-samples", type=int, default=64)
    p.add_argument("--projections", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--timing-batches", default="1,500")
    p.add_argument("--timing-repeats", type=int, default=20)
    p.add_argument("--ae-updates", type=int, default=0, help="0 explicitly skips downstream training")
    p.add_argument("--ae-batch", type=int, default=500)
    p.add_argument("--ae-eval-samples", type=int, default=10000, help="Per message, not total trials")
    p.add_argument("--ebno-offsets", default="-4,-2,0,2")
    p.add_argument("--gradient-samples", type=int, default=512, help="Per message at the common analytic-trained codec; 0 skips")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p


def ints(value):
    return list(map(int, value.split(",")))


def source_files():
    files = list((ROOT / "conditional_drifting").rglob("*.py"))
    files += [Path(__file__), ROOT / "scripts/run_local_gradient_fidelity.py",
              ROOT / "scripts/run_symbolic_awgn_benchmark.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}


def ensure_manifest(path, manifest, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(f"Use --resume with the identical configuration: {path}")
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Configuration, source or runtime changed. Use a new output directory.")
    else:
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("Output directory is nonempty but has no matching manifest")
        write_json(path, manifest)


def train(trainer, path, save_every, resume):
    trace = []
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        trainer.restore(checkpoint["trainer"])
        trace = checkpoint["trace"]
    cfg = trainer.cfg
    while trainer.update < cfg.updates and (not cfg.train_seconds or trainer.train_seconds < cfg.train_seconds):
        sync(trainer.device)
        start = time.perf_counter()
        info = trainer.step()
        sync(trainer.device)
        trainer.train_seconds += time.perf_counter() - start
        info["cumulative_train_seconds"] = trainer.train_seconds
        trace.append(info)
        if trainer.update % save_every == 0:
            save(path, dict(trainer=trainer.state(), trace=trace))
            print(f"[train] {cfg.channel} {cfg.family} seed={cfg.seed} update={trainer.update} loss={info['loss']:.6g}", flush=True)
    save(path, dict(trainer=trainer.state(), trace=trace))


def main():
    args = parser().parse_args()
    positive = (args.eval_size, args.anchors, args.projections, args.eval_batch, args.timing_repeats,
                args.ae_batch, args.ae_eval_samples, args.save_every, args.threads)
    if min(positive) < 1 or args.anchor_samples < 2 or min(args.ae_updates, args.gradient_samples) < 0:
        raise ValueError("Invalid evaluation or checkpoint counts")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
    channels, families, seeds = args.channels.split(","), args.families.split(","), ints(args.seeds)
    if not set(channels) <= {"AWGN", "Rayleigh", "SSPA", "TDL"} or not set(families) <= set(FAMILIES):
        raise ValueError("Unknown channel or family")
    if len(set(channels)) != len(channels) or len(set(families)) != len(families) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate tasks")
    step_map = dict(flow=ints(args.flow_steps), diffusion=ints(args.ddim_steps), consistency=ints(args.consistency_steps))
    if any(min(v) < 1 or len(v) != len(set(v)) for v in step_map.values()):
        raise ValueError("Invalid NFE grid")
    if not set(step_map["consistency"]) <= {1, 2} or min(step_map["diffusion"]) < 2 or min(ints(args.timing_batches)) < 1:
        raise ValueError("Unsupported sampler or timing grid")
    overrides = {} if args.family_config is None else json.loads(args.family_config.read_text())
    if not set(overrides) <= set(FAMILIES):
        raise ValueError("Unknown override family")
    protected = {"channel", "seed", "family", "parameters", "batch_size", "updates", "train_seconds", "inference_weights"}
    configs = []
    for channel in channels:
        for seed in seeds:
            for family in families:
                extra = overrides.get(family, {})
                if protected & set(extra):
                    raise ValueError("Per-family overrides cannot change common comparison budgets")
                cfg = FastConfig(channel=channel, seed=seed, family=family, parameters=args.parameters,
                    batch_size=args.batch_size, updates=args.updates, train_seconds=args.train_seconds,
                    inference_weights=args.inference_weights,
                    **dict({"learning_rate": args.learning_rate}, **extra))
                cfg.validate()
                configs.append(cfg)
    official = load_sources(args.source_dir)
    wflow = load_official(args.wflow_source) if "sinkhorn" in families else None
    settings = {k: v for k, v in vars(args).items() if k not in ("resume", "out_dir", "source_dir", "wflow_source", "family_config")}
    manifest = dict(schema=1, settings=settings, configs=[asdict(c) for c in configs], overrides=overrides,
        channel_specs={c: channel_spec(c) for c in channels}, sources=official.records, adapter_files=source_files(),
        wflow=None if wflow is None else dict(commit=OFFICIAL_COMMIT, loss_sha256=sha256(args.wflow_source / "drift_loss_ot.py")),
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, threads=args.threads,
            device=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
            precision="float32", compile=False, cuda_graphs=False))
    # Normalize tuples and other JSON-compatible sequences before exact resume comparison.
    manifest = json.loads(json.dumps(manifest))
    ensure_manifest(args.out_dir / "manifest.json", manifest, args.resume)
    offsets = list(map(float, args.ebno_offsets.split(",")))
    rows = []
    for channel in channels:
        for seed in seeds:
            base = args.out_dir / channel / f"seed{seed}"
            reference = None
            if args.ae_updates:
                reference = train_codec(base / "analytic/codec.pt", channel,
                    lambda x: physical_channel(channel, x), args.device, seed + 200000,
                    args.ae_updates, args.ae_batch, args.save_every, args.resume)
                encoder, decoder, messages, seconds = reference
                result = dict(channel=channel, seed=seed, family="analytic", nfe=0, ae_train_seconds=seconds,
                    analytic_channel_evaluation=evaluate_codec(channel, encoder, decoder, messages,
                        args.ae_eval_samples, args.eval_batch, seed + 400000, offsets))
                write_json(base / "analytic/result.json", result)
                rows.append(result)
            for cfg in (c for c in configs if c.channel == channel and c.seed == seed):
                directory = base / cfg.family
                steps = step_map.get(cfg.family, [1])
                trainer = FastTrainer(cfg, args.device, official, wflow)
                train(trainer, directory / "generator.pt", args.save_every, args.resume)
                for nfe in steps:
                    result_path = directory / f"nfe{nfe}/result.json"
                    if result_path.exists() and args.resume:
                        rows.append(json.loads(result_path.read_text()))
                        continue
                    implant = FastImplant(trainer, nfe)
                    result = dict(channel=channel, seed=seed, family=cfg.family, nfe=nfe,
                        config=asdict(cfg), inference_parameters=parameter_count(implant.model),
                        critic_parameters=0 if trainer.critic is None else parameter_count(trainer.critic),
                        ema_parameters=parameter_count(trainer.ema),
                        target_parameters=0 if trainer.target is None else parameter_count(trainer.target),
                        train_seconds=trainer.train_seconds, completed_updates=trainer.update, counts=trainer.counts,
                        distribution=distribution_metrics(channel, implant, args.device, seed + 300000,
                            args.eval_size, args.anchors, args.anchor_samples, args.projections, args.eval_batch),
                        timing=timings(implant, trainer.n, args.device, seed + 600000,
                            ints(args.timing_batches), args.timing_repeats))
                    if reference is not None:
                        encoder, decoder, messages, _ = reference
                        result["analytic_codec_distribution"] = distribution_metrics(channel, implant, args.device,
                            seed + 500000, args.eval_size, args.anchors, args.anchor_samples, args.projections,
                            args.eval_batch, codes=normalized_codebook(encoder, messages))
                        if args.gradient_samples:
                            result["common_codec_gradients"] = common_codec_gradients(channel, implant,
                                encoder, decoder, messages, args.gradient_samples, args.eval_batch, seed + 800000)
                        encoder, decoder, messages, seconds = train_codec(result_path.parent / "codec.pt", channel,
                            implant, args.device, seed + 200000, args.ae_updates, args.ae_batch,
                            args.save_every, args.resume)
                        result["ae_train_seconds"] = seconds
                        result["analytic_channel_evaluation"] = evaluate_codec(channel, encoder, decoder, messages,
                            args.ae_eval_samples, args.eval_batch, seed + 400000, offsets)
                    write_json(result_path, result)
                    rows.append(result)
                    print(f"[done] {channel} {cfg.family} seed={seed} nfe={nfe}", flush=True)
    write_json(args.out_dir / "results.json", dict(rows=rows, num_rows=len(rows), manifest="manifest.json"))
    print(json.dumps(dict(result=str(args.out_dir / "results.json"), num_rows=len(rows))))


if __name__ == "__main__":
    main()
