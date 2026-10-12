"""Bounded AWGN development of consistency and clipped-WGAN channel adapters."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer, parameter_count
from conditional_drifting.baselines.fast_evaluation import distribution_metrics, sync, save, write_json
from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.sspa_trajectory import isolated_rng
from scripts.run_fast_channel_comparison import source_files, ensure_manifest


RECIPES = {
    "ct_default": dict(family="consistency"),
    "ct_coarse": dict(family="consistency", consistency_end_scales=20),
    "ct_coarse_lr": dict(family="consistency", consistency_end_scales=20, learning_rate=.001),
    "ct_coarse_lr_scale": dict(family="consistency", consistency_end_scales=20, learning_rate=.001,
                                consistency_sigma_data=1.),
    "wgan_default": dict(family="wgan"),
    "wgan_low_lr": dict(family="wgan", learning_rate=.00005),
    "wgan_wide_clip": dict(family="wgan", critic_clip=.1),
    "wgan_wide_clip_low_lr": dict(family="wgan", critic_clip=.1, learning_rate=.00005),
}


@torch.no_grad()
def moments(trainer, model, nfe, seed, anchors=64, samples=512):
    with isolated_rng(seed):
        x = torch.randn(anchors, trainer.n, device=trainer.device)
        repeated = x.repeat_interleave(samples, 0)
        y = torch.cat([trainer.sample(v, nfe, model=model) for v in repeated.split(1024)])
        y = y.reshape(anchors, samples, trainer.n)
        variance = y.var(1).mean().item()
        expected_variance = trainer.spec["noise_std"] ** 2
        return dict(mean_rmse=(y.mean(1)-x).square().mean().sqrt().item(),
            variance=variance, expected_variance=expected_variance,
            variance_ratio=variance/expected_variance,
            mean_mc_rmse_floor=(expected_variance/samples)**.5,
            anchors=anchors, samples=samples)


def evaluate(trainer, seed):
    rows = []
    for label, model in (("ema", trainer.ema), ("raw", trainer.model)):
        for nfe in ([1, 2] if trainer.cfg.family == "consistency" else [1]):
            sampler = lambda x: trainer.sample(x, nfe, model=model)
            metrics = distribution_metrics("AWGN", sampler, trainer.device, seed, 4096, 64, 256, 64, 1024)
            moment = moments(trainer, model, nfe, seed + 1)
            ratio = metrics["conditional"]["swd"] / metrics["analytic_floor"]["swd"]
            rows.append(dict(weights=label, nfe=nfe, distribution=metrics, moments=moment,
                conditional_swd_ratio=ratio,
                quality_gate=ratio <= 1.5 and moment["mean_rmse"] <= .1 and .8 <= moment["variance_ratio"] <= 1.2))
    return rows


def run_trial(cfg, directory, official, args):
    result_path = directory / "result.json"
    if result_path.exists() and args.resume:
        return json.loads(result_path.read_text())
    trainer = FastTrainer(cfg, args.device, official)
    history, trace = [], []
    checkpoint_path = directory / "generator.pt"
    if checkpoint_path.exists():
        if not args.resume:
            raise FileExistsError(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trainer.restore(state["trainer"])
        history, trace = state["history"], state["trace"]
    start_trial = time.perf_counter()
    while trainer.update < cfg.updates:
        sync(trainer.device)
        start = time.perf_counter()
        row = trainer.step()
        sync(trainer.device)
        trainer.train_seconds += time.perf_counter()-start
        if trainer.update % 100 == 0:
            trace.append(dict(**row, train_seconds=trainer.train_seconds))
        if trainer.update % args.eval_every == 0 or trainer.update == cfg.updates:
            measurements = evaluate(trainer, cfg.seed + 900000)
            history.append(dict(update=trainer.update, train_seconds=trainer.train_seconds, rows=measurements))
            primary = next(v for v in measurements if v["weights"] == "ema" and v["nfe"] == 1)
            print(f"[development] {directory.name} seed={cfg.seed} update={trainer.update} "
                  f"swd_ratio={primary['conditional_swd_ratio']:.3f} "
                  f"mean_rmse={primary['moments']['mean_rmse']:.4f} "
                  f"variance_ratio={primary['moments']['variance_ratio']:.3f}", flush=True)
        if trainer.update % args.save_every == 0 or trainer.update == cfg.updates:
            save(checkpoint_path, dict(trainer=trainer.state(), history=history, trace=trace))
    result = dict(config=asdict(cfg), history=history, trace=trace, counts=trainer.counts,
        parameters=parameter_count(trainer.model), train_seconds=trainer.train_seconds,
        invocation_seconds=time.perf_counter()-start_trial,
        primary=next(v for v in history[-1]["rows"] if v["weights"] == "ema" and v["nfe"] == 1))
    write_json(result_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--device", default="cuda")
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--confirmation-seeds", default="9102,9103")
    p.add_argument("--recipes", default=",".join(RECIPES))
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.updates, args.batch_size, args.eval_every, args.save_every) <= 0:
        raise ValueError("Positive training/checkpoint intervals required")
    torch.set_num_threads(2)
    names = args.recipes.split(",")
    confirmation_seeds = list(map(int, args.confirmation_seeds.split(","))) if args.confirmation_seeds else []
    if len(set(names)) != len(names) or not set(names) <= set(RECIPES):
        raise ValueError("Invalid recipes")
    if args.seed in confirmation_seeds or len(set(confirmation_seeds)) != len(confirmation_seeds):
        raise ValueError("Confirmation seeds must be unique and separate from selection")
    official = load_sources(args.source_dir)
    configs = {name: FastConfig(channel="AWGN", seed=args.seed, batch_size=args.batch_size,
        updates=args.updates, **RECIPES[name]) for name in names}
    for cfg in configs.values():
        cfg.validate()
    hashes = source_files()
    hashes[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    manifest = dict(schema=1, configs={k: asdict(v) for k, v in configs.items()}, sources=official.records,
        adapter_files=hashes, eval_every=args.eval_every, save_every=args.save_every,
        confirmation_seeds=confirmation_seeds,
        selection="Lowest final EMA one-step conditional SWD on development seed; no raw/early-checkpoint selection",
        quality_gate=dict(max_swd_to_floor=1.5, max_mean_rmse=.1, variance_ratio=[.8, 1.2]),
        runtime=dict(torch=str(torch.__version__), device=str(args.device), threads=2,
            hardware=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"))
    ensure_manifest(args.out_dir / "manifest.json", manifest, args.resume)
    rows = {}
    for name, cfg in configs.items():
        rows[name] = run_trial(cfg, args.out_dir / "development" / name, official, args)
    selected = {}
    for family in ("consistency", "wgan"):
        eligible = [name for name, cfg in configs.items() if cfg.family == family]
        if eligible:
            selected[family] = min(eligible, key=lambda name: rows[name]["primary"]["distribution"]["conditional"]["swd"])
    write_json(args.out_dir / "selection.json", dict(selected=selected, rule=manifest["selection"]))
    confirmations = {}
    for family, name in selected.items():
        for seed in confirmation_seeds:
            confirmations[f"{family}_seed{seed}"] = run_trial(replace(configs[name], seed=seed),
                args.out_dir / "confirmation" / f"{name}_seed{seed}", official, args)
    write_json(args.out_dir / "results.json", dict(development=rows, selected=selected, confirmation=confirmations))
    print(json.dumps(dict(result=str(args.out_dir / "results.json"), selected=selected)), flush=True)


if __name__ == "__main__":
    main()
