"""Counted EDM pretraining followed by official consistency training/distillation."""

import argparse
from copy import deepcopy
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

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer, physical_channel
from conditional_drifting.baselines.fast_evaluation import save, write_json, sync
from conditional_drifting.baselines.official_sources import load_sources, definitions, sha256
from conditional_drifting.sspa_trajectory import isolated_rng
from scripts.run_fast_channel_comparison import source_files, ensure_manifest
from scripts.run_fast_baseline_development import evaluate


def loss_step(trainer, kind, denoiser, sigma_sampler, teacher=None):
    if kind == "ct":
        return trainer.step()
    cfg = trainer.cfg
    x = torch.randn(cfg.batch_size, trainer.n, device=trainer.device, generator=trainer.condition_rng)
    y = physical_channel(cfg.channel, x)
    if kind == "edm":
        sigmas, weights = sigma_sampler.sample(len(x), trainer.device)
        loss = (denoiser.training_losses(trainer.model, y, sigmas, model_kwargs={"condition": x})["loss"] * weights).mean()
    else:
        loss = trainer.consistency.consistency_losses(trainer.model, y, 40, model_kwargs={"condition": x},
            target_model=trainer.target, teacher_model=teacher, teacher_diffusion=denoiser)["loss"].mean()
    trainer.optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1., error_if_nonfinite=True)
    trainer.optimizer.step()
    with torch.no_grad():
        for ema, model in zip(trainer.ema.parameters(), trainer.model.parameters()):
            ema.lerp_(model, 1-cfg.ema_decay)
        if kind == "cd":
            for target, model in zip(trainer.target.parameters(), trainer.model.parameters()):
                target.lerp_(model, .05)
    trainer.update += 1
    trainer.counts["oracle_outputs"] += len(x)
    trainer.counts["input_anchors"] += len(x)
    trainer.counts["generator_updates"] += 1
    if kind == "cd":
        trainer.counts["target_network_outputs"] += len(x)
        trainer.counts["teacher_network_outputs"] = trainer.counts.get("teacher_network_outputs", 0) + 2*len(x)
    return dict(update=trainer.update, loss=loss.item(), gradient_norm=norm.item())


def run_stage(trainer, kind, directory, official, sigma_sampler, args, teacher=None):
    path = directory / "checkpoint.pt"
    trace, history = [], []
    if path.exists():
        if not args.resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["kind"] != kind:
            raise ValueError("Stage mismatch")
        trainer.restore(state["trainer"])
        trace, history = state["trace"], state["history"]
    denoiser = official.cm.KarrasDenoiser(sigma_data=trainer.cfg.consistency_sigma_data,
        distillation=False, loss_norm="l2", weight_schedule="karras")
    while trainer.update < trainer.cfg.updates:
        sync(trainer.device)
        start = time.perf_counter()
        row = loss_step(trainer, kind, denoiser, sigma_sampler, teacher)
        sync(trainer.device)
        trainer.train_seconds += time.perf_counter()-start
        if trainer.update % 100 == 0:
            trace.append(dict(**row, train_seconds=trainer.train_seconds))
        if trainer.update % args.eval_every == 0 or trainer.update == trainer.cfg.updates:
            # EDM is a denoiser, not a one-step generator. Only evaluate CT/CD here.
            rows = [] if kind == "edm" else evaluate(trainer, args.seed + 900000)
            history.append(dict(update=trainer.update, rows=rows))
            message = f"[warmstart] {kind} seed={trainer.cfg.seed} update={trainer.update} loss={row['loss']:.6g}"
            if rows:
                primary = next(v for v in rows if v["weights"] == "ema" and v["nfe"] == 1)
                message += f" swd_ratio={primary['conditional_swd_ratio']:.3f} variance_ratio={primary['moments']['variance_ratio']:.3f}"
            print(message, flush=True)
        if trainer.update % args.save_every == 0 or trainer.update == trainer.cfg.updates:
            save(path, dict(kind=kind, trainer=trainer.state(), trace=trace, history=history))
    return dict(kind=kind, config=asdict(trainer.cfg), trace=trace, history=history,
        counts=trainer.counts, train_seconds=trainer.train_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.updates, args.batch_size, args.eval_every, args.save_every) <= 0:
        raise ValueError("Positive budgets and checkpoint intervals required")
    torch.set_num_threads(2)
    official = load_sources(args.source_dir)
    namespace = dict(vars(official.cm))
    definitions(args.source_dir / "consistency_models/cm/resample.py", ("LogNormalSampler",), namespace)
    sigma_sampler = namespace["LogNormalSampler"]()
    cfg = FastConfig(channel="AWGN", family="consistency", seed=args.seed,
        updates=args.updates, batch_size=args.batch_size)
    cfg.validate()
    hashes = source_files()
    for name in ("run_consistency_warmstart_development.py", "run_fast_baseline_development.py"):
        hashes["scripts/"+name] = sha256(ROOT / "scripts" / name)
    manifest = dict(config=asdict(cfg), sources=official.records, adapter_files=hashes,
        edm=dict(sigma_sampling="official LogNormalSampler: mean -1.2, std 1.2", loss="karras weighted L2",
                 initialized_from="random", optimizer="Adam", learning_rate=cfg.learning_rate),
        ct=dict(initialized_from="EDM EMA", teacher_in_loss=False, curriculum="adaptive progressive 2..150"),
        cd=dict(initialized_from="EDM EMA", teacher_in_loss="frozen EDM EMA, Heun", scales=40, target_ema=.95),
        accounting="Each CT/CD row includes the entire EDM pretraining budget; EDM is shared only for execution",
        eval_every=args.eval_every, save_every=args.save_every,
        runtime=dict(torch=str(torch.__version__), device=str(args.device), threads=2,
            hardware=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"))
    ensure_manifest(args.out_dir / "manifest.json", manifest, args.resume)
    edm = FastTrainer(cfg, args.device, official)
    pretraining = run_stage(edm, "edm", args.out_dir / "edm", official, sigma_sampler, args)
    teacher = deepcopy(edm.ema).eval().requires_grad_(False)
    denoiser = official.cm.KarrasDenoiser(distillation=False, loss_norm="l2")
    teacher_check = []
    with isolated_rng(args.seed+950000), torch.no_grad():
        x = torch.randn(4096, edm.n, device=edm.device)
        y = physical_channel("AWGN", x)
        variance = edm.spec["noise_std"]**2
        for sigma in (.01, .1, 1., 10., 80.):
            noisy = y + sigma*torch.randn_like(y)
            truth = x + variance/(variance+sigma**2)*(noisy-x)
            estimated = denoiser.denoise(teacher, noisy, x.new_full((len(x),), sigma), condition=x)[1]
            teacher_check.append(dict(sigma=sigma, posterior_mean_rmse=(estimated-truth).square().mean().sqrt().item()))
    pretraining["analytic_posterior_mean_check"] = teacher_check
    rows = {}
    for kind in ("ct", "cd"):
        trainer = FastTrainer(replace(cfg, seed=cfg.seed+10000), args.device, official)
        for model in (trainer.model, trainer.ema, trainer.target):
            model.load_state_dict(teacher.state_dict())
        result = run_stage(trainer, kind, args.out_dir / kind, official, sigma_sampler, args, teacher)
        result["total_train_seconds"] = pretraining["train_seconds"] + result["train_seconds"]
        result["total_oracle_outputs"] = pretraining["counts"]["oracle_outputs"] + result["counts"]["oracle_outputs"]
        result["primary"] = next(v for v in result["history"][-1]["rows"] if v["weights"] == "ema" and v["nfe"] == 1)
        rows[kind] = result
        write_json(args.out_dir / kind / "result.json", result)
    write_json(args.out_dir / "results.json", dict(pretraining=pretraining, rows=rows))


if __name__ == "__main__":
    main()
