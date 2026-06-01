from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conditional_drifting.baselines.diffusion import DiffusionConfig, sample_ddim, sample_ddpm, train_conditional_diffusion
from conditional_drifting.baselines.gan import GANConfig, train_conditional_gan
from conditional_drifting.baselines.paper_wgan import PaperWGANConfig, PaperWGANGenerator, train_paper_wgan
from conditional_drifting.channels import channel_registry
from conditional_drifting.model import ConditionalDriftingGenerator
from conditional_drifting.training import (
    DriftingConfig,
    sample_drifting_target,
    select_device,
    set_seed,
    train_conditional_drifting,
)


@dataclass(frozen=True)
class PaperTimingPreset:
    n: int
    ebn0_db: float
    rate: float
    diffusion_hidden_dim: int
    paper_wgan_hidden_dim: int
    dataset_size: int
    batch_size: int
    diffusion_epochs: int
    drifting_epochs: int
    paper_wgan_epochs: int
    reference_eval_size: int
    diffusion_steps: int = 100


PAPER2309_TIMING_PRESETS: dict[str, PaperTimingPreset] = {
    "AWGN": PaperTimingPreset(
        n=7,
        ebn0_db=5.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=110,
        paper_wgan_hidden_dim=128,
        dataset_size=10_000_000,
        batch_size=5_000,
        diffusion_epochs=30,
        drifting_epochs=30,
        paper_wgan_epochs=30,
        reference_eval_size=10_000_000,
    ),
    "Rayleigh": PaperTimingPreset(
        n=7,
        ebn0_db=12.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=128,
        paper_wgan_hidden_dim=256,
        dataset_size=10_000_000,
        batch_size=5_000,
        diffusion_epochs=30,
        drifting_epochs=30,
        paper_wgan_epochs=30,
        reference_eval_size=10_000_000,
    ),
    "SSPA": PaperTimingPreset(
        n=8,
        ebn0_db=8.0,
        rate=6.0 / 8.0,
        diffusion_hidden_dim=110,
        paper_wgan_hidden_dim=256,
        dataset_size=10_000_000,
        batch_size=4_096,
        diffusion_epochs=160,
        drifting_epochs=160,
        paper_wgan_epochs=160,
        reference_eval_size=10_000_000,
    ),
}

OPTFIB_SUITE_PRESET = {
    "n": 2,
    "noise_std": 0.3,
    "dataset_size": 120_000,
    "batch_size": 512,
    "drifting_epochs": 60,
    "diffusion_epochs": 60,
    "paper_wgan_epochs": 60,
    "gan_epochs": 120,
    "diffusion_hidden_dim": 128,
    "paper_wgan_hidden_dim": 128,
    "reference_eval_size": 100_000,
    "diffusion_steps": 100,
}


WFLOW_TIMING_VARIANTS: dict[str, dict[str, object]] = {
    "kernel_target": {
        "drift_field": "kernel",
        "conditioning_mode": "none",
        "target_kernel_scale": 1.0,
    },
    "kernel_joint": {
        "drift_field": "kernel",
        "conditioning_mode": "joint",
        "condition_kernel_scale": 0.5,
        "target_kernel_scale": 1.0,
        "target_kernel_mode": "raw",
    },
    "joint_sinkhorn": {
        "drift_field": "sinkhorn",
        "conditioning_mode": "joint",
        "condition_kernel_scale": 0.5,
        "target_kernel_scale": 1.0,
    },
    "fiber_sinkhorn": {
        "drift_field": "fiber_sinkhorn",
        "conditioning_mode": "none",
        "target_kernel_scale": 1.0,
    },
}


def ebno_to_noise(ebn0_db: float, rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / math.sqrt(2.0 * rate * ebn0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training and inference-time benchmark for suite-aligned channel generators")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--channel", type=str, default="AWGN")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prepare-mode", type=str, default="train", choices=("init", "train"))
    parser.add_argument(
        "--preset",
        type=str,
        default="paper2309",
        choices=("paper2309", "optfib_suite", "custom"),
        help="Use suite-aligned benchmark settings instead of raw custom dimensions.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="drifting_residual,drifting_direct,ddpm,ddim100,ddim50,ddim20,ddim10,paper_wgan",
        help="Comma-separated methods to time.",
    )
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-batches", type=int, default=40)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--paper-wgan-hidden-dim", type=int, default=128)
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--drifting-epochs", type=int, default=-1)
    parser.add_argument("--paper-wgan-epochs", type=int, default=-1)
    parser.add_argument("--gan-epochs", type=int, default=120)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sinkhorn-min-epsilon", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fiber-generated-samples", type=int, default=4)
    parser.add_argument("--fiber-positive-samples", type=int, default=4)
    parser.add_argument("--fiber-reference-samples", type=int, default=4)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of training workload to time before extrapolating full training cost.",
    )
    parser.add_argument(
        "--reference-eval-size",
        type=int,
        default=-1,
        help="Projected benchmark evaluation sample count for total-time estimates.",
    )
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args()


def make_output_dir(custom_out_dir: str) -> str:
    if custom_out_dir:
        os.makedirs(custom_out_dir, exist_ok=True)
        return custom_out_dir
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", f"inference_timing_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_training_stats(
    *,
    mode: str,
    train_seconds: float,
    timed_dataset_size: int,
    full_dataset_size: int,
    batch_size: int,
    epochs: int,
) -> dict[str, float | str | int]:
    timed_steps_per_epoch = math.ceil(timed_dataset_size / batch_size)
    timed_total_steps = timed_steps_per_epoch * epochs
    timed_total_samples = timed_dataset_size * epochs
    full_steps_per_epoch = math.ceil(full_dataset_size / batch_size)
    full_total_steps = full_steps_per_epoch * epochs
    full_total_samples = full_dataset_size * epochs
    effective_fraction = timed_total_steps / full_total_steps if full_total_steps > 0 else 1.0
    projected_full_train_seconds = train_seconds / effective_fraction if effective_fraction > 0 else float("inf")
    return {
        "mode": mode,
        "timed_train_seconds": train_seconds,
        "timed_train_minutes": train_seconds / 60.0,
        "timed_train_hours": train_seconds / 3600.0,
        "projected_full_train_seconds": projected_full_train_seconds,
        "projected_full_train_minutes": projected_full_train_seconds / 60.0,
        "projected_full_train_hours": projected_full_train_seconds / 3600.0,
        "epochs": epochs,
        "timed_dataset_size": timed_dataset_size,
        "full_dataset_size": full_dataset_size,
        "timed_steps_per_epoch": timed_steps_per_epoch,
        "timed_total_steps": timed_total_steps,
        "timed_total_samples": timed_total_samples,
        "full_steps_per_epoch": full_steps_per_epoch,
        "full_total_steps": full_total_steps,
        "full_total_samples": full_total_samples,
        "effective_train_fraction": effective_fraction,
        "train_steps_per_second": timed_total_steps / train_seconds if train_seconds > 0 else float("inf"),
        "train_samples_per_second": timed_total_samples / train_seconds if train_seconds > 0 else float("inf"),
    }


def scale_dataset_size(dataset_size: int, batch_size: int, fraction: float) -> int:
    if fraction >= 1.0:
        return dataset_size
    scaled = int(math.ceil((dataset_size * fraction) / batch_size) * batch_size)
    return max(batch_size, min(dataset_size, scaled))


def timed_generation(
    generate_batch: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    n: int,
    batch_size: int,
    num_batches: int,
    warmup_repeats: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    total_samples = batch_size * num_batches

    for _ in range(warmup_repeats):
        for _ in range(num_batches):
            x = torch.randn(batch_size, n, device=device)
            _ = generate_batch(x)
        sync_device(device)

    repeat_seconds: list[float] = []
    for _ in range(repeats):
        sync_device(device)
        start = time.perf_counter()
        for _ in range(num_batches):
            x = torch.randn(batch_size, n, device=device)
            _ = generate_batch(x)
        sync_device(device)
        repeat_seconds.append(time.perf_counter() - start)

    mean_sec = sum(repeat_seconds) / len(repeat_seconds)
    std_sec = 0.0
    if len(repeat_seconds) > 1:
        variance = sum((value - mean_sec) ** 2 for value in repeat_seconds) / len(repeat_seconds)
        std_sec = variance ** 0.5
    return {
        "repeat_seconds": repeat_seconds,
        "mean_seconds": mean_sec,
        "std_seconds": std_sec,
        "samples_per_second": total_samples / mean_sec,
        "milliseconds_per_sample": (mean_sec / total_samples) * 1000.0,
        "total_samples_per_repeat": total_samples,
    }


def resolve_configs(args: argparse.Namespace) -> dict[str, int | float]:
    if args.preset == "paper2309":
        if args.channel not in PAPER2309_TIMING_PRESETS:
            raise ValueError(f"paper2309 preset only supports AWGN, Rayleigh, SSPA; got {args.channel}")
        preset = PAPER2309_TIMING_PRESETS[args.channel]
        return {
            "n": preset.n,
            "noise_std": ebno_to_noise(preset.ebn0_db, preset.rate),
            "dataset_size": preset.dataset_size,
            "batch_size": preset.batch_size,
            "drifting_epochs": preset.drifting_epochs,
            "diffusion_epochs": preset.diffusion_epochs,
            "paper_wgan_epochs": preset.paper_wgan_epochs,
            "diffusion_hidden_dim": preset.diffusion_hidden_dim,
            "paper_wgan_hidden_dim": preset.paper_wgan_hidden_dim,
            "diffusion_steps": preset.diffusion_steps,
            "reference_eval_size": args.reference_eval_size if args.reference_eval_size > 0 else preset.reference_eval_size,
        }
    if args.preset == "optfib_suite":
        return {
            "n": OPTFIB_SUITE_PRESET["n"],
            "noise_std": OPTFIB_SUITE_PRESET["noise_std"],
            "dataset_size": OPTFIB_SUITE_PRESET["dataset_size"],
            "batch_size": OPTFIB_SUITE_PRESET["batch_size"],
            "drifting_epochs": OPTFIB_SUITE_PRESET["drifting_epochs"],
            "diffusion_epochs": OPTFIB_SUITE_PRESET["diffusion_epochs"],
            "paper_wgan_epochs": OPTFIB_SUITE_PRESET["paper_wgan_epochs"],
            "diffusion_hidden_dim": OPTFIB_SUITE_PRESET["diffusion_hidden_dim"],
            "paper_wgan_hidden_dim": OPTFIB_SUITE_PRESET["paper_wgan_hidden_dim"],
            "diffusion_steps": OPTFIB_SUITE_PRESET["diffusion_steps"],
            "reference_eval_size": args.reference_eval_size if args.reference_eval_size > 0 else OPTFIB_SUITE_PRESET["reference_eval_size"],
        }
    return {
        "n": args.n,
        "noise_std": args.noise_std,
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "drifting_epochs": args.drifting_epochs if args.drifting_epochs > 0 else args.epochs,
        "diffusion_epochs": args.epochs,
        "paper_wgan_epochs": args.paper_wgan_epochs if args.paper_wgan_epochs > 0 else args.epochs,
        "diffusion_hidden_dim": args.hidden_dim,
        "paper_wgan_hidden_dim": args.paper_wgan_hidden_dim,
        "diffusion_steps": args.diffusion_steps,
        "reference_eval_size": args.reference_eval_size if args.reference_eval_size > 0 else args.batch_size * args.num_batches,
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    out_dir = make_output_dir(args.out_dir)

    methods = [part.strip() for part in args.methods.split(",") if part.strip()]
    channel_map = channel_registry()
    if args.channel not in channel_map:
        raise ValueError(f"Unknown channel: {args.channel}")
    channel_fn = channel_map[args.channel]
    resolved = resolve_configs(args)
    timed_dataset_size = scale_dataset_size(
        int(resolved["dataset_size"]),
        int(resolved["batch_size"]),
        args.train_fraction,
    )

    drifting_cfg = DriftingConfig(
        n=int(resolved["n"]),
        noise_std=float(resolved["noise_std"]),
        dataset_size=timed_dataset_size,
        epochs=int(resolved["drifting_epochs"]),
        batch_size=int(resolved["batch_size"]),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        is_residual=True,
        sinkhorn_min_epsilon=args.sinkhorn_min_epsilon,
        sinkhorn_iterations=args.sinkhorn_iterations,
        fiber_generated_samples=args.fiber_generated_samples,
        fiber_positive_samples=args.fiber_positive_samples,
        fiber_reference_samples=args.fiber_reference_samples,
    )
    diffusion_cfg = DiffusionConfig(
        n=int(resolved["n"]),
        noise_std=float(resolved["noise_std"]),
        dataset_size=timed_dataset_size,
        epochs=int(resolved["diffusion_epochs"]),
        batch_size=int(resolved["batch_size"]),
        hidden_dim=int(resolved["diffusion_hidden_dim"]),
        num_steps=int(resolved["diffusion_steps"]),
        is_residual=False if args.preset == "paper2309" else True,
        pred_type="v" if args.preset == "paper2309" else "epsilon",
        beta_schedule="cosine-zf" if args.preset == "paper2309" else "cosine",
        learning_rate=1e-4 if args.preset == "paper2309" else 1e-3,
        ema_decay=0.9 if args.preset == "paper2309" else 0.995,
    )
    if args.preset == "paper2309":
        diffusion_cfg.learning_rate_schedule = {
            "AWGN": ((10, 1e-3), (20, 1e-4)),
            "Rayleigh": ((10, 1e-3), (20, 1e-4)),
        }.get(args.channel)
    paper_wgan_cfg = PaperWGANConfig(
        n=int(resolved["n"]),
        noise_std=float(resolved["noise_std"]),
        dataset_size=timed_dataset_size,
        batch_size=int(resolved["batch_size"]),
        epochs=int(resolved["paper_wgan_epochs"]),
        eval_size=int(resolved["reference_eval_size"]),
        hidden_dim=int(resolved["paper_wgan_hidden_dim"]),
    )
    gan_cfg = GANConfig(
        n=int(resolved["n"]),
        noise_std=float(resolved["noise_std"]),
        dataset_size=timed_dataset_size,
        epochs=args.gan_epochs,
        batch_size=int(resolved["batch_size"]),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        mode="gan_fa",
    )

    models: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    training: dict[str, dict[str, float | str | int]] = {}
    method_meta: dict[str, dict[str, str | float]] = {}

    if any(method in methods for method in ("drifting_residual", "drifting_direct")):
        for method_name, is_residual in (("drifting_residual", True), ("drifting_direct", False)):
            if method_name not in methods:
                continue
            cfg = DriftingConfig(**{**drifting_cfg.__dict__, "is_residual": is_residual})
            prep_start = time.perf_counter()
            if args.prepare_mode == "train":
                model, artifacts = train_conditional_drifting(channel_fn, cfg, device)
                note = f"trained_{len(artifacts.history)}epochs"
            else:
                model = ConditionalDriftingGenerator(
                    condition_dim=int(resolved["n"]),
                    output_dim=int(resolved["n"]),
                    latent_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim,
                ).to(device)
                model.eval()
                note = "initialized_only"
            train_seconds = time.perf_counter() - prep_start
            training[method_name] = build_training_stats(
                mode=note,
                train_seconds=train_seconds,
                timed_dataset_size=cfg.dataset_size,
                full_dataset_size=int(resolved["dataset_size"]),
                batch_size=int(resolved["batch_size"]),
                epochs=cfg.epochs,
            )
            method_meta[method_name] = {
                "family": "drifting",
                "target_mode": "residual" if is_residual else "direct_y",
            }
            models[method_name] = (
                lambda x, model=model, is_residual=is_residual: sample_drifting_target(
                    model,
                    x,
                    is_residual=is_residual,
                )
            )

    if any(method in methods for method in WFLOW_TIMING_VARIANTS):
        for method_name, variant_args in WFLOW_TIMING_VARIANTS.items():
            if method_name not in methods:
                continue
            cfg = DriftingConfig(
                **{
                    **drifting_cfg.__dict__,
                    **variant_args,
                    "is_residual": False,
                }
            )
            prep_start = time.perf_counter()
            if args.prepare_mode == "train":
                model, artifacts = train_conditional_drifting(channel_fn, cfg, device)
                note = f"trained_{len(artifacts.history)}epochs"
            else:
                model = ConditionalDriftingGenerator(
                    condition_dim=int(resolved["n"]),
                    output_dim=int(resolved["n"]),
                    latent_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim,
                ).to(device)
                model.eval()
                note = "initialized_only"
            train_seconds = time.perf_counter() - prep_start
            training[method_name] = build_training_stats(
                mode=note,
                train_seconds=train_seconds,
                timed_dataset_size=cfg.dataset_size,
                full_dataset_size=int(resolved["dataset_size"]),
                batch_size=int(resolved["batch_size"]),
                epochs=cfg.epochs,
            )
            method_meta[method_name] = {
                "family": "wflow_drifting",
                "sampler": "one_shot",
                "target_mode": "direct_y",
                **{key: str(value) for key, value in variant_args.items()},
            }
            models[method_name] = (
                lambda x, model=model: sample_drifting_target(
                    model,
                    x,
                    is_residual=False,
                )
            )

    if any(method in methods for method in ("ddpm", "ddim100", "ddim50", "ddim20", "ddim10")):
        prep_start = time.perf_counter()
        if args.prepare_mode == "train":
            diffusion_model, diffusion_state = train_conditional_diffusion(channel_fn, diffusion_cfg, device)
            note = f"trained_{len(diffusion_state['history'])}epochs"
        else:
            from conditional_drifting.baselines.diffusion import ConditionalDiffusionMLP, cosine_beta_schedule

            diffusion_model = ConditionalDiffusionMLP(
                int(resolved["n"]),
                int(resolved["diffusion_hidden_dim"]),
                int(resolved["diffusion_steps"]),
            ).to(device)
            diffusion_model.eval()
            betas = cosine_beta_schedule(
                int(resolved["diffusion_steps"]),
                clamp_max=None if diffusion_cfg.beta_schedule == "cosine-zf" else 0.999,
            )
            alphas = 1.0 - betas
            alphas_prod = torch.cumprod(alphas, dim=0)
            diffusion_state = {
                "betas": betas.cpu(),
                "alphas": alphas.cpu(),
                "alphas_prod": alphas_prod.cpu(),
                "alphas_bar_sqrt": torch.sqrt(alphas_prod).cpu(),
                "one_minus_alphas_bar_sqrt": torch.sqrt(1.0 - alphas_prod).cpu(),
                "pred_type": diffusion_cfg.pred_type,
                "is_residual": diffusion_cfg.is_residual,
            }
            diffusion_model._diffusion_state = diffusion_state
            note = "initialized_only"
        train_seconds = time.perf_counter() - prep_start
        training["diffusion_shared"] = build_training_stats(
            mode=note,
            train_seconds=train_seconds,
            timed_dataset_size=diffusion_cfg.dataset_size,
            full_dataset_size=int(resolved["dataset_size"]),
            batch_size=int(resolved["batch_size"]),
            epochs=int(resolved["diffusion_epochs"]),
        )
        target_mode = "residual" if diffusion_cfg.is_residual else "direct_y"
        if "ddpm" in methods:
            method_meta["ddpm"] = {"family": "diffusion", "sampler": "ddpm", "target_mode": target_mode}
            models["ddpm"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddpm(model, x, state)
        if "ddim100" in methods:
            method_meta["ddim100"] = {"family": "diffusion", "sampler": "ddim100", "target_mode": target_mode}
            models["ddim100"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 100)
        if "ddim50" in methods:
            method_meta["ddim50"] = {"family": "diffusion", "sampler": "ddim50", "target_mode": target_mode}
            models["ddim50"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 50)
        if "ddim20" in methods:
            method_meta["ddim20"] = {"family": "diffusion", "sampler": "ddim20", "target_mode": target_mode}
            models["ddim20"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 20)
        if "ddim10" in methods:
            method_meta["ddim10"] = {"family": "diffusion", "sampler": "ddim10", "target_mode": target_mode}
            models["ddim10"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 10)

    if "paper_wgan" in methods:
        prep_start = time.perf_counter()
        if args.prepare_mode == "train":
            paper_wgan_model, paper_wgan_artifacts = train_paper_wgan(channel_fn, paper_wgan_cfg, device)
            note = f"trained_{len(paper_wgan_artifacts['history'])}epochs"
        else:
            paper_wgan_model = PaperWGANGenerator(
                int(resolved["n"]),
                int(resolved["paper_wgan_hidden_dim"]),
            ).to(device)
            paper_wgan_model.eval()
            note = "initialized_only"
        train_seconds = time.perf_counter() - prep_start
        training["paper_wgan"] = build_training_stats(
            mode=note,
            train_seconds=train_seconds,
            timed_dataset_size=paper_wgan_cfg.dataset_size,
            full_dataset_size=int(resolved["dataset_size"]),
            batch_size=int(resolved["batch_size"]),
            epochs=int(resolved["paper_wgan_epochs"]),
        )
        method_meta["paper_wgan"] = {"family": "wgan", "sampler": "one_shot", "target_mode": "direct_y"}
        models["paper_wgan"] = lambda x, model=paper_wgan_model: model(x)

    if "gan" in methods:
        from conditional_drifting.baselines.gan import ConditionalGANGenerator

        prep_start = time.perf_counter()
        if args.prepare_mode == "train":
            gan_model, gan_artifacts = train_conditional_gan(channel_fn, gan_cfg, device)
            note = f"trained_{len(gan_artifacts['history'])}epochs"
        else:
            gan_model = ConditionalGANGenerator(
                int(resolved["n"]),
                args.latent_dim,
                args.hidden_dim,
            ).to(device)
            gan_model.eval()
            note = "initialized_only"
        train_seconds = time.perf_counter() - prep_start
        training["gan"] = build_training_stats(
            mode=note,
            train_seconds=train_seconds,
            timed_dataset_size=gan_cfg.dataset_size,
            full_dataset_size=int(resolved["dataset_size"]),
            batch_size=int(resolved["batch_size"]),
            epochs=args.gan_epochs,
        )
        method_meta["gan"] = {"family": "gan", "sampler": "one_shot", "target_mode": "direct_y"}
        models["gan"] = lambda x, model=gan_model: model.sample_y(x)

    results = {
        "channel": args.channel,
        "device": str(device),
        "seed": args.seed,
        "preset": args.preset,
        "prepare_mode": args.prepare_mode,
        "warmup_repeats": args.warmup_repeats,
        "repeats": args.repeats,
        "timing_batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "reference_eval_size": int(resolved["reference_eval_size"]),
        "resolved_config": {
            "n": int(resolved["n"]),
            "noise_std": float(resolved["noise_std"]),
            "dataset_size": int(resolved["dataset_size"]),
            "timed_dataset_size": timed_dataset_size,
            "batch_size": int(resolved["batch_size"]),
            "drifting_epochs": int(resolved["drifting_epochs"]),
            "diffusion_epochs": int(resolved["diffusion_epochs"]),
            "paper_wgan_epochs": int(resolved["paper_wgan_epochs"]),
            "diffusion_steps": int(resolved["diffusion_steps"]),
            "reference_eval_size": int(resolved["reference_eval_size"]),
            "train_fraction_requested": args.train_fraction,
        },
        "training": training,
        "methods": {},
    }

    for method in methods:
        if method not in models:
            raise ValueError(f"Unsupported timing method: {method}")
        print(f"[timing] method={method} device={device} channel={args.channel}")
        stats = timed_generation(
            models[method],
            device=device,
            n=int(resolved["n"]),
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            warmup_repeats=args.warmup_repeats,
            repeats=args.repeats,
        )
        if method.startswith("dd"):
            train_seconds = float(training["diffusion_shared"]["projected_full_train_seconds"])
        elif method.startswith("drifting_"):
            train_seconds = float(training[method]["projected_full_train_seconds"])
        elif method in WFLOW_TIMING_VARIANTS:
            train_seconds = float(training[method]["projected_full_train_seconds"])
        elif method == "paper_wgan":
            train_seconds = float(training["paper_wgan"]["projected_full_train_seconds"])
        elif method == "gan":
            train_seconds = float(training["gan"]["projected_full_train_seconds"])
        else:
            train_seconds = 0.0
        projected_eval_seconds = int(resolved["reference_eval_size"]) / float(stats["samples_per_second"])
        results["methods"][method] = {
            **method_meta.get(method, {}),
            **stats,
            "projected_full_train_seconds": train_seconds,
            "projected_full_train_minutes": train_seconds / 60.0,
            "projected_full_train_hours": train_seconds / 3600.0,
            "projected_eval_seconds": projected_eval_seconds,
            "projected_eval_minutes": projected_eval_seconds / 60.0,
            "projected_eval_hours": projected_eval_seconds / 3600.0,
            "projected_total_benchmark_seconds": train_seconds + projected_eval_seconds,
            "projected_total_benchmark_minutes": (train_seconds + projected_eval_seconds) / 60.0,
            "projected_total_benchmark_hours": (train_seconds + projected_eval_seconds) / 3600.0,
        }

    if "drifting_residual" in results["methods"]:
        base = float(results["methods"]["drifting_residual"]["mean_seconds"])
        for method, stats in results["methods"].items():
            results["methods"][method]["eval_slowdown_vs_drifting_residual"] = float(stats["mean_seconds"]) / base

    if "drifting_direct" in results["methods"]:
        base = float(results["methods"]["drifting_direct"]["mean_seconds"])
        for method, stats in results["methods"].items():
            results["methods"][method]["eval_slowdown_vs_drifting_direct"] = float(stats["mean_seconds"]) / base

    out_path = os.path.join(out_dir, "timing_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"output_path": out_path, "methods": list(results["methods"].keys())}, indent=2))


if __name__ == "__main__":
    main()
