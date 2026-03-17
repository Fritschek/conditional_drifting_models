from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import Callable

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conditional_drifting.baselines.diffusion import DiffusionConfig, sample_ddim, sample_ddpm, train_conditional_diffusion
from conditional_drifting.baselines.gan import GANConfig, train_conditional_gan
from conditional_drifting.channels import channel_registry
from conditional_drifting.training import (
    DriftingConfig,
    sample_drifting_target,
    select_device,
    set_seed,
    train_conditional_drifting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training and inference-time benchmark for conditional channel generators")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--channel", type=str, default="AWGN")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prepare-mode", type=str, default="train", choices=("init", "train"))
    parser.add_argument(
        "--methods",
        type=str,
        default="drifting_residual,drifting_direct,ddpm,ddim100,ddim50,ddim10,gan",
    )
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-batches", type=int, default=40)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--gan-epochs", type=int, default=120)
    parser.add_argument("--diffusion-steps", type=int, default=100)
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

    drifting_cfg = DriftingConfig(
        n=args.n,
        noise_std=args.noise_std,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        is_residual=True,
    )
    diffusion_cfg = DiffusionConfig(
        n=args.n,
        noise_std=args.noise_std,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_steps=args.diffusion_steps,
        is_residual=True,
    )
    gan_cfg = GANConfig(
        n=args.n,
        noise_std=args.noise_std,
        dataset_size=args.dataset_size,
        epochs=args.gan_epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        mode="gan_fa",
    )

    models: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    training: dict[str, dict[str, float | str | int]] = {}
    method_meta: dict[str, dict[str, str | float]] = {}

    if any(method in methods for method in ("drifting_residual", "drifting_direct")):
        from conditional_drifting.model import ConditionalDriftingGenerator

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
                    condition_dim=args.n,
                    output_dim=args.n,
                    latent_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim,
                ).to(device)
                model.eval()
                note = "initialized_only"
            train_seconds = time.perf_counter() - prep_start
            training[method_name] = {
                "mode": note,
                "train_seconds": train_seconds,
            }
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

    if any(method in methods for method in ("ddpm", "ddim100", "ddim50", "ddim20", "ddim10")):
        prep_start = time.perf_counter()
        if args.prepare_mode == "train":
            diffusion_model, diffusion_state = train_conditional_diffusion(channel_fn, diffusion_cfg, device)
            note = f"trained_{len(diffusion_state['history'])}epochs"
        else:
            from conditional_drifting.baselines.diffusion import ConditionalDiffusionMLP, cosine_beta_schedule

            diffusion_model = ConditionalDiffusionMLP(args.n, args.hidden_dim, args.diffusion_steps).to(device)
            diffusion_model.eval()
            betas = cosine_beta_schedule(args.diffusion_steps, clamp_max=0.999)
            alphas = 1.0 - betas
            alphas_prod = torch.cumprod(alphas, dim=0)
            diffusion_state = {
                "betas": betas.cpu(),
                "alphas": alphas.cpu(),
                "alphas_prod": alphas_prod.cpu(),
                "alphas_bar_sqrt": torch.sqrt(alphas_prod).cpu(),
                "one_minus_alphas_bar_sqrt": torch.sqrt(1.0 - alphas_prod).cpu(),
                "pred_type": "epsilon",
                "is_residual": True,
            }
            diffusion_model._diffusion_state = diffusion_state
            note = "initialized_only"
        train_seconds = time.perf_counter() - prep_start
        training["diffusion_shared"] = {
            "mode": note,
            "train_seconds": train_seconds,
        }
        if "ddpm" in methods:
            method_meta["ddpm"] = {"family": "diffusion", "sampler": "ddpm", "target_mode": "residual"}
            models["ddpm"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddpm(model, x, state)
        if "ddim100" in methods:
            method_meta["ddim100"] = {"family": "diffusion", "sampler": "ddim100", "target_mode": "residual"}
            models["ddim100"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 100)
        if "ddim50" in methods:
            method_meta["ddim50"] = {"family": "diffusion", "sampler": "ddim50", "target_mode": "residual"}
            models["ddim50"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 50)
        if "ddim20" in methods:
            method_meta["ddim20"] = {"family": "diffusion", "sampler": "ddim20", "target_mode": "residual"}
            models["ddim20"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 20)
        if "ddim10" in methods:
            method_meta["ddim10"] = {"family": "diffusion", "sampler": "ddim10", "target_mode": "residual"}
            models["ddim10"] = lambda x, model=diffusion_model, state=diffusion_state: sample_ddim(model, x, state, 10)

    if any(method in methods for method in ("gan", "gan_fa")):
        from conditional_drifting.baselines.gan import ConditionalGANGenerator

        prep_start = time.perf_counter()
        if args.prepare_mode == "train":
            gan_model, gan_artifacts = train_conditional_gan(channel_fn, gan_cfg, device)
            note = f"trained_{len(gan_artifacts['history'])}epochs"
        else:
            gan_model = ConditionalGANGenerator(args.n, args.latent_dim, args.hidden_dim).to(device)
            gan_model.eval()
            note = "initialized_only"
        train_seconds = time.perf_counter() - prep_start
        training["gan"] = {
            "mode": note,
            "train_seconds": train_seconds,
        }
        method_meta["gan"] = {"family": "gan", "sampler": "one_shot", "target_mode": "direct_y"}
        models["gan"] = lambda x, model=gan_model: model.sample_y(x)

    results = {
        "channel": args.channel,
        "device": str(device),
        "seed": args.seed,
        "prepare_mode": args.prepare_mode,
        "warmup_repeats": args.warmup_repeats,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
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
            n=args.n,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            warmup_repeats=args.warmup_repeats,
            repeats=args.repeats,
        )
        train_seconds = 0.0
        if method.startswith("dd"):
            train_seconds = float(training["diffusion_shared"]["train_seconds"])
        elif method.startswith("drifting_"):
            train_seconds = float(training[method]["train_seconds"])
        elif method == "gan":
            train_seconds = float(training["gan"]["train_seconds"])
        results["methods"][method] = {
            **method_meta.get(method, {}),
            **stats,
            "train_seconds": train_seconds,
            "train_plus_eval_seconds": train_seconds + float(stats["mean_seconds"]),
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
