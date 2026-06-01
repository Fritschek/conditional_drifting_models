from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplcache"))

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, ROOT)

from conditional_drifting.baselines import (
    DiffusionConfig,
    GANConfig,
    PaperWGANConfig,
    evaluate_diffusion_model,
    evaluate_gan_model,
    evaluate_paper_wgan,
    train_conditional_diffusion,
    train_conditional_gan,
    train_paper_wgan,
)
from conditional_drifting.benchmark import BenchmarkConfig
from conditional_drifting.channels import channel_registry
from conditional_drifting.training import evaluate_residual_model, select_device, set_seed, train_conditional_drifting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional baseline comparison for a single channel")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "TDL", "OptFib"])
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=20_000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument(
        "--include-gan",
        action="store_true",
        help="Also run the later standalone GAN baseline.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to repo results/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    out_dir = args.out_dir or os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)

    channel_fn = channel_registry()[args.channel]
    drift_residual_cfg = BenchmarkConfig(
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        is_residual=True,
    )
    drift_direct_cfg = BenchmarkConfig(
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        is_residual=False,
    )
    diffusion_cfg = DiffusionConfig(dataset_size=args.dataset_size, epochs=args.epochs, batch_size=args.batch_size, eval_size=args.eval_size, num_steps=args.num_steps)
    gan_cfg = GANConfig(dataset_size=args.dataset_size, epochs=args.epochs, batch_size=args.batch_size, eval_size=args.eval_size)
    paper_wgan_cfg = PaperWGANConfig(
        n=drift_residual_cfg.n,
        noise_std=drift_residual_cfg.noise_std,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_size=args.eval_size,
    )

    drift_residual_model, _ = train_conditional_drifting(channel_fn, drift_residual_cfg, device)
    drift_residual_eval = evaluate_residual_model(drift_residual_model, channel_fn, drift_residual_cfg, device)
    drift_direct_model, _ = train_conditional_drifting(channel_fn, drift_direct_cfg, device)
    drift_direct_eval = evaluate_residual_model(drift_direct_model, channel_fn, drift_direct_cfg, device)

    diff_model, _ = train_conditional_diffusion(channel_fn, diffusion_cfg, device)
    ddpm_eval = evaluate_diffusion_model(diff_model, channel_fn, diffusion_cfg, device, use_ddim=False)
    ddim_eval = evaluate_diffusion_model(diff_model, channel_fn, diffusion_cfg, device, use_ddim=True, ddim_steps=args.ddim_steps)

    paper_wgan_model, _ = train_paper_wgan(channel_fn, paper_wgan_cfg, device)
    paper_wgan_eval = evaluate_paper_wgan(paper_wgan_model, channel_fn, paper_wgan_cfg, device)

    summary = {
        "channel": args.channel,
        "device": str(device),
        "drifting_residual_swd": drift_residual_eval["swd"],
        "drifting_direct_swd": drift_direct_eval["swd"],
        "drifting_swd": drift_residual_eval["swd"],
        "ddpm_swd": ddpm_eval["swd"],
        "ddim_swd": ddim_eval["swd"],
        "paper_wgan_swd": paper_wgan_eval["swd"],
    }
    if args.include_gan:
        gan_model, _ = train_conditional_gan(channel_fn, gan_cfg, device)
        gan_eval = evaluate_gan_model(gan_model, channel_fn, gan_cfg, device)
        summary["gan_swd"] = gan_eval["swd"]
    out_json = os.path.join(out_dir, f"baseline_compare_{args.channel.lower()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    output = {**summary, "json": out_json}
    if plt is not None:
        labels = ["Drift Res", "Drift Dir", f"DDPM({args.num_steps})", f"DDIM({args.ddim_steps})", "Paper WGAN"]
        values = [
            summary["drifting_residual_swd"],
            summary["drifting_direct_swd"],
            summary["ddpm_swd"],
            summary["ddim_swd"],
            summary["paper_wgan_swd"],
        ]
        colors = ["#cf4446", "#f28e2b", "#1f77b4", "#ff7f0e", "#59a14f"]
        if args.include_gan:
            labels.append("GAN")
            values.append(summary["gan_swd"])
            colors.append("#2ca02c")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Residual SWD")
        ax.set_title(f"Optional baseline comparison on {args.channel}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"baseline_compare_{args.channel.lower()}.png")
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output["figure"] = out_png
    else:
        print("[benchmark] matplotlib not available; skipping plot generation", flush=True)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
