from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ROOT)

from conditional_drifting.baselines import DiffusionConfig, GANConfig, evaluate_diffusion_model, evaluate_gan_model, train_conditional_diffusion, train_conditional_gan
from conditional_drifting.benchmark import BenchmarkConfig
from conditional_drifting.channels import channel_registry
from conditional_drifting.training import evaluate_residual_model, select_device, set_seed, train_conditional_drifting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional baseline comparison for a single channel")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "SSPA", "OptFib"])
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=20_000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    channel_fn = channel_registry()[args.channel]
    drift_cfg = BenchmarkConfig(dataset_size=args.dataset_size, epochs=args.epochs, batch_size=args.batch_size, eval_size=args.eval_size)
    diffusion_cfg = DiffusionConfig(dataset_size=args.dataset_size, epochs=args.epochs, batch_size=args.batch_size, eval_size=args.eval_size, num_steps=args.num_steps)
    gan_cfg = GANConfig(dataset_size=args.dataset_size, epochs=args.epochs, batch_size=args.batch_size, eval_size=args.eval_size)

    drift_model, _ = train_conditional_drifting(channel_fn, drift_cfg, device)
    drift_eval = evaluate_residual_model(drift_model, channel_fn, drift_cfg, device)

    diff_model, _ = train_conditional_diffusion(channel_fn, diffusion_cfg, device)
    ddpm_eval = evaluate_diffusion_model(diff_model, channel_fn, diffusion_cfg, device, use_ddim=False)
    ddim_eval = evaluate_diffusion_model(diff_model, channel_fn, diffusion_cfg, device, use_ddim=True, ddim_steps=args.ddim_steps)

    gan_model, _ = train_conditional_gan(channel_fn, gan_cfg, device)
    gan_eval = evaluate_gan_model(gan_model, channel_fn, gan_cfg, device)

    summary = {
        "channel": args.channel,
        "device": str(device),
        "drifting_swd": drift_eval["swd"],
        "ddpm_swd": ddpm_eval["swd"],
        "ddim_swd": ddim_eval["swd"],
        "gan_swd": gan_eval["swd"],
    }
    out_json = os.path.join(ROOT, "results", f"baseline_compare_{args.channel.lower()}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    labels = ["Drifting", f"DDPM({args.num_steps})", f"DDIM({args.ddim_steps})", "GAN"]
    values = [summary["drifting_swd"], summary["ddpm_swd"], summary["ddim_swd"], summary["gan_swd"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#cf4446", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Residual SWD")
    ax.set_title(f"Optional baseline comparison on {args.channel}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_png = os.path.join(ROOT, "results", f"baseline_compare_{args.channel.lower()}.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({**summary, "json": out_json, "figure": out_png}, indent=2))


if __name__ == "__main__":
    main()
