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

from conditional_drifting.channels import channel_registry
from conditional_drifting.training import DriftingConfig, evaluate_residual_model, select_device, set_seed, train_conditional_drifting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWGN quickstart for conditional drifting")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-size", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=8_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    cfg = DriftingConfig(
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
    )
    awgn = channel_registry()["AWGN"]
    model, artifacts = train_conditional_drifting(awgn, cfg, device)
    evaluation = evaluate_residual_model(model, awgn, cfg, device)

    history = artifacts.history
    residual_true = evaluation["residual_true"].reshape(-1)
    residual_pred = evaluation["residual_pred"].reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot([h["epoch"] for h in history], [h["loss"] for h in history], lw=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE to drift target")

    axes[1].hist(residual_true, bins=80, density=True, alpha=0.55, label="true residual")
    axes[1].hist(residual_pred, bins=80, density=True, alpha=0.55, label="drifting residual")
    axes[1].set_title(f"Residual histogram, SWD={evaluation['swd']:.4f}")
    axes[1].set_xlabel("Residual")
    axes[1].legend()

    y_true = evaluation["y_true"]
    y_pred = evaluation["y_pred"]
    axes[2].scatter(y_true[:, 0], y_true[:, 1], s=8, alpha=0.25, label="true AWGN")
    axes[2].scatter(y_pred[:, 0], y_pred[:, 1], s=8, alpha=0.25, label="drifting")
    axes[2].set_title("Channel output samples")
    axes[2].set_xlabel("y[0]")
    axes[2].set_ylabel("y[1]")
    axes[2].legend()

    fig.suptitle(f"Conditional drifting quickstart on AWGN ({device})")
    fig.tight_layout()
    fig_path = os.path.join(ROOT, "results", "awgn_quickstart.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "device": str(device),
        "seed": args.seed,
        "config": artifacts.config,
        "drifting_swd": evaluation["swd"],
        "figure": fig_path,
    }
    json_path = os.path.join(ROOT, "results", "awgn_quickstart.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
