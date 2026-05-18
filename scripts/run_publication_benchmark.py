from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ROOT)

from conditional_drifting.benchmark import BenchmarkConfig, run_single_channel_benchmark
from conditional_drifting.channels import channel_registry
from conditional_drifting.training import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication-style multi-seed benchmark for conditional drifting")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,OptFib")
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-size", type=int, default=20_000)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--drift-field", type=str, default="kernel", choices=["kernel", "sinkhorn", "fiber_sinkhorn"])
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--min-bandwidth", type=float, default=0.2)
    parser.add_argument("--max-drift-norm", type=float, default=2.0)
    parser.add_argument("--repulsive-weight", type=float, default=1.0)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=None)
    parser.add_argument("--sinkhorn-min-epsilon", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fiber-generated-samples", type=int, default=4)
    parser.add_argument("--fiber-positive-samples", type=int, default=4)
    parser.add_argument("--fiber-reference-samples", type=int, default=4)
    parser.add_argument("--swd-projections", type=int, default=256)
    parser.add_argument("--optfib-L", type=float, default=5000.0)
    parser.add_argument("--optfib-gamma", type=float, default=1.27)
    parser.add_argument("--optfib-kstep", type=int, default=50)
    parser.add_argument("--optfib-pn-dbm", type=float, default=-21.3)
    return parser.parse_args()


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def bootstrap_ci(values: list[float], alpha: float = 0.05, n_boot: int = 4000, seed: int = 12345) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[indices].mean(axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)

    optfib_params = {
        "L": args.optfib_L,
        "gamma": args.optfib_gamma,
        "Kstep": args.optfib_kstep,
        "Pn_dBm": args.optfib_pn_dbm,
        "use_noise_std": False,
    }
    channels = channel_registry(optfib_params)
    requested = [name.strip() for name in args.channels.split(",") if name.strip()]
    unknown = [name for name in requested if name not in channels]
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", f"publication_multiseed_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    cfg = BenchmarkConfig(
        noise_std=args.noise_std,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        drift_field=args.drift_field,
        drift_scale=args.drift_scale,
        min_bandwidth=args.min_bandwidth,
        max_drift_norm=args.max_drift_norm,
        repulsive_weight=args.repulsive_weight,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_min_epsilon=args.sinkhorn_min_epsilon,
        sinkhorn_iterations=args.sinkhorn_iterations,
        fiber_generated_samples=args.fiber_generated_samples,
        fiber_positive_samples=args.fiber_positive_samples,
        fiber_reference_samples=args.fiber_reference_samples,
        swd_projections=args.swd_projections,
    )

    meta = {
        "timestamp": timestamp,
        "device": str(device),
        "seeds": seeds,
        "channels": requested,
        "config": cfg.__dict__,
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rows = []
    for seed in seeds:
        set_seed(seed)
        for channel_name in requested:
            print(f"[seed={seed}] channel={channel_name}")
            result = run_single_channel_benchmark(channel_name, channels[channel_name], cfg, device)
            rows.append(
                {
                    "seed": seed,
                    "channel": channel_name,
                    "device": result["device"],
                    "drifting_swd": result["drifting_swd"],
                    "train_final_loss": result["train_final_loss"],
                    "train_final_drift_norm": result["train_final_drift_norm"],
                }
            )

    with open(os.path.join(out_dir, "per_seed_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_csv(
        os.path.join(out_dir, "per_seed_results.csv"),
        rows,
        ["seed", "channel", "device", "drifting_swd", "train_final_loss", "train_final_drift_norm"],
    )

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["channel"]].append(float(row["drifting_swd"]))

    summary_rows = []
    for channel_name in requested:
        values = grouped[channel_name]
        low, high = bootstrap_ci(values)
        summary_rows.append(
            {
                "channel": channel_name,
                "mean_swd": float(np.mean(values)),
                "std_swd": float(np.std(values, ddof=0)),
                "ci_low": low,
                "ci_high": high,
                "num_seeds": len(values),
            }
        )

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    write_csv(
        os.path.join(out_dir, "summary.csv"),
        summary_rows,
        ["channel", "mean_swd", "std_swd", "ci_low", "ci_high", "num_seeds"],
    )

    means = [row["mean_swd"] for row in summary_rows]
    lows = [row["mean_swd"] - row["ci_low"] for row in summary_rows]
    highs = [row["ci_high"] - row["mean_swd"] for row in summary_rows]
    labels = [row["channel"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, means, color="#2b7bba", alpha=0.9)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Residual SWD")
    ax.set_title("Conditional drifting benchmark (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_plot.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"output_dir": out_dir, "device": str(device), "num_runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
