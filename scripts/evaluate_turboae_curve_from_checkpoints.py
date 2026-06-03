from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.e2e_implants import AnalyticChannelImplant
from scripts.run_e2e_channel_implant_benchmark import build_autoencoder, evaluate_model

VENDORED_TURBO_ROOT = ROOT / "external" / "turbo_mingru_decoder"


MODE_LABELS = {
    "analytic": "Analytic channel training",
    "checkpoint": "Fiber-Sinkhorn training",
}


def parse_int_csv(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def load_turboae_checkpoint(
    checkpoint: Path,
    *,
    turbo_root: Path,
    device: torch.device,
    batch_size: int | None,
):
    payload = torch.load(checkpoint, map_location=device)
    config = dict(payload["config"])
    effective_batch_size = int(batch_size or config["batch_size"])
    encoder, decoder = build_autoencoder(
        turbo_root,
        str(payload.get("model_type", "cnn_turbo")),
        effective_batch_size,
        int(config["sequence_length"]),
        int(config["enc_num_layers"]),
        int(config["dec_num_layers"]),
        int(config["hidden_size_gru"]),
        int(config["num_iteration"]),
        int(config["num_iter_ft_cnn"]),
        device,
    )
    encoder.load_state_dict(payload["encoder_state"])
    decoder.load_state_dict(payload["decoder_state"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, payload, config


def evaluate_checkpoint_curve(
    checkpoint: Path,
    *,
    turbo_root: Path,
    device: torch.device,
    channel: str,
    ebno_values: list[float],
    eval_num_blocks: int,
    batch_size: int | None,
    amp: bool,
    amp_dtype: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoder, decoder, payload, config = load_turboae_checkpoint(
        checkpoint,
        turbo_root=turbo_root,
        device=device,
        batch_size=batch_size,
    )
    effective_batch_size = int(batch_size or config["batch_size"])
    eval_batches = max(1, int(math.ceil(float(eval_num_blocks) / float(effective_batch_size))))
    rate = float(config["sequence_length"]) / float(config["channel_length"])
    implant = AnalyticChannelImplant(channel)

    rows: list[dict[str, Any]] = []
    for ebno_db in ebno_values:
        stats = evaluate_model(
            encoder,
            decoder,
            implant,
            eval_batches=eval_batches,
            batch_size=effective_batch_size,
            sequence_length=int(config["sequence_length"]),
            num_symbols=2,
            ebno_db=float(ebno_db),
            rate=rate,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        rows.append(
            {
                "ebno_db": float(ebno_db),
                "eval_loss": float(stats["loss"]),
                "eval_ber": float(stats["ber"]),
                "eval_bler": float(stats["bler"]),
                "eval_batches": eval_batches,
                "eval_blocks": eval_batches * effective_batch_size,
            }
        )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metrics": payload.get("metrics"),
        "config": config,
        "rate": rate,
        "batch_size": effective_batch_size,
        "eval_batches": eval_batches,
        "eval_blocks": eval_batches * effective_batch_size,
    }
    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "mode",
        "label",
        "ebno_db",
        "eval_loss",
        "eval_ber",
        "eval_bler",
        "eval_batches",
        "eval_blocks",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), float(row["ebno_db"]))].append(row)

    aggregate: list[dict[str, Any]] = []
    for (mode, ebno_db), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out: dict[str, Any] = {
            "mode": mode,
            "label": MODE_LABELS.get(mode, mode),
            "ebno_db": ebno_db,
            "n": len(group),
        }
        for metric in ("eval_loss", "eval_ber", "eval_bler"):
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            out[f"{metric}_ci95"] = ci95(values)
            out[f"{metric}_median"] = float(np.median(values))
        aggregate.append(out)
    return aggregate


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "label",
        "ebno_db",
        "n",
        "eval_loss_mean",
        "eval_loss_std",
        "eval_loss_ci95",
        "eval_loss_median",
        "eval_ber_mean",
        "eval_ber_std",
        "eval_ber_ci95",
        "eval_ber_median",
        "eval_bler_mean",
        "eval_bler_std",
        "eval_bler_ci95",
        "eval_bler_median",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True)
    colors = {"analytic": "#2f6f8f", "checkpoint": "#b54a3a"}
    markers = {"analytic": "o", "checkpoint": "s"}
    for mode in ("analytic", "checkpoint"):
        pts = sorted(grouped.get(mode, []), key=lambda row: float(row["ebno_db"]))
        if not pts:
            continue
        x = np.array([float(row["ebno_db"]) for row in pts])
        for ax, metric, ylabel in [
            (axes[0], "eval_ber", "BER"),
            (axes[1], "eval_bler", "BLER"),
        ]:
            y = np.array([max(float(row[f"{metric}_mean"]), 1e-8) for row in pts])
            yerr = np.array([float(row[f"{metric}_ci95"]) for row in pts])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker=markers.get(mode, "o"),
                linewidth=1.7,
                markersize=4.2,
                capsize=2.5,
                color=colors.get(mode),
                label=MODE_LABELS.get(mode, mode),
            )
            ax.set_yscale("log")
            ax.set_xlabel(r"$E_b/N_0$ [dB]")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", linewidth=0.45, alpha=0.38)
    axes[0].legend(loc="best", fontsize=7.6, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BER/BLER curves from trained TurboAE checkpoints.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--modes", type=str, default="analytic,checkpoint")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "TDL", "OptFib"])
    parser.add_argument("--ebno-values", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--eval-num-blocks", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--turbo-root", type=Path, default=VENDORED_TURBO_ROOT)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = parse_int_csv(args.seeds) if args.seeds else list(range(args.seed_start, args.seed_start + args.num_seeds))
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    ebno_values = parse_float_csv(args.ebno_values)
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else None

    all_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in modes:
            checkpoint = args.suite_dir / f"seed{seed}" / f"{mode}_l{args.length}" / "best_eval.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing TurboAE checkpoint: {checkpoint}")
            print(f"[curve] seed={seed} mode={mode} checkpoint={checkpoint}", flush=True)
            rows, meta = evaluate_checkpoint_curve(
                checkpoint,
                turbo_root=args.turbo_root,
                device=device,
                channel=args.channel,
                ebno_values=ebno_values,
                eval_num_blocks=args.eval_num_blocks,
                batch_size=batch_size,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
            )
            metadata.append({"seed": seed, "mode": mode, **meta})
            for row in rows:
                all_rows.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "label": MODE_LABELS.get(mode, mode),
                        "checkpoint": str(checkpoint),
                        **row,
                    }
                )

    aggregate = aggregate_rows(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "turboae_curve_per_seed.csv", all_rows)
    write_aggregate_csv(args.out_dir / "turboae_curve_summary.csv", aggregate)
    (args.out_dir / "turboae_curve_results.json").write_text(
        json.dumps(
            {
                "suite_dir": str(args.suite_dir),
                "channel": args.channel,
                "length": args.length,
                "seeds": seeds,
                "modes": modes,
                "ebno_values": ebno_values,
                "eval_num_blocks_requested": args.eval_num_blocks,
                "metadata": metadata,
                "summary": aggregate,
            },
            indent=2,
        )
    )
    plot_curves(aggregate, args.out_dir / "turboae_ber_bler_curve.pdf")
    print(
        json.dumps(
            {
                "per_seed_csv": str(args.out_dir / "turboae_curve_per_seed.csv"),
                "summary_csv": str(args.out_dir / "turboae_curve_summary.csv"),
                "figure": str(args.out_dir / "turboae_ber_bler_curve.pdf"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
