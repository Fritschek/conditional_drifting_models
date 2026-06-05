from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CHANNEL_ORDER = ["AWGN", "Rayleigh", "SSPA", "TDL"]
VARIANT_ORDER = ["fiber_sinkhorn", "wgan", "diffusion_ddim100"]
VARIANT_LABELS = {
    "fiber_sinkhorn": "Condition-wise Sinkhorn",
    "wgan": "WGAN",
    "diffusion_ddim100": "DDIM-100",
}
VARIANT_TO_TIMING = {
    "fiber_sinkhorn": "fiber_sinkhorn",
    "wgan": "paper_wgan",
    "diffusion_ddim100": "ddim100",
}
COLORS = {
    "fiber_sinkhorn": "#9d755d",
    "wgan": "#f58518",
    "diffusion_ddim100": "#4c78a8",
}
MARKERS = {
    "fiber_sinkhorn": "P",
    "wgan": "v",
    "diffusion_ddim100": "*",
}
DEFAULT_TARGETS = {
    "AWGN": 1e-3,
    "Rayleigh": 1e-3,
    "SSPA": 1e-4,
    "TDL": 2e-3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNR-penalty versus inference-latency operating points.")
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=ROOT / "results/journal_wflow_curves_20260602_220608/journal_wflow_curve_per_seed.csv",
    )
    parser.add_argument("--timing-csv", type=Path, default=ROOT / "Journal_version/timing_summary_flat.csv")
    parser.add_argument("--out-pdf", type=Path, default=ROOT / "Journal_version/figures/wflow_snr_latency_tradeoff.pdf")
    parser.add_argument("--out-png", type=Path, default=ROOT / "Journal_version/figures/wflow_snr_latency_tradeoff.png")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "results/journal_strategy_diagnostics/snr_latency_tradeoff_points.csv",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(f"{channel}={value:g}" for channel, value in DEFAULT_TARGETS.items()),
        help="Comma-separated channel target SERs, e.g. AWGN=1e-3,Rayleigh=1e-3,SSPA=1e-4,TDL=2e-3.",
    )
    parser.add_argument("--channels", type=str, default=",".join(CHANNEL_ORDER))
    parser.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_targets(text: str) -> dict[str, float]:
    targets: dict[str, float] = {}
    for item in parse_csv_list(text):
        if "=" not in item:
            raise ValueError(f"Invalid target entry {item!r}; expected CHANNEL=value.")
        channel, value = item.split("=", 1)
        targets[channel.strip()] = float(value.strip())
    return targets


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    var = sum((value - center) ** 2 for value in values) / float(len(values) - 1)
    return math.sqrt(var / float(len(values)))


def interpolate_ebno(points: list[tuple[float, float]], target_ser: float) -> float | None:
    positive_points = sorted((x, y) for x, y in points if y > 0.0)
    if len(positive_points) < 2:
        return None
    for (x0, y0), (x1, y1) in zip(positive_points, positive_points[1:]):
        if y0 == target_ser:
            return x0
        if y1 == target_ser:
            return x1
        if (y0 - target_ser) * (y1 - target_ser) > 0:
            continue
        log_y0 = math.log10(y0)
        log_y1 = math.log10(y1)
        log_target = math.log10(target_ser)
        if log_y1 == log_y0:
            return 0.5 * (x0 + x1)
        weight = (log_target - log_y0) / (log_y1 - log_y0)
        return x0 + weight * (x1 - x0)
    return None


def build_curve_map(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], list[tuple[float, float]]]:
    curves: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        curves[(row["channel"], row["variant"], int(row["seed"]))].append((float(row["ebno_db"]), float(row["ser"])))
    return curves


def build_latency_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    latency: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("device") != "cuda":
            continue
        latency[(row["channel"], row["method_key"])] = float(row["milliseconds_per_sample"]) * 1000.0
    return latency


def build_points(
    curve_rows: list[dict[str, str]],
    timing_rows: list[dict[str, str]],
    channels: list[str],
    variants: list[str],
    targets: dict[str, float],
) -> list[dict[str, object]]:
    curves = build_curve_map(curve_rows)
    latency_map = build_latency_map(timing_rows)
    seeds_by_channel = {
        channel: sorted({seed for ch, variant, seed in curves.keys() if ch == channel and variant == "analytic"})
        for channel in channels
    }
    points: list[dict[str, object]] = []
    for channel in channels:
        target = targets[channel]
        for variant in variants:
            penalties: list[float] = []
            used_seeds = 0
            missing_seeds = 0
            for seed in seeds_by_channel[channel]:
                analytic_ebno = interpolate_ebno(curves.get((channel, "analytic", seed), []), target)
                variant_ebno = interpolate_ebno(curves.get((channel, variant, seed), []), target)
                if analytic_ebno is None or variant_ebno is None:
                    missing_seeds += 1
                    continue
                penalties.append(variant_ebno - analytic_ebno)
                used_seeds += 1
            if not penalties:
                raise ValueError(f"No valid SNR-penalty points for {channel}/{variant} at SER={target:g}.")
            timing_key = VARIANT_TO_TIMING[variant]
            points.append(
                {
                    "channel": channel,
                    "variant": variant,
                    "label": VARIANT_LABELS.get(variant, variant),
                    "target_ser": target,
                    "latency_us": latency_map[(channel, timing_key)],
                    "snr_penalty_db_mean": mean(penalties),
                    "snr_penalty_db_sem": sem(penalties),
                    "num_seeds": used_seeds,
                    "missing_seeds": missing_seeds,
                }
            )
    return points


def write_points(path: Path, points: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "channel",
        "variant",
        "label",
        "target_ser",
        "latency_us",
        "snr_penalty_db_mean",
        "snr_penalty_db_sem",
        "num_seeds",
        "missing_seeds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)


def target_label(value: float) -> str:
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10**exponent)
    if abs(mantissa - 1.0) < 1e-9:
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:g}\times10^{{{exponent}}}$"


def plot(points: list[dict[str, object]], channels: list[str], variants: list[str], out_pdf: Path, out_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.35), sharex=True, sharey=True)
    axes_list = list(axes.flat)
    y_values = [
        float(point["snr_penalty_db_mean"]) + sign * float(point["snr_penalty_db_sem"])
        for point in points
        for sign in (-1.0, 1.0)
    ]
    y_min = min(min(y_values) - 0.15, -0.15)
    y_max = max(max(y_values) + 0.2, 0.4)
    for ax, channel in zip(axes_list, channels):
        channel_points = [point for point in points if point["channel"] == channel]
        for point in channel_points:
            variant = str(point["variant"])
            ax.errorbar(
                float(point["latency_us"]),
                float(point["snr_penalty_db_mean"]),
                yerr=float(point["snr_penalty_db_sem"]),
                marker=MARKERS[variant],
                color=COLORS[variant],
                markeredgecolor="#222222",
                markeredgewidth=0.35,
                linestyle="none",
                markersize=6.2 if variant != "diffusion_ddim100" else 8.5,
                capsize=2.4,
                linewidth=0.8,
            )
        target = channel_points[0]["target_ser"]
        ax.axhline(0.0, color="#555555", linewidth=0.7, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlim(0.075, 70.0)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"{channel}  target SER {target_label(float(target))}")
        ax.grid(True, which="major", alpha=0.32)
        ax.grid(True, which="minor", axis="x", alpha=0.16)
        if ax in axes[:, 0]:
            ax.set_ylabel(r"SNR penalty $\Delta E_b/N_0$ [dB]")
        if ax in axes[-1, :]:
            ax.set_xlabel(r"Inference latency per sample [$\mu$s]")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=MARKERS[variant],
            color=COLORS[variant],
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            linestyle="none",
            markersize=6.2 if variant != "diffusion_ddim100" else 8.5,
            label=VARIANT_LABELS[variant],
        )
        for variant in variants
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=True, bbox_to_anchor=(0.5, 1.015))
    fig.text(0.53, 0.045, "lower left is better", ha="center", va="center", color="#555555", fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    channels = parse_csv_list(args.channels)
    variants = parse_csv_list(args.variants)
    targets = parse_targets(args.targets)
    missing_targets = [channel for channel in channels if channel not in targets]
    if missing_targets:
        raise ValueError(f"Missing target SERs for channels: {missing_targets}")
    curve_rows = read_csv(args.curve_csv)
    timing_rows = read_csv(args.timing_csv)
    points = build_points(curve_rows, timing_rows, channels, variants, targets)
    write_points(args.out_csv, points)
    plot(points, channels, variants, args.out_pdf, args.out_png)
    print(
        {
            "figure": str(args.out_pdf),
            "csv": str(args.out_csv),
            "num_points": len(points),
        }
    )


if __name__ == "__main__":
    main()
