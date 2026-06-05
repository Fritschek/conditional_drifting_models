from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

CHANNEL_ORDER = ["AWGN", "Rayleigh", "SSPA", "TDL"]
CHANNEL_COLORS = {
    "AWGN": "#4c78a8",
    "Rayleigh": "#f58518",
    "SSPA": "#54a24b",
    "TDL": "#b279a2",
}
VARIANT_LABELS = {
    "kernel_target": "Kernel target",
    "kernel_joint": "Kernel joint",
    "joint_sinkhorn": "Joint Sinkhorn",
    "fiber_sinkhorn": "Cond. Sinkhorn",
    "wgan": "WGAN",
    "diffusion_ddim100": "DDIM-100",
}
VARIANT_MARKERS = {
    "kernel_target": "s",
    "kernel_joint": "^",
    "joint_sinkhorn": "D",
    "fiber_sinkhorn": "P",
    "wgan": "v",
    "diffusion_ddim100": "*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot journal metric-mismatch and latency diagnostics.")
    parser.add_argument(
        "--wflow-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_paper_hpc_20260519_064227/journal_wflow_suite_results.json",
    )
    parser.add_argument(
        "--wflow-tdl-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_paper_hpc_20260531_143208/journal_wflow_suite_results.json",
    )
    parser.add_argument(
        "--wflow-fiber-fixed-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_fiber_fixed_paper_hpc_20260601_161729/journal_wflow_suite_results.json",
    )
    parser.add_argument(
        "--wflow-sspa-screen-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_sspa_budget_screen_20260602_071106/journal_wflow_suite_results.json",
    )
    parser.add_argument(
        "--ser-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_ser_20260526_082924/journal_wflow_ser_results.json",
    )
    parser.add_argument(
        "--ser-tdl-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_ser_tdl_20260531_151126/journal_wflow_ser_results.json",
    )
    parser.add_argument(
        "--ser-fiber-fixed-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_ser_fiber_fixed_20260601_201405/journal_wflow_ser_results.json",
    )
    parser.add_argument(
        "--ser-sspa-screen-suite",
        type=Path,
        default=ROOT / "results/journal_wflow_ser_sspa_budget_screen_20260602_073353/journal_wflow_ser_results.json",
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=ROOT / "Journal_version/timing_summary_flat.csv",
    )
    parser.add_argument(
        "--wallclock-csv",
        type=Path,
        default=ROOT / "results/journal_equal_wallclock_symbolic_20260604_155906/equal_wallclock_symbolic_summary.csv",
    )
    parser.add_argument("--out-pdf", type=Path, default=ROOT / "Journal_version/figures/wflow_metric_latency_diagnostics.pdf")
    parser.add_argument("--out-png", type=Path, default=ROOT / "Journal_version/figures/wflow_metric_latency_diagnostics.png")
    parser.add_argument("--out-data-dir", type=Path, default=ROOT / "results/journal_strategy_diagnostics")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_block(data: dict, variant: str, channel: str, metric: str, field: str = "mean") -> float:
    return float(data["aggregated"][variant][channel][metric][field])


def build_metric_points(args: argparse.Namespace) -> list[dict[str, object]]:
    wflow = load_json(args.wflow_suite)
    wflow_tdl = load_json(args.wflow_tdl_suite)
    wflow_fiber = load_json(args.wflow_fiber_fixed_suite)
    wflow_sspa = load_json(args.wflow_sspa_screen_suite)
    ser = load_json(args.ser_suite)
    ser_tdl = load_json(args.ser_tdl_suite)
    ser_fiber = load_json(args.ser_fiber_fixed_suite)
    ser_sspa = load_json(args.ser_sspa_screen_suite)

    points: list[dict[str, object]] = []

    def add(channel: str, variant: str, swd_data: dict, ser_data: dict) -> None:
        points.append(
            {
                "channel": channel,
                "variant": variant,
                "direct_swd": metric_block(swd_data, variant, channel, "direct_swd"),
                "direct_swd_sem": metric_block(swd_data, variant, channel, "direct_swd", "sem"),
                "ser": metric_block(ser_data, variant, channel, "final_eval_ser"),
                "ser_sem": metric_block(ser_data, variant, channel, "final_eval_ser", "sem"),
            }
        )

    for channel in ["AWGN", "Rayleigh"]:
        for variant in ["kernel_target", "kernel_joint", "joint_sinkhorn"]:
            add(channel, variant, wflow, ser)
        add(channel, "fiber_sinkhorn", wflow_fiber, ser_fiber)

    add("SSPA", "kernel_target", wflow_sspa, ser_sspa)
    add("SSPA", "fiber_sinkhorn", wflow_sspa, ser_sspa)

    for variant in ["kernel_target", "kernel_joint", "joint_sinkhorn"]:
        add("TDL", variant, wflow_tdl, ser_tdl)
    add("TDL", "fiber_sinkhorn", wflow_fiber, ser_fiber)

    return points


def build_pareto_points(args: argparse.Namespace) -> list[dict[str, object]]:
    timing_rows = read_csv(args.timing_csv)
    wallclock_rows = read_csv(args.wallclock_csv)
    timing_map = {(row["channel"], row["method_key"]): row for row in timing_rows if row["device"] == "cuda"}
    wallclock_map = {(row["channel"], row["variant"]): row for row in wallclock_rows}
    variant_to_timing = {
        "fiber_sinkhorn": "fiber_sinkhorn",
        "wgan": "paper_wgan",
        "diffusion_ddim100": "ddim100",
    }
    points: list[dict[str, object]] = []
    for channel in CHANNEL_ORDER:
        for variant, timing_key in variant_to_timing.items():
            timing = timing_map[(channel, timing_key)]
            wallclock = wallclock_map[(channel, variant)]
            points.append(
                {
                    "channel": channel,
                    "variant": variant,
                    "milliseconds_per_sample": float(timing["milliseconds_per_sample"]),
                    "ser": float(wallclock["eval_ser_mean"]),
                    "ser_sem": float(wallclock["eval_ser_sem"]),
                    "updates": float(wallclock["optimizer_updates_mean"]),
                }
            )
    return points


def write_points(path: Path, points: list[dict[str, object]]) -> None:
    if not points:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(points[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)


def plot(args: argparse.Namespace, metric_points: list[dict[str, object]], pareto_points: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15))
    ax_metric, ax_pareto = axes

    for point in metric_points:
        channel = str(point["channel"])
        variant = str(point["variant"])
        ax_metric.scatter(
            float(point["direct_swd"]),
            float(point["ser"]),
            marker=VARIANT_MARKERS[variant],
            s=42 if variant != "fiber_sinkhorn" else 58,
            color=CHANNEL_COLORS[channel],
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.9,
        )
    ax_metric.set_xscale("log")
    ax_metric.set_yscale("log")
    ax_metric.set_xlabel("Direct-output SWD")
    ax_metric.set_ylabel("Downstream SER")
    ax_metric.set_title("(a) Global SWD can mis-rank surrogates")
    ax_metric.grid(True, which="major", alpha=0.32)
    ax_metric.grid(True, which="minor", alpha=0.12)

    for point in pareto_points:
        channel = str(point["channel"])
        variant = str(point["variant"])
        ax_pareto.scatter(
            float(point["milliseconds_per_sample"]),
            float(point["ser"]),
            marker=VARIANT_MARKERS[variant],
            s=46 if variant != "diffusion_ddim100" else 70,
            color=CHANNEL_COLORS[channel],
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.9,
        )
    ax_pareto.set_xscale("log")
    ax_pareto.set_yscale("log")
    ax_pareto.set_xlabel("Inference time per sample [ms]")
    ax_pareto.set_ylabel("Equal-time downstream SER")
    ax_pareto.set_title("(b) Accuracy-latency operating points")
    ax_pareto.grid(True, which="major", alpha=0.32)
    ax_pareto.grid(True, which="minor", alpha=0.12)

    channel_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CHANNEL_COLORS[ch], markeredgecolor="#222222", markersize=5.5, label=ch)
        for ch in CHANNEL_ORDER
    ]
    variant_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=VARIANT_MARKERS[var],
            color="#555555",
            linestyle="none",
            markerfacecolor="#dddddd",
            markeredgecolor="#222222",
            markersize=5.5,
            label=VARIANT_LABELS[var],
        )
        for var in ["kernel_target", "kernel_joint", "joint_sinkhorn", "fiber_sinkhorn", "wgan", "diffusion_ddim100"]
    ]
    fig.legend(handles=channel_handles, loc="lower center", bbox_to_anchor=(0.29, -0.02), ncol=4, frameon=False)
    fig.legend(handles=variant_handles, loc="lower center", bbox_to_anchor=(0.74, -0.08), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metric_points = build_metric_points(args)
    pareto_points = build_pareto_points(args)
    write_points(args.out_data_dir / "swd_ser_metric_points.csv", metric_points)
    write_points(args.out_data_dir / "latency_pareto_points.csv", pareto_points)
    plot(args, metric_points, pareto_points)
    print(
        json.dumps(
            {
                "metric_points": len(metric_points),
                "pareto_points": len(pareto_points),
                "figure_pdf": str(args.out_pdf),
                "figure_png": str(args.out_png),
                "data_dir": str(args.out_data_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
