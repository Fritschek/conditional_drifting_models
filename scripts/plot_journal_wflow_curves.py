from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.45,
        "figure.dpi": 140,
    }
)


VARIANT_LABELS = {
    "analytic": "Analytic",
    "kernel_target": "Kernel target",
    "kernel_joint": "Kernel joint",
    "joint_sinkhorn": "Joint Sinkhorn",
    "fiber_sinkhorn": "Condition-wise Sinkhorn",
    "fiber_sinkhorn_marginal": "Marginal Sinkhorn",
    "wgan": "WGAN",
    "paper_wgan": "WGAN",
    "diffusion": "Diffusion DDIM",
    "diffusion_ddim100": "Diffusion DDIM-100",
    "ddim100": "Diffusion DDIM-100",
    "diffusion_ddpm": "Diffusion DDPM",
    "ddpm": "Diffusion DDPM",
}

COLORS = {
    "analytic": "#4d4d4d",
    "kernel_target": "#54a24b",
    "kernel_joint": "#b279a2",
    "joint_sinkhorn": "#ff9da6",
    "fiber_sinkhorn": "#9d755d",
    "fiber_sinkhorn_marginal": "#8cd17d",
    "wgan": "#f58518",
    "paper_wgan": "#f58518",
    "diffusion": "#4c78a8",
    "diffusion_ddim100": "#4c78a8",
    "ddim100": "#4c78a8",
    "diffusion_ddpm": "#72b7b2",
    "ddpm": "#72b7b2",
}

MARKERS = {
    "analytic": "o",
    "kernel_target": "s",
    "kernel_joint": "^",
    "joint_sinkhorn": "D",
    "fiber_sinkhorn": "P",
    "fiber_sinkhorn_marginal": "X",
    "wgan": "v",
    "paper_wgan": "v",
    "diffusion": "*",
    "diffusion_ddim100": "*",
    "ddim100": "*",
    "diffusion_ddpm": "h",
    "ddpm": "h",
}

MC_REPORT_FLOORS = {
    ("SSPA", "ber"): 1.67e-6,
    ("SSPA", "ser"): 1.0e-5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot journal BER/SER curves for learned channel surrogates.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("Journal_version/figures"))
    parser.add_argument("--latex-out", type=Path, default=None)
    parser.add_argument("--channels", type=str, default="")
    parser.add_argument("--variants", type=str, default="analytic,fiber_sinkhorn,wgan,diffusion_ddim100")
    parser.add_argument("--metrics", type=str, default="ber,ser", help="Comma-separated metrics to plot: ber,ser.")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def display_curve_values(channel: str, metric: str, means: list[float], sems: list[float]) -> tuple[list[float], list[list[float]], list[bool]]:
    floor = MC_REPORT_FLOORS.get((channel, metric))
    display_means: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    clipped: list[bool] = []
    for mean, sem in zip(means, sems):
        if floor is not None and mean < floor:
            display_means.append(floor)
            lower_errors.append(0.0)
            upper_errors.append(0.0)
            clipped.append(True)
            continue
        display_means.append(mean)
        min_positive = floor if floor is not None else max(mean * 1e-3, 1e-12)
        lower_errors.append(max(0.0, min(sem, mean - min_positive)))
        upper_errors.append(max(0.0, sem))
        clipped.append(False)
    return display_means, [lower_errors, upper_errors], clipped


def plot_channel(channel: str, variants: list[str], metrics: list[str], rows: list[dict[str, str]], out_dir: Path) -> None:
    channel_rows = [row for row in rows if row["channel"] == channel]
    fig_width = 4.9 if len(metrics) == 1 else 8.8
    fig, axes = plt.subplots(1, len(metrics), figsize=(fig_width, 3.2), sharex=True)
    axes_list = list(axes) if isinstance(axes, (list, tuple)) else list(axes.flat) if hasattr(axes, "flat") else [axes]
    metric_labels = {"ber": "BER", "ser": "SER"}
    for ax, metric in zip(axes_list, metrics):
        ylabel = metric_labels[metric]
        for variant in variants:
            pts = sorted(
                [row for row in channel_rows if row["variant"] == variant],
                key=lambda row: float(row["ebno_db"]),
            )
            if not pts:
                continue
            xs = [float(row["ebno_db"]) for row in pts]
            raw_ys = [float(row[f"{metric}_mean"]) for row in pts]
            raw_yerr = [float(row[f"{metric}_sem"]) for row in pts]
            ys, yerr, clipped = display_curve_values(channel, metric, raw_ys, raw_yerr)
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                label=VARIANT_LABELS.get(variant, variant),
                color=COLORS.get(variant),
                marker=MARKERS.get(variant, "o"),
                linewidth=1.45,
                markersize=3.6,
                capsize=2.0,
            )
            floor = MC_REPORT_FLOORS.get((channel, metric))
            if floor is not None:
                for x, is_clipped in zip(xs, clipped):
                    if is_clipped:
                        ax.text(x, floor * 1.25, "$<$", color=COLORS.get(variant), ha="center", va="bottom", fontsize=7)
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_b/N_0$ (dB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", linewidth=0.45, alpha=0.32)
        floor = MC_REPORT_FLOORS.get((channel, metric))
        if floor is not None:
            ax.axhline(floor, color="#777777", linewidth=0.6, linestyle=":", alpha=0.65)
        ax.set_title(f"{channel} {ylabel}")
    axes_list[-1].legend(loc="lower left", frameon=True, framealpha=0.88)
    fig.tight_layout()
    metric_tag = "_".join(metrics)
    stem = out_dir / f"wflow_{channel.lower()}_{metric_tag}_curves"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippet(path: Path, channels: list[str], metrics: list[str]) -> None:
    metric_tag = "_".join(metrics)
    metric_title = "/".join(metric.upper() for metric in metrics)
    metric_text = " and ".join(metric.upper() for metric in metrics)
    lines = [
        "% Auto-generated by scripts/plot_journal_wflow_curves.py.",
        "% Regenerate after the journal W-Flow curve suite has been aggregated.",
        "",
    ]
    for channel in channels:
        stem = f"wflow_{channel.lower()}_{metric_tag}_curves"
        label = f"fig:wflow-{channel.lower()}-{metric_tag.replace('_', '-')}-curves"
        lines.extend(
            [
                r"\begin{figure*}[t]",
                r"\centering",
                rf"\includegraphics[width=0.72\textwidth]{{figures/{stem}.pdf}}",
                rf"\caption{{\textbf{{{channel} {metric_title} curves for learned channel surrogates.}} Symbolic autoencoders are trained through each surrogate at the nominal training point and evaluated on the analytic channel over an $E_b/N_0$ grid. Curves report seed means with standard-error bars for {metric_text}; points below the single-run Monte Carlo resolution are clipped to the dotted reporting floor and marked as upper bounds.}}",
                rf"\label{{{label}}}",
                r"\end{figure*}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)
    variants = parse_csv_list(args.variants)
    metrics = parse_csv_list(args.metrics)
    invalid_metrics = sorted(set(metrics) - {"ber", "ser"})
    if invalid_metrics:
        raise ValueError(f"Unsupported metrics: {invalid_metrics}")
    channels = parse_csv_list(args.channels) or sorted({row["channel"] for row in rows})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for channel in channels:
        plot_channel(channel, variants, metrics, rows, args.out_dir)
    latex_out = args.latex_out or (args.out_dir.parent / "wflow_curve_figures.tex")
    write_latex_snippet(latex_out, channels, metrics)
    print({"channels": channels, "metrics": metrics, "out_dir": str(args.out_dir), "latex_out": str(latex_out)})


if __name__ == "__main__":
    main()
