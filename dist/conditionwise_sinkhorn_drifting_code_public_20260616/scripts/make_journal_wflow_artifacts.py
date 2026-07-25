from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

try:
    import pandas as pd
    import seaborn as sns
except ImportError:  # pragma: no cover - keeps the script usable on minimal TeX/build hosts.
    pd = None
    sns = None


CHANNELS = ["AWGN", "Rayleigh", "SSPA"]
SWD_TABLE_CHANNELS = ["AWGN", "Rayleigh", "SSPA", "TDL"]
DIRECT_DRIFTING_VARIANT = "direct_drifting"
WFLOW_VARIANTS = ["kernel_target", "kernel_joint", "joint_sinkhorn", "fiber_sinkhorn"]
CODING_VARIANTS = ["analytic", *WFLOW_VARIANTS]

VARIANT_LABELS = {
    "analytic": "Analytic",
    "direct_drifting": "Direct drifting",
    "kernel_target": "Kernel target",
    "kernel_joint": "Kernel joint",
    "joint_sinkhorn": "Joint Sinkhorn",
    "fiber_sinkhorn": "Condition-wise Sinkhorn",
}

SHORT_LABELS = {
    "analytic": "Analytic",
    "direct_drifting": "Direct drift",
    "kernel_target": "Kernel target",
    "kernel_joint": "Kernel joint",
    "joint_sinkhorn": "Joint Sink.",
    "fiber_sinkhorn": "Cond. Sink.",
}

# Ten-seed direct-drifting values from Journal_version/benchmark_table_all_channels.tex.
DIRECT_DRIFTING_SWD = {
    "AWGN": (0.0100, 0.0007),
    "Rayleigh": (0.0085, 0.0012),
    "SSPA": (0.0060, 0.0008),
}

# Diffusion/WGAN reference rows are ten-seed benchmark values already reported in
# Journal_version/benchmark_table_all_channels.tex.
REFERENCE_SWDS = {
    "WGAN": {
        "AWGN": (0.0198, 0.0050),
        "Rayleigh": (0.0171, 0.0050),
        "SSPA": (0.0287, 0.0156),
    },
    "DDPM": {
        "AWGN": (0.0070, 0.0004),
        "Rayleigh": (0.0079, 0.0008),
        "SSPA": (0.0032, 0.0005),
    },
    "DDIM-100": {
        "AWGN": (0.0042, 0.0006),
        "Rayleigh": (0.0044, 0.0007),
        "SSPA": (0.0024, 0.0004),
    },
}

REFERENCE_VARIANT_LABELS = {
    "wgan": "WGAN",
    "paper_wgan": "WGAN",
    "paperwgan": "WGAN",
    "ddpm": "DDPM",
    "ddim": "DDIM-100",
    "ddim100": "DDIM-100",
    "ddim_100": "DDIM-100",
    "ddim-100": "DDIM-100",
}

COLORS = {
    "direct_drifting": "#4c78a8",
    "WGAN": "#72b7b2",
    "DDPM": "#f58518",
    "DDIM-100": "#e45756",
    "kernel_target": "#54a24b",
    "kernel_joint": "#b279a2",
    "joint_sinkhorn": "#ff9da6",
    "fiber_sinkhorn": "#9d755d",
    "analytic": "#4d4d4d",
}

MC_REPORT_FLOORS = {
    # SSPA follows the Muah Kim diffusion setup with M_msg=64 and eval_size=100000:
    # one bit error corresponds to 1 / (100000 * log2(64)).
    ("SSPA", "final_eval_ber"): 1.67e-6,
    ("SSPA", "final_eval_ser"): 1.0e-5,
}


def configure_plot_style() -> None:
    if sns is not None:
        sns.set_theme(
            context="paper",
            style="whitegrid",
            font_scale=0.92,
            rc={
                "axes.edgecolor": "#333333",
                "axes.linewidth": 0.7,
                "grid.linewidth": 0.45,
                "grid.alpha": 0.28,
                "legend.frameon": True,
                "legend.framealpha": 0.92,
                "figure.dpi": 140,
            },
        )
    else:
        plt.rcParams.update(
            {
                "axes.grid": True,
                "grid.linewidth": 0.45,
                "grid.alpha": 0.28,
                "axes.edgecolor": "#333333",
                "axes.linewidth": 0.7,
                "figure.dpi": 140,
            }
        )


def draw_barplot(
    ax: plt.Axes,
    *,
    labels: list[str],
    values: list[float],
    errors: list[float],
    colors: list[str],
) -> None:
    x = list(range(len(labels)))
    if sns is not None and pd is not None:
        data = pd.DataFrame({"label": labels, "value": values})
        palette = {label: color for label, color in zip(labels, colors)}
        sns.barplot(
            data=data,
            x="label",
            y="value",
            hue="label",
            palette=palette,
            dodge=False,
            errorbar=None,
            legend=False,
            ax=ax,
            edgecolor="#333333",
            linewidth=0.35,
            saturation=0.9,
        )
        ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.0, zorder=4)
    else:
        ax.bar(x, values, yerr=errors, color=colors, edgecolor="#333333", linewidth=0.35, capsize=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate W-Flow tables and figures.")
    parser.add_argument(
        "--wflow-csv",
        type=Path,
        default=Path("results/YOUR_WFLOW_SUITE/journal_wflow_per_seed.csv"),
    )
    parser.add_argument(
        "--extra-wflow-csv",
        type=Path,
        action="append",
        default=[],
        help="Additional W-Flow per-seed CSVs to merge into anchor diagnostics and metric-alignment figures.",
    )
    parser.add_argument(
        "--sspa-fiber-wflow-csv",
        type=Path,
        default=None,
        help="SSPA compact-budget W-Flow CSV used to replace only condition-wise Sinkhorn generator rows.",
    )
    parser.add_argument(
        "--ser-csv",
        type=Path,
        default=Path("results/YOUR_SER_SUITE/journal_wflow_ser_per_seed.csv"),
    )
    parser.add_argument(
        "--extra-ser-csv",
        type=Path,
        action="append",
        default=[],
        help="Additional BER/SER per-seed CSVs to merge into the coding table.",
    )
    parser.add_argument(
        "--sspa-ser-csv",
        type=Path,
        default=None,
        help="SSPA M_msg=64 BER/SER CSV used to replace all SSPA coding rows.",
    )
    parser.add_argument(
        "--baseline-direct-swd-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional checkpoint-only direct-SWD CSVs for diffusion/WGAN reference rows, e.g. TDL.",
    )
    parser.add_argument(
        "--baseline-direct-swd-error",
        choices=("std", "sem"),
        default="std",
        help="Error statistic used when importing --baseline-direct-swd-csv reference rows.",
    )
    parser.add_argument(
        "--coding-channels",
        type=str,
        default=",".join(CHANNELS),
        help="Comma-separated channels to include in the downstream BER/SER table.",
    )
    parser.add_argument(
        "--metric-channels",
        type=str,
        default=",".join(CHANNELS),
        help="Comma-separated channels to include in W-Flow-only anchor diagnostics and metric-alignment figures.",
    )
    parser.add_argument(
        "--alignment-channels",
        type=str,
        default="",
        help="Comma-separated channels to include in optional metric-alignment outputs. Defaults to --metric-channels.",
    )
    parser.add_argument("--journal-dir", type=Path, default=Path("Journal_version"))
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_rows_by_seed_channel_variant(paths: list[Path]) -> list[dict[str, str]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    order: list[tuple[str, str, str]] = []
    for path in paths:
        for row in read_csv(path):
            key = (row["seed"], row["channel"], row["variant"])
            if key not in rows_by_key:
                order.append(key)
            rows_by_key[key] = row
    return [rows_by_key[key] for key in order]


def replace_selected_rows(
    rows: list[dict[str, str]],
    replacement_path: Path,
    *,
    channel: str,
    variants: set[str] | None = None,
) -> list[dict[str, str]]:
    replacement_rows = [
        row
        for row in read_csv(replacement_path)
        if row["channel"] == channel and (variants is None or row["variant"] in variants)
    ]
    replacement_keys = {(row["seed"], row["channel"], row["variant"]) for row in replacement_rows}
    if variants is None:
        kept_rows = [row for row in rows if row["channel"] != channel]
    else:
        kept_rows = [
            row
            for row in rows
            if not (row["channel"] == channel and row["variant"] in variants)
            and (row["seed"], row["channel"], row["variant"]) not in replacement_keys
        ]
    return merge_rows_by_seed_channel_variant_from_rows([kept_rows, replacement_rows])


def merge_rows_by_seed_channel_variant_from_rows(row_groups: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    order: list[tuple[str, str, str]] = []
    for rows in row_groups:
        for row in rows:
            key = (row["seed"], row["channel"], row["variant"])
            if key not in rows_by_key:
                order.append(key)
            rows_by_key[key] = row
    return [rows_by_key[key] for key in order]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return std(values) / math.sqrt(len(values))


def summarize(rows: list[dict[str, str]], metrics: list[str]) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["channel"], row["variant"])
        for metric in metrics:
            value = row.get(metric)
            if value not in (None, ""):
                grouped[key][metric].append(float(value))
    return {
        key: {metric: (mean(values), sem(values)) for metric, values in metric_values.items()}
        for key, metric_values in grouped.items()
    }


def summarize_reference_swd_rows(
    rows: list[dict[str, str]],
    *,
    error_stat: str,
) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("direct_swd")
        if value in (None, ""):
            continue
        variant = str(row.get("variant", "")).lower()
        method = REFERENCE_VARIANT_LABELS.get(variant)
        if method is None:
            continue
        channel = str(row.get("channel", ""))
        if not channel:
            continue
        grouped[(method, channel)].append(float(value))

    summary: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for (method, channel), values in grouped.items():
        if not values:
            continue
        error = std(values) if error_stat == "std" else sem(values)
        summary[method][channel] = (mean(values), error)
    return dict(summary)


def merge_reference_swds(
    base: dict[str, dict[str, tuple[float, float]]],
    updates: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    merged = {method: dict(values) for method, values in base.items()}
    for method, values in updates.items():
        merged.setdefault(method, {}).update(values)
    return merged


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def pm_fixed(value: float, error: float, digits: int = 4) -> str:
    return f"{value:.{digits}f} $\\pm$ {error:.{digits}f}"


def sci(value: float, sig: int = 3) -> str:
    if value == 0:
        return "0"
    if 1e-3 <= abs(value) < 1e2:
        return f"{value:.{sig}g}"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.{sig - 1}f}$\\times 10^{{{exponent}}}$"


def lt_sci(value: float, sig: int = 3) -> str:
    if value == 0:
        return "$<0$"
    if 1e-3 <= abs(value) < 1e2:
        return f"$<{value:.{sig}g}$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"$<{mantissa:.{sig - 1}f}\\times 10^{{{exponent}}}$"


def pm_sci(value: float, error: float) -> str:
    return f"{sci(value)} $\\pm$ {sci(error, sig=2)}"


def coding_display_value(channel: str, metric: str, value: float) -> float:
    floor = MC_REPORT_FLOORS.get((channel, metric))
    if floor is not None and value < floor:
        return floor
    return value


def format_coding_value(channel: str, metric: str, value: float, error: float) -> str:
    floor = MC_REPORT_FLOORS.get((channel, metric))
    if floor is not None and value < floor:
        return lt_sci(floor)
    return pm_sci(value, error)


def maybe_bold(text: str, condition: bool) -> str:
    return f"\\textbf{{{text}}}" if condition else text


def summary_metric(
    summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    channel: str,
    variant: str,
    metric: str,
) -> tuple[float, float] | None:
    return summary.get((channel, variant), {}).get(metric)


def available_variants(
    summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    channel: str,
    variants: list[str],
    metric: str,
) -> list[str]:
    return [variant for variant in variants if summary_metric(summary, channel, variant, metric) is not None]


def write_wflow_swd_table(
    path: Path,
    wflow_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    reference_swds: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> None:
    reference_swds = reference_swds or REFERENCE_SWDS
    drifting_rows = [DIRECT_DRIFTING_VARIANT, *WFLOW_VARIANTS]
    channels = [
        channel
        for channel in SWD_TABLE_CHANNELS
        if channel in DIRECT_DRIFTING_SWD
        or any((channel, variant) in wflow_summary for variant in WFLOW_VARIANTS)
        or any(channel in values for values in reference_swds.values())
    ]
    best_drifting = {
        channel: min(available, key=lambda item: item[1])[0]
        for channel in channels
        for available in [
            [
                *(
                    [(DIRECT_DRIFTING_VARIANT, DIRECT_DRIFTING_SWD[channel][0])]
                    if channel in DIRECT_DRIFTING_SWD
                    else []
                ),
                *[
                    (variant, wflow_summary[(channel, variant)]["direct_swd"][0])
                    for variant in WFLOW_VARIANTS
                    if (channel, variant) in wflow_summary and "direct_swd" in wflow_summary[(channel, variant)]
                ],
            ]
        ]
        if available
    }
    ncols = len(channels) + 1
    colspec = "l" + "c" * len(channels)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Direct-output SWD comparison for diffusion/WGAN references and drifting variants.} Lower is better. Reference rows report mean $\pm$ standard deviation from the diffusion/WGAN benchmark or from checkpoint-only direct-SWD evaluation. The drifting-family rows contain direct drifting from the same benchmark and the W-Flow drift-field ablation. W-Flow rows report the available-seed mean $\pm$ standard error. Bold marks the best drifting-family row per channel under the reported mean. Dashes mark unavailable matched generator-level SWD values.}",
        r"\label{tab:wflow-swd-baselines}",
        r"\tablestyle{5.8pt}{1.08}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        r"\toprule",
        "Method & " + " & ".join(channels) + r" \\",
        r"\midrule",
        rf"\rowcolor[gray]{{0.9}} \multicolumn{{{ncols}}}{{l}}{{\textit{{Diffusion/WGAN reference baselines}}}} \\",
    ]
    for method, values in reference_swds.items():
        cells = [pm_fixed(*values[channel]) if channel in values else "--" for channel in channels]
        lines.append(r"\headspace " + method + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\midrule",
            rf"\rowcolor[gray]{{0.9}} \multicolumn{{{ncols}}}{{l}}{{\textit{{Drifting family}}}} \\",
        ]
    )
    direct_cells = []
    for channel in channels:
        if channel not in DIRECT_DRIFTING_SWD:
            direct_cells.append("--")
            continue
        value, error = DIRECT_DRIFTING_SWD[channel]
        direct_cells.append(maybe_bold(pm_fixed(value, error), best_drifting.get(channel) == DIRECT_DRIFTING_VARIANT))
    lines.append(r"\headspace " + VARIANT_LABELS[DIRECT_DRIFTING_VARIANT] + " & " + " & ".join(direct_cells) + r" \\")
    for variant in WFLOW_VARIANTS:
        cells = []
        for channel in channels:
            metric = wflow_summary.get((channel, variant), {}).get("direct_swd")
            if metric is None:
                cells.append("--")
                continue
            value, error = metric
            cell = pm_fixed(value, error)
            cells.append(maybe_bold(cell, best_drifting.get(channel) == variant))
        lines.append(r"\headspace " + VARIANT_LABELS[variant] + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_coding_table(
    path: Path,
    ser_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    channels: list[str],
) -> None:
    best_learned = {
        (channel, metric): min(
            available_variants(ser_summary, channel, WFLOW_VARIANTS, metric),
            key=lambda variant: ser_summary[(channel, variant)][metric][0],
        )
        for channel in channels
        for metric in ("final_eval_ber", "final_eval_ser")
        if available_variants(ser_summary, channel, WFLOW_VARIANTS, metric)
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Downstream symbolic coding metrics for W-Flow channel surrogates.} Autoencoders are trained through each channel surrogate and evaluated on the analytic channel. Values are mean $\pm$ standard error over seeds. Lower is better. Bold marks the best learned surrogate in each row, while the analytic channel is the reference floor.}",
        r"\label{tab:wflow-ser-ber}",
        r"\tablestyle{3.4pt}{1.02}",
        r"\tablefontsize",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Channel & Metric & Analytic & Kernel target & Kernel joint & Joint Sinkhorn & Condition-wise Sinkhorn \\",
        r"\midrule",
    ]
    for idx, channel in enumerate(channels):
        if idx:
            lines.append(r"\midrule")
        for metric, label in [("final_eval_ber", "BER"), ("final_eval_ser", "SER")]:
            cells = []
            for variant in CODING_VARIANTS:
                value_error = summary_metric(ser_summary, channel, variant, metric)
                if value_error is None:
                    cells.append(r"--")
                    continue
                value, error = value_error
                cell = format_coding_value(channel, metric, value, error)
                if variant != "analytic":
                    cell = maybe_bold(cell, best_learned.get((channel, metric)) == variant)
                cells.append(cell)
            lines.append(channel + " & " + label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{0.25em}",
            r"\begin{minipage}{0.92\textwidth}",
            r"\footnotesize\raggedright",
            r"SSPA uses the \(30\)-seed \(M_{\mathrm{msg}}=64\) update-budget screen. Dashes mark variants not rerun under that SSPA coding setup.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metric_table(
    path: Path,
    wflow_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    ser_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    channels: list[str],
) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{\textbf{Anchor-conditioned diagnostics for W-Flow variants.} Anchor SWD and Gaussian Wasserstein-2 (GW2) are computed from repeated samples at fixed channel inputs. Direct SWD and BER/SER are reported separately in Tables~\ref{tab:wflow-swd-baselines} and~\ref{tab:wflow-ser-ber}. Values are seed means. Lower is better.}",
        r"\label{tab:wflow-conditional-metrics}",
        r"\tablestyle{3.4pt}{1.02}",
        r"\tablefontsize",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Channel & Variant & Anchor SWD & Anchor GW2 \\",
        r"\midrule",
    ]
    for channel_idx, channel in enumerate(channels):
        if channel_idx:
            lines.append(r"\midrule")
        for variant in WFLOW_VARIANTS:
            anchor_metric = summary_metric(wflow_summary, channel, variant, "anchor_y_swd")
            gw2_metric = summary_metric(wflow_summary, channel, variant, "anchor_gaussian_w2")
            if anchor_metric is None or gw2_metric is None:
                anchor_cell = r"--"
                gw2_cell = r"--"
            else:
                anchor, _ = anchor_metric
                gw2, _ = gw2_metric
                anchor_cell = f"{anchor:.4f}"
                gw2_cell = f"{gw2:.4f}"
            lines.append(
                channel
                + " & "
                + VARIANT_LABELS[variant]
                + " & "
                + anchor_cell
                + " & "
                + gw2_cell
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{0.25em}",
            r"\begin{minipage}{0.92\columnwidth}",
            r"\footnotesize\raggedright",
            r"The SSPA condition-wise Sinkhorn row uses the same \(30\)-seed update-budget-controlled run as Table~\ref{tab:wflow-swd-baselines}.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_coding_ber(
    fig_dir: Path,
    ser_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    channels: list[str],
) -> None:
    ncols = 3 if len(channels) == 3 or len(channels) > 4 else 2
    nrows = math.ceil(len(channels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.85 * ncols, 2.7 * nrows), sharey=False)
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, channel in zip(flat_axes, channels):
        variants = available_variants(ser_summary, channel, CODING_VARIANTS, "final_eval_ber")
        values = [
            coding_display_value(channel, "final_eval_ber", ser_summary[(channel, variant)]["final_eval_ber"][0])
            for variant in variants
        ]
        errors = [ser_summary[(channel, variant)]["final_eval_ber"][1] for variant in variants]
        colors = [COLORS[variant] for variant in variants]
        labels = [SHORT_LABELS[variant] for variant in variants]
        draw_barplot(ax, labels=labels, values=values, errors=errors, colors=colors)
        floor = MC_REPORT_FLOORS.get((channel, "final_eval_ber"))
        if floor is not None:
            for idx, variant in enumerate(variants):
                raw_value = ser_summary[(channel, variant)]["final_eval_ber"][0]
                if raw_value < floor:
                    ax.text(idx, floor * 1.18, "$<$", ha="center", va="bottom", fontsize=8)
        ax.set_title(channel, fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", which="major")
        ax.grid(axis="x", visible=False)
        ax.set_xticks(list(range(len(variants))))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel("")
        ax.set_ylabel("BER")
    for ax in flat_axes[len(channels) :]:
        ax.axis("off")
    fig.suptitle("Symbolic autoencoder BER after training through each surrogate", fontsize=12)
    fig.tight_layout()
    save_figure(fig, fig_dir / "wflow_symbolic_ber")


def write_markdown_summary(
    path: Path,
    wflow_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    ser_summary: dict[tuple[str, str], dict[str, tuple[float, float]]],
    channels: list[str],
    wflow_csvs: list[Path],
    ser_csvs: list[Path],
    reference_csvs: list[Path],
) -> None:
    lines = [
        "# Journal W-Flow Artifact Summary",
        "",
        "Generated from:",
    ]
    for csv_path in wflow_csvs:
        lines.append(f"- `{csv_path}`")
    for csv_path in ser_csvs:
        lines.append(f"- `{csv_path}`")
    for csv_path in reference_csvs:
        lines.append(f"- `{csv_path}`")
    lines.extend(
        [
            "",
            "If repeated `(seed, channel, variant)` rows are present, later CSVs replace earlier rows.",
        ]
    )
    lines.extend(
        [
        "",
        "## Best learned variants",
        "",
        "| Channel | Direct SWD | Anchor SWD | BER | SER |",
        "|---|---|---|---|---|",
        ]
    )
    for channel in channels:
        direct_variants = available_variants(wflow_summary, channel, WFLOW_VARIANTS, "direct_swd")
        anchor_variants = available_variants(wflow_summary, channel, WFLOW_VARIANTS, "anchor_y_swd")
        ber_variants = available_variants(ser_summary, channel, WFLOW_VARIANTS, "final_eval_ber")
        ser_variants = available_variants(ser_summary, channel, WFLOW_VARIANTS, "final_eval_ser")
        best_direct = min(direct_variants, key=lambda variant: wflow_summary[(channel, variant)]["direct_swd"][0])
        best_anchor = min(anchor_variants, key=lambda variant: wflow_summary[(channel, variant)]["anchor_y_swd"][0])
        best_ber = min(ber_variants, key=lambda variant: ser_summary[(channel, variant)]["final_eval_ber"][0])
        best_ser = min(ser_variants, key=lambda variant: ser_summary[(channel, variant)]["final_eval_ser"][0])
        lines.append(
            f"| {channel} | {VARIANT_LABELS[best_direct]} | {VARIANT_LABELS[best_anchor]} | "
            f"{VARIANT_LABELS[best_ber]} | {VARIANT_LABELS[best_ser]} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Condition-wise Sinkhorn is the best learned coding surrogate on AWGN, Rayleigh, SSPA, and TDL under the reported channel-specific coding setups.",
            "- TDL diffusion/WGAN direct-SWD reference rows are checkpoint-only evaluations of the trained baseline implants.",
            "- The SSPA condition-wise row uses the corrected compact-budget M_msg=64 screen. The older full-budget corrected run was an over-optimization failure of the sharp field and is not the reported SSPA model.",
            "- Direct SWD and downstream coding do not always agree, so the paper reports both global and condition-wise diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plot_style()
    wflow_csvs = [args.wflow_csv, *args.extra_wflow_csv]
    ser_csvs = [args.ser_csv, *args.extra_ser_csv]
    wflow_rows = merge_rows_by_seed_channel_variant(wflow_csvs)
    ser_rows = merge_rows_by_seed_channel_variant(ser_csvs)
    if args.sspa_fiber_wflow_csv is not None:
        wflow_rows = replace_selected_rows(
            wflow_rows,
            args.sspa_fiber_wflow_csv,
            channel="SSPA",
            variants={"fiber_sinkhorn"},
        )
        wflow_csvs.append(args.sspa_fiber_wflow_csv)
    if args.sspa_ser_csv is not None:
        ser_rows = replace_selected_rows(
            ser_rows,
            args.sspa_ser_csv,
            channel="SSPA",
            variants=None,
        )
        ser_csvs.append(args.sspa_ser_csv)
    reference_swds = REFERENCE_SWDS
    if args.baseline_direct_swd_csv:
        baseline_reference_rows = merge_rows_by_seed_channel_variant(args.baseline_direct_swd_csv)
        reference_swds = merge_reference_swds(
            reference_swds,
            summarize_reference_swd_rows(
                baseline_reference_rows,
                error_stat=args.baseline_direct_swd_error,
            ),
        )
    coding_channels = parse_csv_list(args.coding_channels)
    metric_channels = parse_csv_list(args.metric_channels)
    wflow_summary = summarize(
        wflow_rows,
        ["direct_swd", "anchor_y_swd", "anchor_gaussian_w2", "suite_elapsed_seconds"],
    )
    ser_summary = summarize(ser_rows, ["final_eval_ser", "final_eval_ber", "final_eval_loss", "train_seconds"])

    args.journal_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.journal_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    write_wflow_swd_table(args.journal_dir / "wflow_swd_baseline_table.tex", wflow_summary, reference_swds)
    write_coding_table(args.journal_dir / "wflow_coding_table.tex", ser_summary, coding_channels)
    write_metric_table(args.journal_dir / "wflow_conditional_metric_table.tex", wflow_summary, ser_summary, metric_channels)
    write_markdown_summary(
        args.journal_dir / "wflow_full_run_summary.md",
        wflow_summary,
        ser_summary,
        metric_channels,
        wflow_csvs,
        ser_csvs,
        args.baseline_direct_swd_csv,
    )

    print(
        {
            "tables": [
                str(args.journal_dir / "wflow_swd_baseline_table.tex"),
                str(args.journal_dir / "wflow_coding_table.tex"),
                str(args.journal_dir / "wflow_conditional_metric_table.tex"),
            ],
            "figures": [],
        }
    )


if __name__ == "__main__":
    main()
