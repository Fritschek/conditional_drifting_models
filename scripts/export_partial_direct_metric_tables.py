from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["AWGN", "Rayleigh", "SSPA", "OptFib"]
ROWS = [
    ("drifting_residual", "Drifting (res.)"),
    ("wgan", "WGAN"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export partial direct-metric rerun results into CSV/Markdown/LaTeX tables."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "generated")
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def write_csv(data: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "method_key", "method_label", "metric_space", "mean", "std", "num_seeds"])
        for channel in CHANNELS:
            block = data["aggregated"][channel]
            for key, label in ROWS:
                stat_block = block[key]
                for metric_space in ("y_swd", "residual_swd"):
                    stat = stat_block[metric_space]
                    writer.writerow([channel, key, label, metric_space, stat["mean"], stat["std"], stat["num_seeds"]])


def write_md(data: dict, out_path: Path) -> None:
    lines = [
        "# Direct-vs-Residual Metric Comparison",
        "",
        f"Source: `{data['suite_dir']}`",
        f"Seeds: `{data['seeds']}`",
        "",
        "## Drifting (Residual) and WGAN in Both Metric Spaces",
        "",
        "| Channel | Method | Direct `y`-space SWD | Residual-space SWD |",
        "|---|---|---:|---:|",
    ]
    for channel in CHANNELS:
        block = data["aggregated"][channel]
        for key, label in ROWS:
            lines.append(
                f"| {channel} | {label} | {fmt(block[key]['y_swd']['mean'], block[key]['y_swd']['std'])} | {fmt(block[key]['residual_swd']['mean'], block[key]['residual_swd']['std'])} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(data: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of direct output-space SWD and residual-space SWD for residual drifting and WGAN over ten seeds. Lower is better.}",
        r"\label{tab:direct-vs-residual-metric}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"Channel & Method & Direct $y$-space SWD & Residual-space SWD \\",
        r"\hline",
    ]
    for channel in CHANNELS:
        block = data["aggregated"][channel]
        for key, label in ROWS:
            lines.append(
                f"{channel} & {label} & {fmt(block[key]['y_swd']['mean'], block[key]['y_swd']['std'])} & {fmt(block[key]['residual_swd']['mean'], block[key]['residual_swd']['std'])} \\\\"
            )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(data, args.out_dir / "partial_direct_metric_summary_flat.csv")
    write_md(data, args.out_dir / "partial_direct_metric_summary.md")
    write_tex(data, args.out_dir / "partial_direct_metric_table.tex")

    manifest = {
        "source_summary": str(args.summary),
        "output_dir": str(args.out_dir),
        "generated_files": [
            "partial_direct_metric_summary_flat.csv",
            "partial_direct_metric_summary.md",
            "partial_direct_metric_table.tex",
        ],
    }
    (args.out_dir / "partial_direct_metric_export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
