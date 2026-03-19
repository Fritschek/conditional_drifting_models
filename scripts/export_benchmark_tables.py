from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPER_CHANNELS = ["AWGN", "Rayleigh", "SSPA"]
ALL_CHANNELS = ["AWGN", "Rayleigh", "SSPA", "OptFib"]
METHOD_ROWS = [
    ("drifting_direct_swd", "Drifting (dir.)"),
    ("drifting_residual_swd", "Drifting (res.)"),
    ("ddpm_swd", "DDPM"),
    ("ddim_100", "DDIM-100"),
    ("ddim_50", "DDIM-50"),
    ("ddim_20", "DDIM-20"),
    ("ddim_10", "DDIM-10"),
    ("paper_wgan_swd", "Paper WGAN"),
]
OPTFIB_ROWS = [
    ("drifting_direct_swd", "Drifting (dir.)"),
    ("drifting_residual_swd", "Drifting (res.)"),
    ("ddpm_swd", "DDPM"),
    ("ddim_swd", "DDIM-100"),
    ("paper_wgan_swd", "Paper WGAN"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export aggregated benchmark results into CSV and LaTeX tables.")
    parser.add_argument("--suite-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper" / "generated")
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def write_csv(data: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block", "channel", "metric_key", "metric_label", "mean", "std", "num_seeds"])
        for channel in PAPER_CHANNELS:
            block = data["aggregated"]["paper_channels"][channel]
            for key, label in METHOD_ROWS:
                if key.startswith("ddim_"):
                    step = key.split("_", 1)[1]
                    stat = block["ddim_swd"][step]
                else:
                    stat = block[key]
                writer.writerow(["paper_channels", channel, key, label, stat["mean"], stat["std"], stat["num_seeds"]])
        optfib = data["aggregated"]["optfib"]
        for key, label in OPTFIB_ROWS:
            stat = optfib[key]
            writer.writerow(["optfib", "OptFib", key, label, stat["mean"], stat["std"], stat["num_seeds"]])


def write_paper_channels_tex(data: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ten-seed SWD benchmark on AWGN, Rayleigh, and SSPA. Lower is better.}",
        r"\label{tab:benchmark-paper-channels}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Method & AWGN & Rayleigh & SSPA \\",
        r"\hline",
    ]
    for key, label in METHOD_ROWS:
        row = [label]
        for channel in PAPER_CHANNELS:
            block = data["aggregated"]["paper_channels"][channel]
            if key.startswith("ddim_"):
                step = key.split("_", 1)[1]
                stat = block["ddim_swd"][step]
            else:
                stat = block[key]
            row.append(fmt(stat["mean"], stat["std"]))
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_tex(data: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ten-seed SWD benchmark across all channels. Lower is better. Entries unavailable for a given channel are marked with dashes.}",
        r"\label{tab:benchmark-all-channels}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & AWGN & Rayleigh & SSPA & OptFib \\",
        r"\hline",
    ]
    for key, label in METHOD_ROWS:
        row = [label]
        for channel in ALL_CHANNELS:
            if channel == "OptFib":
                optfib = data["aggregated"]["optfib"]
                if key in {"ddim_20", "ddim_10"}:
                    row.append("--")
                    continue
                if key == "ddim_100":
                    stat = optfib["ddim_swd"]
                elif key == "ddim_50":
                    row.append("--")
                    continue
                else:
                    opt_key = key
                    stat = optfib.get(opt_key)
                if stat is None:
                    row.append("--")
                else:
                    row.append(fmt(stat["mean"], stat["std"]))
                continue

            block = data["aggregated"]["paper_channels"][channel]
            if key.startswith("ddim_"):
                step = key.split("_", 1)[1]
                stat = block["ddim_swd"][step]
            else:
                stat = block[key]
            row.append(fmt(stat["mean"], stat["std"]))
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_optfib_tex(data: dict, out_path: Path) -> None:
    optfib = data["aggregated"]["optfib"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ten-seed SWD benchmark on OptFib. Lower is better.}",
        r"\label{tab:benchmark-optfib}",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"Method & OptFib \\",
        r"\hline",
    ]
    for key, label in OPTFIB_ROWS:
        stat = optfib[key]
        lines.append(f"{label} & {fmt(stat['mean'], stat['std'])} \\\\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md(data: dict, out_path: Path) -> None:
    lines = ["# Benchmark Summary", ""]
    lines.append(f"Source: `{data['suite_dir']}`")
    lines.append(f"Seeds: `{data['seeds']}`")
    lines.append("")
    lines.append("## Paper Channels")
    lines.append("")
    lines.append("| Method | AWGN | Rayleigh | SSPA |")
    lines.append("|---|---:|---:|---:|")
    for key, label in METHOD_ROWS:
        row = [label]
        for channel in PAPER_CHANNELS:
            block = data["aggregated"]["paper_channels"][channel]
            stat = block["ddim_swd"][key.split('_', 1)[1]] if key.startswith("ddim_") else block[key]
            row.append(fmt(stat["mean"], stat["std"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## OptFib")
    lines.append("")
    lines.append("| Method | OptFib |")
    lines.append("|---|---:|")
    for key, label in OPTFIB_ROWS:
        stat = data["aggregated"]["optfib"][key]
        lines.append(f"| {label} | {fmt(stat['mean'], stat['std'])} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = json.loads(args.suite_summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(data, args.out_dir / "benchmark_summary_flat.csv")
    write_md(data, args.out_dir / "benchmark_summary.md")
    write_paper_channels_tex(data, args.out_dir / "benchmark_table_paper_channels.tex")
    write_optfib_tex(data, args.out_dir / "benchmark_table_optfib.tex")
    write_combined_tex(data, args.out_dir / "benchmark_table_all_channels.tex")

    manifest = {
        "source_summary": str(args.suite_summary),
        "output_dir": str(args.out_dir),
        "generated_files": [
            "benchmark_summary_flat.csv",
            "benchmark_summary.md",
            "benchmark_table_paper_channels.tex",
            "benchmark_table_optfib.tex",
            "benchmark_table_all_channels.tex",
        ],
    }
    (args.out_dir / "benchmark_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
