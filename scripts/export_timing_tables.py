from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METHOD_ORDER = [
    ("drifting_residual", "Drifting (res.)"),
    ("drifting_direct", "Drifting (dir.)"),
    ("kernel_joint", "Joint-kernel drift"),
    ("joint_sinkhorn", "Joint Sinkhorn"),
    ("fiber_sinkhorn", "Condition-wise Sinkhorn"),
    ("paper_wgan", "WGAN"),
    ("ddpm", "DDPM"),
    ("ddim100", "DDIM-100"),
    ("ddim50", "DDIM-50"),
    ("ddim20", "DDIM-20"),
    ("ddim10", "DDIM-10"),
]
ONE_SHOT_METHODS = {
    "drifting_residual",
    "drifting_direct",
    "kernel_joint",
    "joint_sinkhorn",
    "fiber_sinkhorn",
    "paper_wgan",
}
DIFFUSION_METHODS = {"ddpm", "ddim100", "ddim50", "ddim20", "ddim10"}
CHANNEL_ORDER = ["AWGN", "Rayleigh", "SSPA", "TDL", "OptFib"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export timing-suite outputs into CSV and LaTeX tables.")
    parser.add_argument("--timing-suite-summary", type=Path, required=True)
    parser.add_argument(
        "--extra-timing-suite-summary",
        type=Path,
        action="append",
        default=[],
        help="Additional timing-suite summaries to merge channel results from, with later summaries overriding earlier ones.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "paper" / "generated",
    )
    return parser.parse_args()


def format_inference_time(ms_per_sample: float) -> str:
    if ms_per_sample < 0.001:
        return f"{ms_per_sample * 1000.0:.3f} us"
    return f"{ms_per_sample:.3f} ms"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def device_label(device: str) -> str:
    if device.lower() == "cuda":
        return "an NVIDIA RTX 5060 Ti GPU"
    return latex_escape(device.upper())


def format_hours(value: float, *, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if bold else text


def one_shot_best_totals(channels: dict, available_channels: list[str]) -> dict[str, str]:
    winners: dict[str, str] = {}
    for channel in available_channels:
        methods = channels[channel]["methods"]
        candidates = [
            (method_key, float(methods[method_key]["projected_total_benchmark_hours"]))
            for method_key in ONE_SHOT_METHODS
            if method_key in methods
        ]
        if candidates:
            winners[channel] = min(candidates, key=lambda item: item[1])[0]
    return winners


def write_flat_csv(data: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "device",
                "channel",
                "method_key",
                "method_label",
                "samples_per_second",
                "milliseconds_per_sample",
                "inference_time_display",
                "projected_full_train_hours",
                "projected_total_benchmark_hours",
            ]
        )
        for device, channels in data["results"].items():
            for channel in CHANNEL_ORDER:
                if channel not in channels:
                    continue
                methods = channels[channel]["methods"]
                for method_key, method_label in METHOD_ORDER:
                    if method_key not in methods:
                        continue
                    row = methods[method_key]
                    writer.writerow(
                        [
                            device,
                            channel,
                            method_key,
                            method_label,
                            row["samples_per_second"],
                            row["milliseconds_per_sample"],
                            format_inference_time(row["milliseconds_per_sample"]),
                            row["projected_full_train_hours"],
                            row["projected_total_benchmark_hours"],
                        ]
                    )


def write_device_tex(device: str, channels: dict, out_path: Path) -> None:
    available_channels = [channel for channel in CHANNEL_ORDER if channel in channels]
    column_spec = "@{}l" + "ccc" * len(available_channels) + "@{}"
    one_shot_winners = one_shot_best_totals(channels, available_channels)
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{\textbf{{Projected training and inference timing on {device_label(device)}.}} "
        r"Training hours are extrapolated from a 2\% timing run; inference is reported as time per sample.}"
    )
    lines.append(rf"\label{{tab:timing-{latex_escape(device)}}}")
    lines.append(r"\tablestyle{3pt}{1.02}")
    lines.append(r"\tablefontsize")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"\toprule")
    lines.append(
        " & ".join(
            ["Method"]
            + [rf"\multicolumn{{3}}{{c}}{{{latex_escape(channel)}}}" for channel in available_channels]
        )
        + r" \\"
    )
    cmidrules = []
    for idx in range(len(available_channels)):
        start = 2 + idx * 3
        end = start + 2
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))
    lines.append(
        " & ".join(
            [""]
            + [item for _ in available_channels for item in ("Train [h]", "Inf.", "Total [h]")]
        )
        + r" \\"
    )
    lines.append(r"\midrule")

    def append_method_row(method_key: str, method_label: str) -> None:
        row_parts = [latex_escape(method_label)]
        for channel in available_channels:
            method_block = channels[channel]["methods"].get(method_key)
            if method_block is None:
                row_parts.extend(["--", "--", "--"])
                continue
            total = float(method_block["projected_total_benchmark_hours"])
            is_best_one_shot = method_key == one_shot_winners.get(channel)
            row_parts.extend(
                [
                    format_hours(float(method_block["projected_full_train_hours"])),
                    format_inference_time(method_block["milliseconds_per_sample"]),
                    format_hours(total, bold=is_best_one_shot),
                ]
            )
        lines.append(" & ".join(row_parts) + r" \\")

    one_shot_rows = [(key, label) for key, label in METHOD_ORDER if key in ONE_SHOT_METHODS]
    diffusion_rows = [(key, label) for key, label in METHOD_ORDER if key in DIFFUSION_METHODS]
    if one_shot_rows:
        span = 1 + 3 * len(available_channels)
        lines.append(rf"\rowcolor[gray]{{0.9}} \multicolumn{{{span}}}{{l}}{{\textit{{One-shot generators}}}} \\")
        for method_key, method_label in one_shot_rows:
            append_method_row(method_key, method_label)
    if diffusion_rows:
        lines.append(r"\midrule")
        span = 1 + 3 * len(available_channels)
        lines.append(rf"\rowcolor[gray]{{0.9}} \multicolumn{{{span}}}{{l}}{{\textit{{Diffusion samplers}}}} \\")
        for method_key, method_label in diffusion_rows:
            append_method_row(method_key, method_label)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\par\vspace{0.25em}")
    lines.append(r"\begin{minipage}{0.96\textwidth}")
    lines.append(r"\footnotesize\raggedright")
    lines.append(r"All rows use the same 2\% extrapolation protocol. In the one-shot block, bold total times mark the lowest projected total for each channel. The W-Flow rows use the same one-shot generator architecture as direct drifting; their training-time differences come only from the drift-field computation.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table*}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(data: dict, out_path: Path) -> None:
    lines = ["# Timing Summary", ""]
    lines.append(f"Source: `{data['out_dir']}`")
    lines.append(f"Train fraction: `{data['train_fraction']}`")
    lines.append("")
    for device, channels in data["results"].items():
        lines.append(f"## {device.upper()}")
        lines.append("")
        for channel in CHANNEL_ORDER:
            if channel not in channels:
                continue
            lines.append(f"### {channel}")
            lines.append("")
            lines.append("| Method | Train [h] | Inference | Total [h] |")
            lines.append("|---|---:|---:|---:|")
            for method_key, method_label in METHOD_ORDER:
                if method_key not in channels[channel]["methods"]:
                    continue
                method = channels[channel]["methods"][method_key]
                lines.append(
                    f"| {method_label} | {method['projected_full_train_hours']:.3f} | "
                    f"{format_inference_time(method['milliseconds_per_sample'])} | "
                    f"{method['projected_total_benchmark_hours']:.3f} |"
                )
            lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_timing_summaries(base: dict, extras: list[dict]) -> dict:
    merged = json.loads(json.dumps(base))
    merged.setdefault("source_summaries", {})
    merged["source_summaries"]["base"] = base.get("out_dir", "")
    for index, extra in enumerate(extras, start=1):
        merged["source_summaries"][f"extra_{index}"] = extra.get("out_dir", "")
        for device, channels in extra.get("results", {}).items():
            merged.setdefault("results", {}).setdefault(device, {})
            for channel, payload in channels.items():
                merged["results"][device][channel] = payload
    return merged


def main() -> None:
    args = parse_args()
    data = json.loads(args.timing_suite_summary.read_text())
    extras = [json.loads(path.read_text()) for path in args.extra_timing_suite_summary]
    if extras:
        data = merge_timing_summaries(data, extras)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_flat_csv(data, args.out_dir / "timing_summary_flat.csv")
    write_summary_md(data, args.out_dir / "timing_summary.md")
    for device, channels in data["results"].items():
        write_device_tex(device, channels, args.out_dir / f"timing_table_{device}.tex")

    manifest = {
        "source_summary": str(args.timing_suite_summary),
        "extra_timing_summaries": [str(path) for path in args.extra_timing_suite_summary],
        "output_dir": str(args.out_dir),
        "devices": list(data["results"].keys()),
        "generated_files": [
            "timing_summary_flat.csv",
            "timing_summary.md",
            *[f"timing_table_{device}.tex" for device in data["results"].keys()],
        ],
    }
    (args.out_dir / "timing_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
