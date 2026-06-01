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
    ("ddpm", "DDPM"),
    ("ddim100", "DDIM-100"),
    ("ddim50", "DDIM-50"),
    ("ddim20", "DDIM-20"),
    ("ddim10", "DDIM-10"),
    ("paper_wgan", "WGAN"),
]
CHANNEL_ORDER = ["AWGN", "Rayleigh", "SSPA", "OptFib"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export timing-suite outputs into CSV and LaTeX tables.")
    parser.add_argument("--timing-suite-summary", type=Path, required=True)
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
    column_spec = "l" + "ccc" * len(available_channels)
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Projected training and inference timing on {latex_escape(device.upper())}. "
        r"Training hours are extrapolated from a 2\% timing run; inference is reported as time per sample.}}"
    )
    lines.append(rf"\label{{tab:timing-{latex_escape(device)}}}")
    lines.append(rf"\resizebox{{\textwidth}}{{!}}{{%")
    lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"\hline")
    header = ["Method"]
    for channel in available_channels:
        header.extend(
            [
                rf"\multicolumn{{3}}{{c}}{{{latex_escape(channel)}}}",
            ]
        )
    # Build grouped header manually.
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
        cmidrules.append(rf"\cline{{{start}-{end}}}")
    lines.append(" ".join(cmidrules))
    lines.append(
        " & ".join(
            [""]
            + [item for _ in available_channels for item in ("Train [h]", "Inf.", "Total [h]")]
        )
        + r" \\"
    )
    lines.append(r"\hline")
    for method_key, method_label in METHOD_ORDER:
        row_parts = [latex_escape(method_label)]
        for channel in available_channels:
            method_block = channels[channel]["methods"].get(method_key)
            if method_block is None:
                row_parts.extend(["--", "--", "--"])
                continue
            row_parts.extend(
                [
                    f"{method_block['projected_full_train_hours']:.3f}",
                    format_inference_time(method_block["milliseconds_per_sample"]),
                    f"{method_block['projected_total_benchmark_hours']:.3f}",
                ]
            )
        lines.append(" & ".join(row_parts) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
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


def main() -> None:
    args = parse_args()
    data = json.loads(args.timing_suite_summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_flat_csv(data, args.out_dir / "timing_summary_flat.csv")
    write_summary_md(data, args.out_dir / "timing_summary.md")
    for device, channels in data["results"].items():
        write_device_tex(device, channels, args.out_dir / f"timing_table_{device}.tex")

    manifest = {
        "source_summary": str(args.timing_suite_summary),
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
