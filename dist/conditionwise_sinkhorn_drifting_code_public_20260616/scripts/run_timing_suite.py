from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the timing benchmark sequentially across paper channels.")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--devices", type=str, default="cpu,cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,TDL")
    parser.add_argument(
        "--methods",
        type=str,
        default="drifting_residual,drifting_direct,ddpm,ddim100,ddim50,ddim20,ddim10,paper_wgan",
    )
    parser.add_argument("--train-fraction", type=float, default=0.02)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-batches", type=int, default=40)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / f"timing_suite_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_json_from_text(text: str, cmd: list[str]) -> dict:
    stdout = text.strip()
    if stdout:
        lines = stdout.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            candidate = "\n".join(lines[idx:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"Could not parse JSON output from command: {' '.join(cmd)}")


def preset_for_channel(channel: str) -> str:
    if channel in {"AWGN", "Rayleigh", "SSPA", "TDL"}:
        return "paper2309"
    if channel == "OptFib":
        return "optfib_suite"
    raise ValueError(f"Unsupported timing-suite channel: {channel}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    channels = [name.strip() for name in args.channels.split(",") if name.strip()]
    devices = [name.strip() for name in args.devices.split(",") if name.strip()]
    if args.device:
        devices = [args.device]

    commands = []
    for device in devices:
        for channel in channels:
            channel_out_dir = args.out_dir / device.lower().replace(":", "_") / channel.lower()
            cmd = [
                sys.executable,
                "scripts/run_inference_timing_benchmark.py",
                "--device",
                device,
                "--channel",
                channel,
                "--seed",
                str(args.seed),
                "--preset",
                preset_for_channel(channel),
                "--prepare-mode",
                "train",
                "--methods",
                args.methods,
                "--train-fraction",
                str(args.train_fraction),
                "--warmup-repeats",
                str(args.warmup_repeats),
                "--repeats",
                str(args.repeats),
                "--batch-size",
                str(args.batch_size),
                "--num-batches",
                str(args.num_batches),
                "--out-dir",
                str(channel_out_dir),
            ]
            commands.append(
                {
                    "device": device,
                    "channel": channel,
                    "preset": preset_for_channel(channel),
                    "command": cmd,
                }
            )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "out_dir": str(args.out_dir),
                    "devices": devices,
                    "train_fraction": args.train_fraction,
                    "commands": commands,
                },
                indent=2,
            )
        )
        return

    suite_summary: dict[str, object] = {
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "devices": devices,
        "train_fraction": args.train_fraction,
        "results": {},
    }

    for item in commands:
        device = item["device"]
        channel = item["channel"]
        cmd = item["command"]
        print(f"[timing-suite] device={device} channel={channel} preset={item['preset']}", flush=True)
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            sys.stdout.write(completed.stdout)
            sys.stdout.flush()
        if completed.stderr:
            sys.stderr.write(completed.stderr)
            sys.stderr.flush()
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, cmd)
        result = parse_json_from_text(completed.stdout, cmd)
        timing_summary_path = Path(result["output_path"])
        device_key = device.lower().replace(":", "_")
        suite_summary["results"].setdefault(device_key, {})
        suite_summary["results"][device_key][channel] = {
            "preset": item["preset"],
            "timing_summary": str(timing_summary_path),
            "methods": json.loads(timing_summary_path.read_text())["methods"],
        }

    summary_path = args.out_dir / "timing_suite_summary.json"
    summary_path.write_text(json.dumps(suite_summary, indent=2))
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
