from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the full-budget benchmark suite across paper channels and OptFib."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--paper-channels",
        type=str,
        default="AWGN,Rayleigh,SSPA",
        help="Paper-faithful channels for run_paper2309_benchmark.py",
    )
    parser.add_argument(
        "--paper-eval-size",
        type=int,
        default=1_000_000,
        help="Evaluation size override for paper channels. Training budgets remain paper presets.",
    )
    parser.add_argument(
        "--optfib-eval-size",
        type=int,
        default=100_000,
        help="Evaluation size for the OptFib baseline comparison path.",
    )
    parser.add_argument(
        "--optfib-dataset-size",
        type=int,
        default=120_000,
        help="OptFib training dataset size for the later benchmark path.",
    )
    parser.add_argument("--optfib-epochs", type=int, default=60)
    parser.add_argument("--optfib-batch-size", type=int, default=512)
    parser.add_argument("--optfib-num-steps", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / f"full_budget_suite_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prepared commands and manifest without executing them.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> dict:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout = completed.stdout.strip()
    if stdout:
        lines = stdout.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            candidate = "\n".join(lines[idx:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"Could not parse JSON output from command: {' '.join(cmd)}")


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int, fallback_seed: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if num_seeds > 1:
        return list(range(seed_start, seed_start + num_seeds))
    return [fallback_seed]


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "num_seeds": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance), "num_seeds": len(values)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds, args.seed)

    base_env = os.environ.copy()
    base_env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

    manifest = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
        "device": args.device,
        "seeds": seeds,
        "paper_channels": args.paper_channels,
        "paper_eval_size": args.paper_eval_size,
        "optfib_eval_size": args.optfib_eval_size,
    }

    manifest_path = args.out_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        preview = []
        for seed in seeds:
            preview.append(
                {
                    "seed": seed,
                    "paper_command": [
                        sys.executable,
                        "scripts/run_paper2309_benchmark.py",
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--channels",
                        args.paper_channels,
                        "--eval-size",
                        str(args.paper_eval_size),
                        "--include-gan-fa",
                    ],
                    "optfib_command": [
                        sys.executable,
                        "scripts/compare_optional_baselines.py",
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--channel",
                        "OptFib",
                        "--dataset-size",
                        str(args.optfib_dataset_size),
                        "--epochs",
                        str(args.optfib_epochs),
                        "--batch-size",
                        str(args.optfib_batch_size),
                        "--eval-size",
                        str(args.optfib_eval_size),
                        "--num-steps",
                        str(args.optfib_num_steps),
                        "--ddim-steps",
                        str(args.optfib_num_steps),
                    ],
                }
            )
        print(json.dumps({"manifest": str(manifest_path), **manifest, "commands": preview}, indent=2))
        return

    per_seed = []
    for seed in seeds:
        paper_cmd = [
            sys.executable,
            "scripts/run_paper2309_benchmark.py",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--channels",
            args.paper_channels,
            "--eval-size",
            str(args.paper_eval_size),
            "--include-gan-fa",
        ]
        optfib_cmd = [
            sys.executable,
            "scripts/compare_optional_baselines.py",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--channel",
            "OptFib",
            "--dataset-size",
            str(args.optfib_dataset_size),
            "--epochs",
            str(args.optfib_epochs),
            "--batch-size",
            str(args.optfib_batch_size),
            "--eval-size",
            str(args.optfib_eval_size),
            "--num-steps",
            str(args.optfib_num_steps),
            "--ddim-steps",
            str(args.optfib_num_steps),
        ]
        per_seed.append(
            {
                "seed": seed,
                "paper_result": run_command(paper_cmd, ROOT, base_env),
                "optfib_result": run_command(optfib_cmd, ROOT, base_env),
            }
        )

    aggregated: dict[str, dict] = {"paper_channels": {}, "optfib": {}}
    paper_metrics = ["drifting_swd", "ddpm_swd", "paper_wgan_swd", "gan_fa_swd"]
    ddim_keys = [100, 50, 20, 10]
    channel_names = [name.strip() for name in args.paper_channels.split(",") if name.strip()]
    for channel_name in channel_names:
        channel_summary = {}
        for metric in paper_metrics:
            values = []
            for row in per_seed:
                summary_path = Path(row["paper_result"]["output_json"])
                data = json.loads(summary_path.read_text())
                result_block = data["channels"][channel_name]["results"]
                if metric in result_block:
                    values.append(float(result_block[metric]))
            if values:
                channel_summary[metric] = summarize_numeric(values)
        ddim_summary = {}
        for step in ddim_keys:
            values = []
            for row in per_seed:
                summary_path = Path(row["paper_result"]["output_json"])
                data = json.loads(summary_path.read_text())
                values.append(float(data["channels"][channel_name]["results"]["ddim_swd"][str(step)]))
            ddim_summary[str(step)] = summarize_numeric(values)
        channel_summary["ddim_swd"] = ddim_summary
        aggregated["paper_channels"][channel_name] = channel_summary

    optfib_values = {"drifting_swd": [], "ddpm_swd": [], "ddim_swd": [], "gan_swd": []}
    for row in per_seed:
        optfib_json = Path(row["optfib_result"]["json"])
        data = json.loads(optfib_json.read_text())
        for metric in optfib_values:
            optfib_values[metric].append(float(data[metric]))
    aggregated["optfib"] = {metric: summarize_numeric(values) for metric, values in optfib_values.items()}

    summary = {
        "manifest": str(manifest_path),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    summary_path = args.out_dir / "suite_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
