from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-seed full-budget benchmark results into one suite summary.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--paper-channels", type=str, default="AWGN,Rayleigh,SSPA")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


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
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds, args.seed)

    per_seed = []
    for seed in seeds:
        result_path = args.suite_dir / f"seed{seed}_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing per-seed result: {result_path}")
        per_seed.append(json.loads(result_path.read_text()))

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
        "suite_dir": str(args.suite_dir),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    summary_path = args.suite_dir / "suite_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
