from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-seed enhanced-direct benchmark results into one suite summary.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA")
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


def resolve_suite_path(path_str: str, suite_dir: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    parts = path.parts
    suite_name = suite_dir.name
    if suite_name in parts:
        idx = parts.index(suite_name)
        candidate = suite_dir.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate

    candidate = suite_dir / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not resolve copied suite path for {path_str} under {suite_dir}")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds, args.seed)
    channel_names = [name.strip() for name in args.channels.split(",") if name.strip()]

    per_seed = []
    for seed in seeds:
        result_path = args.suite_dir / f"seed{seed}_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing per-seed result: {result_path}")
        per_seed.append(json.loads(result_path.read_text()))

    aggregated: dict[str, dict[str, object]] = {"channels": {}}
    for channel_name in channel_names:
        swd_values = []
        elapsed_values = []
        checkpoints = []
        for row in per_seed:
            summary_path = resolve_suite_path(row["enhanced_direct_result"]["summary"], args.suite_dir)
            data = json.loads(summary_path.read_text())
            channel_block = data["channels"].get(channel_name)
            if not channel_block or "swd" not in channel_block:
                continue
            swd_values.append(float(channel_block["swd"]))
            checkpoints.append(channel_block.get("checkpoint_path"))
            elapsed_values.append(float(row["enhanced_direct_result"].get("suite_elapsed_seconds", float("nan"))))

        aggregated["channels"][channel_name] = {
            "swd": summarize_numeric(swd_values),
            "suite_elapsed_seconds": summarize_numeric([value for value in elapsed_values if not math.isnan(value)]),
            "checkpoints": checkpoints,
        }

    summary = {
        "suite_dir": str(args.suite_dir),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    summary_path = args.suite_dir / "enhanced_direct_suite_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
