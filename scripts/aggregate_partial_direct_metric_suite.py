from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the multi-seed partial direct-metric rerun."
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
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
    suite_name = suite_dir.name
    if suite_name in path.parts:
        idx = path.parts.index(suite_name)
        candidate = suite_dir.joinpath(*path.parts[idx + 1 :])
        if candidate.exists():
            return candidate
    candidate = suite_dir / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve copied suite path for {path_str} under {suite_dir}")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds, args.seed)

    per_seed = []
    for seed in seeds:
        result_path = args.suite_dir / f"seed{seed}_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing per-seed result: {result_path}")
        per_seed.append(json.loads(result_path.read_text()))

    first_summary_path = resolve_suite_path(per_seed[0]["result"]["output_json"], args.suite_dir)
    first_summary = json.loads(first_summary_path.read_text())
    channels = list(first_summary["channels"].keys())

    aggregated: dict[str, dict] = {}
    for channel_name in channels:
        channel_block: dict[str, dict] = {}
        method_names = set()
        for row in per_seed:
            summary_path = resolve_suite_path(row["result"]["output_json"], args.suite_dir)
            data = json.loads(summary_path.read_text())
            method_names.update(
                key for key, value in data["channels"][channel_name].items() if isinstance(value, dict) and "y_swd" in value or key == "diffusion"
            )

        for method_name in sorted(method_names):
            if method_name == "diffusion":
                sub_block = {}
                for sampler_name in ("ddpm", "ddim_100"):
                    y_values = []
                    residual_values = []
                    for row in per_seed:
                        summary_path = resolve_suite_path(row["result"]["output_json"], args.suite_dir)
                        data = json.loads(summary_path.read_text())
                        if method_name in data["channels"][channel_name]:
                            sampler = data["channels"][channel_name][method_name][sampler_name]
                            y_values.append(float(sampler["y_swd"]))
                            residual_values.append(float(sampler["residual_swd"]))
                    if y_values:
                        sub_block[sampler_name] = {
                            "y_swd": summarize_numeric(y_values),
                            "residual_swd": summarize_numeric(residual_values),
                        }
                if sub_block:
                    channel_block[method_name] = sub_block
                continue

            y_values = []
            residual_values = []
            for row in per_seed:
                summary_path = resolve_suite_path(row["result"]["output_json"], args.suite_dir)
                data = json.loads(summary_path.read_text())
                if method_name in data["channels"][channel_name]:
                    values = data["channels"][channel_name][method_name]
                    y_values.append(float(values["y_swd"]))
                    residual_values.append(float(values["residual_swd"]))
            if y_values:
                channel_block[method_name] = {
                    "y_swd": summarize_numeric(y_values),
                    "residual_swd": summarize_numeric(residual_values),
                }

        aggregated[channel_name] = channel_block

    summary = {
        "suite_dir": str(args.suite_dir),
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    summary_path = args.suite_dir / "partial_direct_metric_suite_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
