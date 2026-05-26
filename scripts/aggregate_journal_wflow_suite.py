from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = [
    "swd",
    "direct_swd",
    "residual_swd",
    "train_final_loss",
    "train_final_drift_norm",
    "suite_elapsed_seconds",
    "anchor_y_swd",
    "anchor_y_excess_swd",
    "anchor_residual_swd",
    "anchor_residual_excess_swd",
    "anchor_mean_l2_excess",
    "anchor_cov_fro_excess",
    "anchor_gaussian_w2_excess",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate journal W-Flow seed/variant tasks.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default="kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn")
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,OptFib")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--paired-baseline", type=str, default="kernel_joint")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def summarize_numeric(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "num_seeds": 0,
        }
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    std = math.sqrt(variance)
    sem = std / math.sqrt(len(clean))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "num_seeds": len(clean),
    }


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


def flatten_channel_row(seed: int, variant: str, channel: str, task: dict, summary: dict) -> dict[str, object]:
    channel_block = summary["channels"][channel]
    final_history = channel_block.get("final_history", {})
    row: dict[str, object] = {
        "seed": seed,
        "variant": variant,
        "channel": channel,
        "swd": channel_block.get("swd"),
        "direct_swd": channel_block.get("direct_swd"),
        "residual_swd": channel_block.get("residual_swd"),
        "target_mode": channel_block.get("target_mode"),
        "train_final_loss": final_history.get("loss"),
        "train_final_drift_norm": final_history.get("drift_norm"),
        "suite_elapsed_seconds": task["benchmark_result"].get("suite_elapsed_seconds"),
        "checkpoint_path": channel_block.get("checkpoint_path"),
    }
    for key, value in channel_block.get("anchor_metrics", {}).items():
        row[key] = value
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants)
    channels = parse_csv_list(args.channels)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)

    missing: list[str] = []
    missing_rows: list[str] = []
    task_results: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            result_path = args.suite_dir / f"{variant}_seed{seed}_result.json"
            if not result_path.exists():
                missing.append(str(result_path))
                continue
            task = json.loads(result_path.read_text())
            task_results.append({"variant": variant, "seed": seed, "result": task})
            summary_path = resolve_suite_path(task["benchmark_result"]["summary"], args.suite_dir)
            summary = json.loads(summary_path.read_text())
            for channel in channels:
                if channel in summary.get("channels", {}):
                    rows.append(flatten_channel_row(seed, variant, channel, task, summary))
                else:
                    missing_rows.append(f"{variant} seed {seed} channel {channel} in {summary_path}")

    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} task result files. First missing paths:\n{preview}")
    if missing_rows and not args.allow_missing:
        preview = "\n".join(missing_rows[:20])
        raise ValueError(f"Missing {len(missing_rows)} channel result rows. First missing rows:\n{preview}")

    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        variant = str(row["variant"])
        channel = str(row["channel"])
        for metric in DEFAULT_METRICS:
            value = row.get(metric)
            if value is not None:
                grouped[variant][channel][metric].append(float(value))

    aggregated: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for variant in variants:
        aggregated[variant] = {}
        for channel in channels:
            aggregated[variant][channel] = {
                metric: summarize_numeric(values)
                for metric, values in grouped[variant][channel].items()
            }

    paired: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    baseline = args.paired_baseline
    for variant in variants:
        if variant == baseline:
            continue
        paired[variant] = {}
        for channel in channels:
            paired[variant][channel] = {}
            for metric in DEFAULT_METRICS:
                by_seed: dict[int, float] = {}
                baseline_by_seed: dict[int, float] = {}
                for row in rows:
                    if row["channel"] != channel or row.get(metric) is None:
                        continue
                    if row["variant"] == variant:
                        by_seed[int(row["seed"])] = float(row[metric])
                    elif row["variant"] == baseline:
                        baseline_by_seed[int(row["seed"])] = float(row[metric])
                common = sorted(set(by_seed) & set(baseline_by_seed))
                if common:
                    deltas = [by_seed[seed] - baseline_by_seed[seed] for seed in common]
                    paired[variant][channel][metric] = summarize_numeric(deltas)

    summary = {
        "suite_dir": str(args.suite_dir),
        "variants": variants,
        "channels": channels,
        "seeds": seeds,
        "missing": missing,
        "missing_rows": missing_rows,
        "num_rows": len(rows),
        "paired_baseline": baseline,
        "aggregated": aggregated,
        "paired_deltas_vs_baseline": paired,
        "task_results": task_results,
    }
    summary_path = args.suite_dir / "journal_wflow_suite_results.json"
    csv_path = args.suite_dir / "journal_wflow_per_seed.csv"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_csv(csv_path, rows)
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
