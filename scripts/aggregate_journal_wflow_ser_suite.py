from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = [
    "final_eval_ser",
    "final_eval_ber",
    "final_eval_loss",
    "train_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate journal W-Flow symbolic BER/SER follow-up results.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default="analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn")
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,OptFib")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--paired-baseline", type=str, default="analytic")
    parser.add_argument("--allow-missing", action="store_true")
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


def load_rows(suite_dir: Path, channels: list[str], variants: list[str], seeds: list[int]) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    variant_set = set(variants)
    for channel in channels:
        for seed in seeds:
            result_path = suite_dir / f"ser_{channel.lower()}_seed{seed}_result.json"
            if not result_path.exists():
                missing.append(str(result_path))
                continue
            payload = json.loads(result_path.read_text())
            for run in payload.get("runs", []):
                if run.get("variant") not in variant_set:
                    continue
                row = {
                    "seed": int(run["seed"]),
                    "channel": str(run["channel"]),
                    "variant": str(run["variant"]),
                    "summary": run.get("summary"),
                    "log": run.get("log"),
                    "checkpoint_path": run.get("checkpoint_path"),
                    "final_eval_ser": run.get("final_eval_ser"),
                    "final_eval_ber": run.get("final_eval_ber"),
                    "final_eval_loss": run.get("final_eval_loss"),
                    "train_seconds": run.get("train_seconds"),
                }
                rows.append(row)
    return rows, missing


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants)
    channels = parse_csv_list(args.channels)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)
    rows, missing = load_rows(args.suite_dir, channels, variants, seeds)
    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} SER task result files. First missing paths:\n{preview}")

    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        variant = str(row["variant"])
        channel = str(row["channel"])
        for metric in METRICS:
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
            for metric in METRICS:
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
                    paired[variant][channel][metric] = summarize_numeric(
                        [by_seed[seed] - baseline_by_seed[seed] for seed in common]
                    )

    summary = {
        "suite_dir": str(args.suite_dir),
        "variants": variants,
        "channels": channels,
        "seeds": seeds,
        "missing": missing,
        "num_rows": len(rows),
        "paired_baseline": baseline,
        "aggregated": aggregated,
        "paired_deltas_vs_baseline": paired,
    }
    summary_path = args.suite_dir / "journal_wflow_ser_results.json"
    csv_path = args.suite_dir / "journal_wflow_ser_per_seed.csv"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_csv(csv_path, rows)
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
