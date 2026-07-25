from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TurboAE long-block seeded suite results.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--lengths", type=str, default="64")
    parser.add_argument("--modes", type=str, default="analytic,checkpoint")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int) -> list[int]:
    if seed_text:
        return parse_csv_ints(seed_text)
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
            "count": 0,
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
        "count": len(clean),
    }


def best_eval(history: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in history if row.get("eval_ber") is not None]
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row["eval_ber"]))


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def flatten_run(seed: int, length: int, mode: str, run_summary_path: Path) -> dict[str, Any]:
    data = json.loads(run_summary_path.read_text())
    best = best_eval(data.get("history", []))
    final_eval = data.get("final_eval", {})
    row: dict[str, Any] = {
        "seed": seed,
        "length": length,
        "mode": mode,
        "channel": data.get("channel"),
        "model_type": data.get("model_type"),
        "train_implant": data.get("train_implant"),
        "train_implant_source": data.get("train_implant_source"),
        "eval_implant": data.get("eval_implant"),
        "eval_implant_source": data.get("eval_implant_source"),
        "training_regime": data.get("training_regime"),
        "epochs": data.get("epochs"),
        "rate": data.get("rate"),
        "ebno_db": data.get("ebno_db"),
        "train_seconds": data.get("train_seconds"),
        "summary_path": str(run_summary_path),
        "best_epoch": best.get("epoch"),
        "best_eval_loss": best.get("eval_loss"),
        "best_eval_ber": best.get("eval_ber"),
        "best_eval_bler": best.get("eval_bler"),
        "final_eval_loss": final_eval.get("eval_loss"),
        "final_eval_ber": final_eval.get("eval_ber"),
        "final_eval_bler": final_eval.get("eval_bler"),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
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


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = [
        "best_eval_loss",
        "best_eval_ber",
        "best_eval_bler",
        "final_eval_loss",
        "final_eval_ber",
        "final_eval_bler",
        "train_seconds",
    ]
    for row in rows:
        key = f"length{row['length']}/{row['mode']}"
        for metric in metrics:
            value = float_or_none(row.get(metric))
            if value is not None:
                grouped[key][metric].append(value)

    return {
        group_key: {metric: summarize_numeric(values) for metric, values in metric_values.items()}
        for group_key, metric_values in grouped.items()
    }


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        by_key[(int(row["seed"]), int(row["length"]), str(row["mode"]))] = row

    deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    wins: dict[str, list[float]] = defaultdict(list)
    metrics = ["best_eval_ber", "best_eval_bler", "final_eval_ber", "final_eval_bler"]

    for (seed, length, mode), row in by_key.items():
        if mode == "analytic":
            continue
        analytic = by_key.get((seed, length, "analytic"))
        if analytic is None:
            continue
        group_key = f"length{length}/{mode}_vs_analytic"
        for metric in metrics:
            value = float_or_none(row.get(metric))
            baseline = float_or_none(analytic.get(metric))
            if value is None or baseline is None:
                continue
            deltas[group_key][f"{metric}_delta"].append(value - baseline)
            if baseline > 0:
                deltas[group_key][f"{metric}_ratio"].append(value / baseline)
        value = float_or_none(row.get("best_eval_ber"))
        baseline = float_or_none(analytic.get("best_eval_ber"))
        if value is not None and baseline is not None:
            wins[group_key].append(1.0 if value < baseline else 0.0)

    summary = {
        group_key: {metric: summarize_numeric(values) for metric, values in metric_values.items()}
        for group_key, metric_values in deltas.items()
    }
    for group_key, values in wins.items():
        summary.setdefault(group_key, {})["best_eval_ber_win_rate"] = summarize_numeric(values)
    return summary


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)
    lengths = parse_csv_ints(args.lengths)
    modes = parse_csv_strings(args.modes)

    missing: list[str] = []
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_summary_path = args.suite_dir / f"seed{seed}" / "summary.json"
        if not seed_summary_path.exists():
            missing.append(str(seed_summary_path))
            continue

        seed_summary = json.loads(seed_summary_path.read_text())
        seen_runs: set[tuple[int, str]] = set()
        for run in seed_summary.get("runs", []):
            length = int(run["length"])
            mode = str(run["mode"])
            if length not in lengths or mode not in modes:
                continue
            run_summary_path = Path(run["summary"])
            if not run_summary_path.exists():
                # Let copied result trees resolve relative to the current suite directory.
                candidate = args.suite_dir / f"seed{seed}" / f"{mode}_l{length}" / "summary.json"
                run_summary_path = candidate
            if run_summary_path.exists():
                rows.append(flatten_run(seed, length, mode, run_summary_path))
                seen_runs.add((length, mode))
            else:
                missing.append(str(run_summary_path))

        for length in lengths:
            for mode in modes:
                if (length, mode) in seen_runs:
                    continue
                fallback = args.suite_dir / f"seed{seed}" / f"{mode}_l{length}" / "summary.json"
                if fallback.exists():
                    rows.append(flatten_run(seed, length, mode, fallback))
                else:
                    missing.append(str(fallback))

    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} TurboAE result files. First missing paths:\n{preview}")

    aggregate = grouped_summary(rows)
    paired = paired_summary(rows)
    summary = {
        "suite_dir": str(args.suite_dir),
        "seeds": seeds,
        "lengths": lengths,
        "modes": modes,
        "num_rows": len(rows),
        "missing": missing,
        "aggregated": aggregate,
        "paired_vs_analytic": paired,
    }

    summary_path = args.suite_dir / "turboae_long_block_results.json"
    csv_path = args.suite_dir / "turboae_long_block_per_seed.csv"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_csv(csv_path, rows)
    print(json.dumps({"summary": str(summary_path), "csv": str(csv_path), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
