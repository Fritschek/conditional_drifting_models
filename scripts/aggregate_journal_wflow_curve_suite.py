from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ["ser", "ber", "loss", "cross_entropy_bits", "air_bits_per_message", "normalized_air"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate journal BER/SER curve results for learned channel surrogates.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default="analytic,fiber_sinkhorn,wgan,diffusion_ddim100")
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,TDL")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def summarize(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "num_seeds": 0}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / max(1, len(clean) - 1)
    std = math.sqrt(variance)
    return {"mean": mean, "std": std, "sem": std / math.sqrt(len(clean)), "num_seeds": len(clean)}


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
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants)
    channels = parse_csv_list(args.channels)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)
    variant_set = set(variants)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    missing_rows: list[str] = []
    expected_points: dict[str, set[float]] = defaultdict(set)

    for channel in channels:
        for seed in seeds:
            result_path = args.suite_dir / f"curve_{channel.lower()}_seed{seed}_result.json"
            if not result_path.exists():
                missing.append(str(result_path))
                continue
            payload = json.loads(result_path.read_text())
            task_points = {float(value) for value in payload.get("ebno_values", [])}
            expected_points[channel].update(task_points)
            seen: set[tuple[str, float]] = set()
            for run in payload.get("runs", []):
                variant = str(run.get("variant"))
                if variant not in variant_set:
                    continue
                for point in run.get("curve", []):
                    ebno_db = float(point["ebno_db"])
                    if not task_points:
                        expected_points[channel].add(ebno_db)
                        task_points.add(ebno_db)
                    seen.add((variant, ebno_db))
                    rows.append(
                        {
                            "seed": int(seed),
                            "channel": channel,
                            "variant": variant,
                            "train_ebno_db": run.get("train_ebno_db"),
                            "ebno_db": ebno_db,
                            "ser": point.get("ser"),
                            "ber": point.get("ber"),
                            "loss": point.get("loss"),
                            "cross_entropy_bits": point.get("cross_entropy_bits"),
                            "air_bits_per_message": point.get("air_bits_per_message"),
                            "normalized_air": point.get("normalized_air"),
                            "checkpoint": run.get("checkpoint"),
                            "summary": run.get("summary"),
                        }
                    )
            for variant in variants:
                seen_points = {ebno_db for seen_variant, ebno_db in seen if seen_variant == variant}
                if not seen_points:
                    missing_rows.append(f"{channel} seed {seed} variant {variant} in {result_path}")
                    continue
                missing_points = sorted(task_points - seen_points)
                for ebno_db in missing_points:
                    missing_rows.append(f"{channel} seed {seed} variant {variant} Eb/N0 {ebno_db:g} dB in {result_path}")

    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing curve task result files:\n" + "\n".join(missing[:20]))
    if missing_rows and not args.allow_missing:
        raise ValueError("Missing curve result rows:\n" + "\n".join(missing_rows[:20]))

    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["channel"]), str(row["variant"]), float(row["ebno_db"]))
        for metric in METRICS:
            value = row.get(metric)
            if value is not None:
                grouped[key][metric].append(float(value))

    summary_rows: list[dict[str, object]] = []
    aggregated: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = defaultdict(lambda: defaultdict(dict))
    for channel in channels:
        for variant in variants:
            for ebno_db in sorted(expected_points[channel]):
                key = (channel, variant, ebno_db)
                metric_summary = {metric: summarize(grouped[key][metric]) for metric in METRICS}
                aggregated[channel][variant][str(ebno_db)] = metric_summary
                row = {"channel": channel, "variant": variant, "ebno_db": ebno_db}
                for metric, stats in metric_summary.items():
                    for stat_name, stat_value in stats.items():
                        row[f"{metric}_{stat_name}"] = stat_value
                summary_rows.append(row)

    payload = {
        "suite_dir": str(args.suite_dir),
        "variants": variants,
        "channels": channels,
        "seeds": seeds,
        "missing": missing,
        "missing_rows": missing_rows,
        "num_rows": len(rows),
        "aggregated": aggregated,
    }
    summary_path = args.suite_dir / "journal_wflow_curve_results.json"
    per_seed_csv = args.suite_dir / "journal_wflow_curve_per_seed.csv"
    summary_csv = args.suite_dir / "journal_wflow_curve_summary.csv"
    summary_path.write_text(json.dumps(payload, indent=2))
    write_csv(per_seed_csv, rows)
    write_csv(summary_csv, summary_rows)
    print(json.dumps({"summary": str(summary_path), "per_seed_csv": str(per_seed_csv), "summary_csv": str(summary_csv), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
