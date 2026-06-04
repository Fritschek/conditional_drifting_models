from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1)))


def sem(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(std(values) / math.sqrt(len(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate equal wall-clock symbolic AE implant suite results.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,TDL")
    parser.add_argument("--variants", type=str, default="analytic,fiber_sinkhorn,wgan,diffusion_ddim100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channels = parse_csv_list(args.channels)
    variants = parse_csv_list(args.variants)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_seeds)))
    missing = []
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for channel in channels:
            result_path = args.suite_dir / f"wallclock_{channel.lower()}_seed{seed}_result.json"
            if not result_path.exists():
                missing.append(result_path)
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            by_variant = {run["variant"]: run for run in payload.get("runs", [])}
            for variant in variants:
                if variant not in by_variant:
                    missing.append(result_path.with_name(f"{result_path.stem}:{variant}"))
                    continue
                run = by_variant[variant]
                rows.append(
                    {
                        "seed": int(seed),
                        "channel": channel,
                        "variant": variant,
                        "train_seconds_budget": float(run["train_seconds_budget"]),
                        "train_seconds": float(run["train_seconds"]),
                        "optimizer_updates": int(run["optimizer_updates"]),
                        "channel_calls": int(run["channel_calls"]),
                        "channel_samples": int(run["channel_samples"]),
                        "sampler_steps_per_sample": int(run["sampler_steps_per_sample"]),
                        "effective_sampler_steps": int(run["effective_sampler_steps"]),
                        "eval_ser": float(run["eval_ser"]),
                        "eval_ber": float(run["eval_ber"]),
                        "eval_loss": float(run["eval_loss"]),
                        "eval_air_bits_per_message": float(run["eval_air_bits_per_message"]),
                        "eval_normalized_air": float(run["eval_normalized_air"]),
                    }
                )
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} equal-wall-clock result entries. First missing:\n{preview}")

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["channel"]), str(row["variant"]))].append(row)

    summary_rows = []
    for (channel, variant), group in sorted(groups.items()):
        summary = {"channel": channel, "variant": variant, "num_seeds": len(group)}
        for key in [
            "train_seconds",
            "optimizer_updates",
            "channel_samples",
            "effective_sampler_steps",
            "eval_ser",
            "eval_ber",
            "eval_loss",
            "eval_air_bits_per_message",
            "eval_normalized_air",
        ]:
            values = [float(row[key]) for row in group]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = std(values)
            summary[f"{key}_sem"] = sem(values)
        summary_rows.append(summary)

    args.suite_dir.mkdir(parents=True, exist_ok=True)
    per_seed_csv = args.suite_dir / "equal_wallclock_symbolic_per_seed.csv"
    with per_seed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = args.suite_dir / "equal_wallclock_symbolic_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = args.suite_dir / "equal_wallclock_symbolic_results.json"
    summary_json.write_text(
        json.dumps(
            {
                "suite_dir": str(args.suite_dir),
                "seed_start": int(args.seed_start),
                "num_seeds": int(args.num_seeds),
                "channels": channels,
                "variants": variants,
                "num_rows": len(rows),
                "per_seed_csv": str(per_seed_csv),
                "summary_csv": str(summary_csv),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_json), "csv": str(per_seed_csv), "summary_csv": str(summary_csv), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
