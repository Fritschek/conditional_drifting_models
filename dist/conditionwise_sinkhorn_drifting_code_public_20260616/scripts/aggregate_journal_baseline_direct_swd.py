from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate checkpoint-only baseline direct-SWD evaluations.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--channels", type=str, default="TDL")
    parser.add_argument("--variants", type=str, default="wgan,ddpm,ddim100")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def normalize_variant(variant: str) -> str:
    normalized = variant.lower().replace("-", "")
    if normalized in {"paper_wgan", "paperwgan"}:
        return "wgan"
    if normalized in {"ddim", "ddim100", "ddim_100"}:
        return "ddim100"
    return normalized


def result_path(suite_dir: Path, channel: str, seed: int, variant: str) -> Path:
    return suite_dir / f"baseline_direct_swd_{channel.lower()}_seed{seed}_{variant}.json"


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
    channels = parse_csv_list(args.channels)
    variants = [normalize_variant(variant) for variant in parse_csv_list(args.variants)]
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    grouped: dict[tuple[str, str, str], list[float]] = {}

    for channel in channels:
        for seed in seeds:
            for variant in variants:
                path = result_path(args.suite_dir, channel, seed, variant)
                if not path.exists():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics") or {}
                row: dict[str, object] = {
                    "seed": int(payload["seed"]),
                    "channel": payload["channel"],
                    "variant": payload["variant"],
                    "direct_swd": float(metrics["direct_swd"]),
                    "residual_swd": float(metrics["residual_swd"]),
                    "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                    "checkpoint": payload.get("checkpoint", ""),
                }
                rows.append(row)
                for metric in ("direct_swd", "residual_swd", "elapsed_seconds"):
                    grouped.setdefault((str(row["channel"]), str(row["variant"]), metric), []).append(float(row[metric]))

    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} baseline direct-SWD result files. First missing paths:\n{preview}")

    summary_rows: list[dict[str, object]] = []
    for channel in channels:
        for variant in variants:
            for metric in ("direct_swd", "residual_swd", "elapsed_seconds"):
                values = grouped.get((channel, variant, metric), [])
                if not values:
                    continue
                row: dict[str, object] = {"channel": channel, "variant": variant, "metric": metric}
                row.update(summarize(values))
                summary_rows.append(row)

    per_seed_csv = args.suite_dir / "journal_baseline_direct_swd_per_seed.csv"
    summary_csv = args.suite_dir / "journal_baseline_direct_swd_summary.csv"
    summary_json = args.suite_dir / "journal_baseline_direct_swd_results.json"
    write_csv(per_seed_csv, rows)
    write_csv(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "suite_dir": str(args.suite_dir),
                "channels": channels,
                "variants": variants,
                "seeds": seeds,
                "missing": missing,
                "num_rows": len(rows),
                "per_seed_csv": str(per_seed_csv),
                "summary_csv": str(summary_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_json), "csv": str(per_seed_csv), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
