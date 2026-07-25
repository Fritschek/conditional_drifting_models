from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate journal WGAN/diffusion implant checkpoint suite results.")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--variants", type=str, default="wgan,diffusion")
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,TDL")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--check-checkpoints",
        action="store_true",
        help="Fail if checkpoint files referenced by the result JSONs are unavailable from this filesystem.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def normalize_variant(variant: str) -> str:
    normalized = variant.lower()
    if normalized == "paper_wgan":
        return "wgan"
    return normalized


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


def resolve_local_checkpoint(suite_dir: Path, checkpoint_text: str) -> tuple[Path, bool]:
    checkpoint = Path(checkpoint_text)
    if checkpoint.exists():
        return checkpoint, True
    parts = checkpoint.parts
    if suite_dir.name in parts:
        index = parts.index(suite_dir.name)
        local_checkpoint = suite_dir.joinpath(*parts[index + 1 :])
        if local_checkpoint.exists():
            return local_checkpoint, True
        return local_checkpoint, False
    return checkpoint, False


def main() -> None:
    args = parse_args()
    variants = [normalize_variant(variant) for variant in parse_csv_list(args.variants)]
    channels = parse_csv_list(args.channels)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)
    variant_set = set(variants)

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    missing_rows: list[str] = []
    missing_checkpoints: list[str] = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for channel in channels:
        for seed in seeds:
            result_path = args.suite_dir / f"baseline_{channel.lower()}_seed{seed}_result.json"
            if not result_path.exists():
                missing.append(str(result_path))
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            seen_variants = set()
            for run in payload.get("runs", []):
                variant = normalize_variant(str(run.get("variant")))
                if variant not in variant_set:
                    continue
                seen_variants.add(variant)
                checkpoint_source = str(run.get("checkpoint", ""))
                checkpoint, checkpoint_exists = resolve_local_checkpoint(args.suite_dir, checkpoint_source)
                if not checkpoint_exists:
                    missing_checkpoints.append(f"{channel} seed {seed} variant {variant} missing checkpoint {checkpoint_source}")
                row: dict[str, object] = {
                    "seed": int(seed),
                    "channel": channel,
                    "variant": variant,
                    "checkpoint": str(checkpoint),
                    "checkpoint_source": checkpoint_source,
                    "checkpoint_exists": checkpoint_exists,
                    "summary": run.get("summary"),
                    "trained": run.get("trained"),
                    "elapsed_seconds": run.get("elapsed_seconds"),
                }
                metrics = run.get("metrics") or {}
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        row[f"{metric}_swd"] = value
                        if value is not None:
                            grouped[(channel, variant, str(metric))].append(float(value))
                rows.append(row)
            for variant in variants:
                if variant not in seen_variants:
                    missing_rows.append(f"{channel} seed {seed} variant {variant} in {result_path}")

    if missing and not args.allow_missing:
        raise FileNotFoundError("Missing baseline task result files:\n" + "\n".join(missing[:20]))
    if missing_rows and not args.allow_missing:
        raise ValueError("Missing baseline result rows:\n" + "\n".join(missing_rows[:20]))
    if missing_checkpoints and args.check_checkpoints and not args.allow_missing:
        raise ValueError("Missing baseline checkpoints:\n" + "\n".join(missing_checkpoints[:20]))

    summary_rows: list[dict[str, object]] = []
    for channel in channels:
        for variant in variants:
            metric_names = sorted(metric for row_channel, row_variant, metric in grouped if row_channel == channel and row_variant == variant)
            if not metric_names:
                summary_rows.append({"channel": channel, "variant": variant})
                continue
            for metric in metric_names:
                stats = summarize(grouped[(channel, variant, metric)])
                row = {"channel": channel, "variant": variant, "metric": metric}
                for stat_name, stat_value in stats.items():
                    row[stat_name] = stat_value
                summary_rows.append(row)

    payload = {
        "suite_dir": str(args.suite_dir),
        "variants": variants,
        "channels": channels,
        "seeds": seeds,
        "missing": missing,
        "missing_rows": missing_rows,
        "missing_checkpoints": missing_checkpoints,
        "num_rows": len(rows),
    }
    summary_path = args.suite_dir / "journal_baseline_implant_results.json"
    per_seed_csv = args.suite_dir / "journal_baseline_implant_per_seed.csv"
    summary_csv = args.suite_dir / "journal_baseline_implant_summary.csv"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(per_seed_csv, rows)
    write_csv(summary_csv, summary_rows)
    print(json.dumps({"summary": str(summary_path), "per_seed_csv": str(per_seed_csv), "summary_csv": str(summary_csv), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
