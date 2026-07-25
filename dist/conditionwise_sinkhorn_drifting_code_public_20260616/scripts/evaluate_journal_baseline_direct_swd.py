from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import channel_registry
from conditional_drifting.e2e_implants import load_implant_from_checkpoint
from conditional_drifting.metrics import sliced_wasserstein_distance
from conditional_drifting.paper2309_presets import PAPER2309_PRESETS, ebno_to_noise
from conditional_drifting.training import select_device, set_seed


VARIANT_OFFSETS = {
    "wgan": 10_000,
    "ddpm": 20_000,
    "ddim100": 21_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate direct-output SWD for saved journal WGAN/diffusion baseline "
            "checkpoints without retraining."
        )
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--channels", type=str, default="TDL")
    parser.add_argument("--variants", type=str, default="wgan,ddpm,ddim100")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--eval-size", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--swd-projections", type=int, default=-1)
    parser.add_argument("--diffusion-ddim-steps", type=int, default=100)
    parser.add_argument(
        "--wgan-normalize-condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the paper WGAN training/evaluation convention by normalizing each condition batch.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Only write per-seed result JSON files. Intended for Slurm array workers.",
    )
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
    if normalized in {"wgan", "ddpm"}:
        return normalized
    raise ValueError("Unsupported variant {!r}; expected wgan, ddpm, or ddim100.".format(variant))


def checkpoint_path(suite_dir: Path, variant: str, channel: str, seed: int) -> Path:
    base_variant = "diffusion" if variant in {"ddpm", "ddim100"} else variant
    return suite_dir / base_variant / f"seed{seed}" / "checkpoints" / f"{base_variant}_{channel.lower()}_seed{seed}.pt"


def result_path(out_dir: Path, channel: str, seed: int, variant: str) -> Path:
    return out_dir / f"baseline_direct_swd_{channel.lower()}_seed{seed}_{variant}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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


def summarize(values: list[float]) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "num_seeds": 0}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / max(1, len(clean) - 1)
    std = math.sqrt(variance)
    return {"mean": mean, "std": std, "sem": std / math.sqrt(len(clean)), "num_seeds": len(clean)}


def _normalize_condition(x: torch.Tensor) -> torch.Tensor:
    return x / (x.std() + 1e-12)


@torch.no_grad()
def evaluate_one(
    *,
    checkpoint: Path,
    variant: str,
    channel_fn,
    n: int,
    noise_std: float,
    eval_size: int,
    batch_size: int,
    swd_projections: int,
    metric_seed: int,
    device: torch.device,
    diffusion_ddim_steps: int,
    wgan_normalize_condition: bool,
) -> dict[str, float]:
    if variant == "ddpm":
        implant = load_implant_from_checkpoint(checkpoint, device=device, diffusion_sampler="ddpm", ddim_steps=None)
    elif variant == "ddim100":
        implant = load_implant_from_checkpoint(
            checkpoint,
            device=device,
            diffusion_sampler="ddim",
            ddim_steps=diffusion_ddim_steps,
        )
    else:
        implant = load_implant_from_checkpoint(checkpoint, device=device)

    remaining = int(eval_size)
    y_true_parts: list[torch.Tensor] = []
    y_pred_parts: list[torch.Tensor] = []
    residual_true_parts: list[torch.Tensor] = []
    residual_pred_parts: list[torch.Tensor] = []
    sample_seconds = 0.0

    while remaining > 0:
        current_bs = min(int(batch_size), remaining)
        x = torch.randn(current_bs, n, device=device)
        if variant == "wgan" and wgan_normalize_condition:
            x = _normalize_condition(x)
        y_true = channel_fn(x, noise_std, device)
        t0 = time.perf_counter()
        y_pred = implant(x)
        sample_seconds += time.perf_counter() - t0

        y_true_parts.append(y_true.detach().cpu())
        y_pred_parts.append(y_pred.detach().cpu())
        residual_true_parts.append((y_true - x).detach().cpu())
        residual_pred_parts.append((y_pred - x).detach().cpu())
        remaining -= current_bs

    y_true_all = torch.cat(y_true_parts, dim=0)
    y_pred_all = torch.cat(y_pred_parts, dim=0)
    residual_true_all = torch.cat(residual_true_parts, dim=0)
    residual_pred_all = torch.cat(residual_pred_parts, dim=0)
    direct_swd = sliced_wasserstein_distance(
        y_true_all,
        y_pred_all,
        num_projections=swd_projections,
        seed=metric_seed,
    )
    residual_swd = sliced_wasserstein_distance(
        residual_true_all,
        residual_pred_all,
        num_projections=swd_projections,
        seed=metric_seed,
    )
    return {
        "direct_swd": float(direct_swd),
        "residual_swd": float(residual_swd),
        "sample_seconds": float(sample_seconds),
    }


def collect_existing_rows(out_dir: Path, channels: list[str], seeds: list[int], variants: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for channel in channels:
        for seed in seeds:
            for variant in variants:
                path = result_path(out_dir, channel, seed, variant)
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "seed": int(payload["seed"]),
                        "channel": payload["channel"],
                        "variant": payload["variant"],
                        "direct_swd": float(payload["metrics"]["direct_swd"]),
                        "residual_swd": float(payload["metrics"]["residual_swd"]),
                        "elapsed_seconds": float(payload["elapsed_seconds"]),
                        "checkpoint": payload["checkpoint"],
                    }
                )
    return rows


def write_summary(out_dir: Path, rows: list[dict[str, object]], channels: list[str], variants: list[str]) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        for metric in ("direct_swd", "residual_swd", "elapsed_seconds"):
            grouped.setdefault((str(row["channel"]), str(row["variant"]), metric), []).append(float(row[metric]))

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

    write_csv(out_dir / "journal_baseline_direct_swd_per_seed.csv", rows)
    write_csv(out_dir / "journal_baseline_direct_swd_summary.csv", summary_rows)
    write_json(
        out_dir / "journal_baseline_direct_swd_results.json",
        {
            "out_dir": str(out_dir),
            "channels": channels,
            "variants": variants,
            "num_rows": len(rows),
            "per_seed_csv": str(out_dir / "journal_baseline_direct_swd_per_seed.csv"),
            "summary_csv": str(out_dir / "journal_baseline_direct_swd_summary.csv"),
        },
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    channels = parse_csv_list(args.channels)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds)
    variants: list[str] = []
    for variant in parse_csv_list(args.variants):
        normalized = normalize_variant(variant)
        if normalized not in variants:
            variants.append(normalized)

    channel_fns = channel_registry()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for channel in channels:
        if channel not in PAPER2309_PRESETS:
            raise ValueError(f"Unsupported channel {channel!r}; valid paper presets are {sorted(PAPER2309_PRESETS)}.")
        preset = PAPER2309_PRESETS[channel]
        if channel not in channel_fns:
            raise ValueError(f"Channel {channel!r} is not registered.")
        eval_size = args.eval_size if args.eval_size > 0 else preset.eval_size
        batch_size = args.batch_size if args.batch_size > 0 else preset.batch_size
        swd_projections = args.swd_projections if args.swd_projections > 0 else preset.swd_projections
        noise_std = ebno_to_noise(preset.ebn0_db, preset.rate)

        for seed in seeds:
            for variant in variants:
                ckpt = checkpoint_path(args.suite_dir, variant, channel, seed)
                if not ckpt.exists():
                    raise FileNotFoundError(f"Missing baseline checkpoint: {ckpt}")
                set_seed(seed + VARIANT_OFFSETS[variant])
                start = time.perf_counter()
                print(f"[baseline-direct-swd] channel={channel} seed={seed} variant={variant}", flush=True)
                metrics = evaluate_one(
                    checkpoint=ckpt,
                    variant=variant,
                    channel_fn=channel_fns[channel],
                    n=preset.n,
                    noise_std=noise_std,
                    eval_size=eval_size,
                    batch_size=batch_size,
                    swd_projections=swd_projections,
                    metric_seed=seed,
                    device=device,
                    diffusion_ddim_steps=args.diffusion_ddim_steps,
                    wgan_normalize_condition=args.wgan_normalize_condition,
                )
                elapsed = time.perf_counter() - start
                payload = {
                    "seed": int(seed),
                    "channel": channel,
                    "variant": variant,
                    "checkpoint": str(ckpt),
                    "metrics": metrics,
                    "elapsed_seconds": float(elapsed),
                    "config": {
                        "preset": asdict(preset),
                        "eval_size": int(eval_size),
                        "batch_size": int(batch_size),
                        "swd_projections": int(swd_projections),
                        "noise_std": float(noise_std),
                        "metric_seed": int(seed),
                        "diffusion_ddim_steps": int(args.diffusion_ddim_steps),
                        "wgan_normalize_condition": bool(args.wgan_normalize_condition),
                    },
                }
                write_json(result_path(args.out_dir, channel, seed, variant), payload)
                rows.append(
                    {
                        "seed": int(seed),
                        "channel": channel,
                        "variant": variant,
                        "direct_swd": metrics["direct_swd"],
                        "residual_swd": metrics["residual_swd"],
                        "elapsed_seconds": float(elapsed),
                        "checkpoint": str(ckpt),
                    }
                )
                print(
                    json.dumps(
                        {
                            "channel": channel,
                            "seed": seed,
                            "variant": variant,
                            "direct_swd": metrics["direct_swd"],
                            "elapsed_seconds": elapsed,
                        }
                    ),
                    flush=True,
                )

    if not args.skip_summary:
        rows = collect_existing_rows(args.out_dir, channels, seeds, variants)
        write_summary(args.out_dir, rows, channels, variants)
        print(
            json.dumps(
                {
                    "summary": str(args.out_dir / "journal_baseline_direct_swd_results.json"),
                    "csv": str(args.out_dir / "journal_baseline_direct_swd_per_seed.csv"),
                    "num_rows": len(rows),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
