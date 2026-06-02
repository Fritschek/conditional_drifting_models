from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.baselines import (
    DiffusionConfig,
    PaperWGANConfig,
    evaluate_diffusion_model,
    evaluate_paper_wgan,
    train_conditional_diffusion,
    train_paper_wgan,
)
from conditional_drifting.channels import channel_registry
from conditional_drifting.e2e_implants import save_implant_checkpoint
from conditional_drifting.paper2309_presets import PAPER2309_PRESETS, ebno_to_noise, resolve_paper_diffusion_lr_schedule
from conditional_drifting.training import select_device, set_seed


BASELINE_VARIANTS = {"wgan", "paper_wgan", "diffusion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reusable WGAN/diffusion channel implant checkpoints for journal curves.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--channel", type=str, required=True, choices=sorted(PAPER2309_PRESETS))
    parser.add_argument("--variants", type=str, default="wgan,diffusion")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--dataset-size", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--metric-eval-size", type=int, default=100_000)
    parser.add_argument("--swd-projections", type=int, default=-1)
    parser.add_argument("--diffusion-epochs", type=int, default=-1)
    parser.add_argument("--wgan-epochs", type=int, default=-1)
    parser.add_argument("--diffusion-learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-ddim-steps", type=int, default=100)
    parser.add_argument("--diffusion-eval-batch-size", type=int, default=-1)
    parser.add_argument("--skip-metric-eval", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_variant(variant: str) -> str:
    normalized = variant.lower()
    if normalized == "paper_wgan":
        return "wgan"
    if normalized in {"wgan", "diffusion"}:
        return normalized
    raise ValueError(f"Unsupported baseline variant {variant!r}; valid variants are {sorted(BASELINE_VARIANTS)}.")


def checkpoint_path(suite_dir: Path, variant: str, channel: str, seed: int) -> Path:
    return suite_dir / variant / f"seed{seed}" / "checkpoints" / f"{variant}_{channel.lower()}_seed{seed}.pt"


def variant_summary_path(suite_dir: Path, variant: str, channel: str, seed: int) -> Path:
    return suite_dir / variant / f"seed{seed}" / f"{variant}_{channel.lower()}_seed{seed}_summary.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    channels = channel_registry()
    preset = PAPER2309_PRESETS[args.channel]
    variants = []
    for variant in parse_csv_list(args.variants):
        normalized = normalize_variant(variant)
        if normalized not in variants:
            variants.append(normalized)

    dataset_size = args.dataset_size if args.dataset_size > 0 else preset.dataset_size
    batch_size = args.batch_size if args.batch_size > 0 else preset.batch_size
    metric_eval_size = args.metric_eval_size if args.metric_eval_size > 0 else preset.eval_size
    swd_projections = args.swd_projections if args.swd_projections > 0 else preset.swd_projections
    diffusion_epochs = args.diffusion_epochs if args.diffusion_epochs > 0 else preset.diffusion_epochs
    wgan_epochs = args.wgan_epochs if args.wgan_epochs > 0 else preset.wgan_epochs
    noise_std = ebno_to_noise(preset.ebn0_db, preset.rate)
    args.suite_dir.mkdir(parents=True, exist_ok=True)

    result_path = args.suite_dir / f"baseline_{args.channel.lower()}_seed{args.seed}_result.json"
    payload: dict[str, object] = {
        "seed": int(args.seed),
        "channel": args.channel,
        "variants": variants,
        "suite_dir": str(args.suite_dir),
        "preset": asdict(preset),
        "effective_config": {
            "dataset_size": int(dataset_size),
            "batch_size": int(batch_size),
            "metric_eval_size": int(metric_eval_size),
            "swd_projections": int(swd_projections),
            "diffusion_epochs": int(diffusion_epochs),
            "wgan_epochs": int(wgan_epochs),
            "noise_std": float(noise_std),
        },
        "runs": [],
        "status": "running",
    }
    write_json(result_path, payload)

    start = time.perf_counter()
    for variant in variants:
        set_seed(args.seed + {"wgan": 10_000, "diffusion": 20_000}[variant])
        ckpt = checkpoint_path(args.suite_dir, variant, args.channel, args.seed)
        summary_path = variant_summary_path(args.suite_dir, variant, args.channel, args.seed)
        run_start = time.perf_counter()
        run: dict[str, object] = {
            "variant": variant,
            "checkpoint": str(ckpt),
            "summary": str(summary_path),
            "trained": False,
            "metrics": {},
        }

        if ckpt.exists() and summary_path.exists() and not args.force_retrain:
            print(f"[baseline] reusing {ckpt}", flush=True)
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            run.update({key: cached.get(key) for key in ("metrics", "config") if key in cached})
        elif variant == "wgan":
            cfg = PaperWGANConfig(
                n=preset.n,
                noise_std=noise_std,
                dataset_size=dataset_size,
                batch_size=batch_size,
                epochs=wgan_epochs,
                eval_size=metric_eval_size,
                hidden_dim=preset.wgan_hidden_dim,
                swd_projections=swd_projections,
            )
            print(f"[{args.channel}] training WGAN seed={args.seed}", flush=True)
            model, state = train_paper_wgan(channels[args.channel], cfg, device)
            metrics = {}
            if not args.skip_metric_eval:
                metrics["paper_wgan"] = evaluate_paper_wgan(model, channels[args.channel], cfg, device, metric_seed=args.seed)["swd"]
            save_implant_checkpoint(
                ckpt,
                model,
                family="paper_wgan",
                metadata={"channel": args.channel, "seed": args.seed, "config": asdict(cfg), "history": state["history"]},
            )
            run.update({"trained": True, "config": asdict(cfg), "metrics": metrics, "final_history": state["history"][-1]})
        elif variant == "diffusion":
            cfg = DiffusionConfig(
                n=preset.n,
                noise_std=noise_std,
                dataset_size=dataset_size,
                batch_size=batch_size,
                epochs=diffusion_epochs,
                learning_rate=args.diffusion_learning_rate,
                hidden_dim=preset.diffusion_hidden_dim,
                num_steps=preset.num_steps,
                eval_size=metric_eval_size,
                swd_projections=swd_projections,
                ema_decay=0.9,
                pred_type="v",
                is_residual=False,
                beta_schedule="cosine-zf",
                eval_batch_size=args.diffusion_eval_batch_size if args.diffusion_eval_batch_size > 0 else None,
                learning_rate_schedule=resolve_paper_diffusion_lr_schedule(preset, diffusion_epochs),
            )
            print(f"[{args.channel}] training diffusion seed={args.seed}", flush=True)
            model, state = train_conditional_diffusion(channels[args.channel], cfg, device)
            metrics = {}
            if not args.skip_metric_eval:
                ddim_eval = evaluate_diffusion_model(
                    model,
                    channels[args.channel],
                    cfg,
                    device,
                    use_ddim=True,
                    ddim_steps=args.diffusion_ddim_steps,
                    metric_seed=args.seed,
                )
                metrics[f"ddim{args.diffusion_ddim_steps}"] = ddim_eval["swd"]
            save_implant_checkpoint(
                ckpt,
                model,
                family="diffusion",
                metadata={"channel": args.channel, "seed": args.seed, "config": asdict(cfg), "history": state["history"]},
            )
            run.update({"trained": True, "config": asdict(cfg), "metrics": metrics, "final_history": state["history"][-1]})
        else:
            raise AssertionError(f"Unhandled variant: {variant}")

        run["elapsed_seconds"] = time.perf_counter() - run_start
        write_json(summary_path, run)
        payload["runs"].append(run)
        write_json(result_path, payload)

    payload["elapsed_seconds"] = time.perf_counter() - start
    payload["status"] = "completed"
    write_json(result_path, payload)
    print(json.dumps({"seed": args.seed, "channel": args.channel, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
