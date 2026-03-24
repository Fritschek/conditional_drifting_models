from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conditional_drifting.channels import channel_registry
from conditional_drifting.paper2309_presets import PAPER2309_PRESETS, ebno_to_noise
from conditional_drifting.training import DriftingConfig, evaluate_residual_model, select_device, set_seed, train_conditional_drifting


def _write_summary(out_path: str, summary: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(tmp_path, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enhanced direct drifting on the paper benchmark channels.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA")
    parser.add_argument("--dataset-size", type=int, default=-1)
    parser.add_argument("--eval-size", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--drifting-epochs", type=int, default=-1)
    parser.add_argument("--swd-projections", type=int, default=-1)
    parser.add_argument("--conditioning-mode", type=str, default="joint")
    parser.add_argument("--condition-kernel-scale", type=float, default=0.5)
    parser.add_argument("--target-kernel-scale", type=float, default=1.0)
    parser.add_argument("--target-kernel-mode", type=str, default="raw")
    parser.add_argument("--condition-metric", type=str, default="euclidean")
    parser.add_argument("--adaptive-condition-bandwidth", action="store_true")
    parser.add_argument("--adaptive-target-bandwidth", action="store_true")
    parser.add_argument("--condition-embedding-dim", type=int, default=0)
    parser.add_argument("--positive-queue-size", type=int, default=0)
    parser.add_argument("--positive-reference-size", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    out_path = os.path.abspath(args.out)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else ""
    channels = channel_registry()
    requested = [name.strip() for name in args.channels.split(",") if name.strip()]
    summary: dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "status": "running",
        "completed_channels": [],
        "channels": {},
        "config": {
            "conditioning_mode": args.conditioning_mode,
            "condition_metric": args.condition_metric,
            "condition_kernel_scale": args.condition_kernel_scale,
            "target_kernel_scale": args.target_kernel_scale,
            "target_kernel_mode": args.target_kernel_mode,
            "adaptive_condition_bandwidth": args.adaptive_condition_bandwidth,
            "adaptive_target_bandwidth": args.adaptive_target_bandwidth,
            "condition_embedding_dim": args.condition_embedding_dim,
            "positive_queue_size": args.positive_queue_size,
            "positive_reference_size": args.positive_reference_size,
        },
    }
    _write_summary(out_path, summary)

    for channel_name in requested:
        preset = PAPER2309_PRESETS[channel_name]
        dataset_size = args.dataset_size if args.dataset_size > 0 else preset.dataset_size
        eval_size = args.eval_size if args.eval_size > 0 else preset.eval_size
        batch_size = args.batch_size if args.batch_size > 0 else preset.batch_size
        drifting_epochs = args.drifting_epochs if args.drifting_epochs > 0 else preset.drifting_epochs
        swd_projections = args.swd_projections if args.swd_projections > 0 else preset.swd_projections
        noise_std = ebno_to_noise(preset.ebn0_db, preset.rate)
        cfg = DriftingConfig(
            n=preset.n,
            noise_std=noise_std,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=drifting_epochs,
            eval_size=eval_size,
            swd_projections=swd_projections,
            is_residual=False,
            conditioning_mode=args.conditioning_mode,
            condition_metric=args.condition_metric,
            condition_kernel_scale=args.condition_kernel_scale,
            target_kernel_scale=args.target_kernel_scale,
            target_kernel_mode=args.target_kernel_mode,
            adaptive_condition_bandwidth=args.adaptive_condition_bandwidth,
            adaptive_target_bandwidth=args.adaptive_target_bandwidth,
            condition_embedding_dim=args.condition_embedding_dim,
            positive_queue_size=args.positive_queue_size,
            positive_reference_size=args.positive_reference_size,
        )
        print(f"[{channel_name}] training enhanced direct drifting", flush=True)
        model, artifacts = train_conditional_drifting(channels[channel_name], cfg, device)
        channel_summary = {
            "config": asdict(cfg),
            "final_history": artifacts.history[-1],
        }
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"enhanced_direct_{channel_name.lower()}_seed{args.seed}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "channel": channel_name,
                    "seed": args.seed,
                    "history": artifacts.history,
                },
                ckpt_path,
            )
            channel_summary["checkpoint_path"] = ckpt_path
            _write_summary(out_path, summary)

        summary["channels"][channel_name] = channel_summary
        summary["status"] = f"trained:{channel_name}"
        _write_summary(out_path, summary)
        eval_result = evaluate_residual_model(model, channels[channel_name], cfg, device, metric_seed=args.seed)
        summary["channels"][channel_name]["swd"] = float(eval_result["swd"])
        summary["completed_channels"].append(channel_name)
        summary["status"] = "running"
        _write_summary(out_path, summary)
        print(f"[{channel_name}] swd={eval_result['swd']:.6f}", flush=True)

    summary["status"] = "completed"
    _write_summary(out_path, summary)
    print(json.dumps({"summary": out_path}, indent=2))


if __name__ == "__main__":
    main()
