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
from conditional_drifting.paper2309_presets import PAPER2309_PRESETS, PaperChannelPreset, ebno_to_noise
from conditional_drifting.training import (
    DriftingConfig,
    evaluate_conditional_anchor_metrics,
    evaluate_residual_model,
    select_device,
    set_seed,
    train_conditional_drifting,
)


ENHANCED_DIRECT_PRESETS = {
    **PAPER2309_PRESETS,
    "OptFib": PaperChannelPreset(
        n=2,
        ebn0_db=5.0,
        rate=1.0,
        diffusion_hidden_dim=128,
        wgan_hidden_dim=128,
        diffusion_epochs=60,
        drifting_epochs=60,
        wgan_epochs=60,
        dataset_size=120_000,
        batch_size=512,
        eval_size=100_000,
        swd_projections=256,
    ),
}


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
    parser.add_argument("--drift-field", type=str, default="kernel", choices=["kernel", "sinkhorn", "fiber_sinkhorn"])
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--max-drift-norm", type=float, default=2.0)
    parser.add_argument("--repulsive-weight", type=float, default=1.0)
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
    parser.add_argument("--sinkhorn-epsilon", type=float, default=None)
    parser.add_argument("--sinkhorn-min-epsilon", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fiber-generated-samples", type=int, default=4)
    parser.add_argument("--fiber-positive-samples", type=int, default=4)
    parser.add_argument("--fiber-reference-samples", type=int, default=4)
    parser.add_argument("--anchor-metrics", action="store_true")
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--anchor-samples", type=int, default=64)
    parser.add_argument("--anchor-swd-projections", type=int, default=64)
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
    unknown = [name for name in requested if name not in ENHANCED_DIRECT_PRESETS]
    if unknown:
        raise ValueError(f"Unsupported enhanced-direct channels: {unknown}")
    summary: dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "status": "running",
        "completed_channels": [],
        "channels": {},
        "config": {
            "drift_field": args.drift_field,
            "drift_scale": args.drift_scale,
            "max_drift_norm": args.max_drift_norm,
            "repulsive_weight": args.repulsive_weight,
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
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "sinkhorn_min_epsilon": args.sinkhorn_min_epsilon,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "fiber_generated_samples": args.fiber_generated_samples,
            "fiber_positive_samples": args.fiber_positive_samples,
            "fiber_reference_samples": args.fiber_reference_samples,
            "anchor_metrics": args.anchor_metrics,
            "anchor_count": args.anchor_count,
            "anchor_samples": args.anchor_samples,
            "anchor_swd_projections": args.anchor_swd_projections,
        },
    }
    _write_summary(out_path, summary)

    for channel_name in requested:
        preset = ENHANCED_DIRECT_PRESETS[channel_name]
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
            drift_field=args.drift_field,
            drift_scale=args.drift_scale,
            max_drift_norm=args.max_drift_norm,
            repulsive_weight=args.repulsive_weight,
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
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_min_epsilon=args.sinkhorn_min_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            fiber_generated_samples=args.fiber_generated_samples,
            fiber_positive_samples=args.fiber_positive_samples,
            fiber_reference_samples=args.fiber_reference_samples,
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
        summary["channels"][channel_name]["direct_swd"] = float(eval_result["direct_swd"])
        summary["channels"][channel_name]["residual_swd"] = float(eval_result["residual_swd"])
        summary["channels"][channel_name]["target_mode"] = eval_result["target_mode"]
        if args.anchor_metrics:
            anchor_metrics = evaluate_conditional_anchor_metrics(
                model,
                channels[channel_name],
                cfg,
                device,
                num_anchors=args.anchor_count,
                samples_per_anchor=args.anchor_samples,
                swd_projections=args.anchor_swd_projections,
                metric_seed=args.seed + 1_000,
            )
            summary["channels"][channel_name]["anchor_metrics"] = anchor_metrics
        summary["completed_channels"].append(channel_name)
        summary["status"] = "running"
        _write_summary(out_path, summary)
        print(f"[{channel_name}] swd={eval_result['swd']:.6f}", flush=True)

    summary["status"] = "completed"
    _write_summary(out_path, summary)
    print(json.dumps({"summary": out_path}, indent=2))


if __name__ == "__main__":
    main()
