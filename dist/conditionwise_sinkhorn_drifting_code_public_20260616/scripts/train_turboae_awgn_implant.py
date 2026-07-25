from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from conditional_drifting.channels import channel_registry
from conditional_drifting.paper2309_presets import ebno_to_noise
from conditional_drifting.training import (
    DriftingConfig,
    evaluate_conditional_anchor_metrics,
    evaluate_residual_model,
    select_device,
    set_seed,
    train_conditional_drifting,
)


def _write_json(path: str, payload: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the n=2 AWGN fiber-Sinkhorn implant used by TurboAE.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--ebno-db", type=float, default=4.0)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--drift-field", type=str, default="fiber_sinkhorn", choices=["kernel", "sinkhorn", "fiber_sinkhorn"])
    parser.add_argument("--conditioning-mode", type=str, default="none")
    parser.add_argument("--condition-kernel-scale", type=float, default=0.5)
    parser.add_argument("--target-kernel-scale", type=float, default=1.0)
    parser.add_argument("--target-kernel-mode", type=str, default="raw")
    parser.add_argument("--sinkhorn-min-epsilon", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fiber-generated-samples", type=int, default=4)
    parser.add_argument("--fiber-positive-samples", type=int, default=4)
    parser.add_argument("--fiber-reference-samples", type=int, default=4)
    parser.add_argument("--anchor-metrics", action="store_true")
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--anchor-samples", type=int, default=64)
    parser.add_argument("--anchor-swd-projections", type=int, default=64)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    out_path = os.path.abspath(args.out)
    save_dir = os.path.abspath(args.save_dir)
    channels = channel_registry()
    noise_std = ebno_to_noise(args.ebno_db, args.rate)

    cfg = DriftingConfig(
        n=int(args.n),
        noise_std=float(noise_std),
        dataset_size=int(args.dataset_size),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        eval_size=int(args.eval_size),
        swd_projections=int(args.swd_projections),
        is_residual=False,
        drift_field=args.drift_field,
        conditioning_mode=args.conditioning_mode,
        condition_kernel_scale=float(args.condition_kernel_scale),
        target_kernel_scale=float(args.target_kernel_scale),
        target_kernel_mode=args.target_kernel_mode,
        sinkhorn_min_epsilon=float(args.sinkhorn_min_epsilon),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
        fiber_generated_samples=int(args.fiber_generated_samples),
        fiber_positive_samples=int(args.fiber_positive_samples),
        fiber_reference_samples=int(args.fiber_reference_samples),
    )

    summary: dict[str, object] = {
        "seed": int(args.seed),
        "device": str(device),
        "status": "running",
        "channel": "AWGN",
        "config": asdict(cfg),
        "ebno_db": float(args.ebno_db),
        "rate": float(args.rate),
        "channels": {},
    }
    _write_json(out_path, summary)

    print("[AWGN] training TurboAE n=2 fiber-Sinkhorn implant", flush=True)
    model, artifacts = train_conditional_drifting(channels["AWGN"], cfg, device)

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"enhanced_direct_awgn_seed{args.seed}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "channel": "AWGN",
            "seed": int(args.seed),
            "history": artifacts.history,
        },
        ckpt_path,
    )

    eval_result = evaluate_residual_model(model, channels["AWGN"], cfg, device, metric_seed=args.seed)
    channel_summary = {
        "config": asdict(cfg),
        "final_history": artifacts.history[-1],
        "checkpoint_path": ckpt_path,
        "swd": float(eval_result["swd"]),
        "direct_swd": float(eval_result["direct_swd"]),
        "residual_swd": float(eval_result["residual_swd"]),
        "target_mode": eval_result["target_mode"],
    }
    if args.anchor_metrics:
        channel_summary["anchor_metrics"] = evaluate_conditional_anchor_metrics(
            model,
            channels["AWGN"],
            cfg,
            device,
            num_anchors=int(args.anchor_count),
            samples_per_anchor=int(args.anchor_samples),
            swd_projections=int(args.anchor_swd_projections),
            metric_seed=args.seed + 1_000,
        )

    summary["channels"] = {"AWGN": channel_summary}
    summary["status"] = "completed"
    _write_json(out_path, summary)
    print(json.dumps({"summary": out_path, "checkpoint": ckpt_path}, indent=2))


if __name__ == "__main__":
    main()
