from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import channel_registry
from conditional_drifting.e2e_implants import save_implant_checkpoint
from conditional_drifting.weight_registry import build_artifact_path, register_artifact
from conditional_drifting.training import DriftingConfig, select_device, set_seed, train_conditional_drifting
from conditional_drifting.baselines import DiffusionConfig, PaperWGANConfig, train_conditional_diffusion, train_paper_wgan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a vector-valued channel implant and save it as an end-to-end checkpoint.")
    parser.add_argument("--family", required=True, choices=["drifting_direct", "drifting_residual", "paper_wgan", "diffusion_direct", "diffusion_residual"])
    parser.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "TDL", "OptFib"])
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--ebno-db", type=float, default=None, help="If set for AWGN-style implants, derive noise_std from Eb/N0 instead of using --noise-std.")
    parser.add_argument("--rate", type=float, default=0.5, help="Code rate used when deriving noise_std from --ebno-db.")
    parser.add_argument("--dataset-size", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--pred-type", type=str, default=None)
    parser.add_argument("--beta-schedule", type=str, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--diffusion-profile", type=str, default="paper2309", choices=["paper2309", "muah_symbolic"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay-epoch", type=int, default=0)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--drift-field", type=str, default="kernel", choices=["kernel", "sinkhorn", "fiber_sinkhorn"])
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--max-drift-norm", type=float, default=2.0)
    parser.add_argument("--repulsive-weight", type=float, default=1.0)
    parser.add_argument("--conditional-kernel", action="store_true", help="Use a joint kernel on concatenated [condition, target] during drifting training.")
    parser.add_argument("--conditioning-mode", type=str, default="none", choices=["none", "joint", "product", "local", "soft_local", "radius", "mixture"])
    parser.add_argument("--condition-metric", type=str, default="euclidean", choices=["euclidean", "whitened"])
    parser.add_argument("--condition-kernel-scale", type=float, default=1.0)
    parser.add_argument("--target-kernel-scale", type=float, default=1.0)
    parser.add_argument("--condition-bandwidth", type=float, default=None)
    parser.add_argument("--target-bandwidth", type=float, default=None)
    parser.add_argument("--local-condition-k", type=int, default=32)
    parser.add_argument("--condition-radius", type=float, default=None)
    parser.add_argument("--mixture-alpha", type=float, default=0.5)
    parser.add_argument("--target-kernel-mode", type=str, default="raw", choices=["raw", "raw_plus_residual"])
    parser.add_argument("--residual-target-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-condition-bandwidth", action="store_true")
    parser.add_argument("--adaptive-target-bandwidth", action="store_true")
    parser.add_argument("--adaptive-bandwidth-k", type=int, default=16)
    parser.add_argument("--condition-embedding-dim", type=int, default=0)
    parser.add_argument("--condition-embedding-hidden-dim", type=int, default=64)
    parser.add_argument("--positive-queue-size", type=int, default=0)
    parser.add_argument("--positive-reference-size", type=int, default=0)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=None)
    parser.add_argument("--sinkhorn-min-epsilon", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fiber-generated-samples", type=int, default=4)
    parser.add_argument("--fiber-positive-samples", type=int, default=4)
    parser.add_argument("--fiber-reference-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--register-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    channel_fn = channel_registry()[args.channel]
    noise_std = float(args.noise_std)
    if args.ebno_db is not None:
        ebno_linear = 10.0 ** (float(args.ebno_db) / 10.0)
        noise_std = math.sqrt(1.0 / (2.0 * float(args.rate) * ebno_linear))
    artifact_name = args.register_name or f"{args.family.lower()}_{args.channel.lower()}_seed{args.seed}"
    out_path = (
        Path(args.out)
        if args.out is not None
        else build_artifact_path(
            root=args.weights_root,
            category="channel_implants",
            group=args.channel.lower(),
            name=artifact_name,
            suffix=".pt",
        )
    )

    metadata = {
        "channel": args.channel,
        "noise_std": noise_std,
        "ebno_db": args.ebno_db,
        "rate": args.rate,
        "seed": args.seed,
        "n": args.n,
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "drift_field": args.drift_field,
        "drift_scale": args.drift_scale,
        "max_drift_norm": args.max_drift_norm,
        "repulsive_weight": args.repulsive_weight,
        "conditioning_mode": args.conditioning_mode,
        "condition_metric": args.condition_metric,
        "condition_bandwidth": args.condition_bandwidth,
        "target_bandwidth": args.target_bandwidth,
        "local_condition_k": args.local_condition_k,
        "condition_radius": args.condition_radius,
        "mixture_alpha": args.mixture_alpha,
        "target_kernel_mode": args.target_kernel_mode,
        "residual_target_scale": args.residual_target_scale,
        "adaptive_condition_bandwidth": args.adaptive_condition_bandwidth,
        "adaptive_target_bandwidth": args.adaptive_target_bandwidth,
        "adaptive_bandwidth_k": args.adaptive_bandwidth_k,
        "condition_embedding_dim": args.condition_embedding_dim,
        "positive_queue_size": args.positive_queue_size,
        "positive_reference_size": args.positive_reference_size,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "sinkhorn_min_epsilon": args.sinkhorn_min_epsilon,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "fiber_generated_samples": args.fiber_generated_samples,
        "fiber_positive_samples": args.fiber_positive_samples,
        "fiber_reference_samples": args.fiber_reference_samples,
    }
    train_start = time.perf_counter()

    if args.family.startswith("drifting"):
        is_residual = args.family.endswith("residual")
        conditioning_mode = args.conditioning_mode
        if conditioning_mode == "none" and args.conditional_kernel:
            conditioning_mode = "joint"
        cfg = DriftingConfig(
            n=args.n,
            noise_std=noise_std,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lr_decay_epoch=args.lr_decay_epoch,
            lr_decay_factor=args.lr_decay_factor,
            drift_field=args.drift_field,
            drift_scale=args.drift_scale,
            max_drift_norm=args.max_drift_norm,
            repulsive_weight=args.repulsive_weight,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            is_residual=is_residual,
            use_conditional_kernel=args.conditional_kernel,
            conditioning_mode=conditioning_mode,
            condition_metric=args.condition_metric,
            condition_kernel_scale=args.condition_kernel_scale,
            target_kernel_scale=args.target_kernel_scale,
            condition_bandwidth=args.condition_bandwidth,
            target_bandwidth=args.target_bandwidth,
            local_condition_k=args.local_condition_k,
            condition_radius=args.condition_radius,
            mixture_alpha=args.mixture_alpha,
            target_kernel_mode=args.target_kernel_mode,
            residual_target_scale=args.residual_target_scale,
            adaptive_condition_bandwidth=args.adaptive_condition_bandwidth,
            adaptive_target_bandwidth=args.adaptive_target_bandwidth,
            adaptive_bandwidth_k=args.adaptive_bandwidth_k,
            condition_embedding_dim=args.condition_embedding_dim,
            condition_embedding_hidden_dim=args.condition_embedding_hidden_dim,
            positive_queue_size=args.positive_queue_size,
            positive_reference_size=args.positive_reference_size,
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_min_epsilon=args.sinkhorn_min_epsilon,
            sinkhorn_iterations=args.sinkhorn_iterations,
            fiber_generated_samples=args.fiber_generated_samples,
            fiber_positive_samples=args.fiber_positive_samples,
            fiber_reference_samples=args.fiber_reference_samples,
        )
        model, artifacts = train_conditional_drifting(channel_fn, cfg, device)
        save_implant_checkpoint(out_path, model, family="drifting", metadata=metadata, is_residual=is_residual)
        summary = {"family": args.family, "history": artifacts.history, "config": artifacts.config}

    elif args.family == "paper_wgan":
        cfg = PaperWGANConfig(
            n=args.n,
            noise_std=noise_std,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            eval_size=2000,
            hidden_dim=args.hidden_dim,
            learning_rate_g=1e-4,
            learning_rate_d=1e-4,
            critic_steps=5,
            clip_value=0.01,
            swd_projections=64,
        )
        model, artifacts = train_paper_wgan(channel_fn, cfg, device)
        save_implant_checkpoint(out_path, model, family="paper_wgan", metadata=metadata)
        summary = {"family": args.family, "history": artifacts["history"], "config": cfg.__dict__}

    elif args.family.startswith("diffusion"):
        is_residual = args.family.endswith("residual")
        if args.diffusion_profile == "muah_symbolic":
            pred_type = args.pred_type or "epsilon"
            beta_schedule = args.beta_schedule or "cosine"
            ema_decay = args.ema_decay if args.ema_decay is not None else 0.995
        else:
            pred_type = args.pred_type or ("epsilon" if is_residual else "v")
            beta_schedule = args.beta_schedule or ("cosine" if is_residual else "cosine-zf")
            ema_decay = args.ema_decay if args.ema_decay is not None else (0.995 if is_residual else 0.9)
        cfg = DiffusionConfig(
            n=args.n,
            noise_std=noise_std,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_steps=args.num_steps,
            eval_size=2000,
            swd_projections=64,
            ema_decay=ema_decay,
            pred_type=pred_type,
            is_residual=is_residual,
            beta_schedule=beta_schedule,
        )
        model, state = train_conditional_diffusion(channel_fn, cfg, device)
        save_implant_checkpoint(out_path, model, family="diffusion", metadata=metadata)
        summary = {"family": args.family, "history": state["history"], "config": cfg.__dict__}

    else:
        raise ValueError(f"Unsupported family: {args.family}")

    summary["train_seconds"] = time.perf_counter() - train_start
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    register_artifact(
        name=artifact_name,
        category="channel_implant",
        checkpoint_path=out_path,
        metadata_path=summary_path,
        root=args.weights_root,
        extra={
            "channel": args.channel,
            "family": args.family,
            "seed": args.seed,
        },
    )
    print(json.dumps({"checkpoint": str(out_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
