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
from conditional_drifting.symbolic_ae import apply_code_power_constraint, labels_to_one_hot
from conditional_drifting.training import (
    DriftingConfig,
    evaluate_conditional_anchor_metrics,
    evaluate_residual_model,
    select_device,
    set_seed,
    train_conditional_drifting,
)
from scripts.run_symbolic_awgn_benchmark import load_symbolic_checkpoint


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


@torch.no_grad()
def load_condition_codebook(checkpoint_path: str, device: torch.device) -> list[list[float]]:
    encoder, _, payload = load_symbolic_checkpoint(checkpoint_path, device)
    cfg = payload["config"]
    message_dim = int(cfg["message_dim"])
    labels = torch.arange(message_dim, device=device)
    messages = labels_to_one_hot(labels, message_dim)
    centers = apply_code_power_constraint(encoder(messages), cfg.get("code_power"))
    return centers.detach().cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run enhanced direct drifting on the paper benchmark channels.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA")
    parser.add_argument("--override-n", type=int, default=0, help="Use this channel dimension instead of the preset dimension.")
    parser.add_argument("--override-ebno-db", type=float, default=None, help="Use this Eb/N0 instead of the preset value.")
    parser.add_argument("--override-rate", type=float, default=0.0, help="Use this code rate instead of the preset value.")
    parser.add_argument("--dataset-size", type=int, default=-1)
    parser.add_argument("--eval-size", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--condition-power", type=float, default=0.0)
    parser.add_argument(
        "--condition-input-scale",
        type=float,
        default=0.0,
        help="Scale applied to channel inputs before they enter the drifting generator. Use <=0 for OptFib power normalization.",
    )
    parser.add_argument("--optfib-input-power-dbm", type=float, default=None)
    parser.add_argument("--condition-codebook-checkpoint", type=str, default="")
    parser.add_argument("--condition-jitter-std", type=float, default=0.0)
    parser.add_argument("--condition-context-mode", type=str, default="input", choices=["input", "input_base", "input_base_delta"])
    parser.add_argument("--condition-feature-mode", type=str, default="raw", choices=["raw", "optfib_phase"])
    parser.add_argument("--optfib-gamma", type=float, default=1.27)
    parser.add_argument("--optfib-length", type=float, default=5000.0)
    parser.add_argument("--physics-base-mode", type=str, default="identity", choices=["identity", "optfib", "optfib_noiseless"])
    parser.add_argument("--physics-base-optfib-kstep", type=int, default=20)
    parser.add_argument("--physics-base-optfib-pn-dbm", type=float, default=-21.3)
    parser.add_argument("--drifting-epochs", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--swd-projections", type=int, default=-1)
    parser.add_argument("--latent-input-scale", type=float, default=1.0)
    parser.add_argument(
        "--drift-field",
        type=str,
        default="kernel",
        choices=[
            "kernel",
            "sinkhorn",
            "fiber_sinkhorn",
            "fiber_mmd",
            "fiber_energy",
            "fiber_moment",
            "fiber_energy_moment",
            "fiber_mmd_moment",
        ],
    )
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--max-drift-norm", type=float, default=2.0)
    parser.add_argument("--repulsive-weight", type=float, default=1.0)
    parser.add_argument("--output-init-scale", type=float, default=1.0)
    parser.add_argument("--residual-model", action="store_true")
    parser.add_argument("--conditioning-mode", type=str, default="joint")
    parser.add_argument("--condition-kernel-scale", type=float, default=0.5)
    parser.add_argument("--target-kernel-scale", type=float, default=1.0)
    parser.add_argument("--target-kernel-mode", type=str, default="raw")
    parser.add_argument("--residual-target-scale", type=float, default=1.0)
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
    parser.add_argument("--fiber-moment-mean-weight", type=float, default=1.0)
    parser.add_argument("--fiber-moment-cov-weight", type=float, default=1.0)
    parser.add_argument("--fiber-supervised-weight", type=float, default=0.0)
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
            "condition_power": args.condition_power,
            "condition_input_scale": args.condition_input_scale,
            "optfib_input_power_dbm": args.optfib_input_power_dbm,
            "condition_codebook_checkpoint": args.condition_codebook_checkpoint,
            "condition_jitter_std": args.condition_jitter_std,
            "condition_context_mode": args.condition_context_mode,
            "condition_feature_mode": args.condition_feature_mode,
            "optfib_gamma": args.optfib_gamma,
            "optfib_length": args.optfib_length,
            "physics_base_mode": args.physics_base_mode,
            "physics_base_optfib_kstep": args.physics_base_optfib_kstep,
            "physics_base_optfib_pn_dbm": args.physics_base_optfib_pn_dbm,
            "latent_input_scale": args.latent_input_scale,
            "learning_rate": args.learning_rate,
            "drift_scale": args.drift_scale,
            "max_drift_norm": args.max_drift_norm,
            "repulsive_weight": args.repulsive_weight,
            "output_init_scale": args.output_init_scale,
            "residual_model": args.residual_model,
            "conditioning_mode": args.conditioning_mode,
            "condition_metric": args.condition_metric,
            "condition_kernel_scale": args.condition_kernel_scale,
            "target_kernel_scale": args.target_kernel_scale,
            "target_kernel_mode": args.target_kernel_mode,
            "residual_target_scale": args.residual_target_scale,
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
            "fiber_moment_mean_weight": args.fiber_moment_mean_weight,
            "fiber_moment_cov_weight": args.fiber_moment_cov_weight,
            "fiber_supervised_weight": args.fiber_supervised_weight,
            "anchor_metrics": args.anchor_metrics,
            "anchor_count": args.anchor_count,
            "anchor_samples": args.anchor_samples,
            "anchor_swd_projections": args.anchor_swd_projections,
        },
    }
    _write_summary(out_path, summary)
    condition_codebook = None
    if args.condition_codebook_checkpoint:
        condition_codebook = load_condition_codebook(args.condition_codebook_checkpoint, device)
        summary["config"]["condition_codebook_size"] = len(condition_codebook)
        _write_summary(out_path, summary)

    for channel_name in requested:
        preset = ENHANCED_DIRECT_PRESETS[channel_name]
        channel_dim = int(args.override_n) if int(args.override_n) > 0 else int(preset.n)
        ebn0_db = float(args.override_ebno_db) if args.override_ebno_db is not None else float(preset.ebn0_db)
        rate = float(args.override_rate) if float(args.override_rate) > 0.0 else float(preset.rate)
        dataset_size = args.dataset_size if args.dataset_size > 0 else preset.dataset_size
        eval_size = args.eval_size if args.eval_size > 0 else preset.eval_size
        batch_size = args.batch_size if args.batch_size > 0 else preset.batch_size
        drifting_epochs = args.drifting_epochs if args.drifting_epochs > 0 else preset.drifting_epochs
        swd_projections = args.swd_projections if args.swd_projections > 0 else preset.swd_projections
        noise_std = ebno_to_noise(ebn0_db, rate)
        condition_power = float(args.condition_power) if float(args.condition_power) > 0.0 else None
        if condition_power is None and channel_name == "OptFib" and args.optfib_input_power_dbm is not None:
            condition_power = 10.0 ** ((float(args.optfib_input_power_dbm) - 30.0) / 10.0)
        condition_input_scale = float(args.condition_input_scale)
        if condition_input_scale <= 0.0:
            if channel_name == "OptFib" and condition_power is not None and condition_power > 0.0:
                condition_input_scale = math.sqrt(float(preset.n) / float(condition_power))
            else:
                condition_input_scale = 1.0
        cfg = DriftingConfig(
            n=channel_dim,
            noise_std=noise_std,
            condition_power=condition_power,
            condition_input_scale=condition_input_scale,
            condition_centers=condition_codebook if channel_name == "OptFib" else None,
            condition_jitter_std=args.condition_jitter_std if channel_name == "OptFib" else 0.0,
            condition_context_mode=args.condition_context_mode if channel_name == "OptFib" else "input",
            condition_feature_mode=args.condition_feature_mode if channel_name == "OptFib" else "raw",
            optfib_gamma=args.optfib_gamma,
            optfib_length=args.optfib_length,
            physics_base_mode=args.physics_base_mode if channel_name == "OptFib" else "identity",
            physics_base_optfib_kstep=args.physics_base_optfib_kstep,
            physics_base_optfib_pn_dbm=args.physics_base_optfib_pn_dbm,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=drifting_epochs,
            learning_rate=args.learning_rate,
            latent_input_scale=args.latent_input_scale,
            eval_size=eval_size,
            swd_projections=swd_projections,
            is_residual=bool(args.residual_model),
            output_init_scale=args.output_init_scale,
            drift_field=args.drift_field,
            drift_scale=args.drift_scale,
            max_drift_norm=args.max_drift_norm,
            repulsive_weight=args.repulsive_weight,
            conditioning_mode=args.conditioning_mode,
            condition_metric=args.condition_metric,
            condition_kernel_scale=args.condition_kernel_scale,
            target_kernel_scale=args.target_kernel_scale,
            target_kernel_mode=args.target_kernel_mode,
            residual_target_scale=args.residual_target_scale,
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
            fiber_moment_mean_weight=args.fiber_moment_mean_weight,
            fiber_moment_cov_weight=args.fiber_moment_cov_weight,
            fiber_supervised_weight=args.fiber_supervised_weight,
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
