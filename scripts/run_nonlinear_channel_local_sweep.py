from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import channel_registry, optfib
from conditional_drifting.metrics import (
    conditional_anchor_gaussian_w2,
    conditional_anchor_mean_l2,
    conditional_anchor_swd,
    sliced_wasserstein_distance,
)
from conditional_drifting.paper2309_presets import PaperChannelPreset, ebno_to_noise
from conditional_drifting.training import (
    DriftingConfig,
    evaluate_conditional_anchor_metrics,
    evaluate_residual_model,
    select_device,
    set_seed,
    train_conditional_drifting,
)


CHANNEL_PRESETS = {
    "SSPA": PaperChannelPreset(
        n=8,
        ebn0_db=8.0,
        rate=6.0 / 8.0,
        diffusion_hidden_dim=110,
        wgan_hidden_dim=256,
        diffusion_epochs=160,
        drifting_epochs=160,
        wgan_epochs=160,
        batch_size=4096,
    ),
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


@dataclass(frozen=True)
class SweepSpec:
    name: str
    is_residual: bool = False
    drift_field: str = "kernel"
    conditioning_mode: str = "none"
    condition_kernel_scale: float = 1.0
    target_kernel_scale: float = 1.0
    condition_metric: str = "euclidean"
    target_kernel_mode: str = "raw"
    adaptive_condition_bandwidth: bool = False
    adaptive_target_bandwidth: bool = False
    adaptive_bandwidth_k: int = 16
    local_condition_k: int = 32
    mixture_alpha: float = 0.5
    repulsive_weight: float = 1.0
    residual_target_scale: float = 1.0
    sinkhorn_iterations: int = 10
    fiber_generated_samples: int = 4
    fiber_positive_samples: int = 4
    fiber_reference_samples: int = 4
    base_mode: str = "identity"


SPECS = [
    SweepSpec(
        name="kernel_target",
        is_residual=False,
        drift_field="kernel",
        conditioning_mode="none",
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="kernel_joint",
        is_residual=False,
        drift_field="kernel",
        conditioning_mode="joint",
        condition_kernel_scale=0.5,
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="kernel_product_rawres",
        is_residual=False,
        drift_field="kernel",
        conditioning_mode="product",
        condition_kernel_scale=0.5,
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
    ),
    SweepSpec(
        name="kernel_softlocal",
        is_residual=False,
        drift_field="kernel",
        conditioning_mode="soft_local",
        condition_kernel_scale=0.5,
        target_kernel_scale=1.0,
        local_condition_k=32,
    ),
    SweepSpec(
        name="kernel_softlocal_whitened_adapt",
        is_residual=False,
        drift_field="kernel",
        conditioning_mode="soft_local",
        condition_kernel_scale=0.5,
        target_kernel_scale=1.0,
        condition_metric="whitened",
        adaptive_condition_bandwidth=True,
        adaptive_target_bandwidth=True,
        adaptive_bandwidth_k=8,
        local_condition_k=32,
    ),
    SweepSpec(
        name="joint_sinkhorn",
        is_residual=False,
        drift_field="sinkhorn",
        conditioning_mode="joint",
        condition_kernel_scale=0.5,
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="joint_sinkhorn_rawres",
        is_residual=False,
        drift_field="sinkhorn",
        conditioning_mode="joint",
        condition_kernel_scale=0.5,
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
    ),
    SweepSpec(
        name="fiber_sinkhorn",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="fiber_sinkhorn_rawres",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres_rep0p5",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        repulsive_weight=0.5,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres_rep0p25",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        repulsive_weight=0.25,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres_rep0",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        repulsive_weight=0.0,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres_rscale0p25",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        residual_target_scale=0.25,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres_rscale0p5",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        residual_target_scale=0.5,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawres_rscale2",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        residual_target_scale=2.0,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_polar",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="polar_residual",
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawpolar",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_polar_residual",
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_sinkhorn_more_rawpolar_rscale0p25",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_polar_residual",
        residual_target_scale=0.25,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_mmd_more_rawpolar",
        is_residual=False,
        drift_field="fiber_mmd",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_polar_residual",
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_mmd_more_rawres",
        is_residual=False,
        drift_field="fiber_mmd",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_residual",
        residual_target_scale=0.25,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="fiber_energy_more_rawpolar",
        is_residual=False,
        drift_field="fiber_energy",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        target_kernel_mode="raw_plus_polar_residual",
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="residual_kernel_target",
        is_residual=True,
        drift_field="kernel",
        conditioning_mode="none",
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="residual_kernel_joint",
        is_residual=True,
        drift_field="kernel",
        conditioning_mode="joint",
        condition_kernel_scale=0.25,
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="residual_kernel_softlocal_whitened_adapt",
        is_residual=True,
        drift_field="kernel",
        conditioning_mode="soft_local",
        condition_kernel_scale=0.25,
        target_kernel_scale=1.0,
        condition_metric="whitened",
        adaptive_condition_bandwidth=True,
        adaptive_target_bandwidth=True,
        adaptive_bandwidth_k=8,
        local_condition_k=32,
    ),
    SweepSpec(
        name="residual_joint_sinkhorn",
        is_residual=True,
        drift_field="sinkhorn",
        conditioning_mode="joint",
        condition_kernel_scale=0.25,
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="residual_fiber_sinkhorn",
        is_residual=True,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
    ),
    SweepSpec(
        name="residual_fiber_sinkhorn_more",
        is_residual=True,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
    ),
    SweepSpec(
        name="optfib_physics_fiber_sinkhorn_more",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
        base_mode="optfib_noiseless",
    ),
    SweepSpec(
        name="optfib_physics_fiber_sinkhorn_more_rep0",
        is_residual=False,
        drift_field="fiber_sinkhorn",
        conditioning_mode="none",
        target_kernel_scale=1.0,
        repulsive_weight=0.0,
        fiber_generated_samples=8,
        fiber_positive_samples=8,
        fiber_reference_samples=8,
        base_mode="optfib_noiseless",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Short local SSPA/OptFib drift-field diagnostic sweep.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="SSPA,OptFib")
    parser.add_argument("--specs", type=str, default="")
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--swd-projections", type=int, default=64)
    parser.add_argument("--anchor-count", type=int, default=32)
    parser.add_argument("--anchor-samples", type=int, default=16)
    parser.add_argument("--anchor-swd-projections", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=Path("results/nonlinear_channel_local_sweep"))
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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


def make_config(args: argparse.Namespace, channel: str, spec: SweepSpec) -> DriftingConfig:
    preset = CHANNEL_PRESETS[channel]
    return DriftingConfig(
        n=preset.n,
        noise_std=ebno_to_noise(preset.ebn0_db, preset.rate),
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_size=args.eval_size,
        swd_projections=args.swd_projections,
        is_residual=spec.is_residual,
        drift_field=spec.drift_field,
        drift_scale=1.0,
        max_drift_norm=2.0,
        repulsive_weight=spec.repulsive_weight,
        conditioning_mode=spec.conditioning_mode,
        condition_metric=spec.condition_metric,
        condition_kernel_scale=spec.condition_kernel_scale,
        target_kernel_scale=spec.target_kernel_scale,
        target_kernel_mode=spec.target_kernel_mode,
        residual_target_scale=spec.residual_target_scale,
        adaptive_condition_bandwidth=spec.adaptive_condition_bandwidth,
        adaptive_target_bandwidth=spec.adaptive_target_bandwidth,
        adaptive_bandwidth_k=spec.adaptive_bandwidth_k,
        local_condition_k=spec.local_condition_k,
        mixture_alpha=spec.mixture_alpha,
        sinkhorn_min_epsilon=1e-3,
        sinkhorn_iterations=spec.sinkhorn_iterations,
        fiber_generated_samples=spec.fiber_generated_samples,
        fiber_positive_samples=spec.fiber_positive_samples,
        fiber_reference_samples=spec.fiber_reference_samples,
    )


def make_training_channel(channel_fn, spec: SweepSpec):
    base_mode = str(spec.base_mode or "identity").lower()
    if base_mode == "identity":
        return channel_fn
    if base_mode != "optfib_noiseless":
        raise ValueError(f"Unsupported base mode: {spec.base_mode}")

    def _physics_residual_channel(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
        y = channel_fn(x, noise_std, device)
        base = optfib(x, 0.0, device, use_noise_std=True)
        return y - base

    return _physics_residual_channel


@torch.no_grad()
def sample_physics_base(condition: torch.Tensor, base_mode: str, device: torch.device) -> torch.Tensor:
    mode = str(base_mode or "identity").lower()
    if mode == "identity":
        return condition
    if mode == "optfib_noiseless":
        return optfib(condition, 0.0, device, use_noise_std=True)
    raise ValueError(f"Unsupported base mode: {base_mode}")


@torch.no_grad()
def evaluate_physics_residual_model(
    model,
    channel_fn,
    cfg: DriftingConfig,
    device: torch.device,
    *,
    base_mode: str,
    metric_seed: int = 12345,
) -> dict[str, float]:
    fork_devices = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(metric_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(metric_seed)

        eval_batch_size = min(cfg.batch_size, cfg.eval_size)
        y_true_batches: list[torch.Tensor] = []
        y_pred_batches: list[torch.Tensor] = []
        residual_true_batches: list[torch.Tensor] = []
        residual_pred_batches: list[torch.Tensor] = []
        remaining = cfg.eval_size
        while remaining > 0:
            current_bs = min(eval_batch_size, remaining)
            x = torch.randn(current_bs, cfg.n, device=device)
            base = sample_physics_base(x, base_mode, device)
            y_true = channel_fn(x, cfg.noise_std, device)
            residual_pred = model(x)
            y_pred = base + residual_pred
            residual_true = y_true - base
            y_true_batches.append(y_true.cpu())
            y_pred_batches.append(y_pred.cpu())
            residual_true_batches.append(residual_true.cpu())
            residual_pred_batches.append(residual_pred.cpu())
            remaining -= current_bs

    y_true_cpu = torch.cat(y_true_batches, dim=0)
    y_pred_cpu = torch.cat(y_pred_batches, dim=0)
    residual_true_cpu = torch.cat(residual_true_batches, dim=0)
    residual_pred_cpu = torch.cat(residual_pred_batches, dim=0)
    direct_swd = sliced_wasserstein_distance(
        y_true_cpu,
        y_pred_cpu,
        num_projections=cfg.swd_projections,
        seed=metric_seed,
    )
    residual_swd = sliced_wasserstein_distance(
        residual_true_cpu,
        residual_pred_cpu,
        num_projections=cfg.swd_projections,
        seed=metric_seed,
    )
    return {
        "swd": float(residual_swd),
        "direct_swd": float(direct_swd),
        "residual_swd": float(residual_swd),
        "target_mode": f"{base_mode}_residual",
    }


@torch.no_grad()
def evaluate_physics_anchor_metrics(
    model,
    channel_fn,
    cfg: DriftingConfig,
    device: torch.device,
    *,
    base_mode: str,
    num_anchors: int,
    samples_per_anchor: int,
    swd_projections: int,
    metric_seed: int,
) -> dict[str, float]:
    if num_anchors <= 0:
        raise ValueError("num_anchors must be positive.")
    if samples_per_anchor <= 1:
        raise ValueError("samples_per_anchor must be greater than one.")

    fork_devices = []
    if device.type == "cuda":
        fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(metric_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(metric_seed)

        x_anchor = torch.randn(num_anchors, cfg.n, device=device)
        y_true_a = []
        y_true_b = []
        y_pred = []
        for anchor in x_anchor:
            x_rep = anchor.unsqueeze(0).repeat(samples_per_anchor, 1)
            base = sample_physics_base(x_rep, base_mode, device)
            y_true_a.append(channel_fn(x_rep, cfg.noise_std, device))
            y_true_b.append(channel_fn(x_rep, cfg.noise_std, device))
            y_pred.append(base + model(x_rep))

        y_true_anchor = torch.stack(y_true_a, dim=0)
        y_floor_anchor = torch.stack(y_true_b, dim=0)
        y_pred_anchor = torch.stack(y_pred, dim=0)

    anchor_y_swd = conditional_anchor_swd(
        y_true_anchor,
        y_pred_anchor,
        num_projections=swd_projections,
        seed=metric_seed,
    )
    anchor_y_floor_swd = conditional_anchor_swd(
        y_true_anchor,
        y_floor_anchor,
        num_projections=swd_projections,
        seed=metric_seed + 10_000,
    )
    return {
        "anchor_y_swd": float(anchor_y_swd),
        "anchor_y_floor_swd": float(anchor_y_floor_swd),
        "anchor_y_excess_swd": float(max(anchor_y_swd - anchor_y_floor_swd, 0.0)),
        "anchor_gaussian_w2": float(conditional_anchor_gaussian_w2(y_true_anchor, y_pred_anchor)),
        "anchor_mean_l2": float(conditional_anchor_mean_l2(y_true_anchor, y_pred_anchor)),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    channels = parse_csv_list(args.channels)
    unknown_channels = sorted(set(channels) - set(CHANNEL_PRESETS))
    if unknown_channels:
        raise ValueError(f"Unsupported channels for this sweep: {unknown_channels}")

    spec_names = set(parse_csv_list(args.specs))
    specs = [spec for spec in SPECS if not spec_names or spec.name in spec_names]
    unknown_specs = sorted(spec_names - {spec.name for spec in SPECS})
    if unknown_specs:
        raise ValueError(f"Unknown specs: {unknown_specs}")

    device = select_device(args.device)
    channels_map = channel_registry()
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "channels": channels,
        "specs": [asdict(spec) for spec in specs],
        "args": vars(args),
        "runs": [],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    for channel in channels:
        for spec in specs:
            run_seed = args.seed
            set_seed(run_seed)
            cfg = make_config(args, channel, spec)
            start = time.perf_counter()
            print(f"[local-sweep] channel={channel} spec={spec.name}", flush=True)
            training_channel = make_training_channel(channels_map[channel], spec)
            model, artifacts = train_conditional_drifting(training_channel, cfg, device)
            train_seconds = time.perf_counter() - start
            if str(spec.base_mode or "identity").lower() == "identity":
                eval_result = evaluate_residual_model(
                    model,
                    channels_map[channel],
                    cfg,
                    device,
                    metric_seed=run_seed,
                )
                anchor = evaluate_conditional_anchor_metrics(
                    model,
                    channels_map[channel],
                    cfg,
                    device,
                    num_anchors=args.anchor_count,
                    samples_per_anchor=args.anchor_samples,
                    swd_projections=args.anchor_swd_projections,
                    metric_seed=run_seed + 1000,
                )
            else:
                eval_result = evaluate_physics_residual_model(
                    model,
                    channels_map[channel],
                    cfg,
                    device,
                    base_mode=spec.base_mode,
                    metric_seed=run_seed,
                )
                anchor = evaluate_physics_anchor_metrics(
                    model,
                    channels_map[channel],
                    cfg,
                    device,
                    base_mode=spec.base_mode,
                    num_anchors=args.anchor_count,
                    samples_per_anchor=args.anchor_samples,
                    swd_projections=args.anchor_swd_projections,
                    metric_seed=run_seed + 1000,
                )
            row = {
                "channel": channel,
                "spec": spec.name,
                "seed": run_seed,
                "train_seconds": train_seconds,
                "final_loss": artifacts.history[-1]["loss"],
                "final_drift_norm": artifacts.history[-1]["drift_norm"],
                "direct_swd": eval_result["direct_swd"],
                "residual_swd": eval_result["residual_swd"],
                "anchor_y_swd": anchor["anchor_y_swd"],
                "anchor_y_floor_swd": anchor["anchor_y_floor_swd"],
                "anchor_y_excess_swd": anchor["anchor_y_excess_swd"],
                "anchor_gaussian_w2": anchor["anchor_gaussian_w2"],
                "anchor_mean_l2": anchor["anchor_mean_l2"],
                "drift_field": spec.drift_field,
                "is_residual": spec.is_residual,
                "conditioning_mode": spec.conditioning_mode,
                "target_kernel_mode": spec.target_kernel_mode,
                "residual_target_scale": spec.residual_target_scale,
                "condition_metric": spec.condition_metric,
                "adaptive_condition_bandwidth": spec.adaptive_condition_bandwidth,
                "adaptive_target_bandwidth": spec.adaptive_target_bandwidth,
                "fiber_generated_samples": spec.fiber_generated_samples,
                "fiber_positive_samples": spec.fiber_positive_samples,
                "fiber_reference_samples": spec.fiber_reference_samples,
                "base_mode": spec.base_mode,
            }
            rows.append(row)
            payload["runs"].append({"row": row, "config": asdict(cfg), "history": artifacts.history})
            (args.out_dir / "sweep_results.json").write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            write_csv(args.out_dir / "sweep_results.csv", rows)
            print(
                json.dumps(
                    {
                        "channel": channel,
                        "spec": spec.name,
                        "direct_swd": row["direct_swd"],
                        "anchor_y_swd": row["anchor_y_swd"],
                        "anchor_gaussian_w2": row["anchor_gaussian_w2"],
                    },
                    indent=2,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "summary": str((args.out_dir / "sweep_results.json").resolve()),
                "csv": str((args.out_dir / "sweep_results.csv").resolve()),
                "num_runs": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
