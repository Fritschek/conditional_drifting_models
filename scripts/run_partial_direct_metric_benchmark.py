from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplcache"))
sys.path.insert(0, ROOT)

import torch

from conditional_drifting.baselines import (
    DiffusionConfig,
    PaperWGANConfig,
    sample_ddim,
    sample_ddpm,
    train_conditional_diffusion,
    train_paper_wgan,
)
from conditional_drifting.channels import channel_registry
from conditional_drifting.metrics import sliced_wasserstein_distance
from conditional_drifting.training import DriftingConfig, evaluate_residual_model, select_device, set_seed, train_conditional_drifting


@dataclass(frozen=True)
class MainChannelPreset:
    n: int
    ebn0_db: float
    rate: float
    diffusion_hidden_dim: int
    wgan_hidden_dim: int
    diffusion_epochs: int = 30
    drifting_epochs: int = 30
    wgan_epochs: int = 30
    dataset_size: int = 10_000_000
    batch_size: int = 5_000
    eval_size: int = 10_000_000
    swd_projections: int = 128
    num_steps: int = 100
    diffusion_learning_rate_schedule: tuple[tuple[int, float], ...] | None = None


MAIN_PRESETS: dict[str, MainChannelPreset] = {
    "AWGN": MainChannelPreset(
        n=7,
        ebn0_db=5.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=110,
        wgan_hidden_dim=128,
        diffusion_learning_rate_schedule=((10, 1e-3), (20, 1e-4)),
    ),
    "Rayleigh": MainChannelPreset(
        n=7,
        ebn0_db=12.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=128,
        wgan_hidden_dim=256,
        diffusion_learning_rate_schedule=((10, 1e-3), (20, 1e-4)),
    ),
    "SSPA": MainChannelPreset(
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
}


def ebno_to_noise(ebn0_db: float, rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / math.sqrt(2.0 * rate * ebn0)


def resolve_diffusion_lr_schedule(
    preset: MainChannelPreset,
    diffusion_epochs: int,
) -> tuple[tuple[int, float], ...] | None:
    schedule = preset.diffusion_learning_rate_schedule
    if schedule is None:
        return None
    if sum(stage_epochs for stage_epochs, _ in schedule) == diffusion_epochs:
        return schedule
    return None


def _normalize_condition(x: torch.Tensor) -> torch.Tensor:
    return x / (x.std() + 1e-12)


@torch.no_grad()
def evaluate_wgan_both_spaces(
    generator,
    channel_fn,
    cfg: PaperWGANConfig,
    device: torch.device,
    *,
    metric_seed: int,
) -> dict[str, float]:
    x = _normalize_condition(torch.randn(cfg.eval_size, cfg.n, device=device))
    y_true = channel_fn(x, cfg.noise_std, device)
    y_pred = generator(x)
    y_swd = sliced_wasserstein_distance(y_true, y_pred, num_projections=cfg.swd_projections, seed=metric_seed)
    residual_true = y_true - x
    residual_pred = y_pred - x
    residual_swd = sliced_wasserstein_distance(
        residual_true,
        residual_pred,
        num_projections=cfg.swd_projections,
        seed=metric_seed,
    )
    return {
        "y_swd": float(y_swd),
        "residual_swd": float(residual_swd),
    }


@torch.no_grad()
def evaluate_residual_diffusion_both_spaces(
    model,
    channel_fn,
    cfg: DiffusionConfig,
    device: torch.device,
    *,
    metric_seed: int,
    ddim_steps: int,
) -> dict[str, dict[str, float]]:
    state = getattr(model, "_diffusion_state")
    x = torch.randn(cfg.eval_size, cfg.n, device=device)
    y_true = channel_fn(x, cfg.noise_std, device)

    residual_pred_ddpm = sample_ddpm(model, x, state)
    residual_pred_ddim = sample_ddim(model, x, state, ddim_steps)

    y_pred_ddpm = x + residual_pred_ddpm
    y_pred_ddim = x + residual_pred_ddim
    residual_true = y_true - x

    return {
        "ddpm": {
            "y_swd": float(sliced_wasserstein_distance(y_true, y_pred_ddpm, num_projections=cfg.swd_projections, seed=metric_seed)),
            "residual_swd": float(
                sliced_wasserstein_distance(
                    residual_true,
                    residual_pred_ddpm,
                    num_projections=cfg.swd_projections,
                    seed=metric_seed,
                )
            ),
        },
        "ddim": {
            "y_swd": float(sliced_wasserstein_distance(y_true, y_pred_ddim, num_projections=cfg.swd_projections, seed=metric_seed)),
            "residual_swd": float(
                sliced_wasserstein_distance(
                    residual_true,
                    residual_pred_ddim,
                    num_projections=cfg.swd_projections,
                    seed=metric_seed,
                )
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partial rerun for direct y-space SWD on methods that were previously scored in residual space."
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,OptFib")
    parser.add_argument("--paper-eval-size", type=int, default=1_000_000)
    parser.add_argument("--optfib-eval-size", type=int, default=100_000)
    parser.add_argument("--optfib-dataset-size", type=int, default=120_000)
    parser.add_argument("--optfib-epochs", type=int, default=60)
    parser.add_argument("--optfib-batch-size", type=int, default=512)
    parser.add_argument("--optfib-num-steps", type=int, default=100)
    parser.add_argument(
        "--methods",
        type=str,
        default="drifting_residual,wgan,optfib_diffusion",
        help="Comma-separated subset of: drifting_residual,wgan,optfib_diffusion",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to results/partial_direct_metric_seed<seed>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    channels = channel_registry()
    requested = [name.strip() for name in args.channels.split(",") if name.strip()]
    methods = {name.strip() for name in args.methods.split(",") if name.strip()}
    out_dir = args.out_dir or os.path.join(ROOT, "results", f"partial_direct_metric_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    summary: dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "channels": {},
        "methods": sorted(methods),
        "notes": {
            "purpose": "Rerun only the methods/channels that were previously scored in residual space and report direct y-space SWD.",
            "paper_channel_metric_seed": args.seed,
            "optfib_metric_seed": 12345,
        },
    }

    for channel_name in requested:
        if channel_name in MAIN_PRESETS:
            preset = MAIN_PRESETS[channel_name]
            noise_std = ebno_to_noise(preset.ebn0_db, preset.rate)
            channel_summary: dict[str, object] = {
                "block": "main",
                "preset": asdict(preset),
                "effective_config": {
                    "dataset_size": preset.dataset_size,
                    "eval_size": args.paper_eval_size,
                    "batch_size": preset.batch_size,
                    "drifting_epochs": preset.drifting_epochs,
                    "diffusion_epochs": preset.diffusion_epochs,
                    "wgan_epochs": preset.wgan_epochs,
                },
                "noise_std": noise_std,
            }

            if "drifting_residual" in methods:
                drift_cfg = DriftingConfig(
                    n=preset.n,
                    noise_std=noise_std,
                    dataset_size=preset.dataset_size,
                    batch_size=preset.batch_size,
                    epochs=preset.drifting_epochs,
                    eval_size=args.paper_eval_size,
                    swd_projections=preset.swd_projections,
                    is_residual=True,
                )
                drift_model, drift_artifacts = train_conditional_drifting(channels[channel_name], drift_cfg, device)
                drift_eval = evaluate_residual_model(
                    drift_model,
                    channels[channel_name],
                    drift_cfg,
                    device,
                    metric_seed=args.seed,
                )
                channel_summary["drifting_residual"] = {
                    "config": asdict(drift_cfg),
                    "y_swd": float(
                        sliced_wasserstein_distance(
                            torch.as_tensor(drift_eval["y_true"], device=device),
                            torch.as_tensor(drift_eval["y_pred"], device=device),
                            num_projections=drift_cfg.swd_projections,
                            seed=args.seed,
                        )
                    ),
                    "residual_swd": float(drift_eval["swd"]),
                    "final_history": drift_artifacts.history[-1],
                }

            if "wgan" in methods:
                wgan_cfg = PaperWGANConfig(
                    n=preset.n,
                    noise_std=noise_std,
                    dataset_size=preset.dataset_size,
                    batch_size=preset.batch_size,
                    epochs=preset.wgan_epochs,
                    eval_size=args.paper_eval_size,
                    hidden_dim=preset.wgan_hidden_dim,
                    swd_projections=preset.swd_projections,
                )
                wgan_model, wgan_artifacts = train_paper_wgan(channels[channel_name], wgan_cfg, device)
                wgan_eval = evaluate_wgan_both_spaces(
                    wgan_model,
                    channels[channel_name],
                    wgan_cfg,
                    device,
                    metric_seed=args.seed,
                )
                channel_summary["wgan"] = {
                    "config": asdict(wgan_cfg),
                    **wgan_eval,
                    "final_history": wgan_artifacts["history"][-1],
                }

            summary["channels"][channel_name] = channel_summary
            continue

        if channel_name != "OptFib":
            raise ValueError(f"Unsupported channel: {channel_name}")

        metric_seed = 12345
        channel_summary = {
            "block": "optfib",
            "effective_config": {
                "dataset_size": args.optfib_dataset_size,
                "eval_size": args.optfib_eval_size,
                "batch_size": args.optfib_batch_size,
                "epochs": args.optfib_epochs,
                "num_steps": args.optfib_num_steps,
                "ddim_steps": args.optfib_num_steps,
            },
            "channel_params": {
                "L": 5000.0,
                "gamma": 1.27,
                "Kstep": 50,
                "Pn_dBm": -21.3,
                "use_noise_std": False,
            },
        }

        if "drifting_residual" in methods:
            drift_cfg = DriftingConfig(
                dataset_size=args.optfib_dataset_size,
                epochs=args.optfib_epochs,
                batch_size=args.optfib_batch_size,
                eval_size=args.optfib_eval_size,
                is_residual=True,
            )
            drift_model, drift_artifacts = train_conditional_drifting(channels[channel_name], drift_cfg, device)
            drift_eval = evaluate_residual_model(
                drift_model,
                channels[channel_name],
                drift_cfg,
                device,
                metric_seed=metric_seed,
            )
            channel_summary["drifting_residual"] = {
                "config": asdict(drift_cfg),
                "y_swd": float(
                    sliced_wasserstein_distance(
                        torch.as_tensor(drift_eval["y_true"], device=device),
                        torch.as_tensor(drift_eval["y_pred"], device=device),
                        num_projections=drift_cfg.swd_projections,
                        seed=metric_seed,
                    )
                ),
                "residual_swd": float(drift_eval["swd"]),
                "final_history": drift_artifacts.history[-1],
            }

        if "wgan" in methods:
            wgan_cfg = PaperWGANConfig(
                n=2,
                noise_std=0.3,
                dataset_size=args.optfib_dataset_size,
                batch_size=args.optfib_batch_size,
                epochs=args.optfib_epochs,
                eval_size=args.optfib_eval_size,
                swd_projections=256,
            )
            wgan_model, wgan_artifacts = train_paper_wgan(channels[channel_name], wgan_cfg, device)
            wgan_eval = evaluate_wgan_both_spaces(
                wgan_model,
                channels[channel_name],
                wgan_cfg,
                device,
                metric_seed=metric_seed,
            )
            channel_summary["wgan"] = {
                "config": asdict(wgan_cfg),
                **wgan_eval,
                "final_history": wgan_artifacts["history"][-1],
            }

        if "optfib_diffusion" in methods:
            diff_cfg = DiffusionConfig(
                dataset_size=args.optfib_dataset_size,
                epochs=args.optfib_epochs,
                batch_size=args.optfib_batch_size,
                eval_size=args.optfib_eval_size,
                num_steps=args.optfib_num_steps,
            )
            diff_model, diff_state = train_conditional_diffusion(channels[channel_name], diff_cfg, device)
            diff_eval = evaluate_residual_diffusion_both_spaces(
                diff_model,
                channels[channel_name],
                diff_cfg,
                device,
                metric_seed=metric_seed,
                ddim_steps=args.optfib_num_steps,
            )
            channel_summary["diffusion"] = {
                "config": asdict(diff_cfg),
                "ddpm": diff_eval["ddpm"],
                "ddim_100": diff_eval["ddim"],
                "final_history": diff_state["history"][-1],
            }

        summary["channels"][channel_name] = channel_summary

    out_json = os.path.join(out_dir, f"partial_direct_metric_summary_seed{args.seed}.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"output_json": out_json, "output_dir": out_dir, "channels": requested, "methods": sorted(methods)}, indent=2))


if __name__ == "__main__":
    main()
