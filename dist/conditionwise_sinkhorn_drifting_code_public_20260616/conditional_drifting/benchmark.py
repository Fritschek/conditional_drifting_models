from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .training import DriftingConfig, evaluate_residual_model, train_conditional_drifting


@dataclass
class BenchmarkConfig(DriftingConfig):
    metric_seed: int = 12345


def run_single_channel_benchmark(
    channel_name: str,
    channel_fn,
    cfg: BenchmarkConfig,
    device: torch.device,
) -> dict:
    model, artifacts = train_conditional_drifting(channel_fn, cfg, device)
    evaluation = evaluate_residual_model(
        model,
        channel_fn,
        cfg,
        device,
        metric_seed=cfg.metric_seed,
    )
    final_epoch = artifacts.history[-1]
    return {
        "channel": channel_name,
        "device": str(device),
        "config": asdict(cfg),
        "train_final_loss": final_epoch["loss"],
        "train_final_drift_norm": final_epoch["drift_norm"],
        "drifting_swd": evaluation["swd"],
        "history": artifacts.history,
    }
