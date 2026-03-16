from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from tqdm import tqdm

from .losses import drifting_loss
from .metrics import sliced_wasserstein_distance
from .model import ConditionalDriftingGenerator


@dataclass
class DriftingConfig:
    n: int = 2
    noise_std: float = 0.3
    dataset_size: int = 120_000
    batch_size: int = 512
    epochs: int = 60
    learning_rate: float = 1e-3
    latent_dim: int = 16
    hidden_dim: int = 128
    drift_scale: float = 1.0
    bandwidth: float | None = None
    min_bandwidth: float = 0.2
    max_drift_norm: float | None = 2.0
    repulsive_weight: float = 1.0
    eval_size: int = 20_000
    swd_projections: int = 256


@dataclass
class TrainingArtifacts:
    history: list[dict[str, float]]
    config: dict


def select_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def train_conditional_drifting(
    channel_fn,
    cfg: DriftingConfig,
    device: torch.device,
) -> tuple[ConditionalDriftingGenerator, TrainingArtifacts]:
    model = ConditionalDriftingGenerator(
        condition_dim=cfg.n,
        output_dim=cfg.n,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        loss_values = []
        drift_values = []
        progress = tqdm(range(steps_per_epoch), leave=False, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for _ in progress:
            x = torch.randn(cfg.batch_size, cfg.n, device=device)
            residual_true = channel_fn(x, cfg.noise_std, device) - x
            residual_pred = model(x)
            loss, drift = drifting_loss(
                residual_pred,
                residual_true,
                drift_scale=cfg.drift_scale,
                bandwidth=cfg.bandwidth,
                min_bandwidth=cfg.min_bandwidth,
                max_drift_norm=cfg.max_drift_norm,
                repulsive_weight=cfg.repulsive_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_values.append(float(loss.item()))
            drift_values.append(float(drift.norm(dim=1).mean().item()))
            progress.set_postfix(loss=f"{loss_values[-1]:.3e}", drift=f"{drift_values[-1]:.3e}")

        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(loss_values)),
                "drift_norm": float(np.mean(drift_values)),
            }
        )

    return model, TrainingArtifacts(history=history, config=asdict(cfg))


@torch.no_grad()
def sample_residuals(model: ConditionalDriftingGenerator, condition: torch.Tensor) -> torch.Tensor:
    return model(condition)


@torch.no_grad()
def sample_channel_outputs(model: ConditionalDriftingGenerator, condition: torch.Tensor) -> torch.Tensor:
    return condition + sample_residuals(model, condition)


@torch.no_grad()
def evaluate_residual_model(
    model: ConditionalDriftingGenerator,
    channel_fn,
    cfg: DriftingConfig,
    device: torch.device,
    *,
    metric_seed: int = 12345,
) -> dict:
    x = torch.randn(cfg.eval_size, cfg.n, device=device)
    y_true = channel_fn(x, cfg.noise_std, device)
    y_pred = sample_channel_outputs(model, x)

    residual_true = (y_true - x).cpu().numpy()
    residual_pred = (y_pred - x).cpu().numpy()
    swd = sliced_wasserstein_distance(
        residual_true,
        residual_pred,
        num_projections=cfg.swd_projections,
        seed=metric_seed,
    )

    return {
        "swd": float(swd),
        "x": x.cpu().numpy(),
        "y_true": y_true.cpu().numpy(),
        "y_pred": y_pred.cpu().numpy(),
        "residual_true": residual_true,
        "residual_pred": residual_pred,
    }
