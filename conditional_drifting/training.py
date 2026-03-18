from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
from tqdm import tqdm

from .losses import drifting_loss
from .metrics import sliced_wasserstein_distance
from .model import ConditionalDriftingGenerator


def _disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    return not sys.stderr.isatty()


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
    is_residual: bool = True


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
        progress = tqdm(
            range(steps_per_epoch),
            leave=False,
            desc=f"epoch {epoch + 1}/{cfg.epochs}",
            disable=_disable_tqdm(),
        )
        for _ in progress:
            x = torch.randn(cfg.batch_size, cfg.n, device=device)
            y_true = channel_fn(x, cfg.noise_std, device)
            target_true = y_true - x if cfg.is_residual else y_true
            target_pred = model(x)
            loss, drift = drifting_loss(
                target_pred,
                target_true,
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
        if _disable_tqdm():
            print(
                f"epoch {epoch + 1}/{cfg.epochs}: "
                f"loss={history[-1]['loss']:.6e}, drift={history[-1]['drift_norm']:.6e}",
                flush=True,
            )

    return model, TrainingArtifacts(history=history, config=asdict(cfg))


@torch.no_grad()
def sample_residuals(model: ConditionalDriftingGenerator, condition: torch.Tensor) -> torch.Tensor:
    return model(condition)


@torch.no_grad()
def sample_channel_outputs(model: ConditionalDriftingGenerator, condition: torch.Tensor) -> torch.Tensor:
    return condition + sample_residuals(model, condition)


@torch.no_grad()
def sample_drifting_target(
    model: ConditionalDriftingGenerator,
    condition: torch.Tensor,
    *,
    is_residual: bool,
) -> torch.Tensor:
    generated = model(condition)
    if is_residual:
        return condition + generated
    return generated


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
    y_pred = sample_drifting_target(model, x, is_residual=cfg.is_residual)

    target_true_torch = y_true - x if cfg.is_residual else y_true
    target_pred_torch = y_pred - x if cfg.is_residual else y_pred
    swd = sliced_wasserstein_distance(
        target_true_torch,
        target_pred_torch,
        num_projections=cfg.swd_projections,
        seed=metric_seed,
    )
    target_true = target_true_torch.cpu().numpy()
    target_pred = target_pred_torch.cpu().numpy()

    return {
        "swd": float(swd),
        "x": x.cpu().numpy(),
        "y_true": y_true.cpu().numpy(),
        "y_pred": y_pred.cpu().numpy(),
        "residual_true": (y_true - x).cpu().numpy(),
        "residual_pred": (y_pred - x).cpu().numpy(),
        "target_true": target_true,
        "target_pred": target_pred,
        "target_mode": "residual" if cfg.is_residual else "direct_y",
    }
