from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .losses import drifting_loss
from .metrics import sliced_wasserstein_distance
from .model import ConditionalDriftingGenerator
from .progress import tqdm


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
    lr_decay_epoch: int = 0
    lr_decay_factor: float = 0.1
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
    use_conditional_kernel: bool = False
    conditioning_mode: str = "none"
    condition_metric: str = "euclidean"
    condition_kernel_scale: float = 1.0
    target_kernel_scale: float = 1.0
    condition_bandwidth: float | None = None
    target_bandwidth: float | None = None
    local_condition_k: int = 32
    condition_radius: float | None = None
    mixture_alpha: float = 0.5
    target_kernel_mode: str = "raw"
    residual_target_scale: float = 1.0
    adaptive_condition_bandwidth: bool = False
    adaptive_target_bandwidth: bool = False
    adaptive_bandwidth_k: int = 16
    condition_embedding_dim: int = 0
    condition_embedding_hidden_dim: int = 64
    positive_queue_size: int = 0
    positive_reference_size: int = 0


@dataclass
class TrainingArtifacts:
    history: list[dict[str, float]]
    config: dict


class PositiveSampleQueue:
    def __init__(self, *, max_items: int, feature_dim: int, device: torch.device) -> None:
        self.max_items = max(0, int(max_items))
        self.feature_dim = int(feature_dim)
        self.device = device
        self._x = torch.empty((0, self.feature_dim), device=device)
        self._target = torch.empty((0, self.feature_dim), device=device)

    @property
    def size(self) -> int:
        return int(self._x.shape[0])

    def add(self, x: torch.Tensor, target: torch.Tensor) -> None:
        if self.max_items <= 0:
            return
        x = x.detach()
        target = target.detach()
        self._x = torch.cat((self._x, x), dim=0)
        self._target = torch.cat((self._target, target), dim=0)
        if self._x.shape[0] > self.max_items:
            self._x = self._x[-self.max_items :]
            self._target = self._target[-self.max_items :]

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty positive queue.")
        count = max(1, min(int(count), self.size))
        indices = torch.randperm(self.size, device=self.device)[:count]
        return self._x[indices], self._target[indices]


class ConditionKernelEmbedder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    conditioning_mode = str(cfg.conditioning_mode or "none").lower()
    if conditioning_mode == "none" and cfg.use_conditional_kernel:
        conditioning_mode = "joint"
    use_conditioning = conditioning_mode != "none"
    condition_embedder: ConditionKernelEmbedder | None = None
    if use_conditioning and int(cfg.condition_embedding_dim) > 0:
        condition_embedder = ConditionKernelEmbedder(
            input_dim=cfg.n,
            output_dim=int(cfg.condition_embedding_dim),
            hidden_dim=int(cfg.condition_embedding_hidden_dim),
        ).to(device)
    parameters = list(model.parameters())
    if condition_embedder is not None:
        parameters.extend(condition_embedder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
    scheduler = None
    if int(cfg.lr_decay_epoch) > 0 and 0 < int(cfg.lr_decay_epoch) < int(cfg.epochs):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(cfg.lr_decay_epoch)],
            gamma=float(cfg.lr_decay_factor),
        )
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history: list[dict[str, float]] = []
    positive_queue = PositiveSampleQueue(max_items=cfg.positive_queue_size, feature_dim=cfg.n, device=device)
    reference_size = int(cfg.positive_reference_size) if int(cfg.positive_reference_size) > 0 else cfg.batch_size

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
            positive_queue.add(x, target_true)
            if positive_queue.size > 0 and cfg.positive_queue_size > 0:
                positive_x, positive_target = positive_queue.sample(reference_size)
            else:
                positive_x, positive_target = x, target_true
            kernel_condition = condition_embedder(x) if condition_embedder is not None else x
            kernel_condition_positive = (
                condition_embedder(positive_x) if condition_embedder is not None else positive_x
            )
            loss, drift = drifting_loss(
                target_pred,
                positive_target,
                condition_generated=kernel_condition if use_conditioning else None,
                condition_positive=kernel_condition_positive if use_conditioning else None,
                target_condition_generated=x if use_conditioning else None,
                target_condition_positive=positive_x if use_conditioning else None,
                conditioning_mode=conditioning_mode,
                condition_metric=cfg.condition_metric,
                condition_scale=cfg.condition_kernel_scale,
                target_scale=cfg.target_kernel_scale,
                drift_scale=cfg.drift_scale,
                bandwidth=cfg.bandwidth,
                condition_bandwidth=cfg.condition_bandwidth,
                target_bandwidth=cfg.target_bandwidth,
                local_condition_k=cfg.local_condition_k,
                condition_radius=cfg.condition_radius,
                mixture_alpha=cfg.mixture_alpha,
                target_representation=cfg.target_kernel_mode,
                target_is_residual=cfg.is_residual,
                residual_target_scale=cfg.residual_target_scale,
                adaptive_condition_bandwidth=cfg.adaptive_condition_bandwidth,
                adaptive_target_bandwidth=cfg.adaptive_target_bandwidth,
                adaptive_bandwidth_k=cfg.adaptive_bandwidth_k,
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
                "positive_queue_size": float(positive_queue.size),
            }
        )
        if _disable_tqdm():
            print(
                f"epoch {epoch + 1}/{cfg.epochs}: "
                f"loss={history[-1]['loss']:.6e}, drift={history[-1]['drift_norm']:.6e}",
                flush=True,
            )
        if scheduler is not None:
            scheduler.step()

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
    eval_batch_size = min(cfg.batch_size, cfg.eval_size)
    x_batches: list[torch.Tensor] = []
    y_true_batches: list[torch.Tensor] = []
    y_pred_batches: list[torch.Tensor] = []
    residual_true_batches: list[torch.Tensor] = []
    residual_pred_batches: list[torch.Tensor] = []
    target_true_batches: list[torch.Tensor] = []
    target_pred_batches: list[torch.Tensor] = []

    remaining = cfg.eval_size
    while remaining > 0:
        current_bs = min(eval_batch_size, remaining)
        x = torch.randn(current_bs, cfg.n, device=device)
        y_true = channel_fn(x, cfg.noise_std, device)
        y_pred = sample_drifting_target(model, x, is_residual=cfg.is_residual)

        residual_true = y_true - x
        residual_pred = y_pred - x
        target_true = residual_true if cfg.is_residual else y_true
        target_pred = residual_pred if cfg.is_residual else y_pred

        x_batches.append(x.cpu())
        y_true_batches.append(y_true.cpu())
        y_pred_batches.append(y_pred.cpu())
        residual_true_batches.append(residual_true.cpu())
        residual_pred_batches.append(residual_pred.cpu())
        target_true_batches.append(target_true.cpu())
        target_pred_batches.append(target_pred.cpu())
        remaining -= current_bs

    if device.type == "cuda":
        torch.cuda.empty_cache()

    x_cpu = torch.cat(x_batches, dim=0)
    y_true_cpu = torch.cat(y_true_batches, dim=0)
    y_pred_cpu = torch.cat(y_pred_batches, dim=0)
    residual_true_cpu = torch.cat(residual_true_batches, dim=0)
    residual_pred_cpu = torch.cat(residual_pred_batches, dim=0)
    target_true_cpu = torch.cat(target_true_batches, dim=0)
    target_pred_cpu = torch.cat(target_pred_batches, dim=0)
    swd = sliced_wasserstein_distance(
        target_true_cpu,
        target_pred_cpu,
        num_projections=cfg.swd_projections,
        seed=metric_seed,
    )

    return {
        "swd": float(swd),
        "x": x_cpu.numpy(),
        "y_true": y_true_cpu.numpy(),
        "y_pred": y_pred_cpu.numpy(),
        "residual_true": residual_true_cpu.numpy(),
        "residual_pred": residual_pred_cpu.numpy(),
        "target_true": target_true_cpu.numpy(),
        "target_pred": target_pred_cpu.numpy(),
        "target_mode": "residual" if cfg.is_residual else "direct_y",
    }
