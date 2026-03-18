from __future__ import annotations

import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..metrics import sliced_wasserstein_distance


def _disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    return not sys.stderr.isatty()


@dataclass
class DiffusionConfig:
    n: int = 2
    noise_std: float = 0.3
    dataset_size: int = 120_000
    batch_size: int = 512
    epochs: int = 60
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    num_steps: int = 100
    eval_size: int = 20_000
    swd_projections: int = 256
    ema_decay: float = 0.995
    pred_type: str = "epsilon"
    is_residual: bool = True
    beta_schedule: str = "cosine"
    eval_batch_size: int | None = None
    learning_rate_schedule: tuple[tuple[int, float], ...] | None = None


class SinusoidalTimeEmbedding(nn.Module):
    """Compatibility wrapper for old tests; the legacy baseline uses learned embeddings instead."""

    def __init__(self, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max(dim, 1), dim)
        nn.init.uniform_(self.embedding.weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        indices = t.clamp_max(self.embedding.num_embeddings - 1)
        return self.embedding(indices)


class ConditionalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_steps: int):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.embedding = nn.Embedding(num_steps, out_features)
        nn.init.uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.embedding(t).view(-1, self.out_features) * self.linear(x)


class ConditionalLinearWithCondition(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_steps: int):
        super().__init__()
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.embedding = nn.Embedding(num_steps, out_features)
        nn.init.uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((x, condition), dim=1)
        return self.embedding(t).view(-1, self.out_features) * self.linear(joined)


class ConditionalDiffusionMLP(nn.Module):
    """
    Legacy benchmark architecture from DM_for_learning_channels.

    It predicts residual-space epsilon with:
    - learned per-timestep multiplicative embeddings
    - conditional first layer on [noisy_residual, condition]
    - Softplus activations
    """

    def __init__(self, n: int, hidden_dim: int = 128, num_steps: int = 100):
        super().__init__()
        self.n = n
        self.num_steps = num_steps
        self.layer1 = ConditionalLinearWithCondition(2 * n, hidden_dim, num_steps)
        self.layer2 = ConditionalLinear(hidden_dim, hidden_dim, num_steps)
        self.layer3 = ConditionalLinear(hidden_dim, hidden_dim, num_steps)
        self.output = nn.Linear(hidden_dim, n)

    def forward(self, noisy_residual: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.softplus(self.layer1(noisy_residual, t, condition))
        x = torch.nn.functional.softplus(self.layer2(x, t))
        x = torch.nn.functional.softplus(self.layer3(x, t))
        return self.output(x)


class ExponentialMovingAverage:
    def __init__(self, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, model: nn.Module) -> None:
        self.shadow = {name: param.detach().clone() for name, param in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for name, param in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def cosine_beta_schedule(num_steps: int, s: float = 0.008, clamp_max: float | None = 0.999) -> torch.Tensor:
    grid = torch.linspace(0, 1, num_steps + 1)
    alphas_bar = torch.cos((grid + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
    if clamp_max is not None:
        betas = betas.clamp_max(clamp_max)
    return betas


def _extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, t.to(values.device))
    return out.reshape((t.shape[0],) + (1,) * (len(shape) - 1))


def build_ddim_trajectory(total_steps: int, ddim_steps: int | None) -> list[int]:
    if ddim_steps is None or ddim_steps >= total_steps:
        return list(range(total_steps))
    if ddim_steps <= 0:
        raise ValueError("ddim_steps must be positive")
    if total_steps % ddim_steps == 0:
        skip = total_steps // ddim_steps
        return list(range(skip - 1, total_steps, skip))
    trajectory = np.unique(np.round(np.linspace(0, total_steps - 1, ddim_steps)).astype(int)).tolist()
    trajectory[0] = 0
    trajectory[-1] = total_steps - 1
    return trajectory


def _noise_estimation_loss(
    model: nn.Module,
    residual: torch.Tensor,
    condition: torch.Tensor,
    alphas_bar_sqrt: torch.Tensor,
    one_minus_alphas_bar_sqrt: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    batch_size = residual.shape[0]
    t = torch.randint(0, num_steps, size=(batch_size,), device=residual.device)
    a = _extract(alphas_bar_sqrt, t, residual.shape)
    am1 = _extract(one_minus_alphas_bar_sqrt, t, residual.shape)
    noise = torch.randn_like(residual)
    noisy = residual * a + noise * am1
    pred = model(noisy, t, condition)
    return (noise - pred).square().mean()


def _velocity_estimation_loss(
    model: nn.Module,
    target: torch.Tensor,
    condition: torch.Tensor,
    alphas_bar_sqrt: torch.Tensor,
    one_minus_alphas_bar_sqrt: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    batch_size = target.shape[0]
    t = torch.randint(0, num_steps, size=(batch_size,), device=target.device)
    a = _extract(alphas_bar_sqrt, t, target.shape)
    am1 = _extract(one_minus_alphas_bar_sqrt, t, target.shape)
    noise = torch.randn_like(target)
    noisy = target * a + noise * am1
    velocity = noise * a - target * am1
    pred = model(noisy, t, condition)
    return (velocity - pred).square().mean()


def resolve_learning_rate_schedule(cfg: DiffusionConfig) -> list[tuple[int, float]]:
    if cfg.learning_rate_schedule is None:
        return [(cfg.epochs, cfg.learning_rate)]
    schedule = [(int(epochs), float(lr)) for epochs, lr in cfg.learning_rate_schedule if int(epochs) > 0]
    if not schedule:
        raise ValueError("learning_rate_schedule must contain at least one positive-length stage")
    scheduled_epochs = sum(epochs for epochs, _ in schedule)
    if scheduled_epochs != cfg.epochs:
        raise ValueError(
            f"learning_rate_schedule covers {scheduled_epochs} epochs, but cfg.epochs is {cfg.epochs}"
        )
    return schedule


def train_conditional_diffusion(channel_fn, cfg: DiffusionConfig, device: torch.device) -> tuple[nn.Module, dict]:
    model = ConditionalDiffusionMLP(cfg.n, cfg.hidden_dim, cfg.num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    ema = ExponentialMovingAverage(cfg.ema_decay)
    ema.register(model)

    if cfg.beta_schedule == "cosine":
        betas = cosine_beta_schedule(cfg.num_steps, clamp_max=0.999).to(device)
    elif cfg.beta_schedule == "cosine-zf":
        betas = cosine_beta_schedule(cfg.num_steps, clamp_max=None).to(device)
    else:
        raise ValueError(f"Unsupported beta schedule: {cfg.beta_schedule}")
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - alphas_prod)
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history = []
    lr_schedule = resolve_learning_rate_schedule(cfg)
    schedule_index = 0
    schedule_epoch_limit = lr_schedule[0][0]
    current_lr = lr_schedule[0][1]
    optimizer.param_groups[0]["lr"] = current_lr

    for epoch in range(cfg.epochs):
        if epoch >= schedule_epoch_limit:
            schedule_index += 1
            schedule_epoch_limit += lr_schedule[schedule_index][0]
            current_lr = lr_schedule[schedule_index][1]
            optimizer.param_groups[0]["lr"] = current_lr
        losses = []
        progress = tqdm(
            range(steps_per_epoch),
            leave=False,
            desc=f"diff epoch {epoch + 1}/{cfg.epochs}",
            disable=_disable_tqdm(),
        )
        for _ in progress:
            x = torch.randn(cfg.batch_size, cfg.n, device=device)
            target = channel_fn(x, cfg.noise_std, device)
            if cfg.is_residual:
                target = target - x
            if cfg.pred_type == "epsilon":
                loss = _noise_estimation_loss(model, target, x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, cfg.num_steps)
            elif cfg.pred_type == "v":
                loss = _velocity_estimation_loss(model, target, x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, cfg.num_steps)
            else:
                raise ValueError(f"Unsupported prediction type: {cfg.pred_type}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{losses[-1]:.3e}")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": current_lr})
        if _disable_tqdm():
            print(
                f"diff epoch {epoch + 1}/{cfg.epochs}: "
                f"loss={history[-1]['loss']:.6e}, lr={current_lr:.3e}",
                flush=True,
            )

    model_ema = deepcopy(model).to(device)
    ema.copy_to(model_ema)
    model_ema.eval()

    state = {
        "betas": betas.detach().cpu(),
        "alphas": alphas.detach().cpu(),
        "alphas_prod": alphas_prod.detach().cpu(),
        "alphas_bar_sqrt": alphas_bar_sqrt.detach().cpu(),
        "one_minus_alphas_bar_sqrt": one_minus_alphas_bar_sqrt.detach().cpu(),
        "history": history,
        "ema_decay": cfg.ema_decay,
        "pred_type": cfg.pred_type,
        "is_residual": cfg.is_residual,
        "learning_rate_schedule": [(epochs, lr) for epochs, lr in lr_schedule],
    }
    model_ema._diffusion_state = state
    return model_ema, state


@torch.no_grad()
def sample_ddpm(model: nn.Module, condition: torch.Tensor, state: dict) -> torch.Tensor:
    device = condition.device
    betas = state["betas"].to(device)
    alphas = state["alphas"].to(device)
    alphas_bar_sqrt = state["alphas_bar_sqrt"].to(device)
    one_minus_alphas_bar_sqrt = state["one_minus_alphas_bar_sqrt"].to(device)
    pred_type = state.get("pred_type", "epsilon")

    residual = torch.randn_like(condition)
    for step in reversed(range(len(betas))):
        t = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
        if pred_type == "epsilon":
            eps_factor = (1.0 - _extract(alphas, t, residual.shape)) / _extract(one_minus_alphas_bar_sqrt, t, residual.shape)
            pred_noise = model(residual, t, condition)
            mean = (residual - eps_factor * pred_noise) / _extract(alphas, t, residual.shape).sqrt()
            noise = torch.randn_like(residual)
            sigma_t = _extract(betas, t, residual.shape).sqrt()
            residual = mean + sigma_t * noise
        elif pred_type == "v":
            a = _extract(alphas_bar_sqrt, t, residual.shape)
            am1 = _extract(one_minus_alphas_bar_sqrt, t, residual.shape)
            if step > 0:
                prev_t = torch.full((condition.shape[0],), step - 1, device=device, dtype=torch.long)
                a_next = _extract(alphas_bar_sqrt, prev_t, residual.shape)
                am1_next = _extract(one_minus_alphas_bar_sqrt, prev_t, residual.shape)
            else:
                a_next = torch.ones_like(residual)
                am1_next = torch.zeros_like(residual)
            pred_v = model(residual, t, condition)
            noise = torch.randn_like(residual)
            e_hat = am1 * residual + a * pred_v
            x_hat = a * residual - am1 * pred_v
            residual = (
                a_next * x_hat
                + (am1_next.square() * a / (am1 * a_next)) * e_hat
                + torch.sqrt(1.0 - a.square() / a_next.square()) * am1_next / am1 * noise
            )
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")
    return residual


@torch.no_grad()
def sample_ddim(model: nn.Module, condition: torch.Tensor, state: dict, num_steps: int | None = None) -> torch.Tensor:
    device = condition.device
    alphas_prod = state["alphas_prod"].to(device)
    alphas_bar_sqrt = state["alphas_bar_sqrt"].to(device)
    one_minus_alphas_bar_sqrt = state["one_minus_alphas_bar_sqrt"].to(device)
    pred_type = state.get("pred_type", "epsilon")
    total_steps = len(alphas_prod)
    trajectory = build_ddim_trajectory(total_steps, num_steps)

    residual = torch.randn_like(condition)
    next_traj = [-1] + trajectory[:-1]
    for step, next_step in zip(reversed(trajectory), reversed(next_traj)):
        t = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
        at = _extract(alphas_prod, t, residual.shape)
        if next_step > -1:
            next_t = torch.full((condition.shape[0],), next_step, device=device, dtype=torch.long)
            at_next = _extract(alphas_prod, next_t, residual.shape)
        else:
            at_next = torch.ones_like(residual)
        if pred_type == "epsilon":
            pred_noise = model(residual, t, condition)
            residual = at_next.sqrt() * (residual - pred_noise * (1.0 - at).sqrt()) / at.sqrt() + (1.0 - at_next).sqrt() * pred_noise
        elif pred_type == "v":
            a = _extract(alphas_bar_sqrt, t, residual.shape)
            am1 = _extract(one_minus_alphas_bar_sqrt, t, residual.shape)
            if next_step > -1:
                next_t = torch.full((condition.shape[0],), next_step, device=device, dtype=torch.long)
                a_next = _extract(alphas_bar_sqrt, next_t, residual.shape)
                am1_next = _extract(one_minus_alphas_bar_sqrt, next_t, residual.shape)
            else:
                a_next = torch.ones_like(residual)
                am1_next = torch.zeros_like(residual)
            pred_v = model(residual, t, condition)
            x_hat = a * residual - am1 * pred_v
            residual = a_next * x_hat + am1_next * (residual - a * x_hat) / am1
        else:
            raise ValueError(f"Unsupported prediction type: {pred_type}")
    return residual


@torch.no_grad()
def evaluate_diffusion_model(
    model: nn.Module,
    channel_fn,
    cfg: DiffusionConfig,
    device: torch.device,
    *,
    use_ddim: bool = False,
    ddim_steps: int | None = None,
    metric_seed: int = 12345,
    progress_callback=None,
) -> dict:
    model.eval()
    timings: dict[str, float] = {}
    state = getattr(model, "_diffusion_state")
    eval_batch_size = cfg.eval_batch_size or cfg.eval_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    total_batches = math.ceil(cfg.eval_size / eval_batch_size)
    target_true_parts: list[torch.Tensor] = []
    target_pred_parts: list[torch.Tensor] = []
    chunk_timings = []
    timings["prepare_sec"] = 0.0
    timings["sample_sec"] = 0.0
    timings["to_numpy_sec"] = 0.0

    for batch_idx in range(total_batches):
        batch_size = min(eval_batch_size, cfg.eval_size - batch_idx * eval_batch_size)
        t0 = time.perf_counter()
        x = torch.randn(batch_size, cfg.n, device=device)
        y_true = channel_fn(x, cfg.noise_std, device)
        target_true = y_true if not cfg.is_residual else (y_true - x)
        prepare_sec = time.perf_counter() - t0
        timings["prepare_sec"] += prepare_sec

        t1 = time.perf_counter()
        if use_ddim:
            target_pred = sample_ddim(model, x, state, ddim_steps)
        else:
            target_pred = sample_ddpm(model, x, state)
        sample_sec = time.perf_counter() - t1
        timings["sample_sec"] += sample_sec

        target_true_parts.append(target_true.detach())
        target_pred_parts.append(target_pred.detach())
        chunk_timing = {
            "batch_index": batch_idx + 1,
            "batch_size": batch_size,
            "prepare_sec": prepare_sec,
            "sample_sec": sample_sec,
            "to_numpy_sec": 0.0,
            "total_sec": prepare_sec + sample_sec,
        }
        chunk_timings.append(chunk_timing)
        if progress_callback is not None:
            progress_callback(batch_idx + 1, total_batches, chunk_timing)

    t3 = time.perf_counter()
    target_true_torch = torch.cat(target_true_parts, dim=0)
    target_pred_torch = torch.cat(target_pred_parts, dim=0)
    swd = sliced_wasserstein_distance(target_true_torch, target_pred_torch, num_projections=cfg.swd_projections, seed=metric_seed)
    timings["swd_sec"] = time.perf_counter() - t3
    t2 = time.perf_counter()
    target_true_np = target_true_torch.cpu().numpy()
    target_pred_np = target_pred_torch.cpu().numpy()
    timings["to_numpy_sec"] += time.perf_counter() - t2
    timings["total_sec"] = sum(timings.values())
    return {
        "swd": float(swd),
        "target_true": target_true_np,
        "target_pred": target_pred_np,
        "timings": timings,
        "chunk_timings": chunk_timings,
    }
