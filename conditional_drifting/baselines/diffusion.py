from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..metrics import sliced_wasserstein_distance


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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalDiffusionMLP(nn.Module):
    def __init__(self, n: int, hidden_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * n + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n),
        )

    def forward(self, noisy_residual: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        time_features = self.time_embed(t)
        inputs = torch.cat((noisy_residual, condition, time_features), dim=1)
        return self.net(inputs)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, num_steps, num_steps + 1)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape((t.shape[0],) + (1,) * (len(shape) - 1))


def train_conditional_diffusion(channel_fn, cfg: DiffusionConfig, device: torch.device) -> tuple[nn.Module, dict]:
    model = ConditionalDiffusionMLP(cfg.n, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    betas = cosine_beta_schedule(cfg.num_steps).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars = alpha_bars.sqrt()
    sqrt_one_minus = (1.0 - alpha_bars).sqrt()
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history = []

    for epoch in range(cfg.epochs):
        losses = []
        progress = tqdm(range(steps_per_epoch), leave=False, desc=f"diff epoch {epoch + 1}/{cfg.epochs}")
        for _ in progress:
            x = torch.randn(cfg.batch_size, cfg.n, device=device)
            residual = channel_fn(x, cfg.noise_std, device) - x
            noise = torch.randn_like(residual)
            t = torch.randint(0, cfg.num_steps, (cfg.batch_size,), device=device)
            noisy = _extract(sqrt_alpha_bars, t, residual.shape) * residual + _extract(sqrt_one_minus, t, residual.shape) * noise
            pred_noise = model(noisy, t, x)
            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{losses[-1]:.3e}")
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})

    state = {
        "betas": betas.detach().cpu(),
        "alphas": alphas.detach().cpu(),
        "alpha_bars": alpha_bars.detach().cpu(),
        "history": history,
    }
    model._diffusion_state = state
    return model, state


@torch.no_grad()
def sample_ddpm(model: nn.Module, condition: torch.Tensor, state: dict) -> torch.Tensor:
    device = condition.device
    betas = state["betas"].to(device)
    alphas = state["alphas"].to(device)
    alpha_bars = state["alpha_bars"].to(device)
    residual = torch.randn_like(condition)
    for step in range(len(betas) - 1, -1, -1):
        t = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
        pred_noise = model(residual, t, condition)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]
        mean = (residual - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise) / alpha.sqrt()
        if step > 0:
            residual = mean + beta.sqrt() * torch.randn_like(residual)
        else:
            residual = mean
    return residual


@torch.no_grad()
def sample_ddim(model: nn.Module, condition: torch.Tensor, state: dict, num_steps: int | None = None) -> torch.Tensor:
    device = condition.device
    alpha_bars = state["alpha_bars"].to(device)
    total_steps = len(alpha_bars)
    if num_steps is None or num_steps >= total_steps:
        trajectory = list(range(total_steps - 1, -1, -1))
    else:
        trajectory = np.linspace(total_steps - 1, 0, num=num_steps, dtype=int).tolist()
    residual = torch.randn_like(condition)
    for idx, step in enumerate(trajectory):
        t = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
        pred_noise = model(residual, t, condition)
        alpha_bar = alpha_bars[step]
        x0 = (residual - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        next_step = trajectory[idx + 1] if idx + 1 < len(trajectory) else -1
        alpha_bar_next = alpha_bars[next_step] if next_step >= 0 else torch.tensor(1.0, device=device)
        residual = alpha_bar_next.sqrt() * x0 + (1.0 - alpha_bar_next).sqrt() * pred_noise
    return residual


@torch.no_grad()
def evaluate_diffusion_model(model: nn.Module, channel_fn, cfg: DiffusionConfig, device: torch.device, *, use_ddim: bool = False, ddim_steps: int | None = None, metric_seed: int = 12345) -> dict:
    model.eval()
    x = torch.randn(cfg.eval_size, cfg.n, device=device)
    y_true = channel_fn(x, cfg.noise_std, device)
    residual_true = (y_true - x).cpu().numpy()
    state = getattr(model, "_diffusion_state")
    if use_ddim:
        residual_pred = sample_ddim(model, x, state, ddim_steps).cpu().numpy()
    else:
        residual_pred = sample_ddpm(model, x, state).cpu().numpy()
    swd = sliced_wasserstein_distance(residual_true, residual_pred, num_projections=cfg.swd_projections, seed=metric_seed)
    return {"swd": float(swd), "residual_true": residual_true, "residual_pred": residual_pred}
