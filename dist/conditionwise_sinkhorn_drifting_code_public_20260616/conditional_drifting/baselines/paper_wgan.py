from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..metrics import sliced_wasserstein_distance


@dataclass
class PaperWGANConfig:
    n: int
    noise_std: float
    dataset_size: int
    batch_size: int
    epochs: int
    eval_size: int
    hidden_dim: int = 128
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    critic_steps: int = 5
    clip_value: float = 0.01
    swd_projections: int = 128


class PaperWGANGenerator(nn.Module):
    """Small conditional WGAN generator following the early paper code path."""

    def __init__(self, n: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(condition)
        return self.net(torch.cat((condition, noise), dim=1))


class PaperWGANDiscriminator(nn.Module):
    def __init__(self, n: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sample_y: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((sample_y, condition), dim=1))


def _normalize_condition(x: torch.Tensor) -> torch.Tensor:
    return x / (x.std() + 1e-12)


def train_paper_wgan(channel_fn, cfg: PaperWGANConfig, device: torch.device) -> tuple[PaperWGANGenerator, dict]:
    generator = PaperWGANGenerator(cfg.n, cfg.hidden_dim).to(device)
    discriminator = PaperWGANDiscriminator(cfg.n, cfg.hidden_dim).to(device)
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=cfg.learning_rate_d)
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=cfg.learning_rate_g)
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history = []

    for epoch in range(cfg.epochs):
        d_losses = []
        g_losses = []
        for _ in range(steps_per_epoch):
            for _ in range(cfg.critic_steps):
                x = _normalize_condition(torch.randn(cfg.batch_size, cfg.n, device=device))
                fake_y = generator(x).detach()
                real_y = channel_fn(x, cfg.noise_std, device).detach()
                d_real = discriminator(real_y, x).mean()
                d_fake = discriminator(fake_y, x).mean()
                d_loss = -(d_real - d_fake)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                for parameter in discriminator.parameters():
                    parameter.data.clamp_(-cfg.clip_value, cfg.clip_value)
                d_losses.append(float(d_loss.item()))

            x = _normalize_condition(torch.randn(cfg.batch_size, cfg.n, device=device))
            fake_y = generator(x)
            g_loss = -discriminator(fake_y, x).mean()
            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()
            g_losses.append(float(g_loss.item()))

        history.append(
            {
                "epoch": epoch + 1,
                "g_loss": float(np.mean(g_losses)),
                "d_loss": float(np.mean(d_losses)),
            }
        )

    generator.eval()
    return generator, {"history": history}


@torch.no_grad()
def evaluate_paper_wgan(
    generator: PaperWGANGenerator,
    channel_fn,
    cfg: PaperWGANConfig,
    device: torch.device,
    metric_seed: int = 12345,
) -> dict:
    x = _normalize_condition(torch.randn(cfg.eval_size, cfg.n, device=device))
    y_true = channel_fn(x, cfg.noise_std, device)
    y_pred = generator(x)
    residual_true_torch = y_true - x
    residual_pred_torch = y_pred - x
    swd = sliced_wasserstein_distance(residual_true_torch, residual_pred_torch, num_projections=cfg.swd_projections, seed=metric_seed)
    residual_true = residual_true_torch.cpu().numpy()
    residual_pred = residual_pred_torch.cpu().numpy()
    return {
        "swd": float(swd),
        "residual_true": residual_true,
        "residual_pred": residual_pred,
    }
