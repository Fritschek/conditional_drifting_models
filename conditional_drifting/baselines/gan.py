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
class GANConfig:
    n: int = 2
    noise_std: float = 0.3
    dataset_size: int = 120_000
    batch_size: int = 512
    epochs: int = 60
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    hidden_dim: int = 128
    latent_dim: int = 16
    l1_weight: float = 1.0
    eval_size: int = 20_000
    swd_projections: int = 256


class ConditionalGANGenerator(nn.Module):
    def __init__(self, n: int, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(n + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n),
        )

    def forward(self, condition: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        device = condition.device
        if latent is None:
            latent = torch.randn(condition.shape[0], self.latent_dim, device=device)
        return self.net(torch.cat((condition, latent), dim=1))


class ConditionalGANDiscriminator(nn.Module):
    def __init__(self, n: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, residual: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((residual, condition), dim=1))


def train_conditional_gan(channel_fn, cfg: GANConfig, device: torch.device) -> tuple[nn.Module, dict]:
    generator = ConditionalGANGenerator(cfg.n, cfg.latent_dim, cfg.hidden_dim).to(device)
    discriminator = ConditionalGANDiscriminator(cfg.n, cfg.hidden_dim).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate_d, betas=(0.5, 0.999))
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history = []
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        g_losses = []
        d_losses = []
        progress = tqdm(range(steps_per_epoch), leave=False, desc=f"gan epoch {epoch + 1}/{cfg.epochs}")
        for _ in progress:
            x = torch.randn(cfg.batch_size, cfg.n, device=device)
            residual_real = channel_fn(x, cfg.noise_std, device) - x
            residual_fake = generator(x)

            real_label = torch.ones(cfg.batch_size, 1, device=device)
            fake_label = torch.zeros(cfg.batch_size, 1, device=device)

            d_real = discriminator(residual_real, x)
            d_fake = discriminator(residual_fake.detach(), x)
            d_loss = criterion(d_real, real_label) + criterion(d_fake, fake_label)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            residual_fake = generator(x)
            d_fake_for_g = discriminator(residual_fake, x)
            g_adv = criterion(d_fake_for_g, real_label)
            g_l1 = F.l1_loss(residual_fake, residual_real)
            g_loss = g_adv + cfg.l1_weight * g_l1
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_losses.append(float(d_loss.item()))
            g_losses.append(float(g_loss.item()))
            progress.set_postfix(g=f"{g_losses[-1]:.3e}", d=f"{d_losses[-1]:.3e}")

        history.append({"epoch": epoch + 1, "g_loss": float(np.mean(g_losses)), "d_loss": float(np.mean(d_losses))})

    return generator, {"history": history}


@torch.no_grad()
def evaluate_gan_model(generator: nn.Module, channel_fn, cfg: GANConfig, device: torch.device, metric_seed: int = 12345) -> dict:
    generator.eval()
    x = torch.randn(cfg.eval_size, cfg.n, device=device)
    residual_true = (channel_fn(x, cfg.noise_std, device) - x).cpu().numpy()
    residual_pred = generator(x).cpu().numpy()
    swd = sliced_wasserstein_distance(residual_true, residual_pred, num_projections=cfg.swd_projections, seed=metric_seed)
    return {"swd": float(swd), "residual_true": residual_true, "residual_pred": residual_pred}
