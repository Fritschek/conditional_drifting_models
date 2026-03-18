from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import autograd

from ..metrics import sliced_wasserstein_distance
from ..progress import tqdm


def _disable_tqdm() -> bool:
    value = os.environ.get("TQDM_DISABLE", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    return not sys.stderr.isatty()


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
    mode: str = "gan_fa"
    critic_steps: int = 5
    lambda_gp: float = 10.0
    gen_every: int = 5


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConditionalGANGenerator(nn.Module):
    """
    DM_OptFib-style conditional generator on y | x.

    `forward(condition)` returns generated residuals to preserve the current standalone API.
    `sample_y(condition)` returns generated channel outputs directly, matching the legacy benchmark.
    """

    def __init__(self, n: int, latent_dim: int = 16, hidden_dim: int = 128, num_hidden_layers: int = 2):
        super().__init__()
        self.n = n
        self.latent_dim = latent_dim
        self.in_layer = nn.Linear(n + latent_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, n)
        self.hidden_layers1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layers3 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.bn_layers1 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_hidden_layers)])
        self.bn_layer2 = nn.BatchNorm1d(hidden_dim)
        self.bn_layers3 = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_hidden_layers)])
        self.activation = nn.LeakyReLU(0.2)
        self.apply(_init_weights)

    def sample_y(self, condition: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        if latent is None:
            latent = torch.randn(condition.shape[0], self.latent_dim, device=condition.device)
        inputs = torch.cat((condition, latent), dim=1)
        y = self.activation(self.bn_in(self.in_layer(inputs)))
        for index, layer in enumerate(self.hidden_layers1):
            y = self.activation(self.bn_layers1[index](layer(y)))
        y = self.activation(self.bn_layer2(self.hidden_layer2(y)))
        for index, layer in enumerate(self.hidden_layers3):
            y = self.activation(self.bn_layers3[index](layer(y)))
        return self.out_layer(y) + condition

    def forward(self, condition: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        return self.sample_y(condition, latent) - condition


class ConditionalGANDiscriminator(nn.Module):
    def __init__(self, n: int, hidden_dim: int = 128, num_hidden_layers: int = 3):
        super().__init__()
        input_size = 2 * n
        self.in_layer = nn.Linear(input_size, hidden_dim)
        self.hidden_layers1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        merged = hidden_dim + input_size
        self.hidden_layers2 = nn.ModuleList(
            [
                nn.Linear(merged, merged // 2),
                nn.Linear(merged // 2, merged // 4),
                nn.Linear(merged // 4, merged // 8),
            ]
        )
        self.out_layer = nn.Linear(merged // 8, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.apply(_init_weights)

    def score_y(self, sample_y: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((sample_y, condition), dim=1)
        x = self.activation(self.in_layer(inputs))
        for layer in self.hidden_layers1:
            x = self.activation(layer(x))
        x = torch.cat((x, inputs), dim=1)
        for layer in self.hidden_layers2:
            x = self.activation(layer(x))
        return self.out_layer(x)

    def forward(self, residual: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.score_y(residual + condition, condition)


def _gradient_penalty(discriminator: ConditionalGANDiscriminator, real_y: torch.Tensor, fake_y: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    batch_size = real_y.shape[0]
    alpha = torch.rand(batch_size, 1, device=real_y.device)
    interpolated = (alpha * real_y + (1.0 - alpha) * fake_y).requires_grad_(True)
    scores = discriminator.score_y(interpolated, condition)
    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def train_conditional_gan(channel_fn, cfg: GANConfig, device: torch.device) -> tuple[ConditionalGANGenerator, dict]:
    generator = ConditionalGANGenerator(cfg.n, cfg.latent_dim, cfg.hidden_dim).to(device)
    discriminator = ConditionalGANDiscriminator(cfg.n, cfg.hidden_dim).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.learning_rate_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.learning_rate_d, betas=(0.0, 0.9))
    criterion = nn.BCEWithLogitsLoss()
    steps_per_epoch = math.ceil(cfg.dataset_size / cfg.batch_size)
    history = []

    for epoch in range(cfg.epochs):
        g_losses = []
        d_losses = []
        progress = tqdm(
            range(steps_per_epoch),
            leave=False,
            desc=f"gan epoch {epoch + 1}/{cfg.epochs}",
            disable=_disable_tqdm(),
        )
        for step in progress:
            if cfg.mode == "wgan_gp":
                for _ in range(cfg.critic_steps):
                    x = torch.randn(cfg.batch_size, cfg.n, device=device)
                    with torch.no_grad():
                        real_y = channel_fn(x, cfg.noise_std, device)
                    fake_y = generator.sample_y(x).detach()
                    d_real = discriminator.score_y(real_y, x).mean()
                    d_fake = discriminator.score_y(fake_y, x).mean()
                    gp = _gradient_penalty(discriminator, real_y, fake_y, x)
                    d_loss = d_fake - d_real + cfg.lambda_gp * gp
                    opt_d.zero_grad()
                    d_loss.backward()
                    opt_d.step()
                    d_losses.append(float(d_loss.item()))

                x = torch.randn(cfg.batch_size, cfg.n, device=device)
                fake_y = generator.sample_y(x)
                g_loss = -discriminator.score_y(fake_y, x).mean()
                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                opt_g.step()
                g_losses.append(float(g_loss.item()))
            elif cfg.mode == "gan_fa":
                x = torch.randn(cfg.batch_size, cfg.n, device=device)
                with torch.no_grad():
                    real_y = channel_fn(x, cfg.noise_std, device)

                fake_y_d = generator.sample_y(x).detach()
                d_real = discriminator.score_y(real_y, x)
                d_fake = discriminator.score_y(fake_y_d, x)
                real_labels = torch.rand_like(d_real) * 0.3 + 0.7
                fake_labels = torch.rand_like(d_fake) * 0.3
                d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                d_losses.append(float(d_loss.item()))

                if step % cfg.gen_every == 0:
                    fake_y_g = generator.sample_y(x)
                    d_fake_g = discriminator.score_y(fake_y_g, x)
                    recon_l1 = torch.mean(torch.sum(torch.abs(fake_y_g - real_y), dim=1))
                    g_loss = criterion(d_fake_g, real_labels) + cfg.l1_weight * recon_l1
                    opt_g.zero_grad()
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                    opt_g.step()
                    g_losses.append(float(g_loss.item()))
            else:
                raise ValueError(f"Unknown GAN mode: {cfg.mode}")

            latest_g = g_losses[-1] if g_losses else float("nan")
            progress.set_postfix(g=f"{latest_g:.3e}", d=f"{d_losses[-1]:.3e}")

        history.append(
            {
                "epoch": epoch + 1,
                "g_loss": float(np.mean(g_losses)) if g_losses else float("nan"),
                "d_loss": float(np.mean(d_losses)),
            }
        )
        if _disable_tqdm():
            print(
                f"gan epoch {epoch + 1}/{cfg.epochs}: "
                f"g_loss={history[-1]['g_loss']:.6e}, d_loss={history[-1]['d_loss']:.6e}",
                flush=True,
            )

    generator.eval()
    return generator, {"history": history, "mode": cfg.mode}


@torch.no_grad()
def evaluate_gan_model(generator: ConditionalGANGenerator, channel_fn, cfg: GANConfig, device: torch.device, metric_seed: int = 12345) -> dict:
    generator.eval()
    x = torch.randn(cfg.eval_size, cfg.n, device=device)
    residual_true_torch = channel_fn(x, cfg.noise_std, device) - x
    residual_pred_torch = generator(x)
    swd = sliced_wasserstein_distance(residual_true_torch, residual_pred_torch, num_projections=cfg.swd_projections, seed=metric_seed)
    residual_true = residual_true_torch.cpu().numpy()
    residual_pred = residual_pred_torch.cpu().numpy()
    return {"swd": float(swd), "residual_true": residual_true, "residual_pred": residual_pred}
