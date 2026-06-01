from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalDriftingGenerator(nn.Module):
    """One-shot conditional generator for residual samples e ~ p(e | x)."""

    def __init__(
        self,
        condition_dim: int,
        output_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        latent_input_scale: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_input_scale = float(latent_input_scale)
        input_dim = condition_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, condition: torch.Tensor, latent: torch.Tensor | None = None) -> torch.Tensor:
        device = next(self.parameters()).device
        condition = condition.to(device=device, dtype=torch.float32)
        if latent is None:
            latent = torch.randn(condition.shape[0], self.latent_dim, device=device)
        else:
            latent = latent.to(device=device, dtype=torch.float32)
        latent = latent * self.latent_input_scale
        return self.net(torch.cat((condition, latent), dim=1))
