import torch
from torch import nn


class ConditionalGenerator(nn.Module):
    """One-step channel generator y = G(x,z)."""

    def __init__(self, condition_dim, output_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(x.shape[0], self.latent_dim, device=x.device)
        return self.net(torch.cat([x.float(), z.float()], dim=1))
