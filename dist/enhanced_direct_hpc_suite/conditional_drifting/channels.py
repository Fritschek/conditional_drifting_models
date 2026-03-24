from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

ChannelFn = Callable[[torch.Tensor, float, torch.device], torch.Tensor]


def awgn(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    noise = torch.normal(mean=0.0, std=float(noise_std), size=x.shape, device=device)
    return x + noise


def rayleigh_awgn(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    i, j = x.shape
    fading = (1.0 / math.sqrt(2.0)) * torch.sqrt(
        torch.randn((i, j), device=device).square() + torch.randn((i, j), device=device).square()
    )
    noise = float(noise_std) * torch.randn((i, j), device=device)
    return fading * x + noise


def modeflip_awgn(x: torch.Tensor, noise_std: float, device: torch.device, flip_prob: float = 0.5) -> torch.Tensor:
    """
    Bimodal synthetic channel for mode-collapse stress tests.

    For each sample, draw a Rademacher branch variable s in {+1, -1} and return
    y = s * x + n. For a fixed input x, this induces two separated modes.
    """
    x = x.to(device)
    if not (0.0 <= float(flip_prob) <= 1.0):
        raise ValueError("flip_prob must lie in [0, 1].")
    batch = x.shape[0]
    flips = torch.where(
        torch.rand((batch, 1), device=device) < float(flip_prob),
        torch.full((batch, 1), -1.0, device=device, dtype=x.dtype),
        torch.full((batch, 1), 1.0, device=device, dtype=x.dtype),
    )
    noise = float(noise_std) * torch.randn_like(x, device=device)
    return flips * x + noise


def sspa(x: torch.Tensor, noise_std: float, device: torch.device, p: float = 3.0, a0: float = 1.5, gain: float = 5.0) -> torch.Tensor:
    x = x.to(device)
    if x.shape[1] % 2 != 0:
        raise ValueError("SSPA expects an even channel dimension (I/Q pairs).")
    dim = x.shape[1] // 2
    x_2d = x.reshape(-1, 2)
    amplitude = torch.sum(x_2d.square(), dim=1).sqrt()
    amplitude_ratio = gain / (1.0 + (gain * amplitude / a0) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    x_amp = torch.mul(amplitude_ratio.reshape(-1, 1), x_2d).reshape(-1, 2 * dim)
    return x_amp + (float(noise_std) / math.sqrt(2.0)) * torch.randn_like(x_amp, device=device)


def optfib(
    x: torch.Tensor,
    noise_std: float,
    device: torch.device,
    *,
    L: float = 5000.0,
    gamma: float = 1.27,
    Kstep: int = 50,
    Pn_dBm: float = -21.3,
    use_noise_std: bool = False,
) -> torch.Tensor:
    x = x.to(device)
    if x.shape[1] % 2 != 0:
        raise ValueError("OptFib expects an even channel dimension (I/Q pairs).")

    if use_noise_std:
        sigma_n = float(noise_std) / math.sqrt(float(Kstep))
    else:
        sigma_n = float(np.sqrt((10 ** ((Pn_dBm - 30.0) / 10.0)) / float(Kstep) / 2.0))

    x2 = x.reshape(-1, 2)
    xr = x2[:, 0:1]
    xi = x2[:, 1:2]
    for _ in range(int(Kstep)):
        phase = gamma * (xr.square() + xi.square()) * L / float(Kstep)
        xr_rot = xr * torch.cos(phase) - xi * torch.sin(phase)
        xi_rot = xi * torch.cos(phase) + xr * torch.sin(phase)
        xr = xr_rot + sigma_n * torch.randn_like(xr, device=device)
        xi = xi_rot + sigma_n * torch.randn_like(xi, device=device)
    return torch.cat((xr, xi), dim=1).reshape_as(x)


def channel_registry(optfib_params: dict | None = None, modeflip_params: dict | None = None) -> dict[str, ChannelFn]:
    params = dict(optfib_params or {})
    modeflip_cfg = dict(modeflip_params or {})

    def _optfib(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
        return optfib(x, noise_std, device, **params)

    def _modeflip(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
        return modeflip_awgn(x, noise_std, device, **modeflip_cfg)

    return {
        "AWGN": awgn,
        "Rayleigh": rayleigh_awgn,
        "ModeFlip": _modeflip,
        "SSPA": sspa,
        "OptFib": _optfib,
    }
