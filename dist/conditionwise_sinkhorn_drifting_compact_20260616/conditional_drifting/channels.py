import math

import torch


def awgn(x, noise_std):
    return x + noise_std * torch.randn_like(x)


def rayleigh(x, noise_std):
    fading = torch.sqrt(torch.randn_like(x).square() + torch.randn_like(x).square()) / math.sqrt(2.0)
    return fading * x + noise_std * torch.randn_like(x)


def sspa(x, noise_std, p=3.0, a0=1.5, gain=5.0):
    if x.shape[1] % 2:
        raise ValueError("SSPA expects I/Q pairs, so the dimension must be even.")
    pairs = x.reshape(-1, 2)
    radius = torch.sqrt((pairs * pairs).sum(dim=1))
    gain_factor = gain / (1.0 + (gain * radius / a0) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    y = (gain_factor[:, None] * pairs).reshape_as(x)
    return y + (noise_std / math.sqrt(2.0)) * torch.randn_like(y)


def tdl(x, noise_std):
    """Compact short-block TDL channel used in the paper.

    The input is four complex channel uses represented as eight real I/Q
    coordinates. Three circular taps are used to make the channel memory-bearing
    while keeping the benchmark small.
    """
    if x.shape[1] != 8:
        raise ValueError("The compact TDL benchmark uses n=8 real dimensions.")
    batch = x.shape[0]
    symbols = 4
    taps = [(0, 0.72), (1, 0.20), (3, 0.08)]
    pairs = x.reshape(batch, symbols, 2)
    xr = pairs[:, :, 0]
    xi = pairs[:, :, 1]
    yr = torch.zeros_like(xr)
    yi = torch.zeros_like(xi)
    for shift, power in taps:
        scale = math.sqrt(power / 2.0)
        hr = scale * torch.randn(batch, 1, device=x.device)
        hi = scale * torch.randn(batch, 1, device=x.device)
        xr_shift = torch.roll(xr, shifts=shift, dims=1)
        xi_shift = torch.roll(xi, shifts=shift, dims=1)
        yr = yr + hr * xr_shift - hi * xi_shift
        yi = yi + hr * xi_shift + hi * xr_shift
    y = torch.stack([yr, yi], dim=-1).reshape_as(x)
    return y + noise_std * torch.randn_like(y)


CHANNELS = {
    "AWGN": awgn,
    "Rayleigh": rayleigh,
    "SSPA": sspa,
    "TDL": tdl,
}


def get_channel(name):
    if name not in CHANNELS:
        raise ValueError("Unknown channel %r. Available: %s" % (name, ", ".join(CHANNELS)))
    return CHANNELS[name]

