import math

import torch


def sliced_wasserstein(x, y, projections=128, seed=12345):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    directions = torch.randn(x.shape[1], projections, generator=generator, device=x.device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    x_proj = (x @ directions).sort(dim=0).values
    y_proj = (y @ directions).sort(dim=0).values
    return torch.sqrt(((x_proj - y_proj) ** 2).mean()).item()


def anchor_swd(x, y_true, y_model, anchors=64, samples=64, projections=64):
    if x.shape[0] < anchors * samples:
        anchors = max(1, x.shape[0] // samples)
    values = []
    for i in range(anchors):
        start = i * samples
        stop = start + samples
        values.append(sliced_wasserstein(y_true[start:stop], y_model[start:stop], projections=projections, seed=9000 + i))
    return sum(values) / max(1, len(values))


def bit_errors_from_symbols(pred, target, bits_per_symbol):
    symbol_errors = (pred != target).float()
    ser = symbol_errors.mean().item()
    ber = ser / max(1.0, math.log2(bits_per_symbol))
    return ber, ser

