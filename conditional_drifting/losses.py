from __future__ import annotations

import torch
import torch.nn.functional as F


def median_heuristic_bandwidth(x: torch.Tensor, min_bandwidth: float = 1e-3) -> float:
    if x.shape[0] < 2:
        return float(min_bandwidth)
    distances = torch.cdist(x, x, p=2)
    upper = torch.triu(distances, diagonal=1)
    values = upper[upper > 0]
    if values.numel() == 0:
        return float(min_bandwidth)
    return float(torch.clamp(values.median(), min=min_bandwidth).item())


def compute_kernel_drift(
    generated: torch.Tensor,
    positive: torch.Tensor,
    bandwidth: float | None = None,
    min_bandwidth: float = 1e-3,
    max_drift_norm: float | None = None,
    repulsive_weight: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    generated = generated.float()
    positive = positive.to(device=generated.device, dtype=torch.float32)

    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(
            torch.cat((generated.detach(), positive.detach()), dim=0),
            min_bandwidth=min_bandwidth,
        )
    bandwidth = max(float(bandwidth), float(min_bandwidth))

    sq_dist = torch.cdist(generated, positive, p=2).square()
    weights = torch.exp(-sq_dist / (2.0 * bandwidth * bandwidth))
    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    barycenter = weights @ positive
    drift = barycenter - generated

    if repulsive_weight > 0.0:
        sq_dist_gg = torch.cdist(generated, generated, p=2).square()
        weights_gg = torch.exp(-sq_dist_gg / (2.0 * bandwidth * bandwidth))
        weights_gg.fill_diagonal_(0.0)
        weights_gg = weights_gg / (weights_gg.sum(dim=1, keepdim=True) + eps)
        neighbor_center = weights_gg @ generated
        repulse = neighbor_center - generated
        drift = drift - float(repulsive_weight) * repulse

    if max_drift_norm is not None:
        drift_norm = drift.norm(dim=1, keepdim=True) + eps
        scale = torch.clamp(float(max_drift_norm) / drift_norm, max=1.0)
        drift = drift * scale

    return drift


def drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    drift_scale: float = 1.0,
    bandwidth: float | None = None,
    min_bandwidth: float = 1e-3,
    max_drift_norm: float | None = None,
    repulsive_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    drift = compute_kernel_drift(
        generated,
        positive,
        bandwidth=bandwidth,
        min_bandwidth=min_bandwidth,
        max_drift_norm=max_drift_norm,
        repulsive_weight=repulsive_weight,
    )
    target = (generated + float(drift_scale) * drift).detach()
    loss = F.mse_loss(generated, target)
    return loss, drift
