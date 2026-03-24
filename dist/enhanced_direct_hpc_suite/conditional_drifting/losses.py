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


def _resolve_bandwidth(data: torch.Tensor, bandwidth: float | None, min_bandwidth: float) -> float:
    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(data.detach(), min_bandwidth=min_bandwidth)
    return max(float(bandwidth), float(min_bandwidth))


def _local_bandwidths(data: torch.Tensor, k: int, min_bandwidth: float) -> torch.Tensor:
    if data.shape[0] < 2:
        return torch.full((data.shape[0],), float(min_bandwidth), device=data.device, dtype=data.dtype)
    neighbor_k = max(1, min(int(k), data.shape[0] - 1))
    distances = torch.cdist(data, data, p=2)
    distances.fill_diagonal_(float("inf"))
    kth = torch.topk(distances, k=neighbor_k, dim=1, largest=False).values[:, -1]
    return torch.clamp(kth, min=float(min_bandwidth))


def _gaussian_from_sqdist(
    sq_dist: torch.Tensor,
    *,
    bandwidth: float | None = None,
    left_bandwidths: torch.Tensor | None = None,
    right_bandwidths: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if left_bandwidths is not None or right_bandwidths is not None:
        if left_bandwidths is None or right_bandwidths is None:
            raise ValueError("Adaptive bandwidths require both left and right bandwidth tensors.")
        denom = 2.0 * (left_bandwidths[:, None] * right_bandwidths[None, :] + eps)
        return torch.exp(-sq_dist / denom)
    if bandwidth is None:
        raise ValueError("Either a scalar bandwidth or adaptive bandwidth tensors must be provided.")
    return torch.exp(-sq_dist / (2.0 * float(bandwidth) * float(bandwidth)))


def _whiten_features(generated: torch.Tensor, positive: torch.Tensor, min_bandwidth: float) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.cat((generated, positive), dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    if centered.shape[0] <= 1:
        return generated, positive
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    jitter = float(min_bandwidth) * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    eigvals, eigvecs = torch.linalg.eigh(cov + jitter)
    inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(torch.clamp(eigvals, min=float(min_bandwidth)))) @ eigvecs.T
    whitened = centered @ inv_sqrt
    return whitened[: generated.shape[0]], whitened[generated.shape[0] :]


def _prepare_condition_features(
    condition_generated: torch.Tensor | None,
    condition_positive: torch.Tensor | None,
    *,
    condition_scale: float,
    condition_metric: str,
    min_bandwidth: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if condition_generated is None or condition_positive is None:
        return condition_generated, condition_positive
    scaled_generated = float(condition_scale) * condition_generated
    scaled_positive = float(condition_scale) * condition_positive
    metric = str(condition_metric or "euclidean").lower()
    if metric == "euclidean":
        return scaled_generated, scaled_positive
    if metric == "whitened":
        return _whiten_features(scaled_generated, scaled_positive, min_bandwidth=min_bandwidth)
    raise ValueError(f"Unsupported condition_metric={condition_metric!r}")


def _prepare_target_features(
    generated: torch.Tensor,
    positive: torch.Tensor,
    *,
    target_scale: float,
    target_representation: str,
    target_is_residual: bool,
    target_condition_generated: torch.Tensor | None,
    target_condition_positive: torch.Tensor | None,
    residual_target_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scaled_generated = float(target_scale) * generated
    scaled_positive = float(target_scale) * positive
    representation = str(target_representation or "raw").lower()
    if representation == "raw":
        return scaled_generated, scaled_positive
    if representation == "raw_plus_residual":
        if target_is_residual:
            return scaled_generated, scaled_positive
        if target_condition_generated is None or target_condition_positive is None:
            raise ValueError("target_representation='raw_plus_residual' requires target condition tensors.")
        residual_generated = float(residual_target_scale) * (generated - target_condition_generated)
        residual_positive = float(residual_target_scale) * (positive - target_condition_positive)
        return (
            torch.cat((scaled_generated, residual_generated), dim=1),
            torch.cat((scaled_positive, residual_positive), dim=1),
        )
    raise ValueError(f"Unsupported target_representation={target_representation!r}")


def _resolve_cross_bandwidths(
    generated: torch.Tensor,
    positive: torch.Tensor,
    *,
    adaptive: bool,
    adaptive_k: int,
    bandwidth: float | None,
    min_bandwidth: float,
) -> tuple[float | None, torch.Tensor | None, torch.Tensor | None]:
    if adaptive:
        return (
            None,
            _local_bandwidths(generated.detach(), adaptive_k, min_bandwidth),
            _local_bandwidths(positive.detach(), adaptive_k, min_bandwidth),
        )
    return (
        _resolve_bandwidth(torch.cat((generated, positive), dim=0), bandwidth, min_bandwidth),
        None,
        None,
    )


def compute_kernel_drift(
    generated: torch.Tensor,
    positive: torch.Tensor,
    condition_generated: torch.Tensor | None = None,
    condition_positive: torch.Tensor | None = None,
    target_condition_generated: torch.Tensor | None = None,
    target_condition_positive: torch.Tensor | None = None,
    conditioning_mode: str = "none",
    condition_metric: str = "euclidean",
    condition_scale: float = 1.0,
    target_scale: float = 1.0,
    bandwidth: float | None = None,
    condition_bandwidth: float | None = None,
    target_bandwidth: float | None = None,
    local_condition_k: int = 32,
    condition_radius: float | None = None,
    mixture_alpha: float = 0.5,
    target_representation: str = "raw",
    target_is_residual: bool = False,
    residual_target_scale: float = 1.0,
    adaptive_condition_bandwidth: bool = False,
    adaptive_target_bandwidth: bool = False,
    adaptive_bandwidth_k: int = 16,
    min_bandwidth: float = 1e-3,
    max_drift_norm: float | None = None,
    repulsive_weight: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    generated = generated.float()
    positive = positive.to(device=generated.device, dtype=torch.float32)
    if condition_generated is not None:
        condition_generated = condition_generated.to(device=generated.device, dtype=torch.float32)
    if condition_positive is not None:
        condition_positive = condition_positive.to(device=generated.device, dtype=torch.float32)
    if target_condition_generated is not None:
        target_condition_generated = target_condition_generated.to(device=generated.device, dtype=torch.float32)
    if target_condition_positive is not None:
        target_condition_positive = target_condition_positive.to(device=generated.device, dtype=torch.float32)
    if (condition_generated is None) != (condition_positive is None):
        raise ValueError("Provide both condition_generated and condition_positive or neither.")
    if (target_condition_generated is None) != (target_condition_positive is None):
        raise ValueError("Provide both target_condition_generated and target_condition_positive or neither.")
    conditioning_mode = str(conditioning_mode or "none").lower()
    if conditioning_mode not in {"none", "joint", "product", "local", "soft_local", "radius", "mixture"}:
        raise ValueError(f"Unsupported conditioning_mode={conditioning_mode!r}")
    if conditioning_mode != "none" and condition_generated is None:
        raise ValueError(f"conditioning_mode={conditioning_mode!r} requires condition tensors.")

    target_generated, target_positive = _prepare_target_features(
        generated,
        positive,
        target_scale=target_scale,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        target_condition_generated=target_condition_generated,
        target_condition_positive=target_condition_positive,
        residual_target_scale=residual_target_scale,
    )
    cond_generated, cond_positive = _prepare_condition_features(
        condition_generated,
        condition_positive,
        condition_scale=condition_scale,
        condition_metric=condition_metric,
        min_bandwidth=min_bandwidth,
    )

    target_bw, target_left_bw, target_right_bw = _resolve_cross_bandwidths(
        target_generated,
        target_positive,
        adaptive=adaptive_target_bandwidth,
        adaptive_k=adaptive_bandwidth_k,
        bandwidth=target_bandwidth if target_bandwidth is not None else bandwidth,
        min_bandwidth=min_bandwidth,
    )

    if conditioning_mode == "none":
        sq_dist = torch.cdist(target_generated, target_positive, p=2).square()
        weights = _gaussian_from_sqdist(
            sq_dist,
            bandwidth=target_bw,
            left_bandwidths=target_left_bw,
            right_bandwidths=target_right_bw,
            eps=eps,
        )
    elif conditioning_mode == "joint":
        generated_kernel = torch.cat((cond_generated, target_generated), dim=1)
        positive_kernel = torch.cat((cond_positive, target_positive), dim=1)
        joint_bw, joint_left_bw, joint_right_bw = _resolve_cross_bandwidths(
            generated_kernel,
            positive_kernel,
            adaptive=adaptive_target_bandwidth or adaptive_condition_bandwidth,
            adaptive_k=adaptive_bandwidth_k,
            bandwidth=bandwidth,
            min_bandwidth=min_bandwidth,
        )
        sq_dist = torch.cdist(generated_kernel, positive_kernel, p=2).square()
        weights = _gaussian_from_sqdist(
            sq_dist,
            bandwidth=joint_bw,
            left_bandwidths=joint_left_bw,
            right_bandwidths=joint_right_bw,
            eps=eps,
        )
    else:
        condition_bw, condition_left_bw, condition_right_bw = _resolve_cross_bandwidths(
            cond_generated,
            cond_positive,
            adaptive=adaptive_condition_bandwidth,
            adaptive_k=adaptive_bandwidth_k,
            bandwidth=condition_bandwidth,
            min_bandwidth=min_bandwidth,
        )
        sq_dist_x = torch.cdist(cond_generated, cond_positive, p=2).square()
        sq_dist_y = torch.cdist(target_generated, target_positive, p=2).square()
        kx = _gaussian_from_sqdist(
            sq_dist_x,
            bandwidth=condition_bw,
            left_bandwidths=condition_left_bw,
            right_bandwidths=condition_right_bw,
            eps=eps,
        )
        ky = _gaussian_from_sqdist(
            sq_dist_y,
            bandwidth=target_bw,
            left_bandwidths=target_left_bw,
            right_bandwidths=target_right_bw,
            eps=eps,
        )

        if conditioning_mode == "product":
            weights = kx * ky
        elif conditioning_mode == "mixture":
            alpha = float(mixture_alpha)
            weights = alpha * ky + (1.0 - alpha) * (kx * ky)
        elif conditioning_mode == "radius":
            radius = condition_radius
            if radius is None:
                radius = condition_bw if condition_bw is not None else _resolve_bandwidth(
                    torch.cat((cond_generated, cond_positive), dim=0),
                    condition_bandwidth,
                    min_bandwidth,
                )
            mask = torch.cdist(cond_generated, cond_positive, p=2) <= float(radius)
            weights = ky * mask.to(dtype=generated.dtype)
        else:
            k = max(1, min(int(local_condition_k), positive.shape[0]))
            neighbor_idx = torch.topk(sq_dist_x, k=k, dim=1, largest=False).indices
            mask = torch.zeros_like(sq_dist_y, dtype=torch.bool)
            mask.scatter_(1, neighbor_idx, True)
            if conditioning_mode == "local":
                weights = ky * mask.to(dtype=generated.dtype)
            elif conditioning_mode == "soft_local":
                weights = ky * kx * mask.to(dtype=generated.dtype)
            else:
                raise ValueError(f"Unsupported conditioning_mode={conditioning_mode!r}")

    weights = weights / (weights.sum(dim=1, keepdim=True) + eps)
    barycenter = weights @ positive
    drift = barycenter - generated

    if repulsive_weight > 0.0:
        if conditioning_mode == "joint":
            sq_dist_gg = torch.cdist(generated_kernel, generated_kernel, p=2).square()
            joint_self_bw, joint_self_left_bw, joint_self_right_bw = _resolve_cross_bandwidths(
                generated_kernel,
                generated_kernel,
                adaptive=adaptive_target_bandwidth or adaptive_condition_bandwidth,
                adaptive_k=adaptive_bandwidth_k,
                bandwidth=bandwidth,
                min_bandwidth=min_bandwidth,
            )
            weights_gg = _gaussian_from_sqdist(
                sq_dist_gg,
                bandwidth=joint_self_bw,
                left_bandwidths=joint_self_left_bw,
                right_bandwidths=joint_self_right_bw,
                eps=eps,
            )
        elif conditioning_mode in {"product", "local", "soft_local", "radius", "mixture"}:
            sq_dist_x_gg = torch.cdist(cond_generated, cond_generated, p=2).square()
            sq_dist_y_gg = torch.cdist(target_generated, target_generated, p=2).square()
            condition_self_bw, condition_self_left_bw, condition_self_right_bw = _resolve_cross_bandwidths(
                cond_generated,
                cond_generated,
                adaptive=adaptive_condition_bandwidth,
                adaptive_k=adaptive_bandwidth_k,
                bandwidth=condition_bandwidth,
                min_bandwidth=min_bandwidth,
            )
            target_self_bw, target_self_left_bw, target_self_right_bw = _resolve_cross_bandwidths(
                target_generated,
                target_generated,
                adaptive=adaptive_target_bandwidth,
                adaptive_k=adaptive_bandwidth_k,
                bandwidth=target_bandwidth if target_bandwidth is not None else bandwidth,
                min_bandwidth=min_bandwidth,
            )
            kx_gg = _gaussian_from_sqdist(
                sq_dist_x_gg,
                bandwidth=condition_self_bw,
                left_bandwidths=condition_self_left_bw,
                right_bandwidths=condition_self_right_bw,
                eps=eps,
            )
            ky_gg = _gaussian_from_sqdist(
                sq_dist_y_gg,
                bandwidth=target_self_bw,
                left_bandwidths=target_self_left_bw,
                right_bandwidths=target_self_right_bw,
                eps=eps,
            )
            if conditioning_mode == "product":
                weights_gg = kx_gg * ky_gg
            elif conditioning_mode == "mixture":
                alpha = float(mixture_alpha)
                weights_gg = alpha * ky_gg + (1.0 - alpha) * (kx_gg * ky_gg)
            elif conditioning_mode == "radius":
                radius = condition_radius
                if radius is None:
                    radius = condition_self_bw if condition_self_bw is not None else _resolve_bandwidth(
                        cond_generated,
                        condition_bandwidth,
                        min_bandwidth,
                    )
                weights_gg = ky_gg * (torch.cdist(cond_generated, cond_generated, p=2) <= float(radius)).to(dtype=generated.dtype)
            else:
                k = max(1, min(int(local_condition_k), generated.shape[0]))
                neighbor_idx = torch.topk(sq_dist_x_gg + torch.eye(generated.shape[0], device=generated.device) * float("inf"), k=k, dim=1, largest=False).indices
                mask = torch.zeros_like(sq_dist_y_gg, dtype=torch.bool)
                mask.scatter_(1, neighbor_idx, True)
                weights_gg = ky_gg * mask.to(dtype=generated.dtype)
                if conditioning_mode == "soft_local":
                    weights_gg = weights_gg * kx_gg
        else:
            repulse_bw, repulse_left_bw, repulse_right_bw = _resolve_cross_bandwidths(
                target_generated,
                target_generated,
                adaptive=adaptive_target_bandwidth,
                adaptive_k=adaptive_bandwidth_k,
                bandwidth=target_bandwidth if target_bandwidth is not None else bandwidth,
                min_bandwidth=min_bandwidth,
            )
            sq_dist_gg = torch.cdist(target_generated, target_generated, p=2).square()
            weights_gg = _gaussian_from_sqdist(
                sq_dist_gg,
                bandwidth=repulse_bw,
                left_bandwidths=repulse_left_bw,
                right_bandwidths=repulse_right_bw,
                eps=eps,
            )
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
    condition_generated: torch.Tensor | None = None,
    condition_positive: torch.Tensor | None = None,
    target_condition_generated: torch.Tensor | None = None,
    target_condition_positive: torch.Tensor | None = None,
    conditioning_mode: str = "none",
    condition_metric: str = "euclidean",
    condition_scale: float = 1.0,
    target_scale: float = 1.0,
    drift_scale: float = 1.0,
    bandwidth: float | None = None,
    condition_bandwidth: float | None = None,
    target_bandwidth: float | None = None,
    local_condition_k: int = 32,
    condition_radius: float | None = None,
    mixture_alpha: float = 0.5,
    target_representation: str = "raw",
    target_is_residual: bool = False,
    residual_target_scale: float = 1.0,
    adaptive_condition_bandwidth: bool = False,
    adaptive_target_bandwidth: bool = False,
    adaptive_bandwidth_k: int = 16,
    min_bandwidth: float = 1e-3,
    max_drift_norm: float | None = None,
    repulsive_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    drift = compute_kernel_drift(
        generated,
        positive,
        condition_generated=condition_generated,
        condition_positive=condition_positive,
        target_condition_generated=target_condition_generated,
        target_condition_positive=target_condition_positive,
        conditioning_mode=conditioning_mode,
        condition_metric=condition_metric,
        condition_scale=condition_scale,
        target_scale=target_scale,
        bandwidth=bandwidth,
        condition_bandwidth=condition_bandwidth,
        target_bandwidth=target_bandwidth,
        local_condition_k=local_condition_k,
        condition_radius=condition_radius,
        mixture_alpha=mixture_alpha,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        residual_target_scale=residual_target_scale,
        adaptive_condition_bandwidth=adaptive_condition_bandwidth,
        adaptive_target_bandwidth=adaptive_target_bandwidth,
        adaptive_bandwidth_k=adaptive_bandwidth_k,
        min_bandwidth=min_bandwidth,
        max_drift_norm=max_drift_norm,
        repulsive_weight=repulsive_weight,
    )
    target = (generated + float(drift_scale) * drift).detach()
    loss = F.mse_loss(generated, target)
    return loss, drift
