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
    if representation in {"polar_residual", "raw_plus_polar_residual"}:
        if target_is_residual:
            raise ValueError(f"target_representation={target_representation!r} requires direct target values.")
        if target_condition_generated is None or target_condition_positive is None:
            raise ValueError(f"target_representation={target_representation!r} requires target condition tensors.")
        polar_generated = _complex_polar_residual_features(
            generated,
            target_condition_generated,
            scale=residual_target_scale,
        )
        polar_positive = _complex_polar_residual_features(
            positive,
            target_condition_positive,
            scale=residual_target_scale,
        )
        if representation == "polar_residual":
            return polar_generated, polar_positive
        return (
            torch.cat((scaled_generated, polar_generated), dim=1),
            torch.cat((scaled_positive, polar_positive), dim=1),
        )
    raise ValueError(f"Unsupported target_representation={target_representation!r}")


def _complex_polar_residual_features(
    values: torch.Tensor,
    conditions: torch.Tensor,
    *,
    scale: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Represent complex I/Q outputs by radial and phase residuals to conditions."""
    if values.shape[-1] % 2 != 0:
        raise ValueError("Polar residual features require an even I/Q feature dimension.")
    if values.shape != conditions.shape:
        raise ValueError("Polar residual features require values and conditions with matching shapes.")

    pair_count = values.shape[-1] // 2
    values_2d = values.reshape(*values.shape[:-1], pair_count, 2)
    conditions_2d = conditions.reshape(*conditions.shape[:-1], pair_count, 2)

    vr = values_2d[..., 0]
    vi = values_2d[..., 1]
    cr = conditions_2d[..., 0]
    ci = conditions_2d[..., 1]

    value_radius = torch.sqrt(vr.square() + vi.square() + eps)
    condition_radius = torch.sqrt(cr.square() + ci.square() + eps)
    radial_residual = value_radius - condition_radius
    denom = value_radius * condition_radius + eps
    cos_delta = torch.clamp((vr * cr + vi * ci) / denom, min=-1.0, max=1.0)
    sin_delta = torch.clamp((vi * cr - vr * ci) / denom, min=-1.0, max=1.0)
    features = torch.stack((radial_residual, sin_delta, cos_delta), dim=-1)
    return float(scale) * features.reshape(*values.shape[:-1], pair_count * 3)


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


def _resolve_sinkhorn_epsilon(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    epsilon: float | None,
    min_epsilon: float,
) -> float:
    if epsilon is not None:
        return max(float(epsilon), float(min_epsilon))
    if source_features.numel() == 0 or target_features.numel() == 0:
        return float(min_epsilon)
    with torch.no_grad():
        costs = 0.5 * torch.cdist(source_features.detach(), target_features.detach(), p=2).square()
        values = costs[costs > 0]
        if values.numel() == 0:
            return float(min_epsilon)
        return float(torch.clamp(values.median(), min=float(min_epsilon)).item())


def _sinkhorn_barycentric_projection(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    target_values: torch.Tensor,
    *,
    epsilon: float | None,
    min_epsilon: float,
    iterations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    source_features = source_features.detach().float()
    target_features = target_features.detach().to(device=source_features.device, dtype=torch.float32)
    target_values = target_values.detach().to(device=source_features.device, dtype=torch.float32)
    n_source = source_features.shape[0]
    n_target = target_features.shape[0]
    if n_source == 0 or n_target == 0:
        raise ValueError("Sinkhorn projection requires non-empty source and target batches.")

    regularization = _resolve_sinkhorn_epsilon(source_features, target_features, epsilon, min_epsilon)
    cost = 0.5 * torch.cdist(source_features, target_features, p=2).square()
    scaled_cost = cost / regularization
    # Row shifts are absorbed by Sinkhorn's source scaling and improve numerical stability.
    scaled_cost = scaled_cost - scaled_cost.amin(dim=1, keepdim=True)
    kernel = torch.exp(-scaled_cost).clamp_min(eps)

    source_mass = torch.full((n_source,), 1.0 / n_source, device=source_features.device, dtype=kernel.dtype)
    target_mass = torch.full((n_target,), 1.0 / n_target, device=source_features.device, dtype=kernel.dtype)
    u = torch.ones_like(source_mass)
    v = torch.ones_like(target_mass)
    for _ in range(max(1, int(iterations))):
        u = source_mass / (kernel @ v + eps)
        v = target_mass / (kernel.T @ u + eps)

    coupling = u[:, None] * kernel * v[None, :]
    row_weights = coupling / (coupling.sum(dim=1, keepdim=True) + eps)
    return row_weights @ target_values


def _batched_sinkhorn_barycentric_projection(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    target_values: torch.Tensor,
    *,
    epsilon: float | None,
    min_epsilon: float,
    iterations: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    source_features = source_features.detach().float()
    target_features = target_features.detach().to(device=source_features.device, dtype=torch.float32)
    target_values = target_values.detach().to(device=source_features.device, dtype=torch.float32)
    batch_size, n_source, _ = source_features.shape
    n_target = target_features.shape[1]
    if batch_size == 0 or n_source == 0 or n_target == 0:
        raise ValueError("Batched Sinkhorn projection requires non-empty source and target batches.")

    cost = 0.5 * torch.cdist(source_features, target_features, p=2).square()
    if epsilon is not None:
        regularization = max(float(epsilon), float(min_epsilon))
    else:
        # Estimate the Sinkhorn scale only from within-condition costs. Flattening
        # the batch before cdist would compare unrelated fibers and create a
        # quadratic global matrix.
        with torch.no_grad():
            values = cost.detach()[cost.detach() > 0]
            if values.numel() == 0:
                regularization = float(min_epsilon)
            else:
                regularization = float(torch.clamp(values.median(), min=float(min_epsilon)).item())
    scaled_cost = cost / regularization
    scaled_cost = scaled_cost - scaled_cost.amin(dim=2, keepdim=True)
    kernel = torch.exp(-scaled_cost).clamp_min(eps)

    source_mass = torch.full((batch_size, n_source), 1.0 / n_source, device=source_features.device, dtype=kernel.dtype)
    target_mass = torch.full((batch_size, n_target), 1.0 / n_target, device=source_features.device, dtype=kernel.dtype)
    u = torch.ones_like(source_mass)
    v = torch.ones_like(target_mass)
    for _ in range(max(1, int(iterations))):
        u = source_mass / (torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1) + eps)
        v = target_mass / (torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + eps)

    coupling = u[:, :, None] * kernel * v[:, None, :]
    row_weights = coupling / (coupling.sum(dim=2, keepdim=True) + eps)
    return torch.bmm(row_weights, target_values)


def _build_sinkhorn_features(
    target_left: torch.Tensor,
    target_right: torch.Tensor,
    condition_left: torch.Tensor | None,
    condition_right: torch.Tensor | None,
    conditioning_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if conditioning_mode == "none":
        return target_left, target_right
    if condition_left is None or condition_right is None:
        raise ValueError(f"conditioning_mode={conditioning_mode!r} requires condition tensors.")
    return torch.cat((condition_left, target_left), dim=1), torch.cat((condition_right, target_right), dim=1)


def _prepare_fiber_target_features(
    values: torch.Tensor,
    conditions: torch.Tensor | None,
    *,
    target_scale: float,
    target_representation: str,
    target_is_residual: bool,
    residual_target_scale: float,
) -> torch.Tensor:
    scaled = float(target_scale) * values
    representation = str(target_representation or "raw").lower()
    if representation == "raw":
        return scaled
    if representation == "raw_plus_residual":
        if target_is_residual:
            return scaled
        if conditions is None:
            raise ValueError("target_representation='raw_plus_residual' requires target condition tensors.")
        residual = float(residual_target_scale) * (values - conditions)
        return torch.cat((scaled, residual), dim=2)
    if representation in {"polar_residual", "raw_plus_polar_residual"}:
        if target_is_residual:
            raise ValueError(f"target_representation={target_representation!r} requires direct target values.")
        if conditions is None:
            raise ValueError(f"target_representation={target_representation!r} requires target condition tensors.")
        polar = _complex_polar_residual_features(
            values,
            conditions,
            scale=residual_target_scale,
        )
        if representation == "polar_residual":
            return polar
        return torch.cat((scaled, polar), dim=2)
    raise ValueError(f"Unsupported target_representation={target_representation!r}")


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
                self_dist_x = sq_dist_x_gg.masked_fill(
                    torch.eye(generated.shape[0], device=generated.device, dtype=torch.bool),
                    float("inf"),
                )
                neighbor_idx = torch.topk(self_dist_x, k=k, dim=1, largest=False).indices
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


def compute_sinkhorn_drift(
    generated: torch.Tensor,
    positive: torch.Tensor,
    condition_generated: torch.Tensor | None = None,
    condition_positive: torch.Tensor | None = None,
    condition_reference: torch.Tensor | None = None,
    target_condition_generated: torch.Tensor | None = None,
    target_condition_positive: torch.Tensor | None = None,
    target_condition_reference: torch.Tensor | None = None,
    generated_reference: torch.Tensor | None = None,
    conditioning_mode: str = "none",
    condition_metric: str = "euclidean",
    condition_scale: float = 1.0,
    target_scale: float = 1.0,
    target_representation: str = "raw",
    target_is_residual: bool = False,
    residual_target_scale: float = 1.0,
    sinkhorn_epsilon: float | None = None,
    sinkhorn_min_epsilon: float = 1e-3,
    sinkhorn_iterations: int = 10,
    max_drift_norm: float | None = None,
    repulsive_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    generated = generated.float()
    positive = positive.to(device=generated.device, dtype=torch.float32)
    if generated_reference is None:
        generated_reference = generated.detach()
    generated_reference = generated_reference.to(device=generated.device, dtype=torch.float32)
    if condition_generated is not None:
        condition_generated = condition_generated.to(device=generated.device, dtype=torch.float32)
    if condition_positive is not None:
        condition_positive = condition_positive.to(device=generated.device, dtype=torch.float32)
    if condition_reference is not None:
        condition_reference = condition_reference.to(device=generated.device, dtype=torch.float32)
    if target_condition_generated is not None:
        target_condition_generated = target_condition_generated.to(device=generated.device, dtype=torch.float32)
    if target_condition_positive is not None:
        target_condition_positive = target_condition_positive.to(device=generated.device, dtype=torch.float32)
    if target_condition_reference is not None:
        target_condition_reference = target_condition_reference.to(device=generated.device, dtype=torch.float32)

    conditioning_mode = str(conditioning_mode or "none").lower()
    if conditioning_mode not in {"none", "joint", "product", "local", "soft_local", "radius", "mixture"}:
        raise ValueError(f"Unsupported conditioning_mode={conditioning_mode!r}")
    if conditioning_mode != "none" and condition_generated is None:
        raise ValueError(f"conditioning_mode={conditioning_mode!r} requires condition tensors.")
    if conditioning_mode != "none" and condition_positive is None:
        raise ValueError(f"conditioning_mode={conditioning_mode!r} requires positive condition tensors.")
    if conditioning_mode != "none" and condition_reference is None:
        condition_reference = condition_generated.detach()
    if target_condition_reference is None and target_condition_generated is not None:
        target_condition_reference = target_condition_generated.detach()

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
    target_self_generated, target_reference = _prepare_target_features(
        generated,
        generated_reference,
        target_scale=target_scale,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        target_condition_generated=target_condition_generated,
        target_condition_positive=target_condition_reference,
        residual_target_scale=residual_target_scale,
    )
    cond_generated, cond_positive = _prepare_condition_features(
        condition_generated,
        condition_positive,
        condition_scale=condition_scale,
        condition_metric=condition_metric,
        min_bandwidth=sinkhorn_min_epsilon,
    )
    cond_self_generated, cond_reference = _prepare_condition_features(
        condition_generated,
        condition_reference,
        condition_scale=condition_scale,
        condition_metric=condition_metric,
        min_bandwidth=sinkhorn_min_epsilon,
    )

    source_features, positive_features = _build_sinkhorn_features(
        target_generated,
        target_positive,
        cond_generated,
        cond_positive,
        conditioning_mode,
    )
    positive_center = _sinkhorn_barycentric_projection(
        source_features,
        positive_features,
        positive,
        epsilon=sinkhorn_epsilon,
        min_epsilon=sinkhorn_min_epsilon,
        iterations=sinkhorn_iterations,
        eps=eps,
    )
    drift = positive_center - generated

    if repulsive_weight > 0.0:
        self_source_features, self_reference_features = _build_sinkhorn_features(
            target_self_generated,
            target_reference,
            cond_self_generated,
            cond_reference,
            conditioning_mode,
        )
        self_center = _sinkhorn_barycentric_projection(
            self_source_features,
            self_reference_features,
            generated_reference,
            epsilon=sinkhorn_epsilon,
            min_epsilon=sinkhorn_min_epsilon,
            iterations=sinkhorn_iterations,
            eps=eps,
        )
        drift = drift - float(repulsive_weight) * (self_center - generated)

    if max_drift_norm is not None:
        drift_norm = drift.norm(dim=1, keepdim=True) + eps
        scale = torch.clamp(float(max_drift_norm) / drift_norm, max=1.0)
        drift = drift * scale

    return drift.detach()


def compute_fiber_sinkhorn_drift(
    generated: torch.Tensor,
    positive: torch.Tensor,
    target_condition_generated: torch.Tensor | None = None,
    target_condition_positive: torch.Tensor | None = None,
    target_condition_reference: torch.Tensor | None = None,
    generated_reference: torch.Tensor | None = None,
    fiber_num_conditions: int | None = None,
    fiber_generated_samples: int = 1,
    fiber_positive_samples: int = 1,
    fiber_reference_samples: int | None = None,
    target_scale: float = 1.0,
    target_representation: str = "raw",
    target_is_residual: bool = False,
    residual_target_scale: float = 1.0,
    sinkhorn_epsilon: float | None = None,
    sinkhorn_min_epsilon: float = 1e-3,
    sinkhorn_iterations: int = 10,
    max_drift_norm: float | None = None,
    repulsive_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    generated = generated.float()
    positive = positive.to(device=generated.device, dtype=torch.float32)
    generated_count = max(1, int(fiber_generated_samples))
    positive_count = max(1, int(fiber_positive_samples))
    if fiber_num_conditions is None:
        if generated.shape[0] % generated_count != 0:
            raise ValueError("Cannot infer fiber_num_conditions from generated samples.")
        fiber_num_conditions = generated.shape[0] // generated_count
    num_conditions = int(fiber_num_conditions)
    if generated.shape[0] != num_conditions * generated_count:
        raise ValueError("Generated sample count does not match fiber_num_conditions * fiber_generated_samples.")
    if positive.shape[0] != num_conditions * positive_count:
        raise ValueError("Positive sample count does not match fiber_num_conditions * fiber_positive_samples.")

    generated_3d = generated.reshape(num_conditions, generated_count, generated.shape[1])
    positive_3d = positive.reshape(num_conditions, positive_count, positive.shape[1])
    cond_generated_3d = (
        target_condition_generated.to(device=generated.device, dtype=torch.float32).reshape(num_conditions, generated_count, generated.shape[1])
        if target_condition_generated is not None
        else None
    )
    cond_positive_3d = (
        target_condition_positive.to(device=generated.device, dtype=torch.float32).reshape(num_conditions, positive_count, positive.shape[1])
        if target_condition_positive is not None
        else None
    )

    generated_features = _prepare_fiber_target_features(
        generated_3d,
        cond_generated_3d,
        target_scale=target_scale,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        residual_target_scale=residual_target_scale,
    )
    positive_features = _prepare_fiber_target_features(
        positive_3d,
        cond_positive_3d,
        target_scale=target_scale,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        residual_target_scale=residual_target_scale,
    )
    positive_center = _batched_sinkhorn_barycentric_projection(
        generated_features,
        positive_features,
        positive_3d,
        epsilon=sinkhorn_epsilon,
        min_epsilon=sinkhorn_min_epsilon,
        iterations=sinkhorn_iterations,
        eps=eps,
    )
    drift = positive_center - generated_3d

    if repulsive_weight > 0.0:
        if generated_reference is None:
            generated_reference = generated.detach()
            reference_count = generated_count
        else:
            generated_reference = generated_reference.to(device=generated.device, dtype=torch.float32)
            reference_count = max(1, int(fiber_reference_samples or generated_count))
        if generated_reference.shape[0] != num_conditions * reference_count:
            raise ValueError("Reference sample count does not match fiber_num_conditions * fiber_reference_samples.")
        reference_3d = generated_reference.reshape(num_conditions, reference_count, generated.shape[1])
        cond_reference_3d = (
            target_condition_reference.to(device=generated.device, dtype=torch.float32).reshape(num_conditions, reference_count, generated.shape[1])
            if target_condition_reference is not None
            else cond_generated_3d
        )
        reference_features = _prepare_fiber_target_features(
            reference_3d,
            cond_reference_3d,
            target_scale=target_scale,
            target_representation=target_representation,
            target_is_residual=target_is_residual,
            residual_target_scale=residual_target_scale,
        )
        self_center = _batched_sinkhorn_barycentric_projection(
            generated_features,
            reference_features,
            reference_3d,
            epsilon=sinkhorn_epsilon,
            min_epsilon=sinkhorn_min_epsilon,
            iterations=sinkhorn_iterations,
            eps=eps,
        )
        drift = drift - float(repulsive_weight) * (self_center - generated_3d)

    drift = drift.reshape(num_conditions * generated_count, generated.shape[1])
    if max_drift_norm is not None:
        drift_norm = drift.norm(dim=1, keepdim=True) + eps
        scale = torch.clamp(float(max_drift_norm) / drift_norm, max=1.0)
        drift = drift * scale
    return drift.detach()


def compute_fiber_cloud_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    target_condition_generated: torch.Tensor | None = None,
    target_condition_positive: torch.Tensor | None = None,
    fiber_num_conditions: int | None = None,
    fiber_generated_samples: int = 1,
    fiber_positive_samples: int = 1,
    target_scale: float = 1.0,
    target_representation: str = "raw",
    target_is_residual: bool = False,
    residual_target_scale: float = 1.0,
    bandwidth: float | None = None,
    min_bandwidth: float = 1e-3,
    objective: str = "mmd",
    moment_mean_weight: float = 1.0,
    moment_cov_weight: float = 1.0,
    supervised_weight: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    generated = generated.float()
    positive = positive.to(device=generated.device, dtype=torch.float32)
    generated_count = max(1, int(fiber_generated_samples))
    positive_count = max(1, int(fiber_positive_samples))
    if fiber_num_conditions is None:
        if generated.shape[0] % generated_count != 0:
            raise ValueError("Cannot infer fiber_num_conditions from generated samples.")
        fiber_num_conditions = generated.shape[0] // generated_count
    num_conditions = int(fiber_num_conditions)
    if generated.shape[0] != num_conditions * generated_count:
        raise ValueError("Generated sample count does not match fiber_num_conditions * fiber_generated_samples.")
    if positive.shape[0] != num_conditions * positive_count:
        raise ValueError("Positive sample count does not match fiber_num_conditions * fiber_positive_samples.")

    generated_3d = generated.reshape(num_conditions, generated_count, generated.shape[1])
    positive_3d = positive.reshape(num_conditions, positive_count, positive.shape[1])
    cond_generated_3d = (
        target_condition_generated.to(device=generated.device, dtype=torch.float32).reshape(num_conditions, generated_count, generated.shape[1])
        if target_condition_generated is not None
        else None
    )
    cond_positive_3d = (
        target_condition_positive.to(device=generated.device, dtype=torch.float32).reshape(num_conditions, positive_count, positive.shape[1])
        if target_condition_positive is not None
        else None
    )

    generated_features = _prepare_fiber_target_features(
        generated_3d,
        cond_generated_3d,
        target_scale=target_scale,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        residual_target_scale=residual_target_scale,
    )
    positive_features = _prepare_fiber_target_features(
        positive_3d,
        cond_positive_3d,
        target_scale=target_scale,
        target_representation=target_representation,
        target_is_residual=target_is_residual,
        residual_target_scale=residual_target_scale,
    )

    objective_name = str(objective or "mmd").lower()
    def moment_loss() -> torch.Tensor:
        generated_mean = generated_features.mean(dim=1)
        positive_mean = positive_features.mean(dim=1)
        mean_loss = (generated_mean - positive_mean).square().sum(dim=1).mean()

        generated_centered = generated_features - generated_mean[:, None, :]
        positive_centered = positive_features - positive_mean[:, None, :]
        generated_denom = max(1, generated_count - 1)
        positive_denom = max(1, positive_count - 1)
        generated_cov = torch.bmm(generated_centered.transpose(1, 2), generated_centered) / generated_denom
        positive_cov = torch.bmm(positive_centered.transpose(1, 2), positive_centered) / positive_denom
        cov_loss = (generated_cov - positive_cov).square().mean(dim=(1, 2)).mean()
        return float(moment_mean_weight) * mean_loss + float(moment_cov_weight) * cov_loss

    if objective_name in {"mmd", "mmd_moment"}:
        flat_features = torch.cat(
            (
                generated_features.reshape(-1, generated_features.shape[-1]).detach(),
                positive_features.reshape(-1, positive_features.shape[-1]).detach(),
            ),
            dim=0,
        )
        resolved_bw = _resolve_bandwidth(flat_features, bandwidth, min_bandwidth=min_bandwidth)
        sq_gg = torch.cdist(generated_features, generated_features, p=2).square()
        sq_gp = torch.cdist(generated_features, positive_features, p=2).square()
        sq_pp = torch.cdist(positive_features, positive_features, p=2).square()
        denom = 2.0 * resolved_bw * resolved_bw + eps
        k_gg = torch.exp(-sq_gg / denom)
        k_gp = torch.exp(-sq_gp / denom)
        k_pp = torch.exp(-sq_pp / denom)
        loss = (k_gg.mean(dim=(1, 2)) + k_pp.mean(dim=(1, 2)) - 2.0 * k_gp.mean(dim=(1, 2))).mean()
        if objective_name == "mmd_moment":
            loss = loss + moment_loss()
    elif objective_name in {"energy", "energy_moment"}:
        d_gg = torch.cdist(generated_features, generated_features, p=2)
        d_gp = torch.cdist(generated_features, positive_features, p=2)
        d_pp = torch.cdist(positive_features, positive_features, p=2)
        loss = (2.0 * d_gp.mean(dim=(1, 2)) - d_gg.mean(dim=(1, 2)) - d_pp.mean(dim=(1, 2))).mean()
        if objective_name == "energy_moment":
            loss = loss + moment_loss()
    elif objective_name == "moment":
        loss = moment_loss()
    else:
        raise ValueError(f"Unsupported fiber cloud objective={objective!r}")

    if float(supervised_weight) > 0.0:
        paired_count = min(generated_count, positive_count)
        paired_generated = generated_features[:, :paired_count, :]
        paired_positive = positive_features[:, :paired_count, :]
        loss = loss + float(supervised_weight) * F.mse_loss(paired_generated, paired_positive)

    drift = positive_3d.mean(dim=1, keepdim=True) - generated_3d
    return loss, drift.reshape(num_conditions * generated_count, generated.shape[1]).detach()


def drifting_loss(
    generated: torch.Tensor,
    positive: torch.Tensor,
    condition_generated: torch.Tensor | None = None,
    condition_positive: torch.Tensor | None = None,
    condition_reference: torch.Tensor | None = None,
    target_condition_generated: torch.Tensor | None = None,
    target_condition_positive: torch.Tensor | None = None,
    target_condition_reference: torch.Tensor | None = None,
    generated_reference: torch.Tensor | None = None,
    drift_field: str = "kernel",
    fiber_num_conditions: int | None = None,
    fiber_generated_samples: int = 1,
    fiber_positive_samples: int = 1,
    fiber_reference_samples: int | None = None,
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
    sinkhorn_epsilon: float | None = None,
    sinkhorn_min_epsilon: float = 1e-3,
    sinkhorn_iterations: int = 10,
    fiber_moment_mean_weight: float = 1.0,
    fiber_moment_cov_weight: float = 1.0,
    fiber_supervised_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    drift_field = str(drift_field or "kernel").lower()
    if drift_field == "kernel":
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
    elif drift_field == "sinkhorn":
        drift = compute_sinkhorn_drift(
            generated,
            positive,
            condition_generated=condition_generated,
            condition_positive=condition_positive,
            condition_reference=condition_reference,
            target_condition_generated=target_condition_generated,
            target_condition_positive=target_condition_positive,
            target_condition_reference=target_condition_reference,
            generated_reference=generated_reference,
            conditioning_mode=conditioning_mode,
            condition_metric=condition_metric,
            condition_scale=condition_scale,
            target_scale=target_scale,
            target_representation=target_representation,
            target_is_residual=target_is_residual,
            residual_target_scale=residual_target_scale,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_min_epsilon=sinkhorn_min_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            max_drift_norm=max_drift_norm,
            repulsive_weight=repulsive_weight,
        )
    elif drift_field == "fiber_sinkhorn":
        drift = compute_fiber_sinkhorn_drift(
            generated,
            positive,
            target_condition_generated=target_condition_generated,
            target_condition_positive=target_condition_positive,
            target_condition_reference=target_condition_reference,
            generated_reference=generated_reference,
            fiber_num_conditions=fiber_num_conditions,
            fiber_generated_samples=fiber_generated_samples,
            fiber_positive_samples=fiber_positive_samples,
            fiber_reference_samples=fiber_reference_samples,
            target_scale=target_scale,
            target_representation=target_representation,
            target_is_residual=target_is_residual,
            residual_target_scale=residual_target_scale,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_min_epsilon=sinkhorn_min_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            max_drift_norm=max_drift_norm,
            repulsive_weight=repulsive_weight,
        )
    elif drift_field in {"fiber_mmd", "fiber_energy", "fiber_moment", "fiber_energy_moment", "fiber_mmd_moment"}:
        loss, drift = compute_fiber_cloud_loss(
            generated,
            positive,
            target_condition_generated=target_condition_generated,
            target_condition_positive=target_condition_positive,
            fiber_num_conditions=fiber_num_conditions,
            fiber_generated_samples=fiber_generated_samples,
            fiber_positive_samples=fiber_positive_samples,
            target_scale=target_scale,
            target_representation=target_representation,
            target_is_residual=target_is_residual,
            residual_target_scale=residual_target_scale,
            bandwidth=target_bandwidth if target_bandwidth is not None else bandwidth,
            min_bandwidth=min_bandwidth,
            objective={
                "fiber_mmd": "mmd",
                "fiber_energy": "energy",
                "fiber_moment": "moment",
                "fiber_energy_moment": "energy_moment",
                "fiber_mmd_moment": "mmd_moment",
            }[drift_field],
            moment_mean_weight=fiber_moment_mean_weight,
            moment_cov_weight=fiber_moment_cov_weight,
            supervised_weight=fiber_supervised_weight,
        )
        return loss, drift
    else:
        raise ValueError(f"Unsupported drift_field={drift_field!r}")
    target = (generated + float(drift_scale) * drift).detach()
    loss = F.mse_loss(generated, target)
    return loss, drift
