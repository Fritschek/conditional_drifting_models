from __future__ import annotations

import numpy as np
import torch


def _normalize_numpy_inputs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension.")
    return x, y


def _normalize_torch_inputs(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32, device=x.device)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension.")
    return x, y


def sliced_wasserstein_distance_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_projections: int = 256,
    seed: int = 12345,
) -> torch.Tensor:
    x, y = _normalize_torch_inputs(x, y)
    rng = np.random.default_rng(seed)
    directions_np = rng.normal(size=(num_projections, x.shape[1]))
    directions_np /= np.linalg.norm(directions_np, axis=1, keepdims=True) + 1e-12
    directions = torch.as_tensor(directions_np, dtype=x.dtype, device=x.device)

    quantile_count = min(len(x), len(y))
    if quantile_count < 2:
        return torch.mean(torch.abs(x - y))

    grid = torch.linspace(0.0, 1.0, steps=quantile_count, device=x.device, dtype=x.dtype)
    proj_x = x @ directions.T
    proj_y = y @ directions.T
    qx = torch.quantile(proj_x, grid, dim=0)
    qy = torch.quantile(proj_y, grid, dim=0)
    return torch.mean(torch.abs(qx - qy))


def sliced_wasserstein_distance(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    *,
    num_projections: int = 256,
    seed: int = 12345,
) -> float:
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        x_tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y_tensor = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, device=x_tensor.device)
        return float(sliced_wasserstein_distance_torch(x_tensor, y_tensor, num_projections=num_projections, seed=seed).item())

    x, y = _normalize_numpy_inputs(x, y)
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_projections, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    quantile_count = min(len(x), len(y))
    if quantile_count < 2:
        return float(np.mean(np.abs(x - y)))
    grid = np.linspace(0.0, 1.0, num=quantile_count, endpoint=True)

    proj_x = x @ directions.T
    proj_y = y @ directions.T
    qx = np.quantile(proj_x, grid, axis=0)
    qy = np.quantile(proj_y, grid, axis=0)
    return float(np.mean(np.abs(qx - qy)))


def conditional_joint_swd(
    x: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    *,
    num_projections: int = 256,
    seed: int = 12345,
) -> float:
    if isinstance(x, torch.Tensor) or isinstance(y_true, torch.Tensor) or isinstance(y_pred, torch.Tensor):
        x_tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y_true_tensor = y_true if isinstance(y_true, torch.Tensor) else torch.as_tensor(y_true, device=x_tensor.device)
        y_pred_tensor = y_pred if isinstance(y_pred, torch.Tensor) else torch.as_tensor(y_pred, device=x_tensor.device)
        x_tensor = x_tensor.to(dtype=torch.float32)
        y_true_tensor = y_true_tensor.to(dtype=torch.float32, device=x_tensor.device)
        y_pred_tensor = y_pred_tensor.to(dtype=torch.float32, device=x_tensor.device)
        joint_true = torch.cat((x_tensor, y_true_tensor), dim=1)
        joint_pred = torch.cat((x_tensor, y_pred_tensor), dim=1)
        return float(
            sliced_wasserstein_distance_torch(
                joint_true,
                joint_pred,
                num_projections=num_projections,
                seed=seed,
            ).item()
        )

    x_np = np.asarray(x, dtype=float)
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if y_true_np.ndim == 1:
        y_true_np = y_true_np[:, None]
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np[:, None]
    joint_true = np.concatenate((x_np, y_true_np), axis=1)
    joint_pred = np.concatenate((x_np, y_pred_np), axis=1)
    return sliced_wasserstein_distance(
        joint_true,
        joint_pred,
        num_projections=num_projections,
        seed=seed,
    )


def conditional_anchor_swd(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    *,
    num_projections: int = 256,
    seed: int = 12345,
) -> float:
    if isinstance(y_true, torch.Tensor) or isinstance(y_pred, torch.Tensor):
        y_true_tensor = y_true if isinstance(y_true, torch.Tensor) else torch.as_tensor(y_true)
        y_pred_tensor = y_pred if isinstance(y_pred, torch.Tensor) else torch.as_tensor(y_pred, device=y_true_tensor.device)
        y_true_tensor = y_true_tensor.to(dtype=torch.float32)
        y_pred_tensor = y_pred_tensor.to(dtype=torch.float32, device=y_true_tensor.device)
        if y_true_tensor.ndim != 3 or y_pred_tensor.ndim != 3:
            raise ValueError("conditional_anchor_swd expects tensors of shape [num_anchors, num_samples, dim].")
        if y_true_tensor.shape != y_pred_tensor.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        values = [
            sliced_wasserstein_distance_torch(
                y_true_tensor[i],
                y_pred_tensor[i],
                num_projections=num_projections,
                seed=seed + i,
            )
            for i in range(y_true_tensor.shape[0])
        ]
        return float(torch.stack(values).mean().item())

    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    if y_true_np.ndim != 3 or y_pred_np.ndim != 3:
        raise ValueError("conditional_anchor_swd expects arrays of shape [num_anchors, num_samples, dim].")
    if y_true_np.shape != y_pred_np.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    values = [
        sliced_wasserstein_distance(
            y_true_np[i],
            y_pred_np[i],
            num_projections=num_projections,
            seed=seed + i,
        )
        for i in range(y_true_np.shape[0])
    ]
    return float(np.mean(values))


def conditional_anchor_residual_swd(
    x_anchor: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    *,
    num_projections: int = 256,
    seed: int = 12345,
) -> float:
    if isinstance(x_anchor, torch.Tensor) or isinstance(y_true, torch.Tensor) or isinstance(y_pred, torch.Tensor):
        x_tensor = x_anchor if isinstance(x_anchor, torch.Tensor) else torch.as_tensor(x_anchor)
        y_true_tensor = y_true if isinstance(y_true, torch.Tensor) else torch.as_tensor(y_true, device=x_tensor.device)
        y_pred_tensor = y_pred if isinstance(y_pred, torch.Tensor) else torch.as_tensor(y_pred, device=x_tensor.device)
        x_tensor = x_tensor.to(dtype=torch.float32)
        y_true_tensor = y_true_tensor.to(dtype=torch.float32, device=x_tensor.device)
        y_pred_tensor = y_pred_tensor.to(dtype=torch.float32, device=x_tensor.device)
        if x_tensor.ndim != 2 or y_true_tensor.ndim != 3 or y_pred_tensor.ndim != 3:
            raise ValueError("conditional_anchor_residual_swd expects x of shape [num_anchors, dim] and y tensors of shape [num_anchors, num_samples, dim].")
        if y_true_tensor.shape != y_pred_tensor.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if x_tensor.shape[0] != y_true_tensor.shape[0] or x_tensor.shape[1] != y_true_tensor.shape[2]:
            raise ValueError("Anchor shape is incompatible with conditional samples.")
        residual_true = y_true_tensor - x_tensor[:, None, :]
        residual_pred = y_pred_tensor - x_tensor[:, None, :]
        return conditional_anchor_swd(
            residual_true,
            residual_pred,
            num_projections=num_projections,
            seed=seed,
        )

    x_np = np.asarray(x_anchor, dtype=float)
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    if x_np.ndim != 2 or y_true_np.ndim != 3 or y_pred_np.ndim != 3:
        raise ValueError("conditional_anchor_residual_swd expects x of shape [num_anchors, dim] and y arrays of shape [num_anchors, num_samples, dim].")
    if y_true_np.shape != y_pred_np.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if x_np.shape[0] != y_true_np.shape[0] or x_np.shape[1] != y_true_np.shape[2]:
        raise ValueError("Anchor shape is incompatible with conditional samples.")
    residual_true = y_true_np - x_np[:, None, :]
    residual_pred = y_pred_np - x_np[:, None, :]
    return conditional_anchor_swd(
        residual_true,
        residual_pred,
        num_projections=num_projections,
        seed=seed,
    )


def _normalize_anchor_torch_inputs(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_true = y_true.to(dtype=torch.float32)
    y_pred = y_pred.to(dtype=torch.float32, device=y_true.device)
    if y_true.ndim != 3 or y_pred.ndim != 3:
        raise ValueError("Expected tensors of shape [num_anchors, num_samples, dim].")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return y_true, y_pred


def _covariance_per_anchor(y: torch.Tensor) -> torch.Tensor:
    centered = y - y.mean(dim=1, keepdim=True)
    denom = max(1, y.shape[1] - 1)
    return centered.transpose(1, 2) @ centered / float(denom)


def _matrix_sqrt_psd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals = torch.clamp(eigvals, min=eps).sqrt()
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)


def conditional_anchor_mean_l2(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> float:
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        y_true = torch.as_tensor(y_true)
        y_pred = torch.as_tensor(y_pred)
    y_true, y_pred = _normalize_anchor_torch_inputs(y_true, y_pred)
    mean_true = y_true.mean(dim=1)
    mean_pred = y_pred.mean(dim=1)
    return float((mean_true - mean_pred).norm(dim=1).mean().item())


def conditional_anchor_cov_fro(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> float:
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        y_true = torch.as_tensor(y_true)
        y_pred = torch.as_tensor(y_pred)
    y_true, y_pred = _normalize_anchor_torch_inputs(y_true, y_pred)
    cov_true = _covariance_per_anchor(y_true)
    cov_pred = _covariance_per_anchor(y_pred)
    return float((cov_true - cov_pred).flatten(start_dim=1).norm(dim=1).mean().item())


def conditional_anchor_gaussian_w2(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        y_true = torch.as_tensor(y_true)
        y_pred = torch.as_tensor(y_pred)
    y_true, y_pred = _normalize_anchor_torch_inputs(y_true, y_pred)
    mean_true = y_true.mean(dim=1)
    mean_pred = y_pred.mean(dim=1)
    cov_true = _covariance_per_anchor(y_true)
    cov_pred = _covariance_per_anchor(y_pred)
    sqrt_true = _matrix_sqrt_psd(cov_true, eps=eps)
    middle = sqrt_true @ cov_pred @ sqrt_true
    sqrt_middle = _matrix_sqrt_psd(middle, eps=eps)
    mean_term = (mean_true - mean_pred).pow(2).sum(dim=1)
    trace_term = torch.diagonal(cov_true + cov_pred - 2.0 * sqrt_middle, dim1=-2, dim2=-1).sum(dim=1)
    w2_sq = torch.clamp(mean_term + trace_term, min=0.0)
    return float(torch.sqrt(w2_sq + eps).mean().item())
