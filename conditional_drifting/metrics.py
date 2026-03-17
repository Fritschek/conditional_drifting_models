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
