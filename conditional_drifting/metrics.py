from __future__ import annotations

import numpy as np


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_projections: int = 256,
    seed: int = 12345,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension.")

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
