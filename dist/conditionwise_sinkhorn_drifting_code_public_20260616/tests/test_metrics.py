import numpy as np
import torch

from conditional_drifting.metrics import sliced_wasserstein_distance, sliced_wasserstein_distance_torch


def test_swd_identity_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 2))
    assert sliced_wasserstein_distance(x, x, num_projections=64, seed=1) == 0.0


def test_swd_increases_for_larger_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256, 2))
    y_small = x + 0.05
    y_large = x + 0.5
    swd_small = sliced_wasserstein_distance(x, y_small, num_projections=64, seed=2)
    swd_large = sliced_wasserstein_distance(x, y_large, num_projections=64, seed=2)
    assert swd_small < swd_large


def test_torch_swd_matches_numpy():
    rng = np.random.default_rng(3)
    x_np = rng.normal(size=(128, 3))
    y_np = x_np + 0.1 * rng.normal(size=(128, 3))
    swd_np = sliced_wasserstein_distance(x_np, y_np, num_projections=32, seed=7)
    swd_torch = sliced_wasserstein_distance_torch(
        torch.from_numpy(x_np),
        torch.from_numpy(y_np),
        num_projections=32,
        seed=7,
    )
    assert abs(swd_np - float(swd_torch.item())) < 1e-5
