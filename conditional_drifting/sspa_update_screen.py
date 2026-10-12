"""Read-only checkpoint probes of conditional SSPA training updates."""

import copy
import importlib.util
import os
from pathlib import Path
import subprocess

import torch

from .sspa_solver_replay import clipped_field, residuals

OFFICIAL_COMMIT = "1ccb34baf57f9f8f4e188460ea3d19b431403293"
METHODS = (
    ("production", 4096, 4, "production"),
    ("log4", 4096, 4, "log"),
    ("log16_equal_outputs", 1024, 16, "log"),
    ("log64", 256, 64, "log"),
    ("log4_equal_inputs", 256, 4, "log"),
    ("log16_equal_inputs", 256, 16, "log"),
    ("normalized_fixed_epsilon", 4096, 4, "normalized_fixed"),
    ("normalized_r005", 4096, 4, "normalized_r005"),
    ("log4_zero_first_moment", 4096, 4, "zero_first_moment"),
)


def load_official(repo):
    repo = Path(repo)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT or subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True):
        raise ValueError("Expected the unmodified pinned W-Flow release")
    os.environ["DRIFT_COMPILE"] = "0"
    spec = importlib.util.spec_from_file_location("official_wflow_update_screen", repo / "drift_loss_ot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def coordinate_scale(generated, positive, reference):
    targets = torch.cat((generated, reference, positive), dim=1)
    distance = torch.cdist(generated, targets)
    scale = (distance.square().mean() / (generated.new_tensor(1.) + 1e-8)).sqrt()
    return (scale / generated.shape[-1] ** .5).clamp_min(1e-3)


def clone_adam(model, state, zero_first_moment=False):
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(copy.deepcopy(state))
    if zero_first_moment:
        for values in optimizer.state.values():
            values["exp_avg"].zero_()
    return optimizer


def panel_moments(outputs, truth_mean):
    outputs = outputs.double()
    mean = outputs.mean(dim=1)
    centered = outputs - mean[:, None]
    variance = centered.square().sum(1) / (outputs.shape[1] - 1)
    return dict(variance=variance.mean(), per_anchor_variance=variance.mean(-1),
                mean_mse=(mean - truth_mean.double()).square().mean())


@torch.no_grad()
def official_field(official, g, p, r, epsilon, iterations=10):
    reg = g.new_tensor(epsilon)
    plans = [official._sinkhorn_plan_log_impl(g, target, reg, num_iter=iterations,
             diag_mask=False, use_quadratic_cost=True) for target in (p, r)]
    centers = [torch.bmm(plan, target) / plan.sum(-1, keepdim=True).clamp_min(1e-12)
               for plan, target in zip(plans, (p, r))]
    raw, clipped = clipped_field(g, *centers)
    errors = torch.stack([residuals(plan) for plan in plans])
    return raw, clipped, errors
