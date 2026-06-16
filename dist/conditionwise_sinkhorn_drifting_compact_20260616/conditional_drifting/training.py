import random

import numpy as np
import torch

from .config import VARIANTS, build_config
from .losses import conditionwise_sinkhorn_loss, direct_kernel_loss, joint_sinkhorn_loss
from .metrics import anchor_swd, sliced_wasserstein
from .model import ConditionalGenerator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sample_inputs(batch_size, dim, device):
    return torch.randn(batch_size, dim, device=device)


def train_generator(channel, channel_name, variant_name, seed=7, device="auto", budget="default", overrides=None):
    cfg = build_config(channel_name, budget=budget, overrides=overrides)
    variant = VARIANTS[variant_name]
    device = get_device(device)
    set_seed(seed)

    model = ConditionalGenerator(cfg["n"], cfg["n"], cfg["latent_dim"], cfg["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    steps = max(1, cfg["dataset_size"] // cfg["batch_size"])
    history = []

    for epoch in range(cfg["epochs"]):
        losses = []
        drifts = []
        for _ in range(steps):
            if variant["loss"] == "conditionwise_sinkhorn":
                anchors = sample_inputs(cfg["batch_size"], cfg["n"], device)
                kg = cfg["fiber_generated_samples"]
                kp = cfg["fiber_positive_samples"]
                x_model = anchors.repeat_interleave(kg, dim=0)
                x_true = anchors.repeat_interleave(kp, dim=0)
                x_reference = anchors.repeat_interleave(kg, dim=0)
                y_model = model(x_model)
                y_true = channel(x_true, cfg["noise_std"])
                with torch.no_grad():
                    y_reference = model(x_reference)
                loss, drift = conditionwise_sinkhorn_loss(
                    anchors,
                    y_model,
                    y_true,
                    y_reference,
                    kg,
                    kp,
                    kg,
                    iterations=cfg["sinkhorn_iterations"],
                    max_drift_norm=cfg["max_drift_norm"],
                )
            else:
                x = sample_inputs(cfg["batch_size"], cfg["n"], device)
                y_model = model(x)
                y_true = channel(x, cfg["noise_std"])
                if variant["loss"] == "sinkhorn":
                    loss, drift = joint_sinkhorn_loss(
                        x,
                        y_model,
                        y_true,
                        iterations=cfg["sinkhorn_iterations"],
                        max_drift_norm=cfg["max_drift_norm"],
                    )
                else:
                    loss, drift = direct_kernel_loss(
                        y_model,
                        y_true,
                        max_drift_norm=cfg["max_drift_norm"],
                        bandwidth_floor=cfg["kernel_min_bandwidth"],
                    )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            drifts.append(float(drift.norm(dim=1).mean().item()))

        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "drift_norm": float(np.mean(drifts)),
        }
        history.append(row)
        print("epoch %d/%d loss=%.6e drift=%.6e" % (epoch + 1, cfg["epochs"], row["loss"], row["drift_norm"]), flush=True)

    return model, history, cfg


@torch.no_grad()
def evaluate_generator(model, channel, channel_name, seed=12345, device="auto", budget="default", overrides=None):
    cfg = build_config(channel_name, budget=budget, overrides=overrides)
    device = get_device(device)
    set_seed(seed)

    xs = []
    true_outputs = []
    model_outputs = []
    remaining = cfg["eval_size"]
    while remaining > 0:
        batch = min(cfg["batch_size"], remaining)
        x = sample_inputs(batch, cfg["n"], device)
        y_true = channel(x, cfg["noise_std"])
        y_model = model(x)
        xs.append(x)
        true_outputs.append(y_true)
        model_outputs.append(y_model)
        remaining -= batch

    x = torch.cat(xs)
    y_true = torch.cat(true_outputs)
    y_model = torch.cat(model_outputs)
    return {
        "direct_swd": sliced_wasserstein(y_true, y_model, cfg["swd_projections"]),
        "anchor_swd": anchor_swd(x, y_true, y_model, projections=64),
    }
