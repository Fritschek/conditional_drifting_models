from __future__ import annotations

import math
from dataclasses import asdict, dataclass
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import (
    conditional_anchor_residual_swd,
    conditional_anchor_swd,
    conditional_anchor_cov_fro,
    conditional_anchor_gaussian_w2,
    conditional_anchor_mean_l2,
    conditional_joint_swd,
    sliced_wasserstein_distance,
)


class SymbolicEncoder(nn.Module):
    """Small Muah-style block encoder for one-hot messages."""

    def __init__(self, message_dim: int, code_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or message_dim)
        self.message_dim = int(message_dim)
        self.code_dim = int(code_dim)
        self.net = nn.Sequential(
            nn.Linear(self.message_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.code_dim),
        )

    @staticmethod
    def power_constraint(codes: torch.Tensor) -> torch.Tensor:
        codes_mean = torch.mean(codes)
        codes_std = torch.std(codes).clamp_min(1e-8)
        return (codes - codes_mean) / codes_std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        codes = self.net(inputs.float())
        return self.power_constraint(codes)


class SymbolicDecoder(nn.Module):
    """Small Muah-style block decoder returning logits over the message alphabet."""

    def __init__(self, message_dim: int, code_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or message_dim)
        self.message_dim = int(message_dim)
        self.code_dim = int(code_dim)
        self.net = nn.Sequential(
            nn.Linear(self.code_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.message_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs.float())


@dataclass
class SymbolicAEConfig:
    message_dim: int = 16
    code_dim: int = 7
    hidden_dim: int = 16
    batch_size: int = 500
    dataset_size: int = 1_000_000
    epochs: int = 10
    learning_rate: float = 1e-3
    grad_clip_norm: float = 1.0
    eval_size: int = 100_000


def sample_message_labels(batch_size: int, message_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, message_dim, (batch_size,), device=device)


def labels_to_one_hot(labels: torch.Tensor, message_dim: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=message_dim).float()


def compute_symbol_error_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions != labels).float().mean().detach().cpu().item())


def train_symbolic_autoencoder(
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    train_implant,
    *,
    cfg: SymbolicAEConfig,
    device: torch.device,
    rate: float,
    ebno_db: float,
    eval_implant=None,
    eval_every: int = 1,
) -> tuple[dict, list[dict[str, float]]]:
    optimizer = torch.optim.NAdam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.learning_rate,
    )
    encoder.train()
    decoder.train()
    steps_per_epoch = max(1, math.ceil(cfg.dataset_size / cfg.batch_size))
    history: list[dict[str, float]] = []
    best_eval_ser: float | None = None
    best_epoch: int | None = None
    best_encoder_state = None
    best_decoder_state = None

    for epoch in range(cfg.epochs):
        losses = []
        sers = []
        for _ in range(steps_per_epoch):
            labels = sample_message_labels(cfg.batch_size, cfg.message_dim, device)
            messages = labels_to_one_hot(labels, cfg.message_dim)

            optimizer.zero_grad()
            encoded = encoder(messages)
            received = train_implant(encoded, ebno_db=ebno_db, rate=rate, device=device)
            logits = decoder(received)
            loss = F.cross_entropy(logits, labels)
            ser = compute_symbol_error_rate(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), cfg.grad_clip_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            sers.append(float(ser))

        eval_ser = None
        if eval_implant is not None and eval_every > 0 and ((epoch + 1) % eval_every == 0 or epoch + 1 == cfg.epochs):
            eval_stats = evaluate_symbolic_autoencoder(
                encoder,
                decoder,
                eval_implant,
                cfg=cfg,
                device=device,
                rate=rate,
                ebno_db=ebno_db,
            )
            eval_ser = eval_stats["ser"]
            if best_eval_ser is None or float(eval_ser) < float(best_eval_ser):
                best_eval_ser = float(eval_ser)
                best_epoch = int(epoch + 1)
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_decoder_state = copy.deepcopy(decoder.state_dict())
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "train_ser": float(np.mean(sers)),
                "eval_ser": None if eval_ser is None else float(eval_ser),
            }
        )
        print(
            f"epoch {epoch + 1}/{cfg.epochs}: "
            f"loss={history[-1]['train_loss']:.6e}, "
            f"train_ser={history[-1]['train_ser']:.6e}, "
            f"eval_ser={history[-1]['eval_ser'] if history[-1]['eval_ser'] is not None else 'skipped'}",
            flush=True,
        )

    summary = {
        "config": asdict(cfg),
        "train_implant": getattr(train_implant, "name", str(train_implant)),
        "eval_implant": getattr(eval_implant, "name", str(eval_implant)) if eval_implant is not None else None,
        "rate": float(rate),
        "ebno_db": float(ebno_db),
        "history": history,
        "best_eval_ser": best_eval_ser,
        "best_epoch": best_epoch,
    }
    if best_encoder_state is not None and best_decoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        decoder.load_state_dict(best_decoder_state)
    return summary, history


@torch.no_grad()
def evaluate_symbolic_autoencoder(
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    eval_implant,
    *,
    cfg: SymbolicAEConfig,
    device: torch.device,
    rate: float,
    ebno_db: float,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    num_batches = max(1, math.ceil(cfg.eval_size / cfg.batch_size))
    losses = []
    sers = []
    for _ in range(num_batches):
        labels = sample_message_labels(cfg.batch_size, cfg.message_dim, device)
        messages = labels_to_one_hot(labels, cfg.message_dim)
        encoded = encoder(messages)
        received = eval_implant(encoded, ebno_db=ebno_db, rate=rate, device=device)
        logits = decoder(received)
        losses.append(float(F.cross_entropy(logits, labels).item()))
        sers.append(compute_symbol_error_rate(logits, labels))
    encoder.train()
    decoder.train()
    return {"loss": float(np.mean(losses)), "ser": float(np.mean(sers))}


@torch.no_grad()
def evaluate_implant_conditional_metrics(
    encoder: SymbolicEncoder,
    implant,
    reference_implant,
    *,
    cfg: SymbolicAEConfig,
    device: torch.device,
    rate: float,
    ebno_db: float,
    num_projections: int = 128,
    seed: int = 12345,
    anchor_samples_per_condition: int = 64,
) -> dict[str, float]:
    encoder.eval()
    num_batches = max(1, math.ceil(cfg.eval_size / cfg.batch_size))
    xs = []
    y_trues = []
    y_preds = []
    for _ in range(num_batches):
        labels = sample_message_labels(cfg.batch_size, cfg.message_dim, device)
        messages = labels_to_one_hot(labels, cfg.message_dim)
        x = encoder(messages)
        y_true = reference_implant(x, ebno_db=ebno_db, rate=rate, device=device)
        y_pred = implant(x, ebno_db=ebno_db, rate=rate, device=device)
        xs.append(x)
        y_trues.append(y_true)
        y_preds.append(y_pred)
    encoder.train()
    x_all = torch.cat(xs, dim=0)
    y_true_all = torch.cat(y_trues, dim=0)
    y_pred_all = torch.cat(y_preds, dim=0)
    residual_true = y_true_all - x_all
    residual_pred = y_pred_all - x_all
    anchor_labels = torch.arange(cfg.message_dim, device=device)
    anchor_messages = labels_to_one_hot(anchor_labels, cfg.message_dim)
    x_anchor = encoder(anchor_messages)
    anchor_true_a = []
    anchor_true_b = []
    anchor_pred = []
    for anchor in x_anchor:
        x_rep = anchor.unsqueeze(0).repeat(int(anchor_samples_per_condition), 1)
        anchor_true_a.append(reference_implant(x_rep, ebno_db=ebno_db, rate=rate, device=device))
        anchor_true_b.append(reference_implant(x_rep, ebno_db=ebno_db, rate=rate, device=device))
        anchor_pred.append(implant(x_rep, ebno_db=ebno_db, rate=rate, device=device))
    y_true_anchor = torch.stack(anchor_true_a, dim=0)
    y_true_anchor_floor = torch.stack(anchor_true_b, dim=0)
    y_pred_anchor = torch.stack(anchor_pred, dim=0)
    anchor_y_swd = conditional_anchor_swd(
        y_true_anchor,
        y_pred_anchor,
        num_projections=num_projections,
        seed=seed,
    )
    anchor_y_floor_swd = conditional_anchor_swd(
        y_true_anchor,
        y_true_anchor_floor,
        num_projections=num_projections,
        seed=seed + 10_000,
    )
    anchor_residual_swd = conditional_anchor_residual_swd(
        x_anchor,
        y_true_anchor,
        y_pred_anchor,
        num_projections=num_projections,
        seed=seed,
    )
    anchor_residual_floor_swd = conditional_anchor_residual_swd(
        x_anchor,
        y_true_anchor,
        y_true_anchor_floor,
        num_projections=num_projections,
        seed=seed + 10_000,
    )
    anchor_mean_l2 = conditional_anchor_mean_l2(y_true_anchor, y_pred_anchor)
    anchor_mean_l2_floor = conditional_anchor_mean_l2(y_true_anchor, y_true_anchor_floor)
    anchor_cov_fro = conditional_anchor_cov_fro(y_true_anchor, y_pred_anchor)
    anchor_cov_fro_floor = conditional_anchor_cov_fro(y_true_anchor, y_true_anchor_floor)
    anchor_gaussian_w2 = conditional_anchor_gaussian_w2(y_true_anchor, y_pred_anchor)
    anchor_gaussian_w2_floor = conditional_anchor_gaussian_w2(y_true_anchor, y_true_anchor_floor)
    eps = 1e-12
    return {
        "joint_xy_swd": conditional_joint_swd(
            x_all,
            y_true_all,
            y_pred_all,
            num_projections=num_projections,
            seed=seed,
        ),
        "anchor_y_swd": anchor_y_swd,
        "anchor_y_floor_swd": anchor_y_floor_swd,
        "anchor_y_excess_swd": max(anchor_y_swd - anchor_y_floor_swd, 0.0),
        "anchor_y_ratio": anchor_y_swd / max(anchor_y_floor_swd, eps),
        "anchor_residual_swd": anchor_residual_swd,
        "anchor_residual_floor_swd": anchor_residual_floor_swd,
        "anchor_residual_excess_swd": max(anchor_residual_swd - anchor_residual_floor_swd, 0.0),
        "anchor_residual_ratio": anchor_residual_swd / max(anchor_residual_floor_swd, eps),
        "anchor_mean_l2": anchor_mean_l2,
        "anchor_mean_l2_floor": anchor_mean_l2_floor,
        "anchor_mean_l2_excess": max(anchor_mean_l2 - anchor_mean_l2_floor, 0.0),
        "anchor_mean_l2_ratio": anchor_mean_l2 / max(anchor_mean_l2_floor, eps),
        "anchor_cov_fro": anchor_cov_fro,
        "anchor_cov_fro_floor": anchor_cov_fro_floor,
        "anchor_cov_fro_excess": max(anchor_cov_fro - anchor_cov_fro_floor, 0.0),
        "anchor_cov_fro_ratio": anchor_cov_fro / max(anchor_cov_fro_floor, eps),
        "anchor_gaussian_w2": anchor_gaussian_w2,
        "anchor_gaussian_w2_floor": anchor_gaussian_w2_floor,
        "anchor_gaussian_w2_excess": max(anchor_gaussian_w2 - anchor_gaussian_w2_floor, 0.0),
        "anchor_gaussian_w2_ratio": anchor_gaussian_w2 / max(anchor_gaussian_w2_floor, eps),
        "y_swd": sliced_wasserstein_distance(
            y_true_all,
            y_pred_all,
            num_projections=num_projections,
            seed=seed,
        ),
        "residual_swd": sliced_wasserstein_distance(
            residual_true,
            residual_pred,
            num_projections=num_projections,
            seed=seed,
        ),
    }
