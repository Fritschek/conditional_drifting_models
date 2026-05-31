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

    def __init__(
        self,
        message_dim: int,
        code_dim: int,
        hidden_dim: int | None = None,
        *,
        hidden_layers: int = 2,
        normalization: str = "standardize",
        output_activation: bool = False,
    ):
        super().__init__()
        hidden = int(hidden_dim or message_dim)
        self.message_dim = int(message_dim)
        self.code_dim = int(code_dim)
        self.normalization = str(normalization or "standardize").lower()
        layers: list[nn.Module] = [nn.Linear(self.message_dim, hidden), nn.ELU()]
        for _ in range(max(1, int(hidden_layers)) - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ELU()])
        layers.append(nn.Linear(hidden, self.code_dim))
        if output_activation:
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    @staticmethod
    def power_constraint(codes: torch.Tensor) -> torch.Tensor:
        codes_mean = torch.mean(codes)
        codes_std = torch.std(codes).clamp_min(1e-8)
        return (codes - codes_mean) / codes_std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        codes = self.net(inputs.float())
        if self.normalization in {"none", "identity"}:
            return codes
        if self.normalization == "standardize":
            return self.power_constraint(codes)
        raise ValueError(f"Unsupported encoder normalization: {self.normalization}")


class SymbolicDecoder(nn.Module):
    """Small Muah-style block decoder returning logits over the message alphabet."""

    def __init__(
        self,
        message_dim: int,
        code_dim: int,
        hidden_dim: int | None = None,
        *,
        hidden_layers: int = 2,
        output_activation: bool = False,
    ):
        super().__init__()
        hidden = int(hidden_dim or message_dim)
        self.message_dim = int(message_dim)
        self.code_dim = int(code_dim)
        layers: list[nn.Module] = [nn.Linear(self.code_dim, hidden), nn.ELU()]
        for _ in range(max(1, int(hidden_layers)) - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ELU()])
        layers.append(nn.Linear(hidden, self.message_dim))
        if output_activation:
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs.float())


@dataclass
class SymbolicAEConfig:
    message_dim: int = 16
    code_dim: int = 7
    hidden_dim: int = 16
    hidden_layers: int = 2
    encoder_normalization: str = "standardize"
    encoder_output_activation: bool = False
    decoder_output_activation: bool = False
    code_power: float | None = None
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


def compute_symbol_bit_error_rate(logits: torch.Tensor, labels: torch.Tensor) -> float:
    num_classes = int(logits.shape[1])
    num_bits = max(1, int(math.ceil(math.log2(num_classes))))
    predictions = torch.argmax(logits, dim=1).to(dtype=torch.long)
    labels = labels.to(dtype=torch.long, device=predictions.device)
    bit_positions = torch.arange(num_bits, device=predictions.device, dtype=torch.long)
    prediction_bits = (predictions[:, None] >> bit_positions[None, :]) & 1
    label_bits = (labels[:, None] >> bit_positions[None, :]) & 1
    return float((prediction_bits != label_bits).float().mean().detach().cpu().item())


def _bit_error_tensor(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    num_bits = max(1, int(math.ceil(math.log2(num_classes))))
    bit_positions = torch.arange(num_bits, device=predictions.device, dtype=torch.long)
    prediction_bits = (predictions.long()[:, None] >> bit_positions[None, :]) & 1
    label_bits = (labels.long()[:, None] >> bit_positions[None, :]) & 1
    return (prediction_bits != label_bits).float()


def cross_entropy_information_stats(loss_nats: float, message_dim: int) -> dict[str, float]:
    max_bits = float(math.log2(message_dim))
    cross_entropy_bits = float(loss_nats) / math.log(2.0)
    air_bits = max_bits - cross_entropy_bits
    normalized_air = air_bits / max_bits if max_bits > 0.0 else float("nan")
    return {
        "cross_entropy_bits": float(cross_entropy_bits),
        "air_bits_per_message": float(air_bits),
        "normalized_air": float(normalized_air),
    }


def _decoder_batch_stats(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor | float]:
    message_dim = int(logits.shape[1])
    labels = labels.to(device=logits.device, dtype=torch.long)
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    true_probs = probs.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, num_classes=message_dim).bool()
    other_logits = logits.masked_fill(mask, float("-inf")).amax(dim=1)
    other_probs = probs.masked_fill(mask, float("-inf")).amax(dim=1)
    bit_errors = _bit_error_tensor(predictions, labels, message_dim)
    return {
        "loss": float(F.cross_entropy(logits, labels).item()),
        "ser": float((predictions != labels).float().mean().item()),
        "ber": float(bit_errors.mean().item()),
        "predictions": predictions,
        "probs": probs,
        "log_probs": log_probs,
        "logit_margin": true_logits - other_logits,
        "prob_margin": true_probs - other_probs,
    }


def _confusion_probabilities(predictions: torch.Tensor, labels: torch.Tensor, message_dim: int) -> torch.Tensor:
    confusion = torch.zeros((message_dim, message_dim), device=predictions.device, dtype=torch.float32)
    for message_idx in range(message_dim):
        mask = labels == int(message_idx)
        if torch.any(mask):
            counts = torch.bincount(predictions[mask], minlength=message_dim).to(dtype=torch.float32)
            confusion[message_idx] = counts / counts.sum().clamp_min(1.0)
    return confusion


def _confusion_tv(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    row_tv = 0.5 * torch.abs(left - right).sum(dim=1)
    return float(row_tv.mean().item()), float(row_tv.max().item())


def apply_code_power_constraint(codes: torch.Tensor, code_power: float | None) -> torch.Tensor:
    if code_power is None or float(code_power) <= 0.0:
        return codes
    sample_power = codes.square().sum(dim=-1).mean().clamp_min(1e-12)
    target_power = torch.as_tensor(float(code_power), dtype=codes.dtype, device=codes.device)
    return codes * torch.sqrt(target_power / sample_power)


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
    best_eval_ber: float | None = None
    best_eval_air_bits: float | None = None
    best_eval_normalized_air: float | None = None
    best_epoch: int | None = None
    best_encoder_state = None
    best_decoder_state = None

    for epoch in range(cfg.epochs):
        losses = []
        sers = []
        bers = []
        for _ in range(steps_per_epoch):
            labels = sample_message_labels(cfg.batch_size, cfg.message_dim, device)
            messages = labels_to_one_hot(labels, cfg.message_dim)

            optimizer.zero_grad()
            encoded = apply_code_power_constraint(encoder(messages), cfg.code_power)
            received = train_implant(encoded, ebno_db=ebno_db, rate=rate, device=device)
            logits = decoder(received)
            loss = F.cross_entropy(logits, labels)
            ser = compute_symbol_error_rate(logits, labels)
            ber = compute_symbol_bit_error_rate(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), cfg.grad_clip_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            sers.append(float(ser))
            bers.append(float(ber))

        eval_ser = None
        eval_ber = None
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
            eval_ber = eval_stats["ber"]
            eval_air_bits = eval_stats["air_bits_per_message"]
            eval_normalized_air = eval_stats["normalized_air"]
            if best_eval_ser is None or float(eval_ser) < float(best_eval_ser):
                best_eval_ser = float(eval_ser)
                best_eval_ber = float(eval_ber)
                best_eval_air_bits = float(eval_air_bits)
                best_eval_normalized_air = float(eval_normalized_air)
                best_epoch = int(epoch + 1)
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_decoder_state = copy.deepcopy(decoder.state_dict())
        else:
            eval_air_bits = None
            eval_normalized_air = None
        train_loss = float(np.mean(losses))
        train_info = cross_entropy_information_stats(train_loss, cfg.message_dim)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "train_air_bits_per_message": train_info["air_bits_per_message"],
                "train_normalized_air": train_info["normalized_air"],
                "train_ser": float(np.mean(sers)),
                "train_ber": float(np.mean(bers)),
                "eval_ser": None if eval_ser is None else float(eval_ser),
                "eval_ber": None if eval_ber is None else float(eval_ber),
                "eval_air_bits_per_message": None if eval_air_bits is None else float(eval_air_bits),
                "eval_normalized_air": None if eval_normalized_air is None else float(eval_normalized_air),
            }
        )
        print(
            f"epoch {epoch + 1}/{cfg.epochs}: "
            f"loss={history[-1]['train_loss']:.6e}, "
            f"train_ser={history[-1]['train_ser']:.6e}, "
            f"train_ber={history[-1]['train_ber']:.6e}, "
            f"eval_ser={history[-1]['eval_ser'] if history[-1]['eval_ser'] is not None else 'skipped'}, "
            f"eval_ber={history[-1]['eval_ber'] if history[-1]['eval_ber'] is not None else 'skipped'}",
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
        "best_eval_ber": best_eval_ber,
        "best_eval_air_bits_per_message": best_eval_air_bits,
        "best_eval_normalized_air": best_eval_normalized_air,
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
    bers = []
    for _ in range(num_batches):
        labels = sample_message_labels(cfg.batch_size, cfg.message_dim, device)
        messages = labels_to_one_hot(labels, cfg.message_dim)
        encoded = apply_code_power_constraint(encoder(messages), cfg.code_power)
        received = eval_implant(encoded, ebno_db=ebno_db, rate=rate, device=device)
        logits = decoder(received)
        losses.append(float(F.cross_entropy(logits, labels).item()))
        sers.append(compute_symbol_error_rate(logits, labels))
        bers.append(compute_symbol_bit_error_rate(logits, labels))
    encoder.train()
    decoder.train()
    loss = float(np.mean(losses))
    stats = {"loss": loss, "ser": float(np.mean(sers)), "ber": float(np.mean(bers))}
    stats.update(cross_entropy_information_stats(loss, cfg.message_dim))
    return stats


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
        x = apply_code_power_constraint(encoder(messages), cfg.code_power)
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
    x_anchor = apply_code_power_constraint(encoder(anchor_messages), cfg.code_power)
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


@torch.no_grad()
def evaluate_decoder_channel_metrics(
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    implant,
    reference_implant,
    *,
    cfg: SymbolicAEConfig,
    device: torch.device,
    rate: float,
    ebno_db: float,
    num_projections: int = 128,
    seed: int = 12345,
    samples_per_message: int = 256,
    boundary_prob_margin: float = 0.05,
) -> dict[str, object]:
    """Compare channels after applying the fixed decoder.

    SER/BER depend on decoder decision regions, not raw Euclidean sample
    geometry. This diagnostic uses the decoder as the test-function class:
    decision confusion rows, soft posterior vectors, and decoder margins.
    """
    encoder.eval()
    decoder.eval()
    message_dim = int(cfg.message_dim)
    labels_anchor = torch.arange(message_dim, device=device)
    messages_anchor = labels_to_one_hot(labels_anchor, message_dim)
    x_anchor = apply_code_power_constraint(encoder(messages_anchor), cfg.code_power)
    labels = labels_anchor.repeat_interleave(int(samples_per_message))
    x_rep = x_anchor.repeat_interleave(int(samples_per_message), dim=0)

    y_true_a = reference_implant(x_rep, ebno_db=ebno_db, rate=rate, device=device)
    y_true_b = reference_implant(x_rep, ebno_db=ebno_db, rate=rate, device=device)
    y_pred = implant(x_rep, ebno_db=ebno_db, rate=rate, device=device)

    logits_true = decoder(y_true_a)
    logits_floor = decoder(y_true_b)
    logits_pred = decoder(y_pred)
    stats_true = _decoder_batch_stats(logits_true, labels)
    stats_floor = _decoder_batch_stats(logits_floor, labels)
    stats_pred = _decoder_batch_stats(logits_pred, labels)

    confusion_true = _confusion_probabilities(stats_true["predictions"], labels, message_dim)
    confusion_floor = _confusion_probabilities(stats_floor["predictions"], labels, message_dim)
    confusion_pred = _confusion_probabilities(stats_pred["predictions"], labels, message_dim)
    confusion_tv, confusion_max_tv = _confusion_tv(confusion_true, confusion_pred)
    confusion_floor_tv, confusion_floor_max_tv = _confusion_tv(confusion_true, confusion_floor)
    confusion_row_tv = 0.5 * torch.abs(confusion_true - confusion_pred).sum(dim=1)
    confusion_floor_row_tv = 0.5 * torch.abs(confusion_true - confusion_floor).sum(dim=1)
    per_message_ser_true = 1.0 - torch.diag(confusion_true)
    per_message_ser_pred = 1.0 - torch.diag(confusion_pred)
    per_message_ser_floor = 1.0 - torch.diag(confusion_floor)
    per_message_ser_gap = per_message_ser_pred - per_message_ser_true

    shape = (message_dim, int(samples_per_message), -1)
    prob_true = stats_true["probs"].reshape(shape)
    prob_floor = stats_floor["probs"].reshape(shape)
    prob_pred = stats_pred["probs"].reshape(shape)
    logprob_true = stats_true["log_probs"].reshape(shape)
    logprob_floor = stats_floor["log_probs"].reshape(shape)
    logprob_pred = stats_pred["log_probs"].reshape(shape)
    prob_margin_true = stats_true["prob_margin"].reshape(message_dim, int(samples_per_message), 1)
    prob_margin_floor = stats_floor["prob_margin"].reshape(message_dim, int(samples_per_message), 1)
    prob_margin_pred = stats_pred["prob_margin"].reshape(message_dim, int(samples_per_message), 1)
    logit_margin_true = stats_true["logit_margin"].reshape(message_dim, int(samples_per_message), 1)
    logit_margin_floor = stats_floor["logit_margin"].reshape(message_dim, int(samples_per_message), 1)
    logit_margin_pred = stats_pred["logit_margin"].reshape(message_dim, int(samples_per_message), 1)

    prob_swd = conditional_anchor_swd(prob_true, prob_pred, num_projections=num_projections, seed=seed)
    prob_floor_swd = conditional_anchor_swd(prob_true, prob_floor, num_projections=num_projections, seed=seed + 1_000)
    logprob_swd = conditional_anchor_swd(logprob_true, logprob_pred, num_projections=num_projections, seed=seed + 2_000)
    logprob_floor_swd = conditional_anchor_swd(logprob_true, logprob_floor, num_projections=num_projections, seed=seed + 3_000)
    prob_margin_swd = conditional_anchor_swd(prob_margin_true, prob_margin_pred, num_projections=1, seed=seed + 4_000)
    prob_margin_floor_swd = conditional_anchor_swd(
        prob_margin_true,
        prob_margin_floor,
        num_projections=1,
        seed=seed + 5_000,
    )
    logit_margin_swd = conditional_anchor_swd(logit_margin_true, logit_margin_pred, num_projections=1, seed=seed + 6_000)
    logit_margin_floor_swd = conditional_anchor_swd(
        logit_margin_true,
        logit_margin_floor,
        num_projections=1,
        seed=seed + 7_000,
    )

    boundary = float(boundary_prob_margin)
    boundary_true = (stats_true["prob_margin"].abs() <= boundary).float().mean()
    boundary_floor = (stats_floor["prob_margin"].abs() <= boundary).float().mean()
    boundary_pred = (stats_pred["prob_margin"].abs() <= boundary).float().mean()
    eps = 1e-12
    true_info = cross_entropy_information_stats(float(stats_true["loss"]), message_dim)
    pred_info = cross_entropy_information_stats(float(stats_pred["loss"]), message_dim)
    floor_info = cross_entropy_information_stats(float(stats_floor["loss"]), message_dim)
    encoder.train()
    decoder.train()
    return {
        "decoder_samples_per_message": float(samples_per_message),
        "decoder_boundary_prob_margin": float(boundary),
        "decoder_true_ser": float(stats_true["ser"]),
        "decoder_pred_ser": float(stats_pred["ser"]),
        "decoder_floor_ser": float(stats_floor["ser"]),
        "decoder_ser_gap": float(stats_pred["ser"] - stats_true["ser"]),
        "decoder_abs_ser_gap": float(abs(stats_pred["ser"] - stats_true["ser"])),
        "decoder_floor_abs_ser_gap": float(abs(stats_floor["ser"] - stats_true["ser"])),
        "decoder_true_ber": float(stats_true["ber"]),
        "decoder_pred_ber": float(stats_pred["ber"]),
        "decoder_floor_ber": float(stats_floor["ber"]),
        "decoder_ber_gap": float(stats_pred["ber"] - stats_true["ber"]),
        "decoder_abs_ber_gap": float(abs(stats_pred["ber"] - stats_true["ber"])),
        "decoder_floor_abs_ber_gap": float(abs(stats_floor["ber"] - stats_true["ber"])),
        "decoder_true_ce": float(stats_true["loss"]),
        "decoder_pred_ce": float(stats_pred["loss"]),
        "decoder_floor_ce": float(stats_floor["loss"]),
        "decoder_ce_gap": float(stats_pred["loss"] - stats_true["loss"]),
        "decoder_abs_ce_gap": float(abs(stats_pred["loss"] - stats_true["loss"])),
        "decoder_floor_abs_ce_gap": float(abs(stats_floor["loss"] - stats_true["loss"])),
        "decoder_true_air_bits": float(true_info["air_bits_per_message"]),
        "decoder_pred_air_bits": float(pred_info["air_bits_per_message"]),
        "decoder_floor_air_bits": float(floor_info["air_bits_per_message"]),
        "decoder_air_bits_gap": float(pred_info["air_bits_per_message"] - true_info["air_bits_per_message"]),
        "decoder_confusion_tv": confusion_tv,
        "decoder_confusion_floor_tv": confusion_floor_tv,
        "decoder_confusion_tv_ratio": float(confusion_tv / max(confusion_floor_tv, eps)),
        "decoder_confusion_max_tv": confusion_max_tv,
        "decoder_confusion_floor_max_tv": confusion_floor_max_tv,
        "decoder_confusion_worst_message": int(torch.argmax(confusion_row_tv).item()),
        "decoder_confusion_row_tv": [float(v) for v in confusion_row_tv.detach().cpu().tolist()],
        "decoder_confusion_floor_row_tv": [float(v) for v in confusion_floor_row_tv.detach().cpu().tolist()],
        "decoder_per_message_ser_true": [float(v) for v in per_message_ser_true.detach().cpu().tolist()],
        "decoder_per_message_ser_pred": [float(v) for v in per_message_ser_pred.detach().cpu().tolist()],
        "decoder_per_message_ser_floor": [float(v) for v in per_message_ser_floor.detach().cpu().tolist()],
        "decoder_per_message_ser_gap": [float(v) for v in per_message_ser_gap.detach().cpu().tolist()],
        "decoder_ser_worst_message": int(torch.argmax(torch.abs(per_message_ser_gap)).item()),
        "decoder_prob_swd": float(prob_swd),
        "decoder_prob_floor_swd": float(prob_floor_swd),
        "decoder_prob_swd_ratio": float(prob_swd / max(prob_floor_swd, eps)),
        "decoder_logprob_swd": float(logprob_swd),
        "decoder_logprob_floor_swd": float(logprob_floor_swd),
        "decoder_logprob_swd_ratio": float(logprob_swd / max(logprob_floor_swd, eps)),
        "decoder_prob_margin_swd": float(prob_margin_swd),
        "decoder_prob_margin_floor_swd": float(prob_margin_floor_swd),
        "decoder_prob_margin_swd_ratio": float(prob_margin_swd / max(prob_margin_floor_swd, eps)),
        "decoder_logit_margin_swd": float(logit_margin_swd),
        "decoder_logit_margin_floor_swd": float(logit_margin_floor_swd),
        "decoder_logit_margin_swd_ratio": float(logit_margin_swd / max(logit_margin_floor_swd, eps)),
        "decoder_boundary_mass_true": float(boundary_true.item()),
        "decoder_boundary_mass_pred": float(boundary_pred.item()),
        "decoder_boundary_mass_floor": float(boundary_floor.item()),
        "decoder_boundary_mass_gap": float((boundary_pred - boundary_true).item()),
        "decoder_boundary_mass_abs_gap": float(torch.abs(boundary_pred - boundary_true).item()),
        "decoder_boundary_mass_floor_abs_gap": float(torch.abs(boundary_floor - boundary_true).item()),
    }
