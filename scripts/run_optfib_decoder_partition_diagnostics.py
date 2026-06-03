from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import optfib
from conditional_drifting.e2e_implants import AnalyticChannelImplant, load_implant_from_checkpoint
from conditional_drifting.symbolic_ae import (
    SymbolicAEConfig,
    evaluate_decoder_channel_metrics,
    evaluate_implant_conditional_metrics,
    labels_to_one_hot,
)
from conditional_drifting.training import select_device, set_seed
from scripts.run_symbolic_awgn_benchmark import load_symbolic_checkpoint


class CallableImplant:
    respects_ebno = False

    def __init__(self, name: str, fn: Callable[[torch.Tensor, torch.device], torch.Tensor]):
        self.name = name
        self._fn = fn

    def __call__(
        self,
        encoded_data: torch.Tensor,
        *,
        ebno_db: float,
        rate: float,
        device: torch.device | None = None,
        **_: object,
    ) -> torch.Tensor:
        del ebno_db, rate
        target_device = device or encoded_data.device
        original_device = encoded_data.device
        out = self._fn(encoded_data.to(target_device), target_device)
        return out.to(original_device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decoder-aware OptFib channel partition diagnostics.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, default=None)
    parser.add_argument("--true-kstep", type=int, default=50)
    parser.add_argument("--ksteps", type=str, default="1,2,5,10,20,50")
    parser.add_argument("--pn-dbm", type=float, default=-21.3)
    parser.add_argument("--gamma", type=float, default=1.27)
    parser.add_argument("--length", type=float, default=5000.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--ebno-db", type=float, default=5.0)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--num-projections", type=int, default=128)
    parser.add_argument("--samples-per-message", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("results/optfib_decoder_partition_diagnostics"))
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def optfib_deterministic_phase(
    x: torch.Tensor,
    *,
    gamma: float,
    length: float,
) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("OptFib deterministic phase requires even I/Q dimension.")
    pairs = x.reshape(-1, x.shape[-1] // 2, 2)
    xr = pairs[..., 0]
    xi = pairs[..., 1]
    phase = float(gamma) * float(length) * (xr.square() + xi.square())
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    yr = xr * cos_phase - xi * sin_phase
    yi = xi * cos_phase + xr * sin_phase
    return torch.stack((yr, yi), dim=-1).reshape_as(x)


def optfib_phase_values(
    x: torch.Tensor,
    *,
    gamma: float,
    length: float,
) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("OptFib phase values require even I/Q dimension.")
    pairs = x.reshape(-1, x.shape[-1] // 2, 2)
    power = pairs.square().sum(dim=-1)
    return float(gamma) * float(length) * power


def phase_wrap_abs(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def codebook_partition_rows(
    encoder,
    cfg: SymbolicAEConfig,
    *,
    device: torch.device,
    gamma: float,
    length: float,
) -> list[dict[str, object]]:
    labels = torch.arange(cfg.message_dim, device=device)
    messages = labels_to_one_hot(labels, cfg.message_dim)
    code = encoder(messages)
    if cfg.code_power is not None and float(cfg.code_power) > 0.0:
        power = code.square().sum(dim=-1).mean().clamp_min(1e-12)
        code = code * torch.sqrt(torch.as_tensor(float(cfg.code_power), device=device) / power)
    deterministic = optfib_deterministic_phase(code, gamma=gamma, length=length)
    phase = optfib_phase_values(code, gamma=gamma, length=length)
    code_dist = torch.cdist(code, code)
    out_dist = torch.cdist(deterministic, deterministic)
    rows: list[dict[str, object]] = []
    for message_idx in range(cfg.message_dim):
        masked = code_dist[message_idx].clone()
        masked[message_idx] = float("inf")
        nearest = int(torch.argmin(masked).item())
        raw_phase_delta = phase[message_idx] - phase[nearest]
        rows.append(
            {
                "message": message_idx,
                "nearest_message": nearest,
                "code_distance": float(code_dist[message_idx, nearest].item()),
                "deterministic_output_distance": float(out_dist[message_idx, nearest].item()),
                "raw_phase_delta_abs": float(raw_phase_delta.abs().max().item()),
                "wrapped_phase_delta_abs": float(phase_wrap_abs(raw_phase_delta).max().item()),
                "code_norm": float(code[message_idx].norm().item()),
                "deterministic_residual_norm": float((deterministic[message_idx] - code[message_idx]).norm().item()),
            }
        )
    return rows


def scalar_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float, str, bool)) or value is None}


def per_message_rows(candidate: str, metrics: dict[str, object]) -> list[dict[str, object]]:
    row_tv = metrics.get("decoder_confusion_row_tv")
    floor_row_tv = metrics.get("decoder_confusion_floor_row_tv")
    true_ser = metrics.get("decoder_per_message_ser_true")
    pred_ser = metrics.get("decoder_per_message_ser_pred")
    floor_ser = metrics.get("decoder_per_message_ser_floor")
    gap = metrics.get("decoder_per_message_ser_gap")
    if not all(isinstance(values, list) for values in [row_tv, floor_row_tv, true_ser, pred_ser, floor_ser, gap]):
        return []
    rows: list[dict[str, object]] = []
    for idx in range(len(row_tv)):
        rows.append(
            {
                "candidate": candidate,
                "message": idx,
                "confusion_row_tv": row_tv[idx],
                "confusion_floor_row_tv": floor_row_tv[idx],
                "ser_true": true_ser[idx],
                "ser_pred": pred_ser[idx],
                "ser_floor": floor_ser[idx],
                "ser_gap": gap[idx],
            }
        )
    return rows


def build_candidates(args: argparse.Namespace, device: torch.device) -> list[tuple[str, object]]:
    pn_watt = 10.0 ** ((float(args.pn_dbm) - 30.0) / 10.0)
    output_noise_std = math.sqrt(pn_watt / 2.0)
    candidates: list[tuple[str, object]] = []
    candidates.append(
        (
            "deterministic_phase",
            CallableImplant(
                "deterministic_phase",
                lambda x, _device: optfib_deterministic_phase(x, gamma=args.gamma, length=args.length),
            ),
        )
    )
    candidates.append(
        (
            "phase_plus_output_awgn",
            CallableImplant(
                "phase_plus_output_awgn",
                lambda x, _device: optfib_deterministic_phase(x, gamma=args.gamma, length=args.length)
                + output_noise_std * torch.randn_like(x),
            ),
        )
    )
    candidates.append(
        (
            "phase_after_input_awgn",
            CallableImplant(
                "phase_after_input_awgn",
                lambda x, _device: optfib_deterministic_phase(
                    x + output_noise_std * torch.randn_like(x),
                    gamma=args.gamma,
                    length=args.length,
                ),
            ),
        )
    )
    for kstep in parse_int_list(args.ksteps):
        candidates.append(
            (
                f"optfib_k{kstep}",
                CallableImplant(
                    f"optfib_k{kstep}",
                    lambda x, device, k=kstep: optfib(x, 0.0, device, Kstep=int(k), Pn_dBm=float(args.pn_dbm)),
                ),
            )
        )
    if args.learned_checkpoint is not None:
        candidates.append(("learned_checkpoint", load_implant_from_checkpoint(args.learned_checkpoint, device=device)))
    return candidates


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    encoder, decoder, payload = load_symbolic_checkpoint(args.ae_checkpoint, device)
    ckpt_cfg = payload["config"]
    cfg = SymbolicAEConfig(
        message_dim=int(ckpt_cfg["message_dim"]),
        code_dim=int(ckpt_cfg["code_dim"]),
        hidden_dim=int(ckpt_cfg["hidden_dim"]),
        hidden_layers=int(ckpt_cfg.get("hidden_layers", 2)),
        encoder_normalization=str(ckpt_cfg.get("encoder_normalization", "standardize")),
        encoder_output_activation=bool(ckpt_cfg.get("encoder_output_activation", False)),
        decoder_output_activation=bool(ckpt_cfg.get("decoder_output_activation", False)),
        code_power=ckpt_cfg.get("code_power"),
        batch_size=int(args.batch_size),
        dataset_size=0,
        epochs=0,
        eval_size=int(args.eval_size),
    )
    reference = AnalyticChannelImplant(
        "OptFib",
        optfib_params={"Kstep": int(args.true_kstep), "Pn_dBm": float(args.pn_dbm)},
    )

    code_rows = codebook_partition_rows(
        encoder,
        cfg,
        device=device,
        gamma=args.gamma,
        length=args.length,
    )
    write_csv(args.out_dir / "codebook_partition.csv", code_rows)

    summary_rows: list[dict[str, object]] = []
    all_per_message_rows: list[dict[str, object]] = []
    full_metrics: dict[str, object] = {}
    for idx, (candidate_name, implant) in enumerate(build_candidates(args, device)):
        print(f"[optfib-partition] {candidate_name}", flush=True)
        channel_metrics = evaluate_implant_conditional_metrics(
            encoder,
            implant,
            reference,
            cfg=cfg,
            device=device,
            rate=args.rate,
            ebno_db=args.ebno_db,
            num_projections=args.num_projections,
            seed=args.seed + 100 * idx,
            anchor_samples_per_condition=args.samples_per_message,
        )
        decoder_metrics = evaluate_decoder_channel_metrics(
            encoder,
            decoder,
            implant,
            reference,
            cfg=cfg,
            device=device,
            rate=args.rate,
            ebno_db=args.ebno_db,
            num_projections=args.num_projections,
            seed=args.seed + 10_000 + 100 * idx,
            samples_per_message=args.samples_per_message,
        )
        full_metrics[candidate_name] = {
            "channel_metrics": channel_metrics,
            "decoder_metrics": decoder_metrics,
        }
        summary_row = {
            "candidate": candidate_name,
            **{f"channel_{key}": value for key, value in scalar_metrics(channel_metrics).items()},
            **scalar_metrics(decoder_metrics),
        }
        summary_rows.append(summary_row)
        all_per_message_rows.extend(per_message_rows(candidate_name, decoder_metrics))
        write_csv(args.out_dir / "optfib_decoder_partition_summary.csv", summary_rows)
        write_csv(args.out_dir / "optfib_decoder_partition_per_message.csv", all_per_message_rows)
        (args.out_dir / "optfib_decoder_partition_full.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "device": str(device),
                    "codebook_partition": code_rows,
                    "candidates": full_metrics,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "candidate": candidate_name,
                    "channel_anchor_y_ratio": channel_metrics.get("anchor_y_ratio"),
                    "decoder_ser_gap": decoder_metrics.get("decoder_ser_gap"),
                    "decoder_confusion_tv_ratio": decoder_metrics.get("decoder_confusion_tv_ratio"),
                    "decoder_prob_margin_swd_ratio": decoder_metrics.get("decoder_prob_margin_swd_ratio"),
                    "decoder_worst_message": decoder_metrics.get("decoder_ser_worst_message"),
                },
                indent=2,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "summary_csv": str((args.out_dir / "optfib_decoder_partition_summary.csv").resolve()),
                "per_message_csv": str((args.out_dir / "optfib_decoder_partition_per_message.csv").resolve()),
                "full_json": str((args.out_dir / "optfib_decoder_partition_full.json").resolve()),
                "codebook_csv": str((args.out_dir / "codebook_partition.csv").resolve()),
                "num_candidates": len(summary_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
