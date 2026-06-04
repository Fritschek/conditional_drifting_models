from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.e2e_implants import AnalyticChannelImplant
from conditional_drifting.symbolic_ae import (
    SymbolicAEConfig,
    SymbolicDecoder,
    SymbolicEncoder,
    apply_code_power_constraint,
    compute_symbol_bit_error_rate,
    compute_symbol_error_rate,
    evaluate_symbolic_autoencoder,
    labels_to_one_hot,
    sample_message_labels,
)
from conditional_drifting.training import set_seed
from scripts.run_journal_wflow_curve_seed_channel import (
    CHANNEL_SETTINGS,
    WFLOW_VARIANTS,
    WGAN_VARIANTS,
    baseline_checkpoint_path,
    diffusion_variant_options,
    is_diffusion_variant,
    parse_csv_list,
    parse_suite_dir_map,
    wflow_checkpoint_path,
)
from scripts.run_symbolic_awgn_benchmark import build_implant_from_spec, save_symbolic_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal wall-clock symbolic AE training through channel implants.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--channel", type=str, required=True, choices=sorted(CHANNEL_SETTINGS))
    parser.add_argument("--variants", type=str, default="analytic,fiber_sinkhorn,wgan,diffusion_ddim100")
    parser.add_argument("--wflow-suite-dir", type=Path, default=None)
    parser.add_argument("--wflow-suite-dir-map", type=str, default="")
    parser.add_argument("--baseline-suite-dir", type=Path, default=None)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--train-seconds", type=float, default=1800.0)
    parser.add_argument("--message-dim", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--encoder-normalization", type=str, default="standardize", choices=["standardize", "none"])
    parser.add_argument("--encoder-output-activation", action="store_true")
    parser.add_argument("--decoder-output-activation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--code-power", type=float, default=0.0)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--train-ebno-db", type=float, default=None)
    parser.add_argument("--diffusion-ddim-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def sync_if_cuda(device: torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def sampler_steps_for_variant(variant: str, default_ddim_steps: int) -> int:
    normalized = variant.lower()
    if is_diffusion_variant(normalized):
        sampler, steps = diffusion_variant_options(normalized, default_ddim_steps)
        if sampler == "ddpm":
            return 100
        return int(default_ddim_steps if steps is None else steps)
    return 1


def build_train_implant(args: argparse.Namespace, variant: str, device: torch.device):
    if variant == "analytic":
        return build_implant_from_spec(
            mode="analytic_channel",
            channel=args.channel,
            checkpoint=None,
            name=None,
            weights_root=str(ROOT / "weights"),
            device=device,
            diffusion_sampler="ddim",
            ddim_steps=args.diffusion_ddim_steps,
        )

    if variant in WFLOW_VARIANTS:
        ckpt = wflow_checkpoint_path(args, variant, args.channel, args.seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing W-Flow checkpoint: {ckpt}")
        return build_implant_from_spec(
            mode="checkpoint",
            channel=args.channel,
            checkpoint=str(ckpt),
            name=None,
            weights_root=str(ROOT / "weights"),
            device=device,
            diffusion_sampler="ddim",
            ddim_steps=args.diffusion_ddim_steps,
        )

    if variant.lower() in WGAN_VARIANTS or is_diffusion_variant(variant):
        ckpt = baseline_checkpoint_path(args, variant, args.channel, args.seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing baseline checkpoint: {ckpt}")
        sampler = "ddim"
        ddim_steps = args.diffusion_ddim_steps
        if is_diffusion_variant(variant):
            sampler, parsed_steps = diffusion_variant_options(variant, args.diffusion_ddim_steps)
            if parsed_steps is not None:
                ddim_steps = int(parsed_steps)
        return build_implant_from_spec(
            mode="checkpoint",
            channel=args.channel,
            checkpoint=str(ckpt),
            name=None,
            weights_root=str(ROOT / "weights"),
            device=device,
            diffusion_sampler=sampler,
            ddim_steps=ddim_steps,
        )

    raise ValueError(f"Unsupported equal-wall-clock variant: {variant}")


def make_config(args: argparse.Namespace, settings: dict[str, float | int | str]) -> SymbolicAEConfig:
    message_dim = int(args.message_dim if args.message_dim > 0 else settings["message_dim"])
    return SymbolicAEConfig(
        message_dim=message_dim,
        code_dim=int(settings["code_dim"]),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        encoder_normalization=str(args.encoder_normalization),
        encoder_output_activation=bool(args.encoder_output_activation),
        decoder_output_activation=bool(args.decoder_output_activation),
        code_power=float(args.code_power) if float(args.code_power) > 0.0 else None,
        batch_size=int(args.batch_size),
        dataset_size=int(args.batch_size),
        epochs=0,
        learning_rate=float(args.learning_rate),
        grad_clip_norm=float(args.grad_clip_norm),
        eval_size=int(args.eval_size),
    )


def train_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    settings: dict[str, float | int | str],
    train_ebno_db: float,
    device: torch.device,
) -> dict[str, object]:
    set_seed(int(args.seed))
    cfg = make_config(args, settings)
    encoder = SymbolicEncoder(
        cfg.message_dim,
        cfg.code_dim,
        cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        normalization=cfg.encoder_normalization,
        output_activation=cfg.encoder_output_activation,
    ).to(device)
    decoder = SymbolicDecoder(
        cfg.message_dim,
        cfg.code_dim,
        cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        output_activation=cfg.decoder_output_activation,
    ).to(device)
    train_implant = build_train_implant(args, variant, device)
    eval_implant = AnalyticChannelImplant(args.channel)
    optimizer = torch.optim.NAdam(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.learning_rate)

    losses: list[float] = []
    train_sers: list[float] = []
    train_bers: list[float] = []
    history: list[dict[str, float]] = []
    sync_if_cuda(device)
    start = time.perf_counter()
    updates = 0
    while True:
        elapsed = time.perf_counter() - start
        if updates > 0 and elapsed >= float(args.train_seconds):
            break
        labels = sample_message_labels(cfg.batch_size, cfg.message_dim, device)
        messages = labels_to_one_hot(labels, cfg.message_dim)
        optimizer.zero_grad(set_to_none=True)
        encoded = apply_code_power_constraint(encoder(messages), cfg.code_power)
        received = train_implant(encoded, ebno_db=train_ebno_db, rate=float(settings["rate"]), device=device)
        logits = decoder(received)
        loss = F.cross_entropy(logits, labels)
        ser = compute_symbol_error_rate(logits, labels)
        ber = compute_symbol_bit_error_rate(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), cfg.grad_clip_norm)
        optimizer.step()
        updates += 1
        losses.append(float(loss.item()))
        train_sers.append(float(ser))
        train_bers.append(float(ber))
        if args.log_every > 0 and (updates % int(args.log_every) == 0):
            sync_if_cuda(device)
            history.append(
                {
                    "updates": float(updates),
                    "elapsed_seconds": float(time.perf_counter() - start),
                    "train_loss": float(np.mean(losses[-int(args.log_every) :])),
                    "train_ser": float(np.mean(train_sers[-int(args.log_every) :])),
                    "train_ber": float(np.mean(train_bers[-int(args.log_every) :])),
                }
            )
            print(
                f"[wallclock] seed={args.seed} channel={args.channel} variant={variant} "
                f"updates={updates} elapsed={history[-1]['elapsed_seconds']:.1f}s "
                f"ser={history[-1]['train_ser']:.4e}",
                flush=True,
            )

    sync_if_cuda(device)
    train_seconds = time.perf_counter() - start
    eval_cfg = SymbolicAEConfig(**{**cfg.__dict__, "batch_size": int(args.eval_batch_size), "eval_size": int(args.eval_size)})
    eval_stats = evaluate_symbolic_autoencoder(
        encoder,
        decoder,
        eval_implant,
        cfg=eval_cfg,
        device=device,
        rate=float(settings["rate"]),
        ebno_db=float(train_ebno_db),
    )
    out_dir = args.suite_dir / args.channel / f"seed{args.seed}" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "symbolic_autoencoder.pt"
    if args.save_checkpoints:
        save_symbolic_checkpoint(checkpoint_path, encoder=encoder, decoder=decoder, config=cfg, summary=eval_stats)

    sampler_steps = sampler_steps_for_variant(variant, int(args.diffusion_ddim_steps))
    result = {
        "seed": int(args.seed),
        "channel": args.channel,
        "variant": variant,
        "train_implant": getattr(train_implant, "name", str(train_implant)),
        "train_seconds_budget": float(args.train_seconds),
        "train_seconds": float(train_seconds),
        "optimizer_updates": int(updates),
        "channel_calls": int(updates),
        "channel_samples": int(updates * cfg.batch_size),
        "sampler_steps_per_sample": int(sampler_steps),
        "effective_sampler_steps": int(updates * cfg.batch_size * sampler_steps),
        "batch_size": int(cfg.batch_size),
        "train_ebno_db": float(train_ebno_db),
        "eval_ser": float(eval_stats["ser"]),
        "eval_ber": float(eval_stats["ber"]),
        "eval_loss": float(eval_stats["loss"]),
        "eval_air_bits_per_message": float(eval_stats["air_bits_per_message"]),
        "eval_normalized_air": float(eval_stats["normalized_air"]),
        "history": history,
        "checkpoint": str(checkpoint_path) if args.save_checkpoints else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    args._wflow_suite_map = parse_suite_dir_map(args.wflow_suite_dir_map)
    settings = CHANNEL_SETTINGS[args.channel]
    train_ebno_db = float(settings["train_ebno_db"] if args.train_ebno_db is None else args.train_ebno_db)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_csv_list(args.variants)
    start = time.perf_counter()
    runs = []
    for variant in variants:
        print(f"[wallclock] running seed={args.seed} channel={args.channel} variant={variant}", flush=True)
        runs.append(train_variant(args, variant=variant, settings=settings, train_ebno_db=train_ebno_db, device=device))
    payload = {
        "seed": int(args.seed),
        "channel": args.channel,
        "variants": variants,
        "wflow_suite_dir": str(args.wflow_suite_dir) if args.wflow_suite_dir is not None else None,
        "wflow_suite_dir_map": {key: str(value) for key, value in args._wflow_suite_map.items()},
        "baseline_suite_dir": str(args.baseline_suite_dir) if args.baseline_suite_dir is not None else None,
        "train_ebno_db": train_ebno_db,
        "train_seconds_budget": float(args.train_seconds),
        "elapsed_seconds": float(time.perf_counter() - start),
        "runs": runs,
    }
    result_path = args.suite_dir / f"wallclock_{args.channel.lower()}_seed{args.seed}_result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"seed": args.seed, "channel": args.channel, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
