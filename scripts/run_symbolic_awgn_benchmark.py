from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.e2e_implants import AnalyticAWGNImplant, AnalyticChannelImplant, load_implant_from_checkpoint
from conditional_drifting.symbolic_ae import (
    SymbolicAEConfig,
    SymbolicDecoder,
    SymbolicEncoder,
    evaluate_symbolic_autoencoder,
    evaluate_implant_conditional_metrics,
    train_symbolic_autoencoder,
)
from conditional_drifting.training import set_seed
from conditional_drifting.weight_registry import build_artifact_path, register_artifact, resolve_registered_artifact


def build_implant_from_spec(
    *,
    mode: str,
    channel: str,
    checkpoint: str | None,
    name: str | None,
    weights_root: str,
    device: torch.device,
    diffusion_sampler: str,
    ddim_steps: int,
):
    if mode == "analytic_awgn":
        return AnalyticAWGNImplant()
    if mode == "analytic_channel":
        return AnalyticChannelImplant(channel)
    if mode == "checkpoint":
        implant_checkpoint = checkpoint
        if name:
            implant_checkpoint = resolve_registered_artifact(name, weights_root)["checkpoint"]
        if not implant_checkpoint:
            raise ValueError("Provide a checkpoint path or registered artifact name for checkpoint implants.")
        return load_implant_from_checkpoint(
            implant_checkpoint,
            device=device,
            diffusion_sampler=diffusion_sampler,
            ddim_steps=ddim_steps,
        )
    raise ValueError(f"Unsupported implant mode: {mode}")


def save_symbolic_checkpoint(
    path: Path,
    *,
    encoder: SymbolicEncoder,
    decoder: SymbolicDecoder,
    config: SymbolicAEConfig,
    summary: dict,
) -> None:
    payload = {
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "config": {
            "message_dim": config.message_dim,
            "code_dim": config.code_dim,
            "hidden_dim": config.hidden_dim,
        },
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_symbolic_checkpoint(path: str | Path, device: torch.device) -> tuple[SymbolicEncoder, SymbolicDecoder, dict]:
    payload = torch.load(Path(path), map_location=device)
    cfg = payload["config"]
    encoder = SymbolicEncoder(
        message_dim=int(cfg["message_dim"]),
        code_dim=int(cfg["code_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    decoder = SymbolicDecoder(
        message_dim=int(cfg["message_dim"]),
        code_dim=int(cfg["code_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    encoder.load_state_dict(payload["encoder_state"])
    decoder.load_state_dict(payload["decoder_state"])
    return encoder, decoder, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate a symbolic block autoencoder with pluggable channel implants.")
    parser.add_argument("--train-implant", type=str, default="analytic_awgn", choices=["analytic_awgn", "analytic_channel", "checkpoint"])
    parser.add_argument("--train-implant-checkpoint", type=str, default=None)
    parser.add_argument("--train-implant-name", type=str, default=None)
    parser.add_argument("--eval-implant", type=str, default="analytic_awgn", choices=["analytic_awgn", "analytic_channel", "checkpoint"])
    parser.add_argument("--eval-implant-checkpoint", type=str, default=None)
    parser.add_argument("--eval-implant-name", type=str, default=None)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "OptFib"])
    parser.add_argument("--diffusion-sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--code-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--rate", type=float, default=4.0 / 7.0)
    parser.add_argument("--ebno-db", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dataset-size", type=int, default=1_000_000)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--save-ae-name", type=str, default=None)
    parser.add_argument("--load-ae-checkpoint", type=str, default=None)
    parser.add_argument("--load-ae-name", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/symbolic_awgn_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)

    if args.load_ae_name:
        ae_checkpoint = resolve_registered_artifact(args.load_ae_name, args.weights_root)["checkpoint"]
    else:
        ae_checkpoint = args.load_ae_checkpoint

    if ae_checkpoint:
        encoder, decoder, checkpoint_payload = load_symbolic_checkpoint(ae_checkpoint, device)
        ckpt_cfg = checkpoint_payload["config"]
        config = SymbolicAEConfig(
            message_dim=int(ckpt_cfg["message_dim"]),
            code_dim=int(ckpt_cfg["code_dim"]),
            hidden_dim=int(ckpt_cfg["hidden_dim"]),
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            grad_clip_norm=args.grad_clip_norm,
            eval_size=args.eval_size,
        )
    else:
        config = SymbolicAEConfig(
            message_dim=args.message_dim,
            code_dim=args.code_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            grad_clip_norm=args.grad_clip_norm,
            eval_size=args.eval_size,
        )
        encoder = SymbolicEncoder(config.message_dim, config.code_dim, config.hidden_dim).to(device)
        decoder = SymbolicDecoder(config.message_dim, config.code_dim, config.hidden_dim).to(device)

    train_implant = build_implant_from_spec(
        mode=args.train_implant,
        channel=args.channel,
        checkpoint=args.train_implant_checkpoint,
        name=args.train_implant_name,
        weights_root=args.weights_root,
        device=device,
        diffusion_sampler=args.diffusion_sampler,
        ddim_steps=args.ddim_steps,
    )
    eval_implant = build_implant_from_spec(
        mode=args.eval_implant,
        channel=args.channel,
        checkpoint=args.eval_implant_checkpoint,
        name=args.eval_implant_name,
        weights_root=args.weights_root,
        device=device,
        diffusion_sampler=args.diffusion_sampler,
        ddim_steps=args.ddim_steps,
    )
    analytic_reference = AnalyticChannelImplant(args.channel)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.perf_counter()
    if args.epochs > 0:
        summary, history = train_symbolic_autoencoder(
            encoder,
            decoder,
            train_implant,
            cfg=config,
            device=device,
            rate=args.rate,
            ebno_db=args.ebno_db,
            eval_implant=eval_implant,
            eval_every=args.eval_every,
        )
    else:
        history = []
        summary = {
            "config": config.__dict__,
            "train_implant": getattr(train_implant, "name", args.train_implant),
            "eval_implant": getattr(eval_implant, "name", args.eval_implant),
            "channel": args.channel,
            "rate": float(args.rate),
            "ebno_db": float(args.ebno_db),
            "history": [],
        }
    summary["train_seconds"] = time.perf_counter() - train_start

    final_eval = evaluate_symbolic_autoencoder(
        encoder,
        decoder,
        eval_implant,
        cfg=config,
        device=device,
        rate=args.rate,
        ebno_db=args.ebno_db,
    )
    summary["final_eval"] = final_eval
    summary["train_implant_channel_metrics_vs_analytic"] = evaluate_implant_conditional_metrics(
        encoder,
        train_implant,
        analytic_reference,
        cfg=config,
        device=device,
        rate=args.rate,
        ebno_db=args.ebno_db,
        num_projections=128,
        seed=args.seed,
    )
    summary["eval_implant_channel_metrics_vs_analytic"] = evaluate_implant_conditional_metrics(
        encoder,
        eval_implant,
        analytic_reference,
        cfg=config,
        device=device,
        rate=args.rate,
        ebno_db=args.ebno_db,
        num_projections=128,
        seed=args.seed,
    )
    summary["seed"] = args.seed
    summary["device"] = str(device)
    summary["channel"] = args.channel
    summary["train_implant_source"] = args.train_implant_name or args.train_implant_checkpoint
    summary["eval_implant_source"] = args.eval_implant_name or args.eval_implant_checkpoint

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    if args.save_ae_name:
        checkpoint_path = build_artifact_path(
            root=args.weights_root,
            category="symbolic_autoencoders",
            group=args.channel.lower(),
            name=args.save_ae_name,
            suffix=".pt",
        )
        save_symbolic_checkpoint(
            checkpoint_path,
            encoder=encoder,
            decoder=decoder,
            config=config,
            summary=summary,
        )
        register_artifact(
            name=args.save_ae_name,
            category="symbolic_autoencoder",
            checkpoint_path=checkpoint_path,
            metadata_path=summary_path,
            root=args.weights_root,
            extra={
                "channel": args.channel,
                "message_dim": config.message_dim,
                "code_dim": config.code_dim,
                "seed": args.seed,
            },
        )
        summary["checkpoint"] = str(checkpoint_path.resolve())
        summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps({"out_dir": str(out_dir.resolve()), "summary": str(summary_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
