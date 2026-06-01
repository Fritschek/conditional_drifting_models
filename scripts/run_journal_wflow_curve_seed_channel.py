from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.e2e_implants import AnalyticChannelImplant
from conditional_drifting.symbolic_ae import SymbolicAEConfig, evaluate_symbolic_autoencoder
from conditional_drifting.training import set_seed
from scripts.run_symbolic_awgn_benchmark import load_symbolic_checkpoint


CHANNEL_SETTINGS = {
    "AWGN": {"message_dim": 16, "code_dim": 7, "rate": 4.0 / 7.0, "train_ebno_db": 5.0, "ebno_values": "0,1,2,3,4,5,6,7,8"},
    "Rayleigh": {"message_dim": 16, "code_dim": 7, "rate": 4.0 / 7.0, "train_ebno_db": 12.0, "ebno_values": "6,8,10,12,14,16,18"},
    "SSPA": {"message_dim": 64, "code_dim": 8, "rate": 6.0 / 8.0, "train_ebno_db": 8.0, "ebno_values": "1,2,3,4,5,6,7,8,9,10,11"},
    "TDL": {"message_dim": 16, "code_dim": 8, "rate": 4.0 / 8.0, "train_ebno_db": 10.0, "ebno_values": "2,4,6,8,10,12,14"},
    # The current OptFib analytic channel uses its own P_n parameter and does not
    # vary with Eb/N0 unless the channel implementation is changed to use noise_std.
    "OptFib": {"message_dim": 16, "code_dim": 2, "rate": 1.0, "train_ebno_db": 5.0, "ebno_values": "5"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train symbolic AEs for one seed/channel and evaluate BER/SER curves.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--channel", type=str, required=True, choices=sorted(CHANNEL_SETTINGS))
    parser.add_argument("--variants", type=str, default="analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn")
    parser.add_argument("--wflow-suite-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--message-dim", type=int, default=0, help="Use <=0 for the channel default.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--encoder-normalization", type=str, default="standardize", choices=["standardize", "none"])
    parser.add_argument("--encoder-output-activation", action="store_true")
    parser.add_argument("--decoder-output-activation", action="store_true")
    parser.add_argument("--ae-dataset-size", type=int, default=1_000_000)
    parser.add_argument("--ae-batch-size", type=int, default=500)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--code-power", type=float, default=0.0)
    parser.add_argument("--optfib-input-power-dbm", type=float, default=None)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-ebno-db", type=float, default=None)
    parser.add_argument("--ebno-values", type=str, default="")
    parser.add_argument("--diffusion-ddim-steps", type=int, default=100)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def checkpoint_path(wflow_suite_dir: Path, variant: str, channel: str, seed: int) -> Path:
    return (
        wflow_suite_dir
        / variant
        / f"seed{seed}"
        / "checkpoints"
        / f"enhanced_direct_{channel.lower()}_seed{seed}.pt"
    )


def build_train_command(args: argparse.Namespace, variant: str, out_dir: Path, ae_checkpoint: Path, train_ebno_db: float) -> list[str]:
    settings = CHANNEL_SETTINGS[args.channel]
    message_dim = int(args.message_dim if args.message_dim > 0 else settings["message_dim"])
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_symbolic_awgn_benchmark.py",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--channel",
        args.channel,
        "--message-dim",
        str(message_dim),
        "--code-dim",
        str(settings["code_dim"]),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--encoder-normalization",
        args.encoder_normalization,
        "--rate",
        str(settings["rate"]),
        "--ebno-db",
        str(train_ebno_db),
        "--batch-size",
        str(args.ae_batch_size),
        "--dataset-size",
        str(args.ae_dataset_size),
        "--eval-size",
        str(args.eval_size),
        "--epochs",
        str(args.ae_epochs),
        "--learning-rate",
        str(args.ae_learning_rate),
        "--code-power",
        str(args.code_power),
        "--eval-every",
        str(args.eval_every),
        "--diffusion-sampler",
        "ddim",
        "--ddim-steps",
        str(args.diffusion_ddim_steps),
        "--save-ae-checkpoint",
        str(ae_checkpoint),
        "--out-dir",
        str(out_dir),
    ]
    if args.optfib_input_power_dbm is not None:
        cmd.extend(["--optfib-input-power-dbm", str(args.optfib_input_power_dbm)])
    if args.encoder_output_activation:
        cmd.append("--encoder-output-activation")
    if args.decoder_output_activation:
        cmd.append("--decoder-output-activation")
    if variant == "analytic":
        cmd.extend(["--train-implant", "analytic_channel", "--eval-implant", "analytic_channel"])
        return cmd

    ckpt = checkpoint_path(args.wflow_suite_dir, variant, args.channel, args.seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing W-Flow checkpoint: {ckpt}")
    cmd.extend(
        [
            "--train-implant",
            "checkpoint",
            "--train-implant-checkpoint",
            str(ckpt),
            "--eval-implant",
            "analytic_channel",
        ]
    )
    return cmd


def run_logged_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def evaluate_curve(
    ae_checkpoint: Path,
    *,
    channel: str,
    rate: float,
    ebno_values: list[float],
    eval_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    encoder, decoder, payload = load_symbolic_checkpoint(ae_checkpoint, device)
    cfg_dict = payload["config"]
    cfg = SymbolicAEConfig(
        message_dim=int(cfg_dict["message_dim"]),
        code_dim=int(cfg_dict["code_dim"]),
        hidden_dim=int(cfg_dict["hidden_dim"]),
        hidden_layers=int(cfg_dict.get("hidden_layers", 2)),
        encoder_normalization=str(cfg_dict.get("encoder_normalization", "standardize")),
        encoder_output_activation=bool(cfg_dict.get("encoder_output_activation", False)),
        decoder_output_activation=bool(cfg_dict.get("decoder_output_activation", False)),
        code_power=cfg_dict.get("code_power"),
        batch_size=int(batch_size),
        dataset_size=int(eval_size),
        epochs=0,
        eval_size=int(eval_size),
    )
    eval_implant = AnalyticChannelImplant(channel)
    curve = []
    for ebno_db in ebno_values:
        set_seed(seed + int(round(100.0 * ebno_db)))
        stats = evaluate_symbolic_autoencoder(
            encoder,
            decoder,
            eval_implant,
            cfg=cfg,
            device=device,
            rate=rate,
            ebno_db=float(ebno_db),
        )
        curve.append(
            {
                "ebno_db": float(ebno_db),
                "ser": float(stats["ser"]),
                "ber": float(stats["ber"]),
                "loss": float(stats["loss"]),
                "cross_entropy_bits": float(stats["cross_entropy_bits"]),
                "air_bits_per_message": float(stats["air_bits_per_message"]),
                "normalized_air": float(stats["normalized_air"]),
            }
        )
    return curve


def main() -> None:
    args = parse_args()
    settings = CHANNEL_SETTINGS[args.channel]
    message_dim = int(args.message_dim if args.message_dim > 0 else settings["message_dim"])
    variants = parse_csv_list(args.variants)
    train_ebno_db = float(settings["train_ebno_db"] if args.train_ebno_db is None else args.train_ebno_db)
    ebno_values = parse_float_list(args.ebno_values or str(settings["ebno_values"]))
    args.message_dim = message_dim
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.suite_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    runs = []
    for variant in variants:
        out_dir = args.suite_dir / args.channel / f"seed{args.seed}" / variant
        ae_checkpoint = out_dir / "symbolic_autoencoder.pt"
        summary_path = out_dir / "summary.json"
        train_log = log_dir / f"curve_train_{args.channel.lower()}_{variant}_seed{args.seed}.log"
        if args.force_retrain or not ae_checkpoint.exists() or not summary_path.exists():
            cmd = build_train_command(args, variant, out_dir, ae_checkpoint, train_ebno_db)
            print(f"[curve] training seed={args.seed} channel={args.channel} variant={variant}", flush=True)
            run_logged_command(cmd, train_log)
        else:
            print(f"[curve] reusing {ae_checkpoint}", flush=True)

        print(f"[curve] evaluating seed={args.seed} channel={args.channel} variant={variant}", flush=True)
        curve = evaluate_curve(
            ae_checkpoint,
            channel=args.channel,
            rate=float(settings["rate"]),
            ebno_values=ebno_values,
            eval_size=args.eval_size,
            batch_size=args.eval_batch_size,
            seed=args.seed,
            device=device,
        )
        runs.append(
            {
                "seed": int(args.seed),
                "channel": args.channel,
                "variant": variant,
                "train_ebno_db": train_ebno_db,
                "checkpoint": str(ae_checkpoint),
                "summary": str(summary_path),
                "train_log": str(train_log),
                "curve": curve,
            }
        )

    payload = {
        "seed": int(args.seed),
        "channel": args.channel,
        "variants": variants,
        "wflow_suite_dir": str(args.wflow_suite_dir),
        "train_ebno_db": train_ebno_db,
        "ebno_values": ebno_values,
        "elapsed_seconds": time.perf_counter() - start,
        "runs": runs,
    }
    result_path = args.suite_dir / f"curve_{args.channel.lower()}_seed{args.seed}_result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"seed": args.seed, "channel": args.channel, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
