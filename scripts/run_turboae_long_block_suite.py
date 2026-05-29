from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED_TURBO_ROOT = ROOT / "external" / "turbo_mingru_decoder"
DEFAULT_TURBO_ROOT = VENDORED_TURBO_ROOT if VENDORED_TURBO_ROOT.exists() else ROOT.parent / "turbo_mingru_decoder"


def parse_int_csv(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_mode_csv(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    unsupported = [mode for mode in modes if mode not in {"analytic", "checkpoint"}]
    if unsupported:
        raise ValueError(f"Unsupported long-block modes: {unsupported}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TurboAE long-block analytic/surrogate channel checks.")
    parser.add_argument("--lengths", type=str, default="64,256,1000")
    parser.add_argument("--modes", type=str, default="analytic,checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", help="n=2 channel implant checkpoint for checkpoint mode.")
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "TDL", "OptFib"])
    parser.add_argument("--model-type", type=str, default="cnn_turbo")
    parser.add_argument("--turbo-root", type=str, default=str(DEFAULT_TURBO_ROOT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--training-regime", type=str, default="overnight_alternate")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dec-bs-fac", type=int, default=1)
    parser.add_argument("--enc-micro-batch-size", type=int, default=0)
    parser.add_argument("--dec-micro-batch-size", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--eval-num-blocks", type=int, default=50_000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--num-iterations", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ebno-db", type=float, default=4.0)
    parser.add_argument("--decoder-ebno-offset-low", type=float, default=-3.5)
    parser.add_argument("--decoder-ebno-offset-high", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=0.5)
    parser.add_argument("--enc-num-layers", type=int, default=2)
    parser.add_argument("--dec-num-layers", type=int, default=5)
    parser.add_argument("--num-iteration", type=int, default=5)
    parser.add_argument("--num-iter-ft-cnn", type=int, default=10)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--enable-gpu-warmup", action="store_true")
    parser.add_argument("--lr-plateau-patience-evals", type=int, default=0)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-max-reductions", type=int, default=0)
    parser.add_argument("--lr-plateau-min-lr", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, *, mode: str, length: int, out_dir: Path) -> list[str]:
    channel_length = int(round(float(length) / float(args.rate)))
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_e2e_channel_implant_benchmark.py",
        "--turbo-root",
        args.turbo_root,
        "--model-type",
        args.model_type,
        "--channel",
        args.channel,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--dec-bs-fac",
        str(args.dec_bs_fac),
        "--enc-micro-batch-size",
        str(args.enc_micro_batch_size),
        "--dec-micro-batch-size",
        str(args.dec_micro_batch_size),
        "--sample-size",
        str(args.sample_size),
        "--eval-num-blocks",
        str(args.eval_num_blocks),
        "--sequence-length",
        str(length),
        "--channel-length",
        str(channel_length),
        "--training-regime",
        args.training_regime,
        "--epochs",
        str(args.epochs),
        "--num-iterations",
        str(args.num_iterations),
        "--log-every",
        str(args.log_every),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--ebno-db",
        str(args.ebno_db),
        "--decoder-ebno-offset-low",
        str(args.decoder_ebno_offset_low),
        "--decoder-ebno-offset-high",
        str(args.decoder_ebno_offset_high),
        "--enc-num-layers",
        str(args.enc_num_layers),
        "--dec-num-layers",
        str(args.dec_num_layers),
        "--num-iteration",
        str(args.num_iteration),
        "--num-iter-ft-cnn",
        str(args.num_iter_ft_cnn),
        "--eval-batches",
        str(args.eval_batches),
        "--eval-every",
        str(args.eval_every),
        "--save-every",
        str(args.save_every),
        "--lr-plateau-patience-evals",
        str(args.lr_plateau_patience_evals),
        "--lr-plateau-factor",
        str(args.lr_plateau_factor),
        "--lr-plateau-max-reductions",
        str(args.lr_plateau_max_reductions),
        "--lr-plateau-min-lr",
        str(args.lr_plateau_min_lr),
        "--out-dir",
        str(out_dir),
    ]
    if mode == "analytic":
        cmd.extend(["--train-implant", "analytic_channel", "--eval-implant", "analytic_channel"])
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when modes includes checkpoint.")
        cmd.extend(
            [
                "--train-implant",
                "checkpoint",
                "--train-implant-checkpoint",
                args.checkpoint,
                "--eval-implant",
                "analytic_channel",
            ]
        )
    if args.allow_tf32:
        cmd.append("--allow-tf32")
    if args.amp:
        cmd.append("--amp")
        cmd.extend(["--amp-dtype", args.amp_dtype])
    if args.compile_models:
        cmd.append("--compile-models")
        cmd.extend(["--compile-mode", args.compile_mode])
    if args.enable_gpu_warmup:
        cmd.append("--enable-gpu-warmup")
    return cmd


def main() -> None:
    args = parse_args()
    lengths = parse_int_csv(args.lengths)
    modes = parse_mode_csv(args.modes)
    suite_dir = args.out_dir or Path("results") / f"turboae_long_block_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for length in lengths:
        for mode in modes:
            run_dir = suite_dir / f"{mode}_l{length}"
            cmd = build_command(args, mode=mode, length=length, out_dir=run_dir)
            records.append({"mode": mode, "length": length, "out_dir": str(run_dir), "command": cmd})
            print("[long-block] " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=str(ROOT), check=True)

    summaries = []
    if not args.dry_run:
        for record in records:
            summary_path = Path(record["out_dir"]) / "summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                summaries.append(
                    {
                        "mode": record["mode"],
                        "length": record["length"],
                        "summary": str(summary_path),
                        "final_eval": data.get("final_eval", {}),
                        "train_seconds": data.get("train_seconds"),
                    }
                )

    suite_summary = {
        "suite_dir": str(suite_dir),
        "channel": args.channel,
        "model_type": args.model_type,
        "lengths": lengths,
        "modes": modes,
        "checkpoint": args.checkpoint,
        "runs": summaries,
        "commands": records,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(suite_summary, indent=2))
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
