from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHANNEL_SETTINGS = {
    "AWGN": {"code_dim": 7, "rate": 4.0 / 7.0, "ebno_db": 5.0},
    "Rayleigh": {"code_dim": 7, "rate": 4.0 / 7.0, "ebno_db": 12.0},
    "SSPA": {"code_dim": 8, "rate": 6.0 / 8.0, "ebno_db": 8.0},
    "OptFib": {"code_dim": 2, "rate": 1.0, "ebno_db": 5.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symbolic BER/SER follow-up for one journal W-Flow seed/channel.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--channel", type=str, required=True, choices=sorted(CHANNEL_SETTINGS))
    parser.add_argument("--variants", type=str, default="analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn")
    parser.add_argument("--wflow-suite-dir", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--ae-dataset-size", type=int, default=1_000_000)
    parser.add_argument("--ae-batch-size", type=int, default=500)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--diffusion-ddim-steps", type=int, default=100)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_json_from_text(text: str, cmd: list[str]) -> dict:
    lines = text.strip().splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        candidate = "\n".join(lines[idx:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Could not parse JSON output from command: {' '.join(cmd)}")


def checkpoint_path(wflow_suite_dir: Path, variant: str, channel: str, seed: int) -> Path:
    return (
        wflow_suite_dir
        / variant
        / f"seed{seed}"
        / "checkpoints"
        / f"enhanced_direct_{channel.lower()}_seed{seed}.pt"
    )


def build_ser_command(args: argparse.Namespace, variant: str, out_dir: Path) -> list[str]:
    settings = CHANNEL_SETTINGS[args.channel]
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
        str(args.message_dim),
        "--code-dim",
        str(settings["code_dim"]),
        "--hidden-dim",
        str(args.hidden_dim),
        "--rate",
        str(settings["rate"]),
        "--ebno-db",
        str(settings["ebno_db"]),
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
        "--eval-every",
        str(args.eval_every),
        "--weights-root",
        args.weights_root,
        "--diffusion-sampler",
        "ddim",
        "--ddim-steps",
        str(args.diffusion_ddim_steps),
        "--out-dir",
        str(out_dir),
    ]
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


def run_logged_command(cmd: list[str], log_path: Path) -> dict:
    captured_lines: list[str] = []
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
            captured_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return parse_json_from_text("".join(captured_lines), cmd)


def main() -> None:
    args = parse_args()
    variants = parse_csv_list(args.variants)
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.suite_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    run_results = []
    for variant in variants:
        out_dir = args.suite_dir / args.channel / f"seed{args.seed}" / variant
        log_path = log_dir / f"ser_{args.channel.lower()}_{variant}_seed{args.seed}.log"
        print(f"[journal-wflow-ser] seed={args.seed} channel={args.channel} variant={variant}", flush=True)
        runner_result = run_logged_command(build_ser_command(args, variant, out_dir), log_path)
        summary_path = Path(runner_result["summary"])
        summary = json.loads(summary_path.read_text())
        final_eval = summary.get("final_eval", {})
        run_results.append(
            {
                "seed": args.seed,
                "channel": args.channel,
                "variant": variant,
                "summary": str(summary_path),
                "log": str(log_path),
                "final_eval_ser": final_eval.get("ser"),
                "final_eval_ber": final_eval.get("ber"),
                "final_eval_loss": final_eval.get("loss"),
                "train_seconds": summary.get("train_seconds"),
                "checkpoint_path": None
                if variant == "analytic"
                else str(checkpoint_path(args.wflow_suite_dir, variant, args.channel, args.seed)),
            }
        )

    payload = {
        "seed": args.seed,
        "channel": args.channel,
        "variants": variants,
        "wflow_suite_dir": str(args.wflow_suite_dir),
        "elapsed_seconds": time.perf_counter() - start,
        "runs": run_results,
    }
    result_path = args.suite_dir / f"ser_{args.channel.lower()}_seed{args.seed}_result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"seed": args.seed, "channel": args.channel, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
