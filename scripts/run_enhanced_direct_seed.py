from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one enhanced-direct benchmark seed for HPC/SLURM job arrays.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA")
    parser.add_argument("--dataset-size", type=int, default=-1)
    parser.add_argument("--eval-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--drifting-epochs", type=int, default=-1)
    parser.add_argument("--swd-projections", type=int, default=-1)
    parser.add_argument("--conditioning-mode", type=str, default="joint")
    parser.add_argument("--condition-kernel-scale", type=float, default=0.5)
    parser.add_argument("--target-kernel-scale", type=float, default=1.0)
    parser.add_argument("--target-kernel-mode", type=str, default="raw")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Shared output directory for the multi-seed suite.",
    )
    return parser.parse_args()


def parse_json_from_text(text: str, cmd: list[str]) -> dict:
    stdout = text.strip()
    if stdout:
        lines = stdout.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            candidate = "\n".join(lines[idx:])
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"Could not parse JSON output from command: {' '.join(cmd)}")


def run_logged_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> dict:
    start = time.time()
    actual_cmd = list(cmd)
    if actual_cmd and Path(actual_cmd[0]).name.startswith("python") and "-u" not in actual_cmd[1:2]:
        actual_cmd = [actual_cmd[0], "-u", *actual_cmd[1:]]
    actual_env = dict(env)
    actual_env.setdefault("PYTHONUNBUFFERED", "1")

    captured_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[seed-runner] start {dt.datetime.utcnow().isoformat(timespec='seconds')}Z\n")
        handle.write(f"[seed-runner] label: {label}\n")
        handle.write(f"[seed-runner] cwd: {cwd}\n")
        handle.write(f"[seed-runner] command: {' '.join(actual_cmd)}\n\n")
        handle.flush()

        process = subprocess.Popen(
            actual_cmd,
            cwd=str(cwd),
            env=actual_env,
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
        elapsed = time.time() - start
        handle.write(f"\n[seed-runner] exit_code: {return_code}\n")
        handle.write(f"[seed-runner] elapsed_seconds: {elapsed:.3f}\n")
        handle.flush()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, actual_cmd)

    result = parse_json_from_text("".join(captured_lines), actual_cmd)
    result["suite_log"] = str(log_path)
    result["suite_elapsed_seconds"] = elapsed
    return result


def main() -> None:
    args = parse_args()
    args.suite_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    manifest = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
        "device": args.device,
        "seed": args.seed,
        "channels": args.channels,
        "dataset_size": args.dataset_size,
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "drifting_epochs": args.drifting_epochs,
        "swd_projections": args.swd_projections,
        "conditioning_mode": args.conditioning_mode,
        "condition_kernel_scale": args.condition_kernel_scale,
        "target_kernel_scale": args.target_kernel_scale,
        "target_kernel_mode": args.target_kernel_mode,
    }
    manifest_path = args.suite_dir / f"seed{args.seed}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log_path = args.suite_dir / f"seed{args.seed}.log"
    out_dir = args.suite_dir / f"enhanced_direct_seed{args.seed}"
    ckpt_dir = out_dir / "checkpoints"
    out_json = out_dir / f"enhanced_direct_summary_seed{args.seed}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_enhanced_direct_benchmark.py",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--channels",
        args.channels,
        "--eval-size",
        str(args.eval_size),
        "--drifting-epochs",
        str(args.drifting_epochs),
        "--swd-projections",
        str(args.swd_projections),
        "--conditioning-mode",
        args.conditioning_mode,
        "--condition-kernel-scale",
        str(args.condition_kernel_scale),
        "--target-kernel-scale",
        str(args.target_kernel_scale),
        "--target-kernel-mode",
        args.target_kernel_mode,
        "--save-dir",
        str(ckpt_dir),
        "--out",
        str(out_json),
    ]
    if args.dataset_size > 0:
        cmd.extend(["--dataset-size", str(args.dataset_size)])
    if args.batch_size > 0:
        cmd.extend(["--batch-size", str(args.batch_size)])

    result = {
        "manifest": str(manifest_path),
        "seed": args.seed,
        "log": str(log_path),
        "output_dir": str(out_dir),
        "enhanced_direct_result": run_logged_command(cmd, cwd=ROOT, env=env, log_path=log_path, label="enhanced_direct"),
    }
    result_path = args.suite_dir / f"seed{args.seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
