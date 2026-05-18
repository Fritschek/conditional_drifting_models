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


VARIANT_SPECS: dict[str, dict[str, object]] = {
    "kernel_target": {
        "description": "Direct drifting with the original target-space kernel.",
        "args": {
            "drift_field": "kernel",
            "conditioning_mode": "none",
            "target_kernel_scale": 1.0,
        },
    },
    "kernel_joint": {
        "description": "Enhanced direct drifting with the GLOBECOM conditioning-aware joint kernel.",
        "args": {
            "drift_field": "kernel",
            "conditioning_mode": "joint",
            "condition_kernel_scale": 0.5,
            "target_kernel_scale": 1.0,
            "target_kernel_mode": "raw",
        },
    },
    "joint_sinkhorn": {
        "description": "Naive joint Sinkhorn/W-Flow drift on condition-target features.",
        "args": {
            "drift_field": "sinkhorn",
            "conditioning_mode": "joint",
            "condition_kernel_scale": 0.5,
            "target_kernel_scale": 1.0,
        },
    },
    "fiber_sinkhorn": {
        "description": "Fiberwise conditional Sinkhorn drift with repeated samples per condition.",
        "args": {
            "drift_field": "fiber_sinkhorn",
            "conditioning_mode": "none",
            "target_kernel_scale": 1.0,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one journal W-Flow benchmark task for a seed/variant pair.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", type=str, required=True, choices=sorted(VARIANT_SPECS))
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,OptFib")
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drifting-epochs", type=int, default=60)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=None)
    parser.add_argument("--sinkhorn-min-epsilon", type=float, default=1e-3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--fiber-generated-samples", type=int, default=4)
    parser.add_argument("--fiber-positive-samples", type=int, default=4)
    parser.add_argument("--fiber-reference-samples", type=int, default=4)
    parser.add_argument("--anchor-metrics", action="store_true")
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--anchor-samples", type=int, default=64)
    parser.add_argument("--anchor-swd-projections", type=int, default=64)
    parser.add_argument("--suite-dir", type=Path, required=True)
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
    actual_env = dict(env)
    actual_env.setdefault("PYTHONUNBUFFERED", "1")
    actual_env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

    captured_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[journal-wflow] start {dt.datetime.utcnow().isoformat(timespec='seconds')}Z\n")
        handle.write(f"[journal-wflow] label: {label}\n")
        handle.write(f"[journal-wflow] cwd: {cwd}\n")
        handle.write(f"[journal-wflow] command: {' '.join(cmd)}\n\n")
        handle.flush()

        process = subprocess.Popen(
            cmd,
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
        handle.write(f"\n[journal-wflow] exit_code: {return_code}\n")
        handle.write(f"[journal-wflow] elapsed_seconds: {elapsed:.3f}\n")
        handle.flush()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    result = parse_json_from_text("".join(captured_lines), cmd)
    result["suite_log"] = str(log_path)
    result["suite_elapsed_seconds"] = elapsed
    return result


def _add_option(cmd: list[str], name: str, value: object) -> None:
    cmd.extend([f"--{name.replace('_', '-')}", str(value)])


def build_benchmark_command(args: argparse.Namespace, out_json: Path, ckpt_dir: Path) -> list[str]:
    variant_args = dict(VARIANT_SPECS[args.variant]["args"])
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_enhanced_direct_benchmark.py",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--channels",
        args.channels,
        "--dataset-size",
        str(args.dataset_size),
        "--eval-size",
        str(args.eval_size),
        "--batch-size",
        str(args.batch_size),
        "--drifting-epochs",
        str(args.drifting_epochs),
        "--swd-projections",
        str(args.swd_projections),
        "--sinkhorn-min-epsilon",
        str(args.sinkhorn_min_epsilon),
        "--sinkhorn-iterations",
        str(args.sinkhorn_iterations),
        "--fiber-generated-samples",
        str(args.fiber_generated_samples),
        "--fiber-positive-samples",
        str(args.fiber_positive_samples),
        "--fiber-reference-samples",
        str(args.fiber_reference_samples),
        "--save-dir",
        str(ckpt_dir),
        "--out",
        str(out_json),
    ]
    if args.sinkhorn_epsilon is not None:
        cmd.extend(["--sinkhorn-epsilon", str(args.sinkhorn_epsilon)])
    if args.anchor_metrics:
        cmd.extend(
            [
                "--anchor-metrics",
                "--anchor-count",
                str(args.anchor_count),
                "--anchor-samples",
                str(args.anchor_samples),
                "--anchor-swd-projections",
                str(args.anchor_swd_projections),
            ]
        )
    for key, value in variant_args.items():
        _add_option(cmd, key, value)
    return cmd


def main() -> None:
    args = parse_args()
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.suite_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    out_dir = args.suite_dir / args.variant / f"seed{args.seed}"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.variant}_summary_seed{args.seed}.json"
    log_path = log_dir / f"{args.variant}_seed{args.seed}.log"
    manifest_path = args.suite_dir / f"{args.variant}_seed{args.seed}_manifest.json"
    result_path = args.suite_dir / f"{args.variant}_seed{args.seed}_result.json"

    manifest = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
        "seed": args.seed,
        "variant": args.variant,
        "variant_description": VARIANT_SPECS[args.variant]["description"],
        "variant_args": VARIANT_SPECS[args.variant]["args"],
        "channels": args.channels,
        "dataset_size": args.dataset_size,
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "drifting_epochs": args.drifting_epochs,
        "swd_projections": args.swd_projections,
        "anchor_metrics": args.anchor_metrics,
        "anchor_count": args.anchor_count,
        "anchor_samples": args.anchor_samples,
        "anchor_swd_projections": args.anchor_swd_projections,
        "sinkhorn_epsilon": args.sinkhorn_epsilon,
        "sinkhorn_min_epsilon": args.sinkhorn_min_epsilon,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "fiber_generated_samples": args.fiber_generated_samples,
        "fiber_positive_samples": args.fiber_positive_samples,
        "fiber_reference_samples": args.fiber_reference_samples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    cmd = build_benchmark_command(args, out_json, ckpt_dir)
    benchmark_result = run_logged_command(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        log_path=log_path,
        label=f"{args.variant}:seed{args.seed}",
    )
    result = {
        "manifest": str(manifest_path),
        "seed": args.seed,
        "variant": args.variant,
        "log": str(log_path),
        "output_dir": str(out_dir),
        "benchmark_result": benchmark_result,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"seed": args.seed, "variant": args.variant, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
