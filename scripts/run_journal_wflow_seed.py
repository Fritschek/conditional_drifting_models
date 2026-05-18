from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all requested journal W-Flow variants for one seed.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variants", type=str, default="kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn")
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA")
    parser.add_argument("--dataset-size", type=int, default=-1)
    parser.add_argument("--eval-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--drifting-epochs", type=int, default=-1)
    parser.add_argument("--swd-projections", type=int, default=-1)
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


def build_variant_command(args: argparse.Namespace, variant: str) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_journal_wflow_task.py",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--variant",
        variant,
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
        "--anchor-count",
        str(args.anchor_count),
        "--anchor-samples",
        str(args.anchor_samples),
        "--anchor-swd-projections",
        str(args.anchor_swd_projections),
        "--suite-dir",
        str(args.suite_dir),
    ]
    if args.sinkhorn_epsilon is not None:
        cmd.extend(["--sinkhorn-epsilon", str(args.sinkhorn_epsilon)])
    if args.anchor_metrics:
        cmd.append("--anchor-metrics")
    return cmd


def run_command(cmd: list[str]) -> dict:
    captured_lines: list[str] = []
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
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return parse_json_from_text("".join(captured_lines), cmd)


def main() -> None:
    args = parse_args()
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    variants = parse_csv_list(args.variants)
    results = []
    for variant in variants:
        print(f"[journal-wflow-seed] seed={args.seed} variant={variant}", flush=True)
        result = run_command(build_variant_command(args, variant))
        results.append(result)

    payload = {
        "seed": args.seed,
        "variants": variants,
        "channels": parse_csv_list(args.channels),
        "elapsed_seconds": time.perf_counter() - start,
        "results": results,
    }
    result_path = args.suite_dir / f"seed{args.seed}_packed_result.json"
    result_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"seed": args.seed, "result_json": str(result_path)}, indent=2))


if __name__ == "__main__":
    main()
