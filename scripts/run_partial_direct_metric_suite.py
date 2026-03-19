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
    parser = argparse.ArgumentParser(
        description="Run the partial direct-metric rerun across multiple seeds."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA,OptFib")
    parser.add_argument(
        "--methods",
        type=str,
        default="drifting_residual,wgan,optfib_diffusion",
    )
    parser.add_argument("--paper-eval-size", type=int, default=1_000_000)
    parser.add_argument("--optfib-eval-size", type=int, default=100_000)
    parser.add_argument("--optfib-dataset-size", type=int, default=120_000)
    parser.add_argument("--optfib-epochs", type=int, default=60)
    parser.add_argument("--optfib-batch-size", type=int, default=512)
    parser.add_argument("--optfib-num-steps", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / f"partial_direct_metric_suite_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    return parser.parse_args()


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int, fallback_seed: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if num_seeds > 1:
        return list(range(seed_start, seed_start + num_seeds))
    return [fallback_seed]


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


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> dict:
    start = time.time()
    actual_cmd = list(cmd)
    if actual_cmd and Path(actual_cmd[0]).name.startswith("python") and "-u" not in actual_cmd[1:2]:
        actual_cmd = [actual_cmd[0], "-u", *actual_cmd[1:]]
    actual_env = dict(env)
    actual_env.setdefault("PYTHONUNBUFFERED", "1")

    captured_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[suite] start {dt.datetime.utcnow().isoformat(timespec='seconds')}Z\n")
        handle.write(f"[suite] cwd: {cwd}\n")
        handle.write(f"[suite] command: {' '.join(actual_cmd)}\n\n")
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
        handle.write(f"\n[suite] exit_code: {return_code}\n")
        handle.write(f"[suite] elapsed_seconds: {elapsed:.3f}\n")
        handle.flush()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, actual_cmd)

    result = parse_json_from_text("".join(captured_lines), actual_cmd)
    result["suite_log"] = str(log_path)
    result["suite_elapsed_seconds"] = elapsed
    return result


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
        "device": args.device,
        "seeds": seeds,
        "channels": args.channels,
        "methods": args.methods,
        "paper_eval_size": args.paper_eval_size,
        "optfib_eval_size": args.optfib_eval_size,
        "optfib_dataset_size": args.optfib_dataset_size,
        "optfib_epochs": args.optfib_epochs,
        "optfib_batch_size": args.optfib_batch_size,
        "optfib_num_steps": args.optfib_num_steps,
    }
    (args.out_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2))

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

    per_seed = []
    for seed in seeds:
        log_path = args.out_dir / f"seed{seed}.log"
        seed_out_dir = args.out_dir / f"partial_direct_metric_seed{seed}"
        cmd = [
            sys.executable,
            "scripts/run_partial_direct_metric_benchmark.py",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--channels",
            args.channels,
            "--methods",
            args.methods,
            "--paper-eval-size",
            str(args.paper_eval_size),
            "--optfib-eval-size",
            str(args.optfib_eval_size),
            "--optfib-dataset-size",
            str(args.optfib_dataset_size),
            "--optfib-epochs",
            str(args.optfib_epochs),
            "--optfib-batch-size",
            str(args.optfib_batch_size),
            "--optfib-num-steps",
            str(args.optfib_num_steps),
            "--out-dir",
            str(seed_out_dir),
        ]
        result = run_command(cmd, cwd=ROOT, env=env, log_path=log_path)
        seed_result = {
            "seed": seed,
            "log": str(log_path),
            "output_dir": str(seed_out_dir),
            "result": result,
        }
        (args.out_dir / f"seed{seed}_result.json").write_text(json.dumps(seed_result, indent=2))
        per_seed.append(seed_result)

    suite_summary = {
        "suite_dir": str(args.out_dir),
        "seeds": seeds,
        "per_seed": per_seed,
    }
    (args.out_dir / "suite_run_summary.json").write_text(json.dumps(suite_summary, indent=2))
    print(json.dumps({"suite_dir": str(args.out_dir), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
