from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the full-budget benchmark suite across paper channels and OptFib."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--paper-channels",
        type=str,
        default="AWGN,Rayleigh,SSPA",
        help="Paper-faithful channels for run_paper2309_benchmark.py",
    )
    parser.add_argument(
        "--paper-eval-size",
        type=int,
        default=1_000_000,
        help="Evaluation size override for paper channels. Training budgets remain paper presets.",
    )
    parser.add_argument(
        "--optfib-eval-size",
        type=int,
        default=100_000,
        help="Evaluation size for the OptFib baseline comparison path.",
    )
    parser.add_argument(
        "--optfib-dataset-size",
        type=int,
        default=120_000,
        help="OptFib training dataset size for the later benchmark path.",
    )
    parser.add_argument("--optfib-epochs", type=int, default=60)
    parser.add_argument("--optfib-batch-size", type=int, default=512)
    parser.add_argument("--optfib-num-steps", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / f"full_budget_suite_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prepared commands and manifest without executing them.",
    )
    parser.add_argument(
        "--terminal-mode",
        type=str,
        choices=("compact", "stream"),
        default="compact",
        help="How much child output to mirror to the terminal. Full raw logs are always written to files.",
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


def timestamped_line(line: str) -> str:
    ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    if line.endswith("\n"):
        return f"[{ts}] {line}"
    return f"[{ts}] {line}\n"


def is_progress_line(line: str) -> bool:
    lowered = line.lower()
    return any(
        token in lowered
        for token in ("epoch", "chunk", "batch", "iter", "step", "progress", "loss")
    )


def is_event_line(line: str) -> bool:
    lowered = line.lower()
    return any(
        token in lowered
        for token in (
            "starting",
            "complete",
            "completed",
            "done",
            "finished",
            "summary",
            "saved",
            "writing",
            "evaluating",
            "training",
            "benchmark",
            "channel",
        )
    )


def parse_progress_info(line: str) -> tuple[str, int, int] | None:
    lowered = line.lower()
    for kind in ("epoch", "chunk", "batch", "step", "iter"):
        match = re.search(rf"{kind}\s+(\d+)\s*/\s*(\d+)", lowered)
        if match:
            return kind, int(match.group(1)), int(match.group(2))
    return None


def shorten_terminal_line(label: str, line: str, width: int = 140) -> str:
    clean = re.sub(r"\s+", " ", line.strip())
    text = f"[{label}] {clean}"
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def emit_terminal_line(label: str, line: str, terminal_mode: str, status_state: dict[str, bool]) -> None:
    if terminal_mode == "stream":
        sys.stdout.write(line)
        sys.stdout.flush()
        return

    progress = parse_progress_info(line)
    if progress is not None and tqdm is not None:
        kind, current, total = progress
        bar = status_state.get("bar")
        bar_key = (label, kind, total)
        if bar is None or status_state.get("bar_key") != bar_key:
            if bar is not None:
                bar.close()
            bar = tqdm(total=total, desc=f"{label} {kind}", leave=True, dynamic_ncols=True)
            status_state["bar"] = bar
            status_state["bar_key"] = bar_key
            status_state["bar_pos"] = 0
        target_n = max(0, min(current, total))
        increment = target_n - int(status_state.get("bar_pos", 0))
        if increment > 0:
            bar.update(increment)
        status_state["bar_pos"] = target_n
        postfix = shorten_terminal_line("", line, width=100).strip()
        if postfix:
            bar.set_postfix_str(postfix[:100], refresh=False)
        status_state["active"] = True
        return

    message = shorten_terminal_line(label, line)
    if is_progress_line(line):
        sys.stdout.write("\r" + message.ljust(160))
        sys.stdout.flush()
        status_state["active"] = True
        return

    if status_state["active"]:
        bar = status_state.get("bar")
        if bar is not None:
            bar.close()
            status_state["bar"] = None
            status_state["bar_key"] = None
            status_state["bar_pos"] = 0
        sys.stdout.write("\n")
        sys.stdout.flush()
        status_state["active"] = False

    if is_event_line(line):
        sys.stdout.write(message + "\n")
        sys.stdout.flush()


def run_command(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    label: str,
    terminal_mode: str,
    master_log_path: Path | None = None,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    actual_cmd = list(cmd)
    if actual_cmd and Path(actual_cmd[0]).name.startswith("python") and "-u" not in actual_cmd[1:2]:
        actual_cmd = [actual_cmd[0], "-u", *actual_cmd[1:]]
    actual_env = dict(env)
    actual_env.setdefault("PYTHONUNBUFFERED", "1")
    if terminal_mode == "compact":
        actual_env["TQDM_DISABLE"] = "1"
    master_handle = None
    if master_log_path is not None:
        master_log_path.parent.mkdir(parents=True, exist_ok=True)
        master_handle = master_log_path.open("a", encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as log_handle:
        header = (
            f"[suite] start {dt.datetime.utcnow().isoformat(timespec='seconds')}Z\n"
            f"[suite] label: {label}\n"
            f"[suite] cwd: {cwd}\n"
            f"[suite] command: {' '.join(actual_cmd)}\n\n"
        )
        log_handle.write(header)
        log_handle.flush()
        if master_handle is not None:
            master_handle.write(header)
            master_handle.flush()

        captured_lines: list[str] = []
        status_state = {"active": False}
        if terminal_mode == "stream":
            master_fd, slave_fd = pty.openpty()
            process = subprocess.Popen(
                actual_cmd,
                cwd=str(cwd),
                env=actual_env,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
            os.close(slave_fd)
            captured_text = ""
            try:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    if master_fd in ready:
                        try:
                            chunk = os.read(master_fd, 4096)
                        except OSError:
                            chunk = b""
                        if chunk:
                            text = chunk.decode("utf-8", errors="replace")
                            captured_text += text
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            log_handle.write(timestamped_line(text.rstrip("\n")))
                            log_handle.flush()
                            if master_handle is not None:
                                master_handle.write(timestamped_line(text.rstrip("\n")))
                                master_handle.flush()
                    if process.poll() is not None and not ready:
                        break
            finally:
                os.close(master_fd)
            captured_lines = captured_text.splitlines(keepends=True)
            return_code = process.wait()
        else:
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
                emit_terminal_line(label, line, terminal_mode, status_state)
                stamped = timestamped_line(line)
                log_handle.write(stamped)
                log_handle.flush()
                if master_handle is not None:
                    master_handle.write(stamped)
                    master_handle.flush()
            return_code = process.wait()
        elapsed = time.time() - start
        if status_state["active"]:
            bar = status_state.get("bar")
            if bar is not None:
                bar.close()
                status_state["bar"] = None
            sys.stdout.write("\n")
            sys.stdout.flush()
        footer = f"\n[suite] exit_code: {return_code}\n[suite] elapsed_seconds: {elapsed:.3f}\n"
        log_handle.write(footer)
        log_handle.flush()
        if master_handle is not None:
            master_handle.write(footer)
            master_handle.flush()

    if master_handle is not None:
        master_handle.close()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, actual_cmd)

    parsed = parse_json_from_text("".join(captured_lines), actual_cmd)
    parsed["suite_log"] = str(log_path)
    parsed["suite_elapsed_seconds"] = elapsed
    return parsed


def parse_seeds(seed_text: str, seed_start: int, num_seeds: int, fallback_seed: int) -> list[int]:
    if seed_text:
        return [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if num_seeds > 1:
        return list(range(seed_start, seed_start + num_seeds))
    return [fallback_seed]


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "num_seeds": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance), "num_seeds": len(values)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds, args.seed_start, args.num_seeds, args.seed)

    base_env = os.environ.copy()
    base_env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

    manifest = {
        "timestamp_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "cwd": str(ROOT),
        "device": args.device,
        "seeds": seeds,
        "paper_channels": args.paper_channels,
        "paper_eval_size": args.paper_eval_size,
        "optfib_eval_size": args.optfib_eval_size,
    }
    master_log = args.out_dir / "suite_master.log"
    manifest["master_log"] = str(master_log)

    manifest_path = args.out_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.dry_run:
        preview = []
        for seed in seeds:
            paper_log = args.out_dir / f"seed{seed}_paper.log"
            optfib_log = args.out_dir / f"seed{seed}_optfib.log"
            preview.append(
                {
                    "seed": seed,
                    "master_log": str(master_log),
                    "paper_log": str(paper_log),
                    "optfib_log": str(optfib_log),
                    "paper_command": [
                        sys.executable,
                        "scripts/run_paper2309_benchmark.py",
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--channels",
                        args.paper_channels,
                        "--eval-size",
                        str(args.paper_eval_size),
                    ],
                    "optfib_command": [
                        sys.executable,
                        "scripts/compare_optional_baselines.py",
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--channel",
                        "OptFib",
                        "--dataset-size",
                        str(args.optfib_dataset_size),
                        "--epochs",
                        str(args.optfib_epochs),
                        "--batch-size",
                        str(args.optfib_batch_size),
                        "--eval-size",
                        str(args.optfib_eval_size),
                        "--num-steps",
                        str(args.optfib_num_steps),
                        "--ddim-steps",
                        str(args.optfib_num_steps),
                    ],
                }
            )
        print(json.dumps({"manifest": str(manifest_path), **manifest, "commands": preview}, indent=2))
        return

    per_seed = []
    for seed in seeds:
        paper_log = args.out_dir / f"seed{seed}_paper.log"
        optfib_log = args.out_dir / f"seed{seed}_optfib.log"
        paper_cmd = [
            sys.executable,
            "scripts/run_paper2309_benchmark.py",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--channels",
            args.paper_channels,
            "--eval-size",
            str(args.paper_eval_size),
        ]
        optfib_cmd = [
            sys.executable,
            "scripts/compare_optional_baselines.py",
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--channel",
            "OptFib",
            "--dataset-size",
            str(args.optfib_dataset_size),
            "--epochs",
            str(args.optfib_epochs),
            "--batch-size",
            str(args.optfib_batch_size),
            "--eval-size",
            str(args.optfib_eval_size),
            "--num-steps",
            str(args.optfib_num_steps),
            "--ddim-steps",
            str(args.optfib_num_steps),
        ]
        per_seed.append(
            {
                "seed": seed,
                "master_log": str(master_log),
                "paper_log": str(paper_log),
                "optfib_log": str(optfib_log),
                "paper_result": run_command(
                    paper_cmd,
                    ROOT,
                    base_env,
                    paper_log,
                    f"seed={seed} paper",
                    args.terminal_mode,
                    master_log,
                ),
                "optfib_result": run_command(
                    optfib_cmd,
                    ROOT,
                    base_env,
                    optfib_log,
                    f"seed={seed} optfib",
                    args.terminal_mode,
                    master_log,
                ),
            }
        )

    aggregated: dict[str, dict] = {"paper_channels": {}, "optfib": {}}
    paper_metrics = [
        "drifting_direct_swd",
        "drifting_residual_swd",
        "drifting_swd",
        "ddpm_swd",
        "paper_wgan_swd",
    ]
    ddim_keys = [100, 50, 20, 10]
    channel_names = [name.strip() for name in args.paper_channels.split(",") if name.strip()]
    for channel_name in channel_names:
        channel_summary = {}
        for metric in paper_metrics:
            values = []
            for row in per_seed:
                summary_path = Path(row["paper_result"]["output_json"])
                data = json.loads(summary_path.read_text())
                result_block = data["channels"][channel_name]["results"]
                if metric in result_block:
                    values.append(float(result_block[metric]))
            if values:
                channel_summary[metric] = summarize_numeric(values)
        ddim_summary = {}
        for step in ddim_keys:
            values = []
            for row in per_seed:
                summary_path = Path(row["paper_result"]["output_json"])
                data = json.loads(summary_path.read_text())
                values.append(float(data["channels"][channel_name]["results"]["ddim_swd"][str(step)]))
            ddim_summary[str(step)] = summarize_numeric(values)
        channel_summary["ddim_swd"] = ddim_summary
        aggregated["paper_channels"][channel_name] = channel_summary

    optfib_values = {
        "drifting_residual_swd": [],
        "drifting_direct_swd": [],
        "drifting_swd": [],
        "ddpm_swd": [],
        "ddim_swd": [],
        "paper_wgan_swd": [],
    }
    for row in per_seed:
        optfib_json = Path(row["optfib_result"]["json"])
        data = json.loads(optfib_json.read_text())
        for metric in optfib_values:
            optfib_values[metric].append(float(data[metric]))
    aggregated["optfib"] = {metric: summarize_numeric(values) for metric, values in optfib_values.items()}

    summary = {
        "manifest": str(manifest_path),
        "per_seed": per_seed,
        "aggregated": aggregated,
    }
    summary_path = args.out_dir / "suite_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
