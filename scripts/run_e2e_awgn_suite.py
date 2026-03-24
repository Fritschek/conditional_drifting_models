from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_e2e_channel_implant_benchmark.py"


@dataclass(frozen=True)
class RunSpec:
    name: str
    train_implant: str
    train_implant_name: str | None
    eval_implant: str
    eval_implant_name: str | None
    save_e2e_name: str


def default_specs(seed: int) -> list[RunSpec]:
    return [
        RunSpec(
            name="analytic_to_analytic",
            train_implant="analytic_awgn",
            train_implant_name=None,
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_e2e_name=f"cnn_turbo_awgn_analytic_seed{seed}",
        ),
        RunSpec(
            name="drifting_direct_to_analytic",
            train_implant="checkpoint",
            train_implant_name=f"awgn_drift_direct_seed{seed}",
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_e2e_name=f"cnn_turbo_awgn_drift_direct_seed{seed}",
        ),
        RunSpec(
            name="drifting_residual_to_analytic",
            train_implant="checkpoint",
            train_implant_name=f"awgn_drift_residual_seed{seed}",
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_e2e_name=f"cnn_turbo_awgn_drift_residual_seed{seed}",
        ),
        RunSpec(
            name="wgan_to_analytic",
            train_implant="checkpoint",
            train_implant_name=f"awgn_wgan_seed{seed}",
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_e2e_name=f"cnn_turbo_awgn_wgan_seed{seed}",
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AWGN end-to-end CNN suite sequentially.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dec-bs-fac", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--channel-length", type=int, default=128)
    parser.add_argument("--num-iteration", type=int, default=5)
    parser.add_argument("--num-iter-ft-cnn", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ebno-db", type=float, default=4.0)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--methods", type=str, default="analytic,drifting_direct,drifting_residual,wgan")
    return parser


def run_command(cmd: list[str], log_path: Path) -> dict:
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - start
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return {"elapsed_seconds": elapsed, "log_path": str(log_path.resolve())}


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.suite_dir) if args.suite_dir else ROOT / "results" / f"e2e_awgn_suite_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    requested = {part.strip() for part in args.methods.split(",") if part.strip()}
    specs = [spec for spec in default_specs(args.seed) if spec.name.split("_to_")[0] in requested or spec.name == "analytic_to_analytic" and "analytic" in requested]

    manifest = {
        "device": args.device,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dec_bs_fac": args.dec_bs_fac,
        "sample_size": args.sample_size,
        "sequence_length": args.sequence_length,
        "channel_length": args.channel_length,
        "num_iteration": args.num_iteration,
        "num_iter_ft_cnn": args.num_iter_ft_cnn,
        "learning_rate": args.learning_rate,
        "ebno_db": args.ebno_db,
        "eval_batches": args.eval_batches,
        "eval_every": args.eval_every,
        "weights_root": str(Path(args.weights_root).resolve()),
        "specs": [asdict(spec) for spec in specs],
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    suite_results: dict[str, dict] = {"manifest": manifest, "runs": {}}
    for spec in specs:
        out_dir = suite_dir / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = suite_dir / f"{spec.name}.log"
        cmd = [
            sys.executable,
            str(RUNNER),
            "--device",
            args.device,
            "--allow-tf32",
            "--model-type",
            "cnn_turbo",
            "--train-implant",
            spec.train_implant,
            "--eval-implant",
            spec.eval_implant,
            "--training-regime",
            "epoch_alternate",
            "--sequence-length",
            str(args.sequence_length),
            "--channel-length",
            str(args.channel_length),
            "--num-iteration",
            str(args.num_iteration),
            "--num-iter-ft-cnn",
            str(args.num_iter_ft_cnn),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--dec-bs-fac",
            str(args.dec_bs_fac),
            "--sample-size",
            str(args.sample_size),
            "--learning-rate",
            str(args.learning_rate),
            "--ebno-db",
            str(args.ebno_db),
            "--eval-batches",
            str(args.eval_batches),
            "--eval-every",
            str(args.eval_every),
            "--weights-root",
            args.weights_root,
            "--save-e2e-name",
            spec.save_e2e_name,
            "--out-dir",
            str(out_dir),
        ]
        if spec.train_implant_name:
            cmd.extend(["--train-implant-name", spec.train_implant_name])
        if spec.eval_implant_name:
            cmd.extend(["--eval-implant-name", spec.eval_implant_name])

        result = run_command(cmd, log_path)
        summary_path = out_dir / "summary.json"
        run_summary = json.loads(summary_path.read_text())
        suite_results["runs"][spec.name] = {
            "spec": asdict(spec),
            "runner_result": result,
            "summary_path": str(summary_path.resolve()),
            "summary": run_summary,
        }
        (suite_dir / "suite_results.json").write_text(json.dumps(suite_results, indent=2), encoding="utf-8")

    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary": str((suite_dir / "suite_results.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
