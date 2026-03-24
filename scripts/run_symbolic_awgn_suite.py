from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_symbolic_awgn_benchmark.py"


@dataclass(frozen=True)
class RunSpec:
    name: str
    train_implant: str
    train_implant_name: str | None
    eval_implant: str
    eval_implant_name: str | None
    save_ae_name: str


def default_specs(seed: int) -> list[RunSpec]:
    return [
        RunSpec(
            name="analytic_to_analytic",
            train_implant="analytic_awgn",
            train_implant_name=None,
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_ae_name=f"symbolic_awgn_analytic_seed{seed}",
        ),
        RunSpec(
            name="drifting_direct_to_analytic",
            train_implant="checkpoint",
            train_implant_name=f"awgn7_drift_direct_seed{seed}",
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_ae_name=f"symbolic_awgn_drift_direct_seed{seed}",
        ),
        RunSpec(
            name="drifting_residual_to_analytic",
            train_implant="checkpoint",
            train_implant_name=f"awgn7_drift_residual_seed{seed}",
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_ae_name=f"symbolic_awgn_drift_residual_seed{seed}",
        ),
        RunSpec(
            name="wgan_to_analytic",
            train_implant="checkpoint",
            train_implant_name=f"awgn7_wgan_seed{seed}",
            eval_implant="analytic_awgn",
            eval_implant_name=None,
            save_ae_name=f"symbolic_awgn_wgan_seed{seed}",
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the symbolic (4,7) AWGN suite sequentially.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--methods", type=str, default="analytic,drifting_direct,drifting_residual,wgan")
    return parser


def run_command(cmd: list[str], log_path: Path) -> dict:
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
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
    suite_dir = Path(args.suite_dir) if args.suite_dir else ROOT / "results" / f"symbolic_awgn_suite_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    requested = {part.strip() for part in args.methods.split(",") if part.strip()}
    specs = [
        spec
        for spec in default_specs(args.seed)
        if spec.name.split("_to_")[0] in requested or (spec.name == "analytic_to_analytic" and "analytic" in requested)
    ]

    manifest = {
        "seed": args.seed,
        "device": args.device,
        "message_dim": args.message_dim,
        "code_dim": args.code_dim,
        "hidden_dim": args.hidden_dim,
        "rate": args.rate,
        "ebno_db": args.ebno_db,
        "batch_size": args.batch_size,
        "dataset_size": args.dataset_size,
        "eval_size": args.eval_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
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
            "--message-dim",
            str(args.message_dim),
            "--code-dim",
            str(args.code_dim),
            "--hidden-dim",
            str(args.hidden_dim),
            "--rate",
            str(args.rate),
            "--ebno-db",
            str(args.ebno_db),
            "--batch-size",
            str(args.batch_size),
            "--dataset-size",
            str(args.dataset_size),
            "--eval-size",
            str(args.eval_size),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--eval-every",
            str(args.eval_every),
            "--seed",
            str(args.seed),
            "--weights-root",
            args.weights_root,
            "--train-implant",
            spec.train_implant,
            "--eval-implant",
            spec.eval_implant,
            "--save-ae-name",
            spec.save_ae_name,
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
