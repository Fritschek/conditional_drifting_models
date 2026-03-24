from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMPLANT = ROOT / "scripts" / "train_symbol_pair_implant.py"
RUN_AE = ROOT / "scripts" / "run_symbolic_awgn_benchmark.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-budget old direct-drifting ('easy' kernel / conditioning-mode=none) follow-up experiments."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--out-root", type=str, default=str(ROOT / "results"))
    parser.add_argument("--wait-for-suite", type=str, default=None, help="Optional suite_results.json to wait to complete before starting.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--implant-dataset-size", type=int, default=300_000)
    parser.add_argument("--implant-batch-size", type=int, default=512)
    parser.add_argument("--implant-epochs", type=int, default=160)
    parser.add_argument("--ae-dataset-size", type=int, default=1_000_000)
    parser.add_argument("--ae-eval-size", type=int, default=100_000)
    parser.add_argument("--ae-batch-size", type=int, default=500)
    parser.add_argument("--awgn-ae-epochs", type=int, default=10)
    parser.add_argument("--nonlinear-ae-epochs", type=int, default=30)
    parser.add_argument("--eval-every-awgn", type=int, default=1)
    parser.add_argument("--eval-every-nonlinear", type=int, default=2)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def wait_for_suite_completion(suite_results_path: Path, poll_seconds: int) -> None:
    while True:
        if suite_results_path.exists():
            try:
                data = json.loads(suite_results_path.read_text())
                manifest = data.get("manifest", {})
                ae_runs = data.get("ae_runs", [])
                run_specs = manifest.get("run_specs", [])
                if run_specs and len(ae_runs) >= len(run_specs):
                    return
            except Exception:
                pass
        time.sleep(poll_seconds)


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    if args.wait_for_suite:
        wait_for_suite_completion(Path(args.wait_for_suite), args.poll_seconds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"old_direct_fullbudget_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "channel": "AWGN",
            "code_dim": 7,
            "rate": 4.0 / 7.0,
            "ae_epochs": args.awgn_ae_epochs,
            "eval_every": args.eval_every_awgn,
        },
        {
            "channel": "SSPA",
            "code_dim": 8,
            "rate": 0.5,
            "ae_epochs": args.nonlinear_ae_epochs,
            "eval_every": args.eval_every_nonlinear,
        },
        {
            "channel": "OptFib",
            "code_dim": 2,
            "rate": 0.5,
            "ae_epochs": args.nonlinear_ae_epochs,
            "eval_every": args.eval_every_nonlinear,
        },
    ]

    manifest = {
        "seed": args.seed,
        "device": args.device,
        "weights_root": args.weights_root,
        "implant_dataset_size": args.implant_dataset_size,
        "implant_batch_size": args.implant_batch_size,
        "implant_epochs": args.implant_epochs,
        "ae_dataset_size": args.ae_dataset_size,
        "ae_eval_size": args.ae_eval_size,
        "ae_batch_size": args.ae_batch_size,
        "configs": configs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    aggregate: dict[str, object] = {"manifest": manifest, "runs": []}

    for cfg in configs:
        channel = cfg["channel"]
        code_dim = int(cfg["code_dim"])
        rate = float(cfg["rate"])
        ae_epochs = int(cfg["ae_epochs"])
        eval_every = int(cfg["eval_every"])
        channel_token = f"{channel.lower()}{code_dim}"
        implant_name = f"{channel_token}_direct_none_fullbudget_seed{args.seed}"
        ae_name = f"symbolic_{channel.lower()}_drifting_direct_none_fullbudget_seed{args.seed}"

        implant_cmd = [
            sys.executable,
            str(TRAIN_IMPLANT),
            "--family",
            "drifting_direct",
            "--channel",
            channel,
            "--n",
            str(code_dim),
            "--ebno-db",
            "5",
            "--rate",
            str(rate),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--dataset-size",
            str(args.implant_dataset_size),
            "--batch-size",
            str(args.implant_batch_size),
            "--epochs",
            str(args.implant_epochs),
            "--hidden-dim",
            "128",
            "--latent-dim",
            "16",
            "--conditioning-mode",
            "none",
            "--condition-kernel-scale",
            "0.0",
            "--target-kernel-scale",
            "1.0",
            "--weights-root",
            args.weights_root,
            "--register-name",
            implant_name,
        ]
        if args.force_retrain:
            implant_cmd.append("--force")
        run_logged(implant_cmd, out_dir / f"implant_{channel.lower()}_direct_none_fullbudget.log")

        ae_out_dir = out_dir / f"{channel.lower()}_direct_none_fullbudget_e2e"
        ae_cmd = [
            sys.executable,
            str(RUN_AE),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--channel",
            channel,
            "--message-dim",
            "16",
            "--code-dim",
            str(code_dim),
            "--hidden-dim",
            "16",
            "--rate",
            str(rate),
            "--ebno-db",
            "5",
            "--batch-size",
            str(args.ae_batch_size),
            "--dataset-size",
            str(args.ae_dataset_size),
            "--eval-size",
            str(args.ae_eval_size),
            "--epochs",
            str(ae_epochs),
            "--learning-rate",
            "0.001",
            "--eval-every",
            str(eval_every),
            "--train-implant",
            "checkpoint",
            "--train-implant-name",
            implant_name,
            "--eval-implant",
            "analytic_channel",
            "--weights-root",
            args.weights_root,
            "--save-ae-name",
            ae_name,
            "--out-dir",
            str(ae_out_dir),
        ]
        run_logged(ae_cmd, out_dir / f"{channel.lower()}_direct_none_fullbudget.log")

        summary_path = ae_out_dir / "summary.json"
        run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"channel": channel, "missing_summary": True}
        aggregate["runs"].append(run_summary)
        (out_dir / "suite_results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
