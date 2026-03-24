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
RUNNER = ROOT / "scripts" / "train_symbol_pair_implant.py"


@dataclass(frozen=True)
class ImplantSpec:
    family: str
    register_name: str


def default_specs(seed: int) -> list[ImplantSpec]:
    return [
        ImplantSpec("drifting_direct", f"awgn7_drift_direct_seed{seed}"),
        ImplantSpec("drifting_residual", f"awgn7_drift_residual_seed{seed}"),
        ImplantSpec("paper_wgan", f"awgn7_wgan_seed{seed}"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and register symbolic AWGN channel implants for the (4,7) experiment.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--rate", type=float, default=4.0 / 7.0)
    parser.add_argument("--ebno-db", type=float, default=5.0)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--dataset-size", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--drifting-epochs", type=int, default=60)
    parser.add_argument("--wgan-epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--conditional-kernel", action="store_true")
    parser.add_argument("--conditioning-mode", type=str, default="none", choices=["none", "joint", "product", "local", "soft_local", "radius", "mixture"])
    parser.add_argument("--condition-metric", type=str, default="euclidean", choices=["euclidean", "whitened"])
    parser.add_argument("--condition-kernel-scale", type=float, default=1.0)
    parser.add_argument("--target-kernel-scale", type=float, default=1.0)
    parser.add_argument("--condition-bandwidth", type=float, default=None)
    parser.add_argument("--target-bandwidth", type=float, default=None)
    parser.add_argument("--local-condition-k", type=int, default=32)
    parser.add_argument("--condition-radius", type=float, default=None)
    parser.add_argument("--mixture-alpha", type=float, default=0.5)
    parser.add_argument("--target-kernel-mode", type=str, default="raw", choices=["raw", "raw_plus_residual"])
    parser.add_argument("--residual-target-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-condition-bandwidth", action="store_true")
    parser.add_argument("--adaptive-target-bandwidth", action="store_true")
    parser.add_argument("--adaptive-bandwidth-k", type=int, default=16)
    parser.add_argument("--condition-embedding-dim", type=int, default=0)
    parser.add_argument("--condition-embedding-hidden-dim", type=int, default=64)
    parser.add_argument("--methods", type=str, default="drifting_direct,drifting_residual,paper_wgan")
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
    suite_dir = Path(args.suite_dir) if args.suite_dir else ROOT / "results" / f"symbolic_awgn_implants_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    requested = {part.strip() for part in args.methods.split(",") if part.strip()}
    specs = [spec for spec in default_specs(args.seed) if spec.family in requested]
    manifest = {
        "seed": args.seed,
        "device": args.device,
        "n": args.n,
        "rate": args.rate,
        "ebno_db": args.ebno_db,
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "drifting_epochs": args.drifting_epochs,
        "wgan_epochs": args.wgan_epochs,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "conditional_kernel": args.conditional_kernel,
        "conditioning_mode": args.conditioning_mode,
        "condition_metric": args.condition_metric,
        "condition_kernel_scale": args.condition_kernel_scale,
        "target_kernel_scale": args.target_kernel_scale,
        "condition_bandwidth": args.condition_bandwidth,
        "target_bandwidth": args.target_bandwidth,
        "local_condition_k": args.local_condition_k,
        "condition_radius": args.condition_radius,
        "mixture_alpha": args.mixture_alpha,
        "target_kernel_mode": args.target_kernel_mode,
        "residual_target_scale": args.residual_target_scale,
        "adaptive_condition_bandwidth": args.adaptive_condition_bandwidth,
        "adaptive_target_bandwidth": args.adaptive_target_bandwidth,
        "adaptive_bandwidth_k": args.adaptive_bandwidth_k,
        "condition_embedding_dim": args.condition_embedding_dim,
        "condition_embedding_hidden_dim": args.condition_embedding_hidden_dim,
        "weights_root": str(Path(args.weights_root).resolve()),
        "specs": [asdict(spec) for spec in specs],
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: dict[str, dict] = {"manifest": manifest, "runs": {}}
    for spec in specs:
        log_path = suite_dir / f"{spec.family}.log"
        epochs = args.wgan_epochs if spec.family == "paper_wgan" else args.drifting_epochs
        cmd = [
            sys.executable,
            str(RUNNER),
            "--family",
            spec.family,
            "--channel",
            "AWGN",
            "--n",
            str(args.n),
            "--ebno-db",
            str(args.ebno_db),
            "--rate",
            str(args.rate),
            "--dataset-size",
            str(args.dataset_size),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(epochs),
            "--hidden-dim",
            str(args.hidden_dim),
            "--latent-dim",
            str(args.latent_dim),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--weights-root",
            args.weights_root,
            "--register-name",
            spec.register_name,
        ]
        if spec.family.startswith("drifting"):
            if args.conditional_kernel:
                cmd.append("--conditional-kernel")
            if args.conditioning_mode != "none":
                cmd.extend(["--conditioning-mode", args.conditioning_mode])
            cmd.extend(["--condition-metric", args.condition_metric])
            cmd.extend(["--condition-kernel-scale", str(args.condition_kernel_scale)])
            cmd.extend(["--target-kernel-scale", str(args.target_kernel_scale)])
            if args.condition_bandwidth is not None:
                cmd.extend(["--condition-bandwidth", str(args.condition_bandwidth)])
            if args.target_bandwidth is not None:
                cmd.extend(["--target-bandwidth", str(args.target_bandwidth)])
            cmd.extend(["--local-condition-k", str(args.local_condition_k)])
            if args.condition_radius is not None:
                cmd.extend(["--condition-radius", str(args.condition_radius)])
            cmd.extend(["--mixture-alpha", str(args.mixture_alpha)])
            cmd.extend(["--target-kernel-mode", args.target_kernel_mode])
            cmd.extend(["--residual-target-scale", str(args.residual_target_scale)])
            if args.adaptive_condition_bandwidth:
                cmd.append("--adaptive-condition-bandwidth")
            if args.adaptive_target_bandwidth:
                cmd.append("--adaptive-target-bandwidth")
            cmd.extend(["--adaptive-bandwidth-k", str(args.adaptive_bandwidth_k)])
            cmd.extend(["--condition-embedding-dim", str(args.condition_embedding_dim)])
            cmd.extend(["--condition-embedding-hidden-dim", str(args.condition_embedding_hidden_dim)])
        result = run_command(cmd, log_path)
        results["runs"][spec.family] = {"spec": asdict(spec), "runner_result": result}
        (suite_dir / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary": str((suite_dir / "suite_results.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
