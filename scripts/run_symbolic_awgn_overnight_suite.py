from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMPLANT = ROOT / "scripts" / "train_symbol_pair_implant.py"
RUN_AE = ROOT / "scripts" / "run_symbolic_awgn_benchmark.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.weight_registry import resolve_registered_artifact


def fmt_token(value: float | None) -> str:
    if value is None:
        return "auto"
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def channel_token(channel: str, code_dim: int) -> str:
    return f"{channel.lower()}{int(code_dim)}"


@dataclass(frozen=True)
class ImplantSpec:
    name: str
    family: str
    register_name: str
    train_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    name: str
    train_implant: str
    train_implant_name: str | None
    eval_implant: str
    eval_implant_name: str | None
    save_ae_name: str
    benchmark_args: dict[str, Any] = field(default_factory=dict)


def drifting_register_name(
    *,
    channel: str,
    code_dim: int,
    seed: int,
    family: str,
    conditioning_mode: str,
    condition_scale: float,
    target_scale: float = 1.0,
    condition_metric: str = "euclidean",
    condition_bandwidth: float | None = None,
    target_bandwidth: float | None = None,
    local_condition_k: int = 32,
    condition_radius: float | None = None,
    mixture_alpha: float = 0.5,
    target_kernel_mode: str = "raw",
    adaptive_condition_bandwidth: bool = False,
    adaptive_target_bandwidth: bool = False,
    adaptive_bandwidth_k: int = 16,
    condition_embedding_dim: int = 0,
) -> str:
    return (
        f"{channel_token(channel, code_dim)}_{family}_{conditioning_mode}_{condition_metric}_"
        f"cond{fmt_token(condition_scale)}_targ{fmt_token(target_scale)}_"
        f"cbw{fmt_token(condition_bandwidth)}_tbw{fmt_token(target_bandwidth)}_"
        f"k{local_condition_k}_rad{fmt_token(condition_radius)}_mix{fmt_token(mixture_alpha)}_"
        f"{target_kernel_mode.replace('_', '')}_ac{int(adaptive_condition_bandwidth)}_"
        f"at{int(adaptive_target_bandwidth)}_ak{adaptive_bandwidth_k}_emb{condition_embedding_dim}_seed{seed}"
    )


def baseline_register_name(channel: str, code_dim: int, seed: int, name: str) -> str:
    return f"{channel_token(channel, code_dim)}_{name}_seed{seed}"


def common_benchmark_args(args: argparse.Namespace) -> dict[str, Any]:
    diffusion_ddim_steps = args.diffusion_ddim_steps if args.diffusion_ddim_steps > 0 else args.diffusion_num_steps
    return {
        "channel": args.channel,
        "message-dim": 16,
        "code-dim": args.code_dim,
        "hidden-dim": 16,
        "rate": args.rate,
        "ebno-db": args.ebno_db,
        "batch-size": args.ae_batch_size,
        "dataset-size": args.ae_dataset_size,
        "eval-size": args.eval_size,
        "epochs": args.ae_epochs,
        "learning-rate": args.ae_learning_rate,
        "eval-every": args.eval_every,
        "diffusion-sampler": "ddim",
        "ddim-steps": diffusion_ddim_steps,
    }


def build_default_matrix(args: argparse.Namespace) -> tuple[list[ImplantSpec], list[RunSpec]]:
    seed = args.seed
    implant_specs: list[ImplantSpec] = []
    run_specs: list[RunSpec] = []

    benchmark_base = common_benchmark_args(args)

    run_specs.append(
        RunSpec(
            name="analytic_to_analytic",
            train_implant="analytic_channel",
            train_implant_name=None,
            eval_implant="analytic_channel",
            eval_implant_name=None,
            save_ae_name=f"symbolic_{args.channel.lower()}_analytic_seed{seed}",
            benchmark_args=benchmark_base,
        )
    )

    wgan_name = baseline_register_name(args.channel, args.code_dim, seed, "wgan")
    implant_specs.append(
        ImplantSpec(
            name="wgan",
            family="paper_wgan",
            register_name=wgan_name,
            train_args={
                "channel": args.channel,
                "n": args.code_dim,
                "ebno-db": args.ebno_db,
                "rate": args.rate,
                "dataset-size": args.implant_dataset_size,
                "batch-size": args.implant_batch_size,
                "epochs": args.wgan_epochs,
                "hidden-dim": 128,
            },
        )
    )
    run_specs.append(
        RunSpec(
            name="wgan_to_analytic",
            train_implant="checkpoint",
            train_implant_name=wgan_name,
            eval_implant="analytic_channel",
            eval_implant_name=None,
            save_ae_name=f"symbolic_{args.channel.lower()}_wgan_seed{seed}",
            benchmark_args=benchmark_base,
        )
    )

    diffusion_variants = [
        ("diffusion_direct", {"num-steps": args.diffusion_num_steps}),
        ("diffusion_residual", {"num-steps": args.diffusion_num_steps}),
    ]
    for family, train_extra in diffusion_variants:
        reg_name = baseline_register_name(args.channel, args.code_dim, seed, family)
        implant_specs.append(
            ImplantSpec(
                name=family,
                family=family,
                register_name=reg_name,
                train_args={
                    "channel": args.channel,
                    "n": args.code_dim,
                    "ebno-db": args.ebno_db,
                    "rate": args.rate,
                    "dataset-size": args.diffusion_dataset_size,
                    "batch-size": args.diffusion_batch_size,
                    "epochs": args.diffusion_epochs,
                    "hidden-dim": args.diffusion_hidden_dim,
                    "diffusion-profile": args.diffusion_profile,
                    **train_extra,
                },
            )
        )
        run_specs.append(
            RunSpec(
                name=f"{family}_to_analytic",
                train_implant="checkpoint",
                train_implant_name=reg_name,
                eval_implant="analytic_channel",
                eval_implant_name=None,
                save_ae_name=f"symbolic_{args.channel.lower()}_{family}_seed{seed}",
                benchmark_args=benchmark_base,
            )
        )

    direct_candidates = [
        {
            "tag": "none",
            "conditioning_mode": "none",
            "condition_kernel_scale": 0.0,
        },
        {
            "tag": "joint_cond0p5",
            "conditioning_mode": "joint",
            "condition_kernel_scale": 0.5,
        },
        {
            "tag": "product_cond0p5",
            "conditioning_mode": "product",
            "condition_kernel_scale": 0.5,
        },
        {
            "tag": "mixture_cond0p5_a0p5",
            "conditioning_mode": "mixture",
            "condition_kernel_scale": 0.5,
            "mixture_alpha": 0.5,
        },
        {
            "tag": "product_cond0p5_rawplusres",
            "conditioning_mode": "product",
            "condition_kernel_scale": 0.5,
            "target_kernel_mode": "raw_plus_residual",
            "residual_target_scale": 0.5,
        },
        {
            "tag": "product_whitened_cond0p5_adapt",
            "conditioning_mode": "product",
            "condition_kernel_scale": 0.5,
            "condition_metric": "whitened",
            "adaptive_condition_bandwidth": True,
            "adaptive_target_bandwidth": True,
            "adaptive_bandwidth_k": 8,
        },
    ]

    residual_candidates = [
        {
            "tag": "none",
            "conditioning_mode": "none",
            "condition_kernel_scale": 0.0,
        },
        {
            "tag": "joint_cond0p25",
            "conditioning_mode": "joint",
            "condition_kernel_scale": 0.25,
        },
        {
            "tag": "product_cond0p25",
            "conditioning_mode": "product",
            "condition_kernel_scale": 0.25,
        },
        {
            "tag": "softlocal_cond0p25_k16",
            "conditioning_mode": "soft_local",
            "condition_kernel_scale": 0.25,
            "local_condition_k": 16,
        },
        {
            "tag": "local_cond0p25_k16",
            "conditioning_mode": "local",
            "condition_kernel_scale": 0.25,
            "local_condition_k": 16,
        },
        {
            "tag": "mixture_cond0p25_a0p3",
            "conditioning_mode": "mixture",
            "condition_kernel_scale": 0.25,
            "mixture_alpha": 0.3,
        },
        {
            "tag": "product_whitened_cond0p25_adapt_emb8",
            "conditioning_mode": "product",
            "condition_kernel_scale": 0.25,
            "condition_metric": "whitened",
            "adaptive_condition_bandwidth": True,
            "adaptive_target_bandwidth": True,
            "adaptive_bandwidth_k": 8,
            "condition_embedding_dim": 8,
            "condition_embedding_hidden_dim": 16,
        },
    ]

    drifting_base = {
        "channel": args.channel,
        "n": args.code_dim,
        "ebno-db": args.ebno_db,
        "rate": args.rate,
        "dataset-size": args.implant_dataset_size,
        "batch-size": args.implant_batch_size,
        "epochs": args.drifting_epochs,
        "hidden-dim": 128,
        "latent-dim": 16,
        "target-kernel-scale": 1.0,
    }

    for family, candidates in [("drifting_direct", direct_candidates), ("drifting_residual", residual_candidates)]:
        for candidate in candidates:
            candidate_args = dict(candidate)
            candidate_args.pop("tag")
            register_name = drifting_register_name(
                channel=args.channel,
                code_dim=args.code_dim,
                seed=seed,
                family=family,
                conditioning_mode=candidate_args.get("conditioning_mode", "none"),
                condition_scale=candidate_args.get("condition_kernel_scale", 0.0),
                target_scale=candidate_args.get("target_kernel_scale", 1.0),
                condition_metric=candidate_args.get("condition_metric", "euclidean"),
                condition_bandwidth=candidate_args.get("condition_bandwidth"),
                target_bandwidth=candidate_args.get("target_bandwidth"),
                local_condition_k=candidate_args.get("local_condition_k", 32),
                condition_radius=candidate_args.get("condition_radius"),
                mixture_alpha=candidate_args.get("mixture_alpha", 0.5),
                target_kernel_mode=candidate_args.get("target_kernel_mode", "raw"),
                adaptive_condition_bandwidth=candidate_args.get("adaptive_condition_bandwidth", False),
                adaptive_target_bandwidth=candidate_args.get("adaptive_target_bandwidth", False),
                adaptive_bandwidth_k=candidate_args.get("adaptive_bandwidth_k", 16),
                condition_embedding_dim=candidate_args.get("condition_embedding_dim", 0),
            )
            implant_specs.append(
                ImplantSpec(
                    name=f"{family}_{candidate['tag']}",
                    family=family,
                    register_name=register_name,
                    train_args={**drifting_base, **candidate_args},
                )
            )
            run_specs.append(
                RunSpec(
                    name=f"{family}_{candidate['tag']}_to_analytic",
                    train_implant="checkpoint",
                    train_implant_name=register_name,
                    eval_implant="analytic_channel",
                    eval_implant_name=None,
                    save_ae_name=f"symbolic_{args.channel.lower()}_{family}_{candidate['tag']}_seed{seed}",
                    benchmark_args=benchmark_base,
                )
            )

    return implant_specs, run_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a broad overnight symbolic channel surrogate-training suite.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "TDL", "OptFib"])
    parser.add_argument("--code-dim", type=int, default=7)
    parser.add_argument("--rate", type=float, default=4.0 / 7.0)
    parser.add_argument("--ebno-db", type=float, default=5.0)
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--implant-dataset-size", type=int, default=120_000)
    parser.add_argument("--implant-batch-size", type=int, default=512)
    parser.add_argument("--drifting-epochs", type=int, default=60)
    parser.add_argument("--wgan-epochs", type=int, default=60)
    parser.add_argument("--diffusion-dataset-size", type=int, default=300_000)
    parser.add_argument("--diffusion-batch-size", type=int, default=500)
    parser.add_argument("--diffusion-epochs", type=int, default=10)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-num-steps", type=int, default=50)
    parser.add_argument("--diffusion-ddim-steps", type=int, default=0, help="If <= 0, use the full diffusion trajectory.")
    parser.add_argument("--diffusion-profile", type=str, default="muah_symbolic", choices=["paper2309", "muah_symbolic"])
    parser.add_argument("--ae-dataset-size", type=int, default=1_000_000)
    parser.add_argument("--ae-batch-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=1)
    return parser


def run_command(cmd: list[str], log_path: Path) -> dict[str, Any]:
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


def artifact_exists(name: str, weights_root: str) -> bool:
    try:
        resolve_registered_artifact(name, weights_root)
        return True
    except KeyError:
        return False


def filter_specs(items: list[object], allowed_prefixes: set[str]) -> list[object]:
    if not allowed_prefixes:
        return items
    filtered: list[object] = []
    for item in items:
        name = getattr(item, "name")
        base = name.split("_to_")[0]
        if name in allowed_prefixes or base in allowed_prefixes or any(name.startswith(prefix) for prefix in allowed_prefixes):
            filtered.append(item)
    return filtered


def main() -> None:
    args = build_parser().parse_args()
    if args.channel in {"SSPA", "TDL", "OptFib"} and (args.code_dim % 2 != 0):
        raise ValueError(f"{args.channel} requires an even code dimension, got {args.code_dim}.")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.suite_dir) if args.suite_dir else ROOT / "results" / f"symbolic_{args.channel.lower()}_overnight_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    implant_specs, run_specs = build_default_matrix(args)
    requested = {part.strip() for part in args.methods.split(",") if part.strip()}
    implant_specs = filter_specs(implant_specs, requested)
    run_specs = filter_specs(run_specs, requested)

    manifest = {
        "seed": args.seed,
        "device": args.device,
        "channel": args.channel,
        "code_dim": args.code_dim,
        "rate": args.rate,
        "ebno_db": args.ebno_db,
        "weights_root": str(Path(args.weights_root).resolve()),
        "force_retrain": bool(args.force_retrain),
        "implant_specs": [asdict(spec) for spec in implant_specs],
        "run_specs": [asdict(spec) for spec in run_specs],
        "global_args": {
            "implant_dataset_size": args.implant_dataset_size,
            "implant_batch_size": args.implant_batch_size,
            "drifting_epochs": args.drifting_epochs,
            "wgan_epochs": args.wgan_epochs,
            "diffusion_dataset_size": args.diffusion_dataset_size,
            "diffusion_batch_size": args.diffusion_batch_size,
            "diffusion_epochs": args.diffusion_epochs,
            "diffusion_hidden_dim": args.diffusion_hidden_dim,
            "diffusion_num_steps": args.diffusion_num_steps,
            "diffusion_ddim_steps": args.diffusion_ddim_steps,
            "diffusion_profile": args.diffusion_profile,
            "ae_dataset_size": args.ae_dataset_size,
            "ae_batch_size": args.ae_batch_size,
            "eval_size": args.eval_size,
            "ae_epochs": args.ae_epochs,
            "ae_learning_rate": args.ae_learning_rate,
            "eval_every": args.eval_every,
        },
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: dict[str, Any] = {"manifest": manifest, "implant_runs": {}, "ae_runs": {}}

    for spec in implant_specs:
        log_path = suite_dir / f"implant_{spec.name}.log"
        if artifact_exists(spec.register_name, args.weights_root) and not args.force_retrain:
            results["implant_runs"][spec.name] = {
                "spec": asdict(spec),
                "status": "reused",
                "register_name": spec.register_name,
            }
            (suite_dir / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        cmd = [
            sys.executable,
            str(TRAIN_IMPLANT),
            "--family",
            spec.family,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--weights-root",
            args.weights_root,
            "--register-name",
            spec.register_name,
        ]
        for key, value in spec.train_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])

        result = run_command(cmd, log_path)
        results["implant_runs"][spec.name] = {
            "spec": asdict(spec),
            "status": "trained",
            "register_name": spec.register_name,
            "runner_result": result,
        }
        (suite_dir / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for spec in run_specs:
        out_dir = suite_dir / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = suite_dir / f"{spec.name}.log"
        cmd = [
            sys.executable,
            str(RUN_AE),
            "--device",
            args.device,
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
        for key, value in spec.benchmark_args.items():
            cmd.extend([f"--{key}", str(value)])

        result = run_command(cmd, log_path)
        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        results["ae_runs"][spec.name] = {
            "spec": asdict(spec),
            "runner_result": result,
            "summary_path": str(summary_path.resolve()),
            "final_eval_ser": summary["final_eval"]["ser"],
            "final_eval_ber": summary["final_eval"].get("ber"),
            "train_implant_metrics": summary.get("train_implant_channel_metrics_vs_analytic"),
        }
        (suite_dir / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary": str((suite_dir / "suite_results.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
