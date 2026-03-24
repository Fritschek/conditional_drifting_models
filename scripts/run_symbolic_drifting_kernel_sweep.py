from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_IMPLANT = ROOT / "scripts" / "train_symbol_pair_implant.py"
RUN_AE = ROOT / "scripts" / "run_symbolic_awgn_benchmark.py"


@dataclass(frozen=True)
class SweepSpec:
    family: str
    conditioning_mode: str
    condition_scale: float
    target_scale: float
    condition_metric: str
    condition_bandwidth: float | None
    target_bandwidth: float | None
    local_condition_k: int
    condition_radius: float | None
    mixture_alpha: float
    target_kernel_mode: str
    residual_target_scale: float
    adaptive_condition_bandwidth: bool
    adaptive_target_bandwidth: bool
    adaptive_bandwidth_k: int
    condition_embedding_dim: int
    condition_embedding_hidden_dim: int


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def fmt_scale(value: float) -> str:
    text = f"{value:.4g}".replace("-", "m").replace(".", "p")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep conditioning-aware drifting kernel scales on the symbolic AWGN benchmark.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--conditioning-modes", type=str, default="joint")
    parser.add_argument("--condition-metric", type=str, default="euclidean", choices=["euclidean", "whitened"])
    parser.add_argument("--condition-scales", type=str, default="0,0.25,0.5,1,2")
    parser.add_argument("--target-scales", type=str, default="1")
    parser.add_argument("--condition-bandwidths", type=str, default="")
    parser.add_argument("--target-bandwidths", type=str, default="")
    parser.add_argument("--local-condition-ks", type=str, default="32")
    parser.add_argument("--condition-radii", type=str, default="")
    parser.add_argument("--mixture-alphas", type=str, default="0.5")
    parser.add_argument("--target-kernel-mode", type=str, default="raw", choices=["raw", "raw_plus_residual"])
    parser.add_argument("--residual-target-scale", type=float, default=1.0)
    parser.add_argument("--adaptive-condition-bandwidth", action="store_true")
    parser.add_argument("--adaptive-target-bandwidth", action="store_true")
    parser.add_argument("--adaptive-bandwidth-k", type=int, default=16)
    parser.add_argument("--condition-embedding-dim", type=int, default=0)
    parser.add_argument("--condition-embedding-hidden-dim", type=int, default=64)
    parser.add_argument("--families", type=str, default="drifting_direct,drifting_residual")
    parser.add_argument("--implant-epochs", type=int, default=60)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ae-dataset-size", type=int, default=1000000)
    parser.add_argument("--ae-batch-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=100000)
    parser.add_argument("--suite-dir", type=str, default="")
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
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
    suite_dir = Path(args.suite_dir) if args.suite_dir else ROOT / "results" / f"symbolic_drifting_kernel_sweep_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    families = [part.strip() for part in args.families.split(",") if part.strip()]
    conditioning_modes = [part.strip() for part in args.conditioning_modes.split(",") if part.strip()]
    condition_scales = parse_csv_floats(args.condition_scales)
    target_scales = parse_csv_floats(args.target_scales)
    condition_bandwidths = [None] if not args.condition_bandwidths.strip() else parse_csv_floats(args.condition_bandwidths)
    target_bandwidths = [None] if not args.target_bandwidths.strip() else parse_csv_floats(args.target_bandwidths)
    local_condition_ks = [int(part.strip()) for part in args.local_condition_ks.split(",") if part.strip()]
    condition_radii = [None] if not args.condition_radii.strip() else parse_csv_floats(args.condition_radii)
    mixture_alphas = parse_csv_floats(args.mixture_alphas)
    specs = [
        SweepSpec(
            family=family,
            conditioning_mode=conditioning_mode,
            condition_scale=condition_scale,
            target_scale=target_scale,
            condition_metric=args.condition_metric,
            condition_bandwidth=condition_bandwidth,
            target_bandwidth=target_bandwidth,
            local_condition_k=local_condition_k,
            condition_radius=condition_radius,
            mixture_alpha=mixture_alpha,
            target_kernel_mode=args.target_kernel_mode,
            residual_target_scale=args.residual_target_scale,
            adaptive_condition_bandwidth=args.adaptive_condition_bandwidth,
            adaptive_target_bandwidth=args.adaptive_target_bandwidth,
            adaptive_bandwidth_k=args.adaptive_bandwidth_k,
            condition_embedding_dim=args.condition_embedding_dim,
            condition_embedding_hidden_dim=args.condition_embedding_hidden_dim,
        )
        for family in families
        for conditioning_mode in conditioning_modes
        for condition_scale in condition_scales
        for target_scale in target_scales
        for condition_bandwidth in condition_bandwidths
        for target_bandwidth in target_bandwidths
        for local_condition_k in local_condition_ks
        for condition_radius in condition_radii
        for mixture_alpha in mixture_alphas
    ]

    manifest = {
        "seed": args.seed,
        "device": args.device,
        "conditioning_modes": conditioning_modes,
        "condition_metric": args.condition_metric,
        "condition_scales": condition_scales,
        "target_scales": target_scales,
        "condition_bandwidths": condition_bandwidths,
        "target_bandwidths": target_bandwidths,
        "local_condition_ks": local_condition_ks,
        "condition_radii": condition_radii,
        "mixture_alphas": mixture_alphas,
        "target_kernel_mode": args.target_kernel_mode,
        "residual_target_scale": args.residual_target_scale,
        "adaptive_condition_bandwidth": args.adaptive_condition_bandwidth,
        "adaptive_target_bandwidth": args.adaptive_target_bandwidth,
        "adaptive_bandwidth_k": args.adaptive_bandwidth_k,
        "condition_embedding_dim": args.condition_embedding_dim,
        "condition_embedding_hidden_dim": args.condition_embedding_hidden_dim,
        "families": families,
        "implant_epochs": args.implant_epochs,
        "ae_epochs": args.ae_epochs,
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "ae_dataset_size": args.ae_dataset_size,
        "ae_batch_size": args.ae_batch_size,
        "eval_size": args.eval_size,
        "weights_root": str(Path(args.weights_root).resolve()),
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: dict[str, object] = {"manifest": manifest, "runs": []}

    for spec in specs:
        mode_tag = spec.conditioning_mode
        cond_tag = fmt_scale(spec.condition_scale)
        targ_tag = fmt_scale(spec.target_scale)
        cbw_tag = "auto" if spec.condition_bandwidth is None else fmt_scale(spec.condition_bandwidth)
        tbw_tag = "auto" if spec.target_bandwidth is None else fmt_scale(spec.target_bandwidth)
        k_tag = f"k{spec.local_condition_k}"
        radius_tag = "auto" if spec.condition_radius is None else fmt_scale(spec.condition_radius)
        mix_tag = fmt_scale(spec.mixture_alpha)
        metric_tag = spec.condition_metric
        embed_tag = f"emb{spec.condition_embedding_dim}"
        adapt_tag = f"ac{int(spec.adaptive_condition_bandwidth)}_at{int(spec.adaptive_target_bandwidth)}_ak{spec.adaptive_bandwidth_k}"
        target_mode_tag = spec.target_kernel_mode.replace("_", "")
        run_name = f"{spec.family}_{mode_tag}_{metric_tag}_cond{cond_tag}_targ{targ_tag}_cbw{cbw_tag}_tbw{tbw_tag}_{k_tag}_rad{radius_tag}_mix{mix_tag}_{target_mode_tag}_{adapt_tag}_{embed_tag}"
        register_name = f"awgn7_{spec.family}_{mode_tag}_{metric_tag}_cond{cond_tag}_targ{targ_tag}_cbw{cbw_tag}_tbw{tbw_tag}_{k_tag}_rad{radius_tag}_mix{mix_tag}_{target_mode_tag}_{adapt_tag}_{embed_tag}_seed{args.seed}"
        ae_name = f"symbolic_awgn_{spec.family}_{mode_tag}_{metric_tag}_cond{cond_tag}_targ{targ_tag}_cbw{cbw_tag}_tbw{tbw_tag}_{k_tag}_rad{radius_tag}_mix{mix_tag}_{target_mode_tag}_{adapt_tag}_{embed_tag}_seed{args.seed}"

        implant_log = suite_dir / f"{run_name}_implant.log"
        implant_cmd = [
            sys.executable,
            str(TRAIN_IMPLANT),
            "--family",
            spec.family,
            "--channel",
            "AWGN",
            "--n",
            "7",
            "--ebno-db",
            "5",
            "--rate",
            str(4.0 / 7.0),
            "--dataset-size",
            str(args.dataset_size),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.implant_epochs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--weights-root",
            args.weights_root,
            "--register-name",
            register_name,
        ]
        if spec.conditioning_mode == "joint" and spec.condition_scale > 0.0:
            implant_cmd.append("--conditional-kernel")
        implant_cmd.extend(["--conditioning-mode", spec.conditioning_mode])
        implant_cmd.extend(["--condition-metric", spec.condition_metric])
        implant_cmd.extend(["--condition-kernel-scale", str(spec.condition_scale)])
        implant_cmd.extend(["--target-kernel-scale", str(spec.target_scale)])
        if spec.condition_bandwidth is not None:
            implant_cmd.extend(["--condition-bandwidth", str(spec.condition_bandwidth)])
        if spec.target_bandwidth is not None:
            implant_cmd.extend(["--target-bandwidth", str(spec.target_bandwidth)])
        implant_cmd.extend(["--local-condition-k", str(spec.local_condition_k)])
        if spec.condition_radius is not None:
            implant_cmd.extend(["--condition-radius", str(spec.condition_radius)])
        implant_cmd.extend(["--mixture-alpha", str(spec.mixture_alpha)])
        implant_cmd.extend(["--target-kernel-mode", spec.target_kernel_mode])
        implant_cmd.extend(["--residual-target-scale", str(spec.residual_target_scale)])
        if spec.adaptive_condition_bandwidth:
            implant_cmd.append("--adaptive-condition-bandwidth")
        if spec.adaptive_target_bandwidth:
            implant_cmd.append("--adaptive-target-bandwidth")
        implant_cmd.extend(["--adaptive-bandwidth-k", str(spec.adaptive_bandwidth_k)])
        implant_cmd.extend(["--condition-embedding-dim", str(spec.condition_embedding_dim)])
        implant_cmd.extend(["--condition-embedding-hidden-dim", str(spec.condition_embedding_hidden_dim)])
        implant_result = run_command(implant_cmd, implant_log)

        ae_out_dir = suite_dir / run_name
        ae_log = suite_dir / f"{run_name}_ae.log"
        ae_cmd = [
            sys.executable,
            str(RUN_AE),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--train-implant",
            "checkpoint",
            "--train-implant-name",
            register_name,
            "--eval-implant",
            "analytic_awgn",
            "--batch-size",
            str(args.ae_batch_size),
            "--dataset-size",
            str(args.ae_dataset_size),
            "--eval-size",
            str(args.eval_size),
            "--epochs",
            str(args.ae_epochs),
            "--save-ae-name",
            ae_name,
            "--weights-root",
            args.weights_root,
            "--out-dir",
            str(ae_out_dir),
        ]
        ae_result = run_command(ae_cmd, ae_log)
        ae_summary = json.loads((ae_out_dir / "summary.json").read_text())

        results["runs"].append(
            {
                "family": spec.family,
                "conditioning_mode": spec.conditioning_mode,
                "condition_scale": spec.condition_scale,
                "target_scale": spec.target_scale,
                "condition_metric": spec.condition_metric,
                "condition_bandwidth": spec.condition_bandwidth,
                "target_bandwidth": spec.target_bandwidth,
                "local_condition_k": spec.local_condition_k,
                "condition_radius": spec.condition_radius,
                "mixture_alpha": spec.mixture_alpha,
                "target_kernel_mode": spec.target_kernel_mode,
                "residual_target_scale": spec.residual_target_scale,
                "adaptive_condition_bandwidth": spec.adaptive_condition_bandwidth,
                "adaptive_target_bandwidth": spec.adaptive_target_bandwidth,
                "adaptive_bandwidth_k": spec.adaptive_bandwidth_k,
                "condition_embedding_dim": spec.condition_embedding_dim,
                "condition_embedding_hidden_dim": spec.condition_embedding_hidden_dim,
                "run_name": run_name,
                "implant_register_name": register_name,
                "ae_name": ae_name,
                "implant_result": implant_result,
                "ae_result": ae_result,
                "summary_path": str((ae_out_dir / "summary.json").resolve()),
                "final_eval_ser": ae_summary["final_eval"]["ser"],
                "channel_metrics": ae_summary["train_implant_channel_metrics_vs_analytic"],
            }
        )
        (suite_dir / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps({"suite_dir": str(suite_dir.resolve()), "summary": str((suite_dir / "suite_results.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
