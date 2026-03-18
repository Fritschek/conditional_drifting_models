from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".mplcache"))
sys.path.insert(0, ROOT)

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from conditional_drifting.baselines import (
    DiffusionConfig,
    GANConfig,
    PaperWGANConfig,
    evaluate_diffusion_model,
    evaluate_gan_model,
    evaluate_paper_wgan,
    train_conditional_diffusion,
    train_conditional_gan,
    train_paper_wgan,
)
from conditional_drifting.channels import channel_registry
from conditional_drifting.training import DriftingConfig, evaluate_residual_model, select_device, set_seed, train_conditional_drifting


@dataclass(frozen=True)
class PaperChannelPreset:
    n: int
    ebn0_db: float
    rate: float
    diffusion_hidden_dim: int
    wgan_hidden_dim: int
    diffusion_epochs: int = 30
    drifting_epochs: int = 30
    wgan_epochs: int = 30
    dataset_size: int = 10_000_000
    batch_size: int = 5_000
    eval_size: int = 10_000_000
    swd_projections: int = 128
    num_steps: int = 100
    diffusion_learning_rate_schedule: tuple[tuple[int, float], ...] | None = None


PAPER2309_PRESETS: dict[str, PaperChannelPreset] = {
    "AWGN": PaperChannelPreset(
        n=7,
        ebn0_db=5.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=110,
        wgan_hidden_dim=128,
        diffusion_learning_rate_schedule=((10, 1e-3), (20, 1e-4)),
    ),
    "Rayleigh": PaperChannelPreset(
        n=7,
        ebn0_db=12.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=128,
        wgan_hidden_dim=256,
        diffusion_learning_rate_schedule=((10, 1e-3), (20, 1e-4)),
    ),
    "SSPA": PaperChannelPreset(n=8, ebn0_db=8.0, rate=6.0 / 8.0, diffusion_hidden_dim=110, wgan_hidden_dim=256, diffusion_epochs=160, drifting_epochs=160, wgan_epochs=160, batch_size=4096),
}

PAPER2309_REFERENCE_SWD = {
    "AWGN": {"ddpm": 0.012, "ddim_100": 0.008, "ddim_50": 0.011, "ddim_20": 0.022, "ddim_10": 0.043, "wgan": 0.013},
    "Rayleigh": {"ddpm": 0.013, "ddim_100": 0.008, "ddim_50": 0.011, "ddim_20": 0.024, "ddim_10": 0.043, "wgan": 0.019},
    "SSPA": {"ddpm": 0.009, "ddim_100": 0.007, "ddim_50": 0.009, "ddim_20": 0.018, "ddim_10": 0.032, "wgan": 0.104},
}


def ebno_to_noise(ebn0_db: float, rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / math.sqrt(2.0 * rate * ebn0)


def resolve_paper_diffusion_lr_schedule(
    preset: PaperChannelPreset,
    diffusion_epochs: int,
) -> tuple[tuple[int, float], ...] | None:
    schedule = preset.diffusion_learning_rate_schedule
    if schedule is None:
        return None
    if sum(stage_epochs for stage_epochs, _ in schedule) == diffusion_epochs:
        return schedule
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-2309 benchmark runner with exact paper-style presets")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh,SSPA")
    parser.add_argument("--dataset-size", type=int, default=-1, help="Override paper preset dataset size")
    parser.add_argument("--eval-size", type=int, default=-1, help="Override paper preset eval size")
    parser.add_argument("--batch-size", type=int, default=-1, help="Override paper preset batch size")
    parser.add_argument("--diffusion-epochs", type=int, default=-1)
    parser.add_argument("--drifting-epochs", type=int, default=-1)
    parser.add_argument("--wgan-epochs", type=int, default=-1)
    parser.add_argument("--diffusion-eval-batch-size", type=int, default=-1, help="Chunk size for diffusion evaluation generation")
    parser.add_argument("--diffusion-eval-log-every", type=int, default=1, help="Log every N diffusion eval chunks")
    parser.add_argument("--ddim-steps", type=str, default="100,50,20,10")
    parser.add_argument(
        "--methods",
        type=str,
        default="drifting,diffusion,wgan",
        help="Comma-separated subset of: drifting,diffusion,wgan,gan_fa",
    )
    parser.add_argument("--include-gan-fa", action="store_true", help="Also run the later GAN_FA baseline next to paper WGAN")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to results/paper2309_benchmark_seed<seed>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    set_seed(args.seed)
    channels = channel_registry()
    requested = [name.strip() for name in args.channels.split(",") if name.strip()]
    methods = {name.strip() for name in args.methods.split(",") if name.strip()}

    unknown = [name for name in requested if name not in PAPER2309_PRESETS]
    if unknown:
        raise ValueError(f"Unsupported paper2309 channels: {unknown}. Valid: {sorted(PAPER2309_PRESETS)}")

    ddim_steps = [int(part.strip()) for part in args.ddim_steps.split(",") if part.strip()]
    out_dir = args.out_dir or os.path.join(ROOT, "results", f"paper2309_benchmark_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    summary: dict[str, dict] = {
        "paper_reference_swd": PAPER2309_REFERENCE_SWD,
        "channels": {},
        "seed": args.seed,
        "device": str(device),
        "ddim_steps": ddim_steps,
    }

    for channel_name in requested:
        preset = PAPER2309_PRESETS[channel_name]
        dataset_size = args.dataset_size if args.dataset_size > 0 else preset.dataset_size
        eval_size = args.eval_size if args.eval_size > 0 else preset.eval_size
        batch_size = args.batch_size if args.batch_size > 0 else preset.batch_size
        diffusion_epochs = args.diffusion_epochs if args.diffusion_epochs > 0 else preset.diffusion_epochs
        drifting_epochs = args.drifting_epochs if args.drifting_epochs > 0 else preset.drifting_epochs
        wgan_epochs = args.wgan_epochs if args.wgan_epochs > 0 else preset.wgan_epochs
        noise_std = ebno_to_noise(preset.ebn0_db, preset.rate)

        drift_direct_cfg = DriftingConfig(
            n=preset.n,
            noise_std=noise_std,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=drifting_epochs,
            eval_size=eval_size,
            swd_projections=preset.swd_projections,
            is_residual=False,
        )
        drift_residual_cfg = DriftingConfig(
            n=preset.n,
            noise_std=noise_std,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=drifting_epochs,
            eval_size=eval_size,
            swd_projections=preset.swd_projections,
            is_residual=True,
        )
        diffusion_cfg = DiffusionConfig(
            n=preset.n,
            noise_std=noise_std,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=diffusion_epochs,
            learning_rate=1e-4,
            hidden_dim=preset.diffusion_hidden_dim,
            num_steps=preset.num_steps,
            eval_size=eval_size,
            swd_projections=preset.swd_projections,
            ema_decay=0.9,
            pred_type="v",
            is_residual=False,
            beta_schedule="cosine-zf",
            eval_batch_size=args.diffusion_eval_batch_size if args.diffusion_eval_batch_size > 0 else None,
            learning_rate_schedule=resolve_paper_diffusion_lr_schedule(preset, diffusion_epochs),
        )
        wgan_cfg = PaperWGANConfig(
            n=preset.n,
            noise_std=noise_std,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=wgan_epochs,
            eval_size=eval_size,
            hidden_dim=preset.wgan_hidden_dim,
            swd_projections=preset.swd_projections,
        )

        channel_fn = channels[channel_name]
        results: dict[str, object] = {}
        final_history: dict[str, object] = {}

        if "drifting" in methods:
            drift_direct_model, drift_direct_artifacts = train_conditional_drifting(channel_fn, drift_direct_cfg, device)
            drift_direct_eval = evaluate_residual_model(
                drift_direct_model,
                channel_fn,
                drift_direct_cfg,
                device,
                metric_seed=args.seed,
            )
            drift_residual_model, drift_residual_artifacts = train_conditional_drifting(
                channel_fn,
                drift_residual_cfg,
                device,
            )
            drift_residual_eval = evaluate_residual_model(
                drift_residual_model,
                channel_fn,
                drift_residual_cfg,
                device,
                metric_seed=args.seed,
            )
            results["drifting_direct_swd"] = drift_direct_eval["swd"]
            results["drifting_residual_swd"] = drift_residual_eval["swd"]
            # Backward-compatible alias for prior direct-output paper benchmark consumers.
            results["drifting_swd"] = drift_direct_eval["swd"]
            final_history["drifting_direct"] = drift_direct_artifacts.history[-1]
            final_history["drifting_residual"] = drift_residual_artifacts.history[-1]

        if "diffusion" in methods:
            diff_model, diff_state = train_conditional_diffusion(channel_fn, diffusion_cfg, device)
            print(f"[{channel_name}] diffusion training complete; starting evaluation", flush=True)
            eval_timings: dict[str, dict[str, float]] = {}
            eval_chunk_timings: dict[str, list[dict[str, float]]] = {}
            t_eval = time.perf_counter()
            def make_progress_callback(label: str):
                stage_start = time.perf_counter()

                def _callback(done: int, total: int, chunk_timing: dict[str, float]) -> None:
                    if args.diffusion_eval_log_every <= 0:
                        return
                    if done % args.diffusion_eval_log_every != 0 and done != total:
                        return
                    elapsed = time.perf_counter() - stage_start
                    eta = elapsed / done * max(total - done, 0)
                    print(
                        f"[{channel_name}] {label} chunk {done}/{total}: "
                        f"batch={int(chunk_timing['batch_size'])}, "
                        f"sample={chunk_timing['sample_sec']:.2f}s, "
                        f"total={chunk_timing['total_sec']:.2f}s, "
                        f"elapsed={elapsed:.2f}s, eta={eta:.2f}s",
                        flush=True,
                    )

                return _callback

            ddpm_eval = evaluate_diffusion_model(
                diff_model,
                channel_fn,
                diffusion_cfg,
                device,
                use_ddim=False,
                metric_seed=args.seed,
                progress_callback=make_progress_callback("DDPM"),
            )
            eval_timings["ddpm"] = ddpm_eval["timings"]
            eval_chunk_timings["ddpm"] = ddpm_eval["chunk_timings"]
            print(
                f"[{channel_name}] DDPM done: swd={ddpm_eval['swd']:.6f}, "
                f"sample={ddpm_eval['timings']['sample_sec']:.2f}s, swd_eval={ddpm_eval['timings']['swd_sec']:.2f}s",
                flush=True,
            )
            ddim_results = {}
            for steps in ddim_steps:
                ddim_eval = evaluate_diffusion_model(
                    diff_model,
                    channel_fn,
                    diffusion_cfg,
                    device,
                    use_ddim=True,
                    ddim_steps=steps,
                    metric_seed=args.seed,
                    progress_callback=make_progress_callback(f"DDIM-{steps}"),
                )
                ddim_results[str(steps)] = ddim_eval["swd"]
                eval_timings[f"ddim_{steps}"] = ddim_eval["timings"]
                eval_chunk_timings[f"ddim_{steps}"] = ddim_eval["chunk_timings"]
                print(
                    f"[{channel_name}] DDIM-{steps} done: swd={ddim_eval['swd']:.6f}, "
                    f"sample={ddim_eval['timings']['sample_sec']:.2f}s, swd_eval={ddim_eval['timings']['swd_sec']:.2f}s",
                    flush=True,
                )
            results["ddpm_swd"] = ddpm_eval["swd"]
            results["ddim_swd"] = ddim_results
            final_history["diffusion"] = diff_state["history"][-1]
            final_history["diffusion_eval_timings"] = eval_timings
            final_history["diffusion_eval_chunk_timings"] = eval_chunk_timings
            final_history["diffusion_eval_total_sec"] = time.perf_counter() - t_eval

        if "wgan" in methods:
            wgan_model, wgan_state = train_paper_wgan(channel_fn, wgan_cfg, device)
            wgan_eval = evaluate_paper_wgan(wgan_model, channel_fn, wgan_cfg, device, metric_seed=args.seed)
            results["paper_wgan_swd"] = wgan_eval["swd"]
            final_history["paper_wgan"] = wgan_state["history"][-1]

        channel_summary = {
            "preset": asdict(preset),
            "noise_std": noise_std,
            "effective_config": {
                "dataset_size": dataset_size,
                "eval_size": eval_size,
                "batch_size": batch_size,
                "diffusion_epochs": diffusion_epochs,
                "drifting_epochs": drifting_epochs,
                "wgan_epochs": wgan_epochs,
            },
            "paper_reference_swd": PAPER2309_REFERENCE_SWD[channel_name],
            "results": results,
            "final_history": final_history,
        }

        if args.include_gan_fa or "gan_fa" in methods:
            gan_cfg = GANConfig(
                n=preset.n,
                noise_std=noise_std,
                dataset_size=dataset_size,
                batch_size=batch_size,
                epochs=wgan_epochs,
                hidden_dim=preset.wgan_hidden_dim,
                eval_size=eval_size,
                swd_projections=preset.swd_projections,
            )
            gan_model, gan_state = train_conditional_gan(channel_fn, gan_cfg, device)
            gan_eval = evaluate_gan_model(gan_model, channel_fn, gan_cfg, device, metric_seed=args.seed)
            channel_summary["results"]["gan_fa_swd"] = gan_eval["swd"]
            channel_summary["final_history"]["gan_fa"] = gan_state["history"][-1]

        summary["channels"][channel_name] = channel_summary

    out_json = os.path.join(out_dir, "paper2309_benchmark_summary.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    output = {"output_json": out_json, "output_dir": out_dir, "channels": requested}
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        labels = requested
        x = np.arange(len(labels))
        plot_specs = []
        if "diffusion" in methods:
            plot_specs.append(("DDPM", [summary["channels"][name]["results"]["ddpm_swd"] for name in labels]))
        if "wgan" in methods:
            plot_specs.append(("Paper WGAN", [summary["channels"][name]["results"]["paper_wgan_swd"] for name in labels]))
        if "drifting" in methods:
            plot_specs.append(("Drifting Direct", [summary["channels"][name]["results"]["drifting_direct_swd"] for name in labels]))
            plot_specs.append(("Drifting Residual", [summary["channels"][name]["results"]["drifting_residual_swd"] for name in labels]))

        width = 0.8 / max(len(plot_specs), 1)
        offsets = np.linspace(-(len(plot_specs) - 1) / 2.0, (len(plot_specs) - 1) / 2.0, num=len(plot_specs))
        for offset, (label, values) in zip(offsets, plot_specs):
            ax.bar(x + offset * width, values, width=width, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Residual SWD")
        ax.set_title("Paper-2309 preset benchmark")
        if plot_specs:
            ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, "paper2309_benchmark_plot.png")
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output["plot"] = plot_path
    else:
        print("[benchmark] matplotlib not available; skipping plot generation", flush=True)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
