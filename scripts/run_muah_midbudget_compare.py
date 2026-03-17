from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from conditional_drifting.metrics import sliced_wasserstein_distance_torch
from conditional_drifting.training import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mid-budget diffusion comparison using Muah's original code path.")
    parser.add_argument("--muah-repo", type=Path, default=Path("/home/entropy/GitHub/DM_for_learning_channels"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=str, default="AWGN,Rayleigh")
    parser.add_argument("--dataset-size", type=int, default=30000)
    parser.add_argument("--eval-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=str, default="100,50,20,10")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "muah_midbudget_compare.json")
    return parser.parse_args()


def import_muah_modules(muah_repo: Path):
    src = muah_repo / "src"
    sys.path.insert(0, str(src))
    import channel_models  # type: ignore
    import ema  # type: ignore
    import models  # type: ignore
    import trainer  # type: ignore
    import utils  # type: ignore

    return channel_models, ema, models, trainer, utils


def ebno_to_noise(ebn0_db: float, rate: float) -> float:
    ebno = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / math.sqrt(2.0 * rate * ebno)


def build_paper_channel_config(channel_name: str) -> dict[str, float | int | tuple[tuple[int, float], ...] | None]:
    if channel_name == "AWGN":
        return {
            "n": 7,
            "ebn0_db": 5.0,
            "rate": 4.0 / 7.0,
            "hidden_dim": 110,
            "lr_schedule": ((30, 1e-4),),
        }
    if channel_name == "Rayleigh":
        return {
            "n": 7,
            "ebn0_db": 12.0,
            "rate": 4.0 / 7.0,
            "hidden_dim": 128,
            "lr_schedule": ((10, 1e-3), (20, 1e-4)),
        }
    raise ValueError(f"Unsupported channel: {channel_name}")


def train_muah_diffusion(
    channel_name: str,
    channel_fn,
    models,
    trainer,
    ema_mod,
    utils,
    *,
    device,
    dataset_size: int,
    batch_size: int,
    num_steps: int,
) -> tuple[object, dict]:
    cfg = build_paper_channel_config(channel_name)
    n = int(cfg["n"])
    noise_std = ebno_to_noise(float(cfg["ebn0_db"]), float(cfg["rate"]))
    hidden_dim = int(cfg["hidden_dim"])
    lr_schedule = cfg["lr_schedule"]

    betas = utils.make_beta_schedule(schedule="cosine-zf", n_timesteps=num_steps).to(device)
    channel_gen = models.ConditionalModel_w_Condition(num_steps, 0, n, hidden_dim).to(device)
    ema = ema_mod.EMA(0.9)
    ema.register(channel_gen)

    for epochs, lr in lr_schedule:
        tconf = trainer.TrainerConfig_DDM(
            max_epochs=epochs,
            dataset_size=dataset_size,
            batch_size=batch_size,
            noise_std=noise_std,
            learning_rate=lr,
            M=16,
            n=n,
            rate=float(cfg["rate"]),
            num_steps=num_steps,
            betas=betas,
            optim_gen=torch.optim.Adam,
            pred_type="v",
        )
        t = trainer.Trainer_DDM(channel_gen, ema, device, channel_fn, tconf)
        t.train_PreT()

    state = {
        "n": n,
        "noise_std": noise_std,
        "betas": betas.detach().cpu(),
        "alphas": (1.0 - betas).detach().cpu(),
        "alphas_prod": torch.cumprod(1.0 - betas, dim=0).detach().cpu(),
        "alphas_bar_sqrt": torch.sqrt(torch.cumprod(1.0 - betas, dim=0)).detach().cpu(),
        "one_minus_alphas_bar_sqrt": torch.sqrt(1.0 - torch.cumprod(1.0 - betas, dim=0)).detach().cpu(),
        "noise_std_value": noise_std,
    }
    return channel_gen.eval(), state


@torch.no_grad()
def evaluate_muah_diffusion(
    model,
    channel_fn,
    utils,
    *,
    device,
    n: int,
    noise_std: float,
    eval_size: int,
    num_steps: int,
    ddim_steps: list[int],
    state: dict,
) -> dict[str, float]:
    x = torch.randn(eval_size, n, device=device)
    y_true = channel_fn(x, noise_std, device)
    results: dict[str, float] = {}

    ddpm_seq = utils.p_sample_loop_w_Condition(
        model,
        x.size(),
        num_steps,
        state["alphas"].to(device),
        state["betas"].to(device),
        state["alphas_bar_sqrt"].to(device),
        state["one_minus_alphas_bar_sqrt"].to(device),
        x,
        pred_type="v",
        is_light=True,
    )
    results["ddpm"] = float(
        sliced_wasserstein_distance_torch(y_true, ddpm_seq, num_projections=128, seed=7).item()
    )

    for steps in ddim_steps:
        skip = num_steps // steps
        traj = range(skip - 1, num_steps, skip)
        ddim_seq = utils.p_sample_loop_w_Condition_DDIM(
            model,
            x.size(),
            traj,
            state["alphas_prod"].to(device),
            state["alphas_bar_sqrt"].to(device),
            state["one_minus_alphas_bar_sqrt"].to(device),
            x,
            pred_type="v",
        )[-1]
        results[f"ddim_{steps}"] = float(
            sliced_wasserstein_distance_torch(y_true, ddim_seq, num_projections=128, seed=7).item()
        )
    return results


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    channel_models, ema_mod, models, trainer, utils = import_muah_modules(args.muah_repo)

    channel_lookup = {
        "AWGN": channel_models.ch_AWGN,
        "Rayleigh": channel_models.ch_Rayleigh_AWGN,
    }

    ddim_steps = [int(part.strip()) for part in args.ddim_steps.split(",") if part.strip()]
    requested = [name.strip() for name in args.channels.split(",") if name.strip()]
    summary = {
        "muah_repo": str(args.muah_repo),
        "device": str(device),
        "seed": args.seed,
        "dataset_size": args.dataset_size,
        "eval_size": args.eval_size,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "ddim_steps": ddim_steps,
        "channels": {},
    }

    for channel_name in requested:
        model, state = train_muah_diffusion(
            channel_name,
            channel_lookup[channel_name],
            models,
            trainer,
            ema_mod,
            utils,
            device=device,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
        )
        metrics = evaluate_muah_diffusion(
            model,
            channel_lookup[channel_name],
            utils,
            device=device,
            n=state["n"],
            noise_std=state["noise_std_value"],
            eval_size=args.eval_size,
            num_steps=args.num_steps,
            ddim_steps=ddim_steps,
            state=state,
        )
        summary["channels"][channel_name] = metrics
        print(channel_name, metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
