"""
Corrected long-run AWGN comparison:
  - Diffusion (DDPM and DDIM samplers)
  - Drifting (attraction + repulsion objective)

Run:
  MPLCONFIGDIR=/Users/rickfritschek/Documents/GitHub/DM_for_learning_channels/.mplcache \
  /Users/rickfritschek/Documents/GitHub/turbo_mingru_decoder/.venv/bin/python \
  examples/compare_awgn_ddpm_ddim_drifting_corrected.py
"""

import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

from channel_models import ch_AWGN
from models import ConditionalModel_w_Condition
from trainer import TrainerConfig_DDM, Trainer_DDM
from ema import EMA
from drifting import ConditionalDriftingGenerator, TrainerConfig_Drifting, Trainer_Drifting
import utils


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ddim_traj(num_steps, ddim_steps):
    raw = np.linspace(0, num_steps - 1, ddim_steps)
    traj = np.unique(np.round(raw).astype(int)).tolist()
    traj[0] = 0
    traj[-1] = num_steps - 1
    return traj


def l1_hist_distance(a, b, bins=120, lim=2.5):
    ha, edges = np.histogram(a, bins=bins, range=(-lim, lim), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(-lim, lim), density=True)
    return float(np.sum(np.abs(ha - hb)) * (edges[1] - edges[0]))


def sliced_wasserstein_distance(x, y, num_projections=256, seed=12345):
    """
    Sliced Wasserstein-1 distance between two point clouds x, y (N x d).
    """
    assert x.ndim == 2 and y.ndim == 2
    assert x.shape[1] == y.shape[1]
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    d = x.shape[1]
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((num_projections, d))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)

    sw = 0.0
    for v in dirs:
        xp = np.sort(x @ v)
        yp = np.sort(y @ v)
        sw += np.mean(np.abs(xp - yp))
    return float(sw / num_projections)


def ch_awgn_residual(x, noise_std, device):
    return ch_AWGN(x, noise_std, device) - x


def main():
    set_seed(7)
    device = torch.device("cpu")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    # Shared setup
    n = 2
    noise_std = 0.3
    batch_size = 256
    dataset_size = 120_000
    max_epochs = 60
    eval_size = 12_000

    # Diffusion (tuned) setup
    num_steps = 20
    ddim_fast_steps = 10
    betas = utils.make_beta_schedule(
        schedule="cosine",
        n_timesteps=num_steps,
        start=1e-4,
        end=2e-2,
    )
    cfg_ddm = TrainerConfig_DDM(
        max_epochs=max_epochs,
        dataset_size=dataset_size,
        batch_size=batch_size,
        noise_std=noise_std,
        learning_rate=1e-3,
        M=0,
        n=n,
        num_steps=num_steps,
        betas=betas,
        IS_RES=True,
        pred_type="epsilon",
    )
    ddm_model = ConditionalModel_w_Condition(num_steps, 0, n, 128).to(device)
    ddm_ema = EMA(mu=0.995)
    ddm_ema.register(ddm_model)
    ddm_trainer = Trainer_DDM(ddm_model, ddm_ema, device, ch_AWGN, cfg_ddm)
    ddm_loss = ddm_trainer.train_PreT()
    ddm_ema.ema(ddm_model)
    ddm_model.eval()

    # Drifting (corrected objective) setup
    drift_model = ConditionalDriftingGenerator(
        condition_dim=n,
        output_dim=n,
        latent_dim=16,
        hidden_dim=128,
    ).to(device)
    cfg_drift = TrainerConfig_Drifting(
        max_epochs=max_epochs,
        dataset_size=dataset_size,
        batch_size=batch_size,
        noise_std=noise_std,
        learning_rate=1e-3,
        M=0,
        n=n,
        drift_scale=1.0,
        min_bandwidth=0.2,
        max_drift_norm=2.0,
        repulsive_weight=1.0,
    )
    drift_trainer = Trainer_Drifting(drift_model, device, ch_awgn_residual, cfg_drift)
    drift_loss = drift_trainer.train_PreT()
    drift_model.eval()

    # Evaluation
    with torch.no_grad():
        x = torch.randn(eval_size, n, device=device)
        y_true = ch_AWGN(x, noise_std, device)

        seq_ddpm = utils.p_sample_loop_w_Condition(
            ddm_model,
            x.size(),
            cfg_ddm.num_steps,
            cfg_ddm.alphas,
            cfg_ddm.betas,
            cfg_ddm.alphas_bar_sqrt,
            cfg_ddm.one_minus_alphas_bar_sqrt,
            x,
            pred_type=cfg_ddm.pred_type,
        )
        y_ddpm = seq_ddpm[-1] + x

        traj_full = build_ddim_traj(num_steps, num_steps)
        seq_ddim_full = utils.p_sample_loop_w_Condition_DDIM(
            ddm_model,
            x.size(),
            traj_full,
            cfg_ddm.alphas_prod,
            cfg_ddm.alphas_bar_sqrt,
            cfg_ddm.one_minus_alphas_bar_sqrt,
            x,
            pred_type=cfg_ddm.pred_type,
        )
        y_ddim_full = seq_ddim_full[-1] + x

        traj_fast = build_ddim_traj(num_steps, ddim_fast_steps)
        seq_ddim_fast = utils.p_sample_loop_w_Condition_DDIM(
            ddm_model,
            x.size(),
            traj_fast,
            cfg_ddm.alphas_prod,
            cfg_ddm.alphas_bar_sqrt,
            cfg_ddm.one_minus_alphas_bar_sqrt,
            x,
            pred_type=cfg_ddm.pred_type,
        )
        y_ddim_fast = seq_ddim_fast[-1] + x

        # Drifting model predicts residual; convert to channel output.
        y_drift = x + drift_model(x)

    e_true_vec = (y_true - x).cpu().numpy()
    e_ddpm_vec = (y_ddpm - x).cpu().numpy()
    e_ddim_full_vec = (y_ddim_full - x).cpu().numpy()
    e_ddim_fast_vec = (y_ddim_fast - x).cpu().numpy()
    e_drift_vec = (y_drift - x).cpu().numpy()

    e_true = e_true_vec.reshape(-1)
    e_ddpm = e_ddpm_vec.reshape(-1)
    e_ddim_full = e_ddim_full_vec.reshape(-1)
    e_ddim_fast = e_ddim_fast_vec.reshape(-1)
    e_drift = e_drift_vec.reshape(-1)

    metrics = {
        "true_std": float(np.std(e_true)),
        "ddpm_std": float(np.std(e_ddpm)),
        "ddim_full_std": float(np.std(e_ddim_full)),
        "ddim_fast_std": float(np.std(e_ddim_fast)),
        "drifting_std": float(np.std(e_drift)),
        "ddpm_l1": l1_hist_distance(e_ddpm, e_true),
        "ddim_full_l1": l1_hist_distance(e_ddim_full, e_true),
        "ddim_fast_l1": l1_hist_distance(e_ddim_fast, e_true),
        "drifting_l1": l1_hist_distance(e_drift, e_true),
        "ddpm_swd": sliced_wasserstein_distance(e_ddpm_vec, e_true_vec),
        "ddim_full_swd": sliced_wasserstein_distance(e_ddim_full_vec, e_true_vec),
        "ddim_fast_swd": sliced_wasserstein_distance(e_ddim_fast_vec, e_true_vec),
        "drifting_swd": sliced_wasserstein_distance(e_drift_vec, e_true_vec),
    }

    # Figure
    y_true_np = y_true.cpu().numpy()
    y_ddpm_np = y_ddpm.cpu().numpy()
    y_ddim_fast_np = y_ddim_fast.cpu().numpy()
    y_drift_np = y_drift.cpu().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(ddm_loss, linewidth=2, label="Diffusion loss")
    axs[0, 0].plot(drift_loss, linewidth=2, label="Drifting loss")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(alpha=0.2)

    bins = np.linspace(-1.4, 1.4, 120)
    axs[0, 1].hist(e_true, bins=bins, density=True, alpha=0.30, label="True residual")
    axs[0, 1].hist(e_ddpm, bins=bins, density=True, alpha=0.30, label="DDPM (20)")
    axs[0, 1].hist(e_ddim_full, bins=bins, density=True, alpha=0.30, label="DDIM (20)")
    axs[0, 1].hist(e_ddim_fast, bins=bins, density=True, alpha=0.30, label=f"DDIM ({ddim_fast_steps})")
    axs[0, 1].hist(e_drift, bins=bins, density=True, alpha=0.30, label="Drifting")
    axs[0, 1].set_title("Residual Distribution (e = y - x)")
    axs[0, 1].legend(fontsize=8)

    show_n = 2200
    axs[1, 0].scatter(y_true_np[:show_n, 0], y_true_np[:show_n, 1], s=5, alpha=0.2, label="True AWGN")
    axs[1, 0].scatter(y_ddpm_np[:show_n, 0], y_ddpm_np[:show_n, 1], s=5, alpha=0.2, label="DDPM")
    axs[1, 0].scatter(y_ddim_fast_np[:show_n, 0], y_ddim_fast_np[:show_n, 1], s=5, alpha=0.2, label=f"DDIM ({ddim_fast_steps})")
    axs[1, 0].scatter(y_drift_np[:show_n, 0], y_drift_np[:show_n, 1], s=5, alpha=0.2, label="Drifting")
    axs[1, 0].set_title("Channel Output Samples (2D)")
    axs[1, 0].set_xlabel("y[0]")
    axs[1, 0].set_ylabel("y[1]")
    axs[1, 0].legend(fontsize=8)

    methods = ["DDPM(20)", "DDIM(20)", f"DDIM({ddim_fast_steps})", "Drifting"]
    swds = [metrics["ddpm_swd"], metrics["ddim_full_swd"], metrics["ddim_fast_swd"], metrics["drifting_swd"]]
    axs[1, 1].bar(methods, swds, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    axs[1, 1].set_title("Residual Sliced Wasserstein to True (Lower Better)")
    axs[1, 1].set_ylabel("SWD")
    for i, v in enumerate(swds):
        axs[1, 1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Corrected AWGN: Diffusion (DDPM/DDIM) vs Drifting\n"
        "Diffusion objective: epsilon-prediction DDPM, cosine schedule, T=20\n"
        "Drifting objective: attraction + repulsion drift field on residuals",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = os.path.join(ROOT, "results", "awgn_ddpm_ddim_drifting_corrected.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print("SETTINGS")
    print("diffusion_training_objective: epsilon-prediction DDPM")
    print(f"diffusion_num_steps: {num_steps}")
    print(f"ddpm_sampler_steps: {num_steps}")
    print(f"ddim_full_steps: {num_steps}")
    print(f"ddim_fast_steps: {ddim_fast_steps}")
    print("drifting_training_objective: attraction+repulsion drift on residuals")
    print("drifting_repulsive_weight: 1.0")
    print("drifting_drift_scale: 1.0")
    print("drifting_min_bandwidth: 0.2")
    print("drifting_max_drift_norm: 2.0")
    print(f"train_dataset_size: {dataset_size}")
    print(f"epochs: {max_epochs}")
    print(f"eval_samples: {eval_size}")

    print("RESULTS")
    print("primary_metric: sliced_wasserstein_distance_on_2D_residuals")
    print("swd_num_projections: 256")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"figure_path: {out_path}")


if __name__ == "__main__":
    main()
