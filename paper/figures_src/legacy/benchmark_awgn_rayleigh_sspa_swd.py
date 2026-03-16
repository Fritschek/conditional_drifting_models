"""
Benchmark AWGN, Rayleigh, SSPA, and OptFib channels using:
  - Diffusion (DDPM training + DDPM/DDIM sampling)
  - Corrected drifting (attraction + repulsion)
  - Conditional WGAN-GP baseline ("GAN comp")

Primary metric:
  - Sliced Wasserstein distance (SWD) on residual vectors e = y - x

Run:
  MPLCONFIGDIR=/Users/rickfritschek/Documents/GitHub/DM_for_learning_channels/.mplcache \
  /Users/rickfritschek/Documents/GitHub/turbo_mingru_decoder/.venv/bin/python \
  examples/benchmark_awgn_rayleigh_sspa_swd.py
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

from channel_models import ch_AWGN, ch_Rayleigh_AWGN, ch_SSPA, ch_OptFib
from models import ConditionalModel_w_Condition
from trainer import TrainerConfig_DDM, Trainer_DDM
from ema import EMA
from drifting import ConditionalDriftingGenerator, TrainerConfig_Drifting, Trainer_Drifting
import utils


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sliced_wasserstein_distance(x, y, num_projections=256, seed=12345):
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


def build_ddim_traj(num_steps, ddim_steps):
    raw = np.linspace(0, num_steps - 1, ddim_steps)
    traj = np.unique(np.round(raw).astype(int)).tolist()
    traj[0] = 0
    traj[-1] = num_steps - 1
    return traj


def make_residual_channel(channel_fn):
    def _residual_channel(x, noise_std, device):
        return channel_fn(x, noise_std, device) - x
    return _residual_channel


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GeneratorBN(nn.Module):
    """
    DM_OptFib-style generator:
    input = [condition x, latent z], output = channel output y.
    """

    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, is_res=True):
        super().__init__()
        self.output_size = output_size
        self.is_res = is_res

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.hidden_layers1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layers3 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])

        self.bn_in = nn.BatchNorm1d(hidden_size)
        self.bn_layers1 = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers)])
        self.bn_layer2 = nn.BatchNorm1d(hidden_size)
        self.bn_layers3 = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers)])
        self.f = nn.LeakyReLU(0.2)

        self.apply(_init_weights)

    def forward(self, inputs):
        res = inputs[:, :self.output_size] if self.is_res else None
        x = self.f(self.bn_in(self.in_layer(inputs)))
        for i, layer in enumerate(self.hidden_layers1):
            x = self.f(self.bn_layers1[i](layer(x)))
        x = self.f(self.bn_layer2(self.hidden_layer2(x)))
        for i, layer in enumerate(self.hidden_layers3):
            x = self.f(self.bn_layers3[i](layer(x)))
        y = self.out_layer(x)
        if self.is_res:
            y = y + res
        return y


class DiscriminatorRes(nn.Module):
    """
    DM_OptFib-style discriminator:
    input = [sample y, condition x], scalar score.
    """

    def __init__(self, input_size, hidden_size, output_size=1, num_hidden_layers=3):
        super().__init__()
        self.in_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])

        merged = hidden_size + input_size
        self.hidden_layers2 = nn.ModuleList([
            nn.Linear(merged, merged // 2),
            nn.Linear(merged // 2, merged // 4),
            nn.Linear(merged // 4, merged // 8),
        ])
        self.out_layer = nn.Linear(merged // 8, output_size)
        self.f = nn.LeakyReLU(0.2)

        self.apply(_init_weights)

    def forward(self, inputs):
        x = self.f(self.in_layer(inputs))
        for layer in self.hidden_layers1:
            x = self.f(layer(x))
        x = torch.cat((x, inputs), dim=1)
        for layer in self.hidden_layers2:
            x = self.f(layer(x))
        return self.out_layer(x)


def _gradient_penalty(discriminator, real_y, fake_y, condition):
    bsz = real_y.shape[0]
    alpha = torch.rand(bsz, 1, device=real_y.device)
    interp = (alpha * real_y + (1.0 - alpha) * fake_y).requires_grad_(True)
    interp_score = discriminator(torch.cat((interp, condition), dim=1))
    grad = autograd.grad(
        outputs=interp_score,
        inputs=interp,
        grad_outputs=torch.ones_like(interp_score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_norm = grad.view(bsz, -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


def train_conditional_wgan(channel_fn, cfg, device):
    """
    Train a DM_OptFib-style conditional WGAN-GP on y|x.
    """
    n = cfg["n"]
    noise_std = cfg["noise_std"]

    gen = GeneratorBN(
        input_size=n + cfg["gan_latent_dim"],
        hidden_size=cfg["gan_hidden_dim"],
        output_size=n,
        num_hidden_layers=2,
        is_res=True,
    ).to(device)
    disc = DiscriminatorRes(
        input_size=2 * n,
        hidden_size=cfg["gan_hidden_dim"],
        output_size=1,
        num_hidden_layers=3,
    ).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg["gan_lr_g"], betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg["gan_lr_d"], betas=(0.0, 0.9))
    bce = nn.BCEWithLogitsLoss()

    num_batches = max(1, int(np.ceil(cfg["dataset_size"] / cfg["batch_size"])))
    for epoch in range(cfg["gan_epochs"]):
        for it in range(num_batches):
            if cfg["gan_mode"] == "wgan_gp":
                # Discriminator updates
                for _ in range(cfg["gan_critic_steps"]):
                    x = torch.randn(cfg["batch_size"], n, device=device)
                    z = torch.randn(cfg["batch_size"], cfg["gan_latent_dim"], device=device)
                    with torch.no_grad():
                        real_y = channel_fn(x, noise_std, device)
                    fake_y = gen(torch.cat((x, z), dim=1)).detach()

                    d_real = disc(torch.cat((real_y, x), dim=1)).mean()
                    d_fake = disc(torch.cat((fake_y, x), dim=1)).mean()
                    gp = _gradient_penalty(disc, real_y, fake_y, x)
                    d_loss = d_fake - d_real + cfg["gan_lambda_gp"] * gp

                    opt_d.zero_grad()
                    d_loss.backward()
                    opt_d.step()

                # Generator update
                x = torch.randn(cfg["batch_size"], n, device=device)
                z = torch.randn(cfg["batch_size"], cfg["gan_latent_dim"], device=device)
                fake_y = gen(torch.cat((x, z), dim=1))
                g_loss = -disc(torch.cat((fake_y, x), dim=1)).mean()

                opt_g.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                opt_g.step()

            elif cfg["gan_mode"] == "gan_fa":
                x = torch.randn(cfg["batch_size"], n, device=device)
                with torch.no_grad():
                    real_y = channel_fn(x, noise_std, device)

                # Discriminator step
                z_d = torch.randn(cfg["batch_size"], cfg["gan_latent_dim"], device=device)
                fake_y_d = gen(torch.cat((x, z_d), dim=1)).detach()
                d_real = disc(torch.cat((real_y, x), dim=1))
                d_fake = disc(torch.cat((fake_y_d, x), dim=1))
                real_labels = torch.rand_like(d_real) * 0.3 + 0.7  # [0.7, 1.0]
                fake_labels = torch.rand_like(d_fake) * 0.3        # [0.0, 0.3]
                d_loss = bce(d_real, real_labels) + bce(d_fake, fake_labels)

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # Generator step (delayed, DM_OptFib style)
                if it % cfg["gan_gen_every"] == 0:
                    z_g = torch.randn(cfg["batch_size"], cfg["gan_latent_dim"], device=device)
                    fake_y_g = gen(torch.cat((x, z_g), dim=1))
                    d_fake_g = disc(torch.cat((fake_y_g, x), dim=1))
                    recon_l1 = torch.mean(torch.sum(torch.abs(fake_y_g - real_y), dim=1))
                    g_loss = bce(d_fake_g, real_labels) + cfg["gan_l1_weight"] * recon_l1

                    opt_g.zero_grad()
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                    opt_g.step()
            else:
                raise ValueError(f"Unknown gan_mode: {cfg['gan_mode']}")

        if (epoch + 1) % max(1, cfg["gan_epochs"] // 4) == 0:
            print(
                f"  GAN[{cfg['gan_mode']}] epoch {epoch+1}/{cfg['gan_epochs']} "
                f"(d_loss={d_loss.item():.4f}, g_loss={g_loss.item():.4f})"
            )

    gen.eval()

    def _sample_y(condition_x):
        z = torch.randn(condition_x.shape[0], cfg["gan_latent_dim"], device=condition_x.device)
        return gen(torch.cat((condition_x, z), dim=1))

    return _sample_y


def run_channel_benchmark(channel_name, channel_fn, cfg, device):
    n = cfg["n"]
    noise_std = cfg["noise_std"]
    num_steps = cfg["num_steps"]
    ddim_fast_steps = cfg["ddim_fast_steps"]

    betas = utils.make_beta_schedule(
        schedule="cosine",
        n_timesteps=num_steps,
        start=1e-4,
        end=2e-2,
    )

    # Diffusion
    tconf_ddm = TrainerConfig_DDM(
        max_epochs=cfg["epochs"],
        dataset_size=cfg["dataset_size"],
        batch_size=cfg["batch_size"],
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
    Trainer_DDM(ddm_model, ddm_ema, device, channel_fn, tconf_ddm).train_PreT()
    ddm_ema.ema(ddm_model)
    ddm_model.eval()

    # Drifting
    drift_model = ConditionalDriftingGenerator(
        condition_dim=n,
        output_dim=n,
        latent_dim=16,
        hidden_dim=128,
    ).to(device)
    tconf_drift = TrainerConfig_Drifting(
        max_epochs=cfg["epochs"],
        dataset_size=cfg["dataset_size"],
        batch_size=cfg["batch_size"],
        noise_std=noise_std,
        learning_rate=1e-3,
        M=0,
        n=n,
        drift_scale=1.0,
        min_bandwidth=0.2,
        max_drift_norm=2.0,
        repulsive_weight=1.0,
    )
    residual_channel_fn = make_residual_channel(channel_fn)
    Trainer_Drifting(drift_model, device, residual_channel_fn, tconf_drift).train_PreT()
    drift_model.eval()

    # GAN (DM_OptFib-style conditional WGAN-GP on y | x)
    gan_model = train_conditional_wgan(channel_fn, cfg, device)

    with torch.no_grad():
        x = torch.randn(cfg["eval_size"], n, device=device)
        y_true = channel_fn(x, noise_std, device)

        seq_ddpm = utils.p_sample_loop_w_Condition(
            ddm_model,
            x.size(),
            tconf_ddm.num_steps,
            tconf_ddm.alphas,
            tconf_ddm.betas,
            tconf_ddm.alphas_bar_sqrt,
            tconf_ddm.one_minus_alphas_bar_sqrt,
            x,
            pred_type=tconf_ddm.pred_type,
        )
        y_ddpm = seq_ddpm[-1] + x

        seq_ddim_full = utils.p_sample_loop_w_Condition_DDIM(
            ddm_model,
            x.size(),
            build_ddim_traj(num_steps, num_steps),
            tconf_ddm.alphas_prod,
            tconf_ddm.alphas_bar_sqrt,
            tconf_ddm.one_minus_alphas_bar_sqrt,
            x,
            pred_type=tconf_ddm.pred_type,
        )
        y_ddim_full = seq_ddim_full[-1] + x

        seq_ddim_fast = utils.p_sample_loop_w_Condition_DDIM(
            ddm_model,
            x.size(),
            build_ddim_traj(num_steps, ddim_fast_steps),
            tconf_ddm.alphas_prod,
            tconf_ddm.alphas_bar_sqrt,
            tconf_ddm.one_minus_alphas_bar_sqrt,
            x,
            pred_type=tconf_ddm.pred_type,
        )
        y_ddim_fast = seq_ddim_fast[-1] + x

        y_drift = x + drift_model(x)
        y_gan = gan_model(x)

    e_true = (y_true - x).cpu().numpy()
    e_ddpm = (y_ddpm - x).cpu().numpy()
    e_ddim_full = (y_ddim_full - x).cpu().numpy()
    e_ddim_fast = (y_ddim_fast - x).cpu().numpy()
    e_drift = (y_drift - x).cpu().numpy()
    e_gan = (y_gan - x).cpu().numpy()

    result = {
        "channel": channel_name,
        "true_std": float(np.std(e_true)),
        "ddpm_std": float(np.std(e_ddpm)),
        "ddim_full_std": float(np.std(e_ddim_full)),
        "ddim_fast_std": float(np.std(e_ddim_fast)),
        "drifting_std": float(np.std(e_drift)),
        "gan_std": float(np.std(e_gan)),
        "ddpm_swd": sliced_wasserstein_distance(e_ddpm, e_true),
        "ddim_full_swd": sliced_wasserstein_distance(e_ddim_full, e_true),
        "ddim_fast_swd": sliced_wasserstein_distance(e_ddim_fast, e_true),
        "drifting_swd": sliced_wasserstein_distance(e_drift, e_true),
        "gan_swd": sliced_wasserstein_distance(e_gan, e_true),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="AWGN/Rayleigh/SSPA/OptFib SWD benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--ddim-fast-steps", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=80_000)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--eval-size", type=int, default=10_000)
    parser.add_argument("--gan-epochs", type=int, default=60)
    parser.add_argument("--gan-hidden-dim", type=int, default=128)
    parser.add_argument("--gan-latent-dim", type=int, default=16)
    parser.add_argument("--gan-critic-steps", type=int, default=5)
    parser.add_argument("--gan-lambda-gp", type=float, default=10.0)
    parser.add_argument("--gan-lr-g", type=float, default=1e-4)
    parser.add_argument("--gan-lr-d", type=float, default=1e-4)
    parser.add_argument("--gan-mode", type=str, default="gan_fa", choices=["wgan_gp", "gan_fa"])
    parser.add_argument("--gan-gen-every", type=int, default=5)
    parser.add_argument("--gan-l1-weight", type=float, default=1.0)
    parser.add_argument(
        "--plot-fast-ddim",
        action="store_true",
        help="Include DDIM-fast (e.g., DDIM(20)) in the bar plot.",
    )
    parser.add_argument("--no-optfib", action="store_true")
    parser.add_argument(
        "--channels",
        type=str,
        default="AWGN,Rayleigh,SSPA,OptFib",
        help="Comma-separated subset from {AWGN,Rayleigh,SSPA,OptFib}.",
    )
    parser.add_argument("--optfib-L", type=float, default=5000.0)
    parser.add_argument("--optfib-gamma", type=float, default=1.27)
    parser.add_argument("--optfib-kstep", type=int, default=50)
    parser.add_argument("--optfib-pn-dbm", type=float, default=-21.3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    cfg = {
        "seed": args.seed,
        "n": 2,
        "noise_std": 0.3,
        "dataset_size": args.dataset_size,
        "epochs": args.epochs,
        "batch_size": 256,
        "eval_size": args.eval_size,
        "num_steps": args.num_steps,
        "ddim_fast_steps": args.ddim_fast_steps,
        "gan_epochs": args.gan_epochs,
        "gan_hidden_dim": args.gan_hidden_dim,
        "gan_latent_dim": args.gan_latent_dim,
        "gan_critic_steps": args.gan_critic_steps,
        "gan_lambda_gp": args.gan_lambda_gp,
        "gan_lr_g": args.gan_lr_g,
        "gan_lr_d": args.gan_lr_d,
        "gan_mode": args.gan_mode,
        "gan_gen_every": args.gan_gen_every,
        "gan_l1_weight": args.gan_l1_weight,
        "optfib_params": {
            "L": args.optfib_L,
            "gamma": args.optfib_gamma,
            "Kstep": args.optfib_kstep,
            "Pn_dBm": args.optfib_pn_dbm,
        },
    }

    all_channels = [
        ("AWGN", ch_AWGN),
        ("Rayleigh", ch_Rayleigh_AWGN),
        ("SSPA", ch_SSPA),
    ]
    if not args.no_optfib:
        def _optfib_channel(x, noise_std, device):
            return ch_OptFib(
                x,
                noise_std,
                device,
                L=args.optfib_L,
                gamma=args.optfib_gamma,
                Kstep=args.optfib_kstep,
                Pn_dBm=args.optfib_pn_dbm,
                use_noise_std=False,
            )
        all_channels.append(("OptFib", _optfib_channel))

    requested = {c.strip() for c in args.channels.split(",") if c.strip()}
    valid = {name for name, _ in all_channels}
    unknown = sorted(requested - valid)
    if unknown:
        raise ValueError(f"Unknown channels requested: {unknown}. Valid: {sorted(valid)}")
    channels = [(name, fn) for name, fn in all_channels if name in requested]
    if not channels:
        raise ValueError("No channels selected. Check --channels.")

    results = []
    for name, fn in channels:
        print(f"Running channel: {name}")
        result = run_channel_benchmark(name, fn, cfg, device)
        results.append(result)
        print(json.dumps(result, indent=2))

    suffix = (
        f"t{cfg['num_steps']}_ddim{cfg['ddim_fast_steps']}"
        f"_{cfg['gan_mode']}_ge{cfg['gan_epochs']}"
    )
    out_json = os.path.join(ROOT, "results", f"benchmark_channels_swd_gan_{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2)

    # Plot SWD bars per channel
    methods = ["ddpm_swd", "ddim_full_swd", "drifting_swd", "gan_swd"]
    method_labels = [
        f"DDPM({cfg['num_steps']})",
        f"DDIM({cfg['num_steps']})",
        "Drifting",
        "WGAN",
    ]
    if args.plot_fast_ddim:
        methods.insert(2, "ddim_fast_swd")
        method_labels.insert(2, f"DDIM({cfg['ddim_fast_steps']})")
    x = np.arange(len(channels))
    w = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, (m, label, c) in enumerate(zip(methods, method_labels, colors)):
        vals = [r[m] for r in results]
        ax.bar(x + (i - (len(methods) - 1) / 2.0) * w, vals, width=w, label=label, color=c)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in channels])
    ax.set_ylabel("SWD")
    ax.set_title(
        "Residual SWD to True Channel (Lower Better)\n"
        f"T={cfg['num_steps']}, DDIM-fast={cfg['ddim_fast_steps']}"
    )
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    out_fig = os.path.join(ROOT, "results", f"benchmark_channels_swd_gan_{suffix}.png")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=160)
    plt.close(fig)

    print("BENCHMARK_COMPLETE")
    print(f"results_json: {out_json}")
    print(f"results_fig: {out_fig}")


if __name__ == "__main__":
    main()
