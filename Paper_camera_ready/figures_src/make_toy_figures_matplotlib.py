import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1] / "figures"
ROOT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "target1": "#4c6ef5",
    "target2": "#2b8a8a",
    "gen1": "#f08c00",
    "gen2": "#9c36b5",
    "drift": "#2f9e44",
    "axis": "#9aa0a6",
    "panel": "#fcfcfd",
    "edge": "#d4d7dd",
}

MU1 = np.array([-0.55, 0.62])
MU2 = np.array([0.72, -0.62])
SIG1 = np.array([0.16, 0.18])
SIG2 = np.array([0.18, 0.16])


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sample_cloud(rng, center, scales, n):
    return rng.normal(loc=center, scale=scales, size=(n, 2))


def pairwise_dist(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def drift_targets(generated, positive, repulsive_weight=0.8, drift_scale=0.9, max_norm=0.22, min_bw=0.12):
    pts = np.vstack([generated, positive])
    ds = pairwise_dist(pts, pts)
    bw = np.median(ds[np.triu_indices_from(ds, k=1)])
    if not np.isfinite(bw):
        bw = min_bw
    bw = max(min_bw, float(bw))

    d_gp = pairwise_dist(generated, positive)
    w_gp = np.exp(-(d_gp**2) / (2.0 * bw * bw))
    w_gp /= np.clip(w_gp.sum(axis=1, keepdims=True), 1e-9, None)
    pos_bar = w_gp @ positive
    drift = pos_bar - generated

    d_gg = pairwise_dist(generated, generated)
    w_gg = np.exp(-(d_gg**2) / (2.0 * bw * bw))
    np.fill_diagonal(w_gg, 0.0)
    w_gg /= np.clip(w_gg.sum(axis=1, keepdims=True), 1e-9, None)
    gen_bar = w_gg @ generated
    drift -= repulsive_weight * (gen_bar - generated)

    norms = np.linalg.norm(drift, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.clip(norms, 1e-9, None))
    drift = drift * scale
    return generated + drift_scale * drift


def evolve_particles(rng, steps=14, n=6):
    target1 = sample_cloud(rng, MU1, SIG1, n)
    target2 = sample_cloud(rng, MU2, SIG2, n)
    init1 = sample_cloud(rng, np.array([-0.55, -0.10]), np.array([0.20, 0.18]), n)
    init2 = sample_cloud(rng, np.array([0.35, 0.22]), np.array([0.22, 0.18]), n)
    drift1 = drift_targets(init1, target1, repulsive_weight=0.9, drift_scale=0.9, max_norm=0.30)
    drift2 = drift_targets(init2, target2, repulsive_weight=0.9, drift_scale=0.9, max_norm=0.30)
    cur1, cur2 = init1.copy(), init2.copy()
    for _ in range(steps):
        cur1 = drift_targets(cur1, target1, repulsive_weight=0.9, drift_scale=0.9, max_norm=0.30)
        cur2 = drift_targets(cur2, target2, repulsive_weight=0.9, drift_scale=0.9, max_norm=0.30)
    return target1, target2, init1, init2, drift1, drift2, cur1, cur2


def init_mlp(rng, inp=3, hidden=12, out=2):
    return {
        "W1": rng.uniform(-0.5, 0.5, size=(hidden, inp)),
        "b1": np.zeros(hidden),
        "W2": rng.uniform(-0.35, 0.35, size=(out, hidden)),
        "b2": np.zeros(out),
    }


def forward(params, x):
    pre1 = x @ params["W1"].T + params["b1"]
    h1 = np.tanh(pre1)
    y = h1 @ params["W2"].T + params["b2"]
    return pre1, h1, y


def sample_generator(params, cond_id, latents):
    cond_col = -np.ones((len(latents), 1)) if cond_id == 0 else np.ones((len(latents), 1))
    x = np.concatenate([cond_col, latents], axis=1)
    _, _, y = forward(params, x)
    return y


def train_tiny_mlp(rng, steps=1400, lr=0.035, batch_per_cond=18):
    params = init_mlp(rng)
    init_params = {k: v.copy() for k, v in params.items()}
    target_pool = {
        0: sample_cloud(rng, MU1, SIG1, 160),
        1: sample_cloud(rng, MU2, SIG2, 160),
    }

    for _ in range(steps):
        grads = {k: np.zeros_like(v) for k, v in params.items()}
        for cond_id, mu in enumerate([MU1, MU2]):
            z = rng.normal(size=(batch_per_cond, 2))
            cond_col = -np.ones((batch_per_cond, 1)) if cond_id == 0 else np.ones((batch_per_cond, 1))
            x = np.concatenate([cond_col, z], axis=1)
            pre1, h1, y = forward(params, x)
            idx = rng.choice(len(target_pool[cond_id]), size=batch_per_cond, replace=False)
            positives = target_pool[cond_id][idx]
            tgt = drift_targets(y, positives)
            dy = (y - tgt) / (2.0 * batch_per_cond)
            grads["W2"] += dy.T @ h1
            grads["b2"] += dy.sum(axis=0)
            dh = (dy @ params["W2"]) * (1.0 - np.tanh(pre1) ** 2)
            grads["W1"] += dh.T @ x
            grads["b1"] += dh.sum(axis=0)
        for k in params:
            params[k] -= lr * grads[k]

    k = np.arange(16)
    viz_latents = np.stack(
        [
            np.cos(k * 0.7) + 0.2 * np.sin(k),
            np.sin(k * 0.55) - 0.15 * np.cos(0.8 * k),
        ],
        axis=1,
    )
    target_viz_1 = sample_cloud(rng, MU1, SIG1, 16)
    target_viz_2 = sample_cloud(rng, MU2, SIG2, 16)
    init1 = sample_generator(init_params, 0, viz_latents)
    init2 = sample_generator(init_params, 1, viz_latents)
    final1 = sample_generator(params, 0, viz_latents)
    final2 = sample_generator(params, 1, viz_latents)

    init_err_1 = np.mean(np.linalg.norm(init1 - MU1, axis=1))
    init_err_2 = np.mean(np.linalg.norm(init2 - MU2, axis=1))
    final_err_1 = np.mean(np.linalg.norm(final1 - MU1, axis=1))
    final_err_2 = np.mean(np.linalg.norm(final2 - MU2, axis=1))

    stats = (
        f"init_err_cond1={init_err_1:.4f}\n"
        f"init_err_cond2={init_err_2:.4f}\n"
        f"final_err_cond1={final_err_1:.4f}\n"
        f"final_err_cond2={final_err_2:.4f}\n"
    )
    (ROOT / "drifting_tiny_mlp_demo_stats.txt").write_text(stats)
    return target_viz_1, target_viz_2, init1, init2, final1, final2


def oracle_ddim_step(x_t, mu, sigma_data, alpha_bar_t, alpha_bar_prev):
    var_t = alpha_bar_t * (sigma_data**2) + (1.0 - alpha_bar_t)
    k = math.sqrt(alpha_bar_t) * (sigma_data**2) / var_t
    x0_hat = mu + k * (x_t - math.sqrt(alpha_bar_t) * mu)
    if alpha_bar_prev >= 1.0:
        return x0_hat
    eps_hat = (x_t - math.sqrt(alpha_bar_t) * x0_hat) / math.sqrt(1.0 - alpha_bar_t)
    return math.sqrt(alpha_bar_prev) * x0_hat + math.sqrt(1.0 - alpha_bar_prev) * eps_hat


def simulate_diffusion(rng, mu, sigma_data=0.18, n_samples=12, steps=100):
    # Keep the overall corruption level comparable as T changes so the snapshots
    # remain visually interpretable when we increase the reverse chain length.
    betas = np.linspace(0.003, 0.047, steps) * (50.0 / steps)
    alpha_bars = [1.0]
    acc = 1.0
    for beta in betas:
        acc *= 1.0 - beta
        alpha_bars.append(acc)
    states = [rng.normal(size=(n_samples, 2))]
    current = states[0]
    for t in range(steps, 0, -1):
        nxt = oracle_ddim_step(current, mu, sigma_data, alpha_bars[t], alpha_bars[t - 1])
        states.append(nxt)
        current = nxt
    snapshot_t = [100, 50, 20, 0]
    idxs = [steps - t for t in snapshot_t]
    return [states[i] for i in idxs], snapshot_t


def add_target_ellipses(ax):
    for mu, color in [(MU1, COLORS["target1"]), (MU2, COLORS["target2"])]:
        ax.add_patch(
            Ellipse(
                mu,
                width=0.84,
                height=0.68,
                facecolor=color,
                edgecolor=color,
                alpha=0.08,
                linestyle="--",
                linewidth=1.2,
            )
        )


def style_axis(ax, title):
    ax.set_title(title, pad=8)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0.0, color=COLORS["axis"], lw=0.8, zorder=0)
    ax.axvline(0.0, color=COLORS["axis"], lw=0.8, zorder=0)
    for side in ax.spines:
        ax.spines[side].set_color(COLORS["edge"])
        ax.spines[side].set_linewidth(0.9)
    ax.text(1.34, -0.09, r"$e_1$", ha="right", va="top", fontsize=10)
    ax.text(0.05, 1.25, r"$e_2$", ha="left", va="top", fontsize=10)


def add_panel_arrows(fig, axes):
    for left, right in zip(axes[:-1], axes[1:]):
        lbox = left.get_position()
        rbox = right.get_position()
        start = (lbox.x1 + 0.008, 0.5 * (lbox.y0 + lbox.y1))
        end = (rbox.x0 - 0.008, 0.5 * (rbox.y0 + rbox.y1))
        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#7a7d84",
        )
        fig.add_artist(arrow)


def scatter_set(ax, pts, color, label=None, alpha=0.95, s=22, lw=0.5):
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=s,
        c=color,
        edgecolors="#202124",
        linewidths=lw,
        alpha=alpha,
        label=label,
        zorder=3,
    )


def common_legend(fig, labels):
    handles = []
    for color, label in labels:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="#202124",
                markeredgewidth=0.5,
                markersize=5,
                label=label,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=9,
    )


def save_figure(fig, stem):
    fig.savefig(ROOT / f"{stem}.pdf")
    fig.savefig(ROOT / f"{stem}.png", dpi=240)
    plt.close(fig)


def make_particle_figure():
    rng = np.random.default_rng(7)
    target1, target2, init1, init2, drift1, drift2, final1, final2 = evolve_particles(rng)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.23, top=0.86, wspace=0.10)
    titles = ["(a) Initial Particles", "(b) One Drift Update", "(c) Repeated Updates"]
    for ax, title in zip(axes, titles):
        style_axis(ax, title)
        add_target_ellipses(ax)
    scatter_set(axes[0], target1, COLORS["target1"], alpha=0.35)
    scatter_set(axes[0], target2, COLORS["target2"], alpha=0.35)
    scatter_set(axes[0], init1, COLORS["gen1"])
    scatter_set(axes[0], init2, COLORS["gen2"])

    scatter_set(axes[1], target1, COLORS["target1"], alpha=0.55)
    scatter_set(axes[1], target2, COLORS["target2"], alpha=0.55)
    scatter_set(axes[1], init1, COLORS["gen1"])
    scatter_set(axes[1], init2, COLORS["gen2"])
    scatter_set(axes[1], drift1, COLORS["target1"], alpha=0.95, s=18)
    scatter_set(axes[1], drift2, COLORS["target2"], alpha=0.95, s=18)
    for p, q in zip(init1, drift1):
        axes[1].annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["drift"]))
    for p, q in zip(init2, drift2):
        axes[1].annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["drift"]))
    axes[1].text(0.0, -1.18, r"$\tilde e=\mathrm{stopgrad}(\hat e+\eta V(\hat e))$", ha="center", va="top", fontsize=10)

    scatter_set(axes[2], target1, COLORS["target1"], alpha=0.28)
    scatter_set(axes[2], target2, COLORS["target2"], alpha=0.28)
    scatter_set(axes[2], final1, COLORS["gen1"])
    scatter_set(axes[2], final2, COLORS["gen2"])

    common_legend(
        fig,
        [
            (COLORS["target1"], r"target, $x^{(1)}$"),
            (COLORS["target2"], r"target, $x^{(2)}$"),
            (COLORS["gen1"], r"generated, $x^{(1)}$"),
            (COLORS["gen2"], r"generated, $x^{(2)}$"),
        ],
    )
    add_panel_arrows(fig, axes)
    save_figure(fig, "drifting_conditional_toy_figure_mpl")


def make_mlp_figure():
    rng = np.random.default_rng(11)
    target1, target2, init1, init2, final1, final2 = train_tiny_mlp(rng)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.23, top=0.86, wspace=0.10)
    titles = ["(a) Target", "(b) Init MLP", "(c) Drift-Trained MLP"]
    for ax, title in zip(axes, titles):
        style_axis(ax, title)
        add_target_ellipses(ax)
    scatter_set(axes[0], target1, COLORS["target1"])
    scatter_set(axes[0], target2, COLORS["target2"])
    scatter_set(axes[1], init1, COLORS["gen1"])
    scatter_set(axes[1], init2, COLORS["gen2"])
    scatter_set(axes[2], target1, COLORS["target1"], alpha=0.22)
    scatter_set(axes[2], target2, COLORS["target2"], alpha=0.22)
    scatter_set(axes[2], final1, COLORS["gen1"])
    scatter_set(axes[2], final2, COLORS["gen2"])
    common_legend(
        fig,
        [
            (COLORS["target1"], r"target, $x^{(1)}$"),
            (COLORS["target2"], r"target, $x^{(2)}$"),
            (COLORS["gen1"], r"MLP, $x^{(1)}$"),
            (COLORS["gen2"], r"MLP, $x^{(2)}$"),
        ],
    )
    save_figure(fig, "drifting_tiny_mlp_demo_mpl")


def make_diffusion_figure():
    rng = np.random.default_rng(23)
    target1 = sample_cloud(rng, MU1, np.array([0.18, 0.18]), 12)
    target2 = sample_cloud(rng, MU2, np.array([0.18, 0.18]), 12)
    snaps1, snapshot_t = simulate_diffusion(rng, MU1)
    snaps2, _ = simulate_diffusion(rng, MU2)
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.23, top=0.86, wspace=0.08)
    titles = [
        rf"(a) $t={snapshot_t[0]}$",
        rf"(b) $t={snapshot_t[1]}$",
        rf"(c) $t={snapshot_t[2]}$",
        rf"(d) $t={snapshot_t[3]}$",
    ]
    for ax, title in zip(axes, titles):
        style_axis(ax, title)
        add_target_ellipses(ax)
    for ax, s1, s2 in zip(axes, snaps1, snaps2):
        scatter_set(ax, s1, COLORS["gen1"])
        scatter_set(ax, s2, COLORS["gen2"])
    scatter_set(axes[-1], target1, COLORS["target1"], alpha=0.22)
    scatter_set(axes[-1], target2, COLORS["target2"], alpha=0.22)
    common_legend(
        fig,
        [
            (COLORS["target1"], r"target, $x^{(1)}$"),
            (COLORS["target2"], r"target, $x^{(2)}$"),
            (COLORS["gen1"], r"samples, $x^{(1)}$"),
            (COLORS["gen2"], r"samples, $x^{(2)}$"),
        ],
    )
    add_panel_arrows(fig, axes)
    save_figure(fig, "toy_diffusion_oracle_demo_mpl")


def main():
    configure_matplotlib()
    make_particle_figure()
    make_mlp_figure()
    make_diffusion_figure()


if __name__ == "__main__":
    main()
