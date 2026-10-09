"""Illustrate Proposition 5 in the SWD/gradient research note.

Deterministic Gaussian quadrature checks arithmetic, not the theorem.
No learned models, empirical channel results, or cluster jobs are used.
Run from any directory with Python, NumPy, and Matplotlib available.
"""

from pathlib import Path
import json
import math
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "swd-theory-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "theory_artifacts"
SIGMA = 0.5
KAPPA = 1.0
T0 = math.pi / 6
A = 2 / math.sqrt(3)


def gaussian_quadrature(order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return math.sqrt(2) * nodes, weights / math.sqrt(math.pi)


def expected_loss(mean, order=128):
    nodes, weights = gaussian_quadrature(order)
    logits = KAPPA * (np.asarray(mean)[..., None] + SIGMA * nodes)
    return np.sum(np.logaddexp(0.0, -logits) * weights, axis=-1)


def loss_mean_derivative(mean):
    nodes, weights = gaussian_quadrature(128)
    logits = KAPPA * (mean + SIGMA * nodes)
    return float(-KAPPA * np.sum(np.exp(-np.logaddexp(0.0, logits)) * weights))


def surrogate_mean(t, n):
    omega = 4 * math.pi * n
    return np.cos(t) + A / omega * np.sin(omega * np.sin(t))


def true_ber(t):
    return 0.5 * math.erfc(math.cos(t) / (SIGMA * math.sqrt(2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mean0 = math.cos(T0)
    risk0 = float(expected_loss(mean0))
    grad_p = -0.5 * loss_mean_derivative(mean0)
    grad_q = -grad_p
    rows = []
    quadrature_error = abs(risk0 - float(expected_loss(mean0, order=64)))
    assert quadrature_error < 1e-11

    for n in (1, 2, 4, 8, 16, 32, 64):
        omega = 4 * math.pi * n
        h = 1e-4 / omega
        derivative_fd = float(
            (expected_loss(surrogate_mean(T0 + h, n))
             - expected_loss(surrogate_mean(T0 - h, n))) / (2 * h)
        )
        step = 0.2 / omega
        t_after = T0 - step * grad_q
        risk_p_after = float(expected_loss(math.cos(t_after)))
        risk_q_after = float(expected_loss(surrogate_mean(t_after, n)))
        codeword_mean_error = abs(float(surrogate_mean(T0, n)) - mean0)
        assert codeword_mean_error < 1e-13
        assert abs(derivative_fd - grad_q) < 1e-8
        assert risk_p_after > risk0
        assert risk_q_after < risk0
        assert true_ber(t_after) > true_ber(T0)
        rows.append({
            "n": n,
            "uniform_conditional_w1_bound": A / omega,
            "codeword_mean_error_floating_point": codeword_mean_error,
            "true_encoder_gradient": grad_p,
            "surrogate_encoder_gradient": grad_q,
            "surrogate_gradient_centered_difference": derivative_fd,
            "gradient_absolute_error": abs(grad_p - grad_q),
            "sgd_step_size": step,
            "true_ce_before": risk0,
            "true_ce_after": risk_p_after,
            "surrogate_ce_after": risk_q_after,
            "true_ber_before": true_ber(T0),
            "true_ber_after": true_ber(t_after),
        })

    payload = {
        "status": "Deterministic illustration of a proved constructed example; not empirical paper results",
        "sigma": SIGMA,
        "kappa": KAPPA,
        "t0": T0,
        "quadrature_order": 128,
        "quadrature_64_vs_128_difference_at_t0": quadrature_error,
        "rows": rows,
    }
    (OUT / "swd_gradient_counterexample.json").write_text(json.dumps(payload, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
    offsets = np.linspace(-0.02, 0.02, 401)
    angles = T0 + offsets
    axes[0].plot(offsets, expected_loss(np.cos(angles)) - risk0, color="#176b87", lw=2.4, label="True channel")
    axes[0].plot(offsets, expected_loss(surrogate_mean(angles, 4)) - risk0, color="#c65b34", lw=2.4, label="Surrogate (n = 4)")
    axes[0].axhline(0, color="#aab0b5", lw=0.8)
    axes[0].axvline(0, color="#aab0b5", lw=0.8)
    axes[0].scatter([0], [0], color="#262d33", zorder=4, s=30)
    axes[0].set(xlabel="Encoder angle change (radians)", ylabel="Cross-entropy change from current codebook", title="Exact codeword laws, opposite slopes")
    axes[0].legend(frameon=False)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ns = [row["n"] for row in rows]
    axes[1].loglog(ns, [row["uniform_conditional_w1_bound"] for row in rows], "o-", color="#176b87", lw=2, label="Uniform conditional W1 bound")
    axes[1].loglog(ns, [row["gradient_absolute_error"] for row in rows], "s-", color="#c65b34", lw=2, label="Absolute encoder-gradient error")
    axes[1].set(xlabel="Oscillation index n", ylabel="Discrepancy (each in its own units)", title="Distribution convergence, persistent gradient error")
    axes[1].set_xticks(ns, [str(n) for n in ns])
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("A power-preserving Gaussian channel counterexample", fontsize=15)
    fig.savefig(OUT / "swd_gradient_counterexample.png", dpi=180)
    fig.savefig(OUT / "swd_gradient_counterexample.svg")
    plt.close(fig)
    print(json.dumps({"checks": "passed", "quadrature_difference": quadrature_error, "n4": rows[2], "output_directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
