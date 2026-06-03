from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "Analytic train": "Analytic",
    "Drifting (dir.)": "Drifting",
    "Drifting (dir., standard)": "Drifting, standard",
    "Drifting (dir., cond. kernel)": "Drifting, cond. kernel",
    "Diffusion (dir.)": "Diffusion",
    "Paper WGAN": "WGAN",
}

COLORS = {
    "Analytic train": "#333333",
    "Drifting (dir.)": "#4c78a8",
    "Drifting (dir., standard)": "#4c78a8",
    "Drifting (dir., cond. kernel)": "#54a24b",
    "Diffusion (dir.)": "#f58518",
    "Paper WGAN": "#b279a2",
}

MARKERS = {
    "Analytic train": "o",
    "Drifting (dir.)": "s",
    "Drifting (dir., standard)": "s",
    "Drifting (dir., cond. kernel)": "P",
    "Diffusion (dir.)": "^",
    "Paper WGAN": "D",
}

LINESTYLES = {
    "Analytic train": "-",
    "Drifting (dir.)": "-",
    "Drifting (dir., standard)": "--",
    "Drifting (dir., cond. kernel)": "-",
    "Diffusion (dir.)": "-",
    "Paper WGAN": "-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AWGN BER/SER comparison curves from saved CSV data.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-stem", type=Path, required=True)
    parser.add_argument("--title", type=str, default="AWGN symbolic coding curves")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    return grouped


def configure_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.25, alpha=0.18)
    ax.set_xlim(-0.2, 8.2)


def plot_curves(rows: list[dict[str, str]], out_stem: Path, title: str) -> None:
    grouped = grouped_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45), sharex=True)

    for label, pts in grouped.items():
        pts = sorted(pts, key=lambda row: float(row["ebno_db"]))
        xs = [float(row["ebno_db"]) for row in pts]
        display = LABELS.get(label, label)
        for ax, metric in [(axes[0], "ber"), (axes[1], "ser")]:
            ys = [max(float(row[metric]), 1e-7) for row in pts]
            ax.plot(
                xs,
                ys,
                label=display,
                color=COLORS.get(label),
                marker=MARKERS.get(label, "o"),
                linestyle=LINESTYLES.get(label, "-"),
                linewidth=1.65,
                markersize=4.0,
                markeredgewidth=0.4,
            )

    configure_axes(axes[0], "BER")
    configure_axes(axes[1], "SER")
    axes[0].set_title("Bit error rate")
    axes[1].set_title("Symbol error rate")
    axes[1].legend(loc="best", fontsize=7.4, frameon=True)
    fig.suptitle(title, y=1.04, fontsize=11.0)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_curves(read_rows(args.csv), args.out_stem, args.title)
    print({"pdf": str(args.out_stem.with_suffix(".pdf")), "png": str(args.out_stem.with_suffix(".png"))})


if __name__ == "__main__":
    main()
