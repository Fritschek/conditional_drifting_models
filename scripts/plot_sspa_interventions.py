#!/usr/bin/env python3
"""Plot all continuation seeds, including failed-fidelity branches."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional-drifting-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"control": "Unchanged control", "log_floor10": "Log solver, floored kernel",
          "log10": "Unfloored log solver (10)", "log100": "Unfloored log solver (100)",
          "log10_beta0": "Log (10), beta1 = 0", "log10_lr01": "Log (10), learning rate / 10",
          "log10_drift01": "Log (10), drift scale / 10", "official64": "Official-style normalized K64"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.suite_dir / "audit.json").read_text())
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = ("#2166ac", "#b35806", "#00896b")
    for key, label in (("anchor_swd", "Conditional SWD"),
                       ("conditional_variance", "Average conditional variance"),
                       ("anchor_mean_l2", "Conditional mean L2")):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharex=True, sharey=True)
        for ax, branch in zip(axes.flat, manifest["branches"]):
            for color, seed in zip(colors, manifest["seeds"]):
                rows = json.loads((args.suite_dir / branch / f"seed{seed}/trajectory.json").read_text())
                x = [r["update"] / 1000 for r in rows]
                ax.plot(x, [r[key] for r in rows], color=color, lw=1.4, label=str(seed), marker=".", markersize=2)
                if branch == "official64":
                    ax.plot(x, [r["ema"][key] for r in rows], color=color, lw=1.2, ls="--")
            if key == "conditional_variance":
                parent = json.loads((args.suite_dir / branch / "seed9001/result200000.json").read_text())["parent"]
                ax.axhline(parent["config"]["noise_std"] ** 2 / 2, color="0.4", ls=":", lw=1)
            else:
                ax.axhline(rows[0][key + "_floor"], color="0.4", ls=":", lw=1)
            ax.axvline(200, color="0.7", lw=.7)
            ax.set_title(LABELS[branch], fontsize=11)
            ax.set_yscale("log")
            ax.set_xlim(78, 395)
            ax.grid(axis="y", alpha=.18)
        axes[0, 0].legend(title="Development seed", frameon=False, fontsize=9)
        fig.supxlabel("Total training updates (thousands)", y=.055)
        fig.supylabel(label)
        fig.suptitle("SSPA continuation from 80k: all branches and seeds", fontsize=14)
        fig.text(.5, .015, "Dotted: analytic variance or sampling floor. Dashed: EMA for K64. "
                 "Branches failing the 200k gate stop there.", ha="center", fontsize=9)
        fig.tight_layout(rect=(.01, .09, 1, .96))
        for extension in ("pdf", "png"):
            fig.savefig(args.suite_dir / f"trajectories_{key}.{extension}", dpi=160)
        plt.close(fig)
    print(f"Plotted {len(audit['rows'])} audited endpoint rows.")


if __name__ == "__main__":
    main()
