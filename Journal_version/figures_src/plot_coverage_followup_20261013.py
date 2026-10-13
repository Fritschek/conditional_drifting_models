"""Plot generator-block effects and the reversed ordinary-input metric ordering."""

import json
import os
from pathlib import Path
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/coverage-followup-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Journal_version/evidence/coverage_followup_20261013/coverage/results.json"
data = json.loads(SOURCE.read_text())
colors = ("#17669d", "#aa5a19", "#557b35")
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
for seed, color in zip((9201, 9202, 9203), colors):
    blocks = [b for b in data["blocks"] if b["generator_seed"] == seed]
    for ax, family in zip(axes[:2], ("gaussian", "sinkhorn")):
        values = [statistics.mean(b["curves"][-1]["variants"][family+"_"+law]["ser"] for b in blocks)
                  for law in ("normal", "broad")]
        ax.plot((0, 1), values, marker="o", color=color, linewidth=2, label=str(seed))
        ax.set_yscale("log")
        ax.set_ylim(5e-6, 2e-3)
    gens = {g["variant"]: g for g in data["generators"] if g["generator_seed"] == seed}
    ratios = []
    for law in ("normal", "broad"):
        d = gens["sinkhorn_"+law]["evaluation"]["random_inputs"]
        ratios.append(d["conditional"]["swd"]/d["analytic_floor"]["swd"])
    axes[2].plot((0, 1), ratios, marker="o", color=color, linewidth=2, label=str(seed))
for ax, title in zip(axes, ("Gaussian: transfer improves", "Sinkhorn: mean transfer improves", "Sinkhorn: ordinary-input error rises")):
    ax.set_title(title, fontsize=10, pad=12)
    ax.set_xticks((0, 1), ("Normal inputs", "Broader inputs"))
    ax.set_xlim(-.12, 1.12)
    ax.grid(axis="y", alpha=.2, which="both")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Physical-channel SER; mean of 3 paired AEs")
axes[2].set_ylabel("Conditional SWD / sampling floor")
axes[2].set_ylim(1.1, 1.6)
axes[2].legend(title="Generator seed", frameon=False, fontsize=9, title_fontsize=9, loc="lower right")
fig.suptitle("Coverage replication: each line is one fresh generator-seed block", fontsize=13, y=.99)
fig.text(.5, .025, "SER at nominal software 8 dB (= 11.01 dB input-referenced Eb/N0). Individual AE outcomes remain mixed.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .09, 1, .94))
out = ROOT / "Journal_version/evidence/coverage_followup_assessment_20261013.png"
fig.savefig(out, dpi=180)
plt.close(fig)
print(out)
