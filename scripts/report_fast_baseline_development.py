"""Summarize the bounded baseline runs without dropping failed recipes."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional-drifting-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", type=Path, default=ROOT / "results/fast_baseline_development_20261012")
    p.add_argument("--warmstarts", nargs="+", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    development = json.loads((args.development / "results.json").read_text())
    warmstarts = [json.loads((v / "results.json").read_text()) for v in args.warmstarts]
    lines = ["# AWGN baseline development results", "",
        "All final EMA one-step results are retained, including failed recipes. These are development",
        "checks, not downstream SER/BER results or a final across-family ranking.", "",
        "| Recipe | Seed | SWD / floor | Mean RMSE | Variance / truth | Gate | Training seconds | Oracle outputs |",
        "|---|---:|---:|---:|---:|---|---:|---:|"]
    def append(name, row, seconds=None, outputs=None):
        value = row["primary"]
        lines.append(f"| {name} | {row['config']['seed']} | {value['conditional_swd_ratio']:.4f} | "
            f"{value['moments']['mean_rmse']:.5f} | {value['moments']['variance_ratio']:.4f} | "
            f"{'pass' if value['quality_gate'] else 'fail'} | "
            f"{row['train_seconds'] if seconds is None else seconds:.2f} | "
            f"{row['counts']['oracle_outputs'] if outputs is None else outputs} |")
    for group in ("development", "confirmation"):
        for name, row in development[group].items():
            append(name, row)
    for run in warmstarts:
        for kind, row in run["rows"].items():
            append("EDM + " + kind.upper(), row, row["total_train_seconds"], row["total_oracle_outputs"])
    lines += ["", "Warm-start student seeds equal their pretraining seed plus 10000. Both student rows",
        "include the complete teacher/pretraining cost. The teacher is shared only in execution.", "",
        "The development gate requires conditional SWD/floor <= 1.5, conditional mean RMSE <= .1,",
        "and variance/truth in [.8,1.2]. These thresholds are not confidence intervals.", "",
        "## WGAN EMA lag", "",
        "| Outer updates | Generator updates | Weights | SWD / floor | Mean RMSE | Variance / truth |",
        "|---:|---:|---|---:|---:|---:|"]
    for history in development["development"]["wgan_default"]["history"]:
        for value in history["rows"]:
            lines.append(f"| {history['update']} | {history['update']//5} | {value['weights']} | "
                f"{value['conditional_swd_ratio']:.4f} | {value['moments']['mean_rmse']:.5f} | "
                f"{value['moments']['variance_ratio']:.4f} |")
    lines += ["", "EMA decay .999 is applied per generator update. Initial-weight contributions are",
        f"{.999**1000:.3%} after 1000 generator updates and {.999**4000:.3%} after 4000.",
        "This explains a substantial part of the short pilot's weakness, without changing the",
        "generator sampling mechanism or claiming adversarial training universally succeeds.", "",
        "## Sources", "", f"- {args.development}/results.json"]
    lines.extend(f"- {v}/results.json" for v in args.warmstarts)
    (args.out_dir / "results.md").write_text("\n".join(lines)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    curves = [
        ("CT, original curriculum", development["development"]["ct_default"], "ema", "#777777", "-", 0),
        ("CT, coarser curriculum", development["development"]["ct_coarse"], "ema", "#b66d00", "-", 0),
        ("WGAN, raw", development["development"]["wgan_default"], "raw", "#007c91", "--", 0),
        ("WGAN, EMA", development["development"]["wgan_default"], "ema", "#007c91", "-", 0),
        ("Consistency distillation, EMA", warmstarts[0]["rows"]["cd"], "ema", "#873aad",
         "-", warmstarts[0]["pretraining"]["counts"]["oracle_outputs"]),
    ]
    for label, row, weights, color, style, exposure in curves:
        xs, values = [], []
        for history in row["history"]:
            v = next(v for v in history["rows"] if v["weights"] == weights and v["nfe"] == 1)
            xs.append((history["update"]*row["config"]["batch_size"]+exposure)/1e6)
            values.append((v["conditional_swd_ratio"], v["moments"]["mean_rmse"], v["moments"]["variance_ratio"]))
        for index, ax in enumerate(axes):
            ax.plot(xs, [v[index] for v in values], label=label, color=color, linestyle=style, marker="o", markersize=3)
    for index, (ax, title) in enumerate(zip(axes, ("Conditional SWD / sampling floor", "Conditional mean RMSE", "Conditional variance / truth"))):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training oracle outputs (millions)", fontsize=9)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if index < 2:
            ax.set_yscale("log")
        if index != 1:
            ax.axhline(1., color="black", linewidth=.7, linestyle=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .16, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(args.out_dir / f"awgn_development.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
