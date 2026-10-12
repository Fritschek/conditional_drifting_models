"""Audit and summarize the fixed SSPA input-coverage intervention."""

import argparse
import csv
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
import numpy as np
import torch

from conditional_drifting.baselines.fast_evaluation import write_json
from conditional_drifting.baselines.official_sources import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.suite_dir/"manifest.json").read_text())
    data = json.loads((args.suite_dir/"results.json").read_text())
    if data["manifest_sha256"] != sha256(args.suite_dir/"manifest.json"):
        raise ValueError("Manifest hash mismatch")
    for name, digest in manifest["parent_files"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Parent changed: {name}")
    for name, digest in manifest["parent"]["adapter_files"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Numerical source changed: {name}")
    if sha256(ROOT/"scripts/run_sspa_input_coverage.py") != manifest["script_sha256"]:
        raise ValueError("Intervention runner changed")
    variants = ("analytic", "gaussian_normal", "gaussian_broad", "sinkhorn_normal", "sinkhorn_broad")
    seeds = manifest["ae_seeds"]
    rows = {(v["variant"], v["ae_seed"]): v for v in data["rows"]}
    expected = {(v, s) for v in variants for s in seeds}
    if set(rows) != expected or len(data["rows"]) != len(expected):
        raise ValueError("Missing or duplicate AE rows")
    settings = manifest["parent"]["settings"]
    trials = 64*settings["ae_eval_samples"]
    hashes = {}
    for key, row in rows.items():
        if abs(row["mean_squared_norm"]-8) > 1e-4:
            raise ValueError("AE power normalization changed")
        for point in row["analytic_curve"]:
            if point["trials"] != trials or point["ser"] != point["errors"]/trials or point["ber"] != point["bit_errors"]/(6*trials):
                raise ValueError("SER/BER accounting mismatch")
        if not row["reused_parent_codec"]:
            path = args.suite_dir/key[0]/f"ae{key[1]}"/"codec.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["update"] != settings["ae_updates"] or state["seed"] != key[1]:
                raise ValueError("AE checkpoint incomplete")
            hashes[str(path)] = sha256(path)
    generators = {v["variant"]: v for v in data["generators"]}
    for family in ("gaussian", "sinkhorn"):
        broad = generators[family+"_broad"]["cost"]
        normal = generators[family+"_normal"]["cost"]
        if broad["config"] != normal["config"] or broad["counts"] != normal["counts"]:
            raise ValueError("Generator budgets/settings differ")
        path = args.suite_dir/(family+"_broad")/"generator.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
        if state["update"] != broad["config"]["updates"] or state["counts"] != broad["counts"]:
            raise ValueError("Generator checkpoint incomplete")
        if state["coverage"]["law"] != "broad" or state["coverage"]["broad_anchors"] != broad["broad_anchors"]:
            raise ValueError("Coverage checkpoint mismatch")
        if not all(torch.isfinite(v).all() for v in state["ema"].values()):
            raise ValueError("Nonfinite generator weights")
        hashes[str(path)] = sha256(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = dict(passed=True, ae_rows=len(rows), new_ae_checkpoints=len(rows)-sum(r["reused_parent_codec"] for r in rows.values()),
        result_sha256=sha256(args.suite_dir/"results.json"), checkpoint_hashes=hashes)
    write_json(args.out_dir/"audit.json", audit)
    flat = [dict(variant=r["variant"], ae_seed=r["ae_seed"], **point) for r in data["rows"] for point in r["analytic_curve"]]
    with (args.out_dir/"curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# SSPA input-coverage results", "", "One generator seed and paired AE initializations. No checkpoint selection.",
        "All downstream curves use analytic SSPA. Normal and broad generators have identical",
        "architectures, output-query counts and optimization settings; only training input law changes.", "",
        "| Variant | AE seed | SER 4 dB | SER 6 dB | SER 8 dB | Surrogate SER 8 dB | Maximum input norm | Own mean RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in variants:
        for seed in seeds:
            r = rows[variant, seed]
            ser = [v["ser"] for v in r["analytic_curve"]]
            lines.append(f"| {variant} | {seed} | {ser[0]:.6g} | {ser[1]:.6g} | {ser[2]:.6g} | "
                f"{r['surrogate_nominal']['ser']:.6g} | {max(r['codebook_norms']):.5g} | {r['own_moments']['mean_rmse']:.5g} |")
    lines += ["", "## Generator accuracy and cost", "",
        "| Variant | Random anchor SWD / floor | Mean RMSE on old Gaussian codebook | Generator seconds | Broad inputs / total inputs |",
        "|---|---:|---:|---:|---|"]
    for variant, r in generators.items():
        d, c = r["evaluation"]["random_inputs"], r["cost"]
        lines.append(f"| {variant} | {d['conditional']['swd']/d['analytic_floor']['swd']:.5g} | "
            f"{r['evaluation']['old_gaussian_codebook']['mean_rmse']:.5g} | {c['train_seconds']:.2f} | "
            f"{c.get('broad_anchors', 0)} / {c['counts']['input_anchors']} |")
    lines += ["", "## Radial accuracy", "", "| Variant | Input norm | Mean RMSE | Variance / analytic |",
        "|---|---:|---:|---:|"]
    for variant, r in generators.items():
        for point in r["evaluation"]["radial"]:
            lines.append(f"| {variant} | {point['radius']:.5g} | {point['mean_rmse']:.5g} | {point['variance_ratio']:.5g} |")
    lines += ["", "## Limits", "", "Three AE initializations do not replace independent generator seeds.",
        "The broad-law fraction and radius were fixed before training. No conclusions about the",
        "optimal mixture or universal applicability follow. Ordinary-input error may increase.",
        "Mean-error measurements include finite-sample noise, with floors retained in raw JSON.",
        "Zero observed errors are kept as zero in tables and omitted from log axes, not floored.",
        "No peak-power constraint, extra physical output restriction or downstream feedback was added.",
        "Expected training-query squared norm increases from 8 to 48.6667; equal oracle counts",
        "are not equal query energy. The physical noise and AE average-power constraint are unchanged.",
        f"Source: `{args.suite_dir}`. Audit passed for {len(rows)} rows and unchanged parent artifacts."]
    (args.out_dir/"results.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = {"normal": "#c75a32", "broad": "#008877"}
    seed_colors = ("#2166ac", "#87459c", "#a27700")
    for col, family in enumerate(("gaussian", "sinkhorn")):
        ax = axes[0, col]
        for law in ("normal", "broad"):
            points = generators[family+"_"+law]["evaluation"]["radial"]
            ax.plot([r["radius"] for r in points], [r["mean_rmse"] for r in points], color=colors[law],
                marker="o", label="Gaussian inputs" if law == "normal" else "Broader input mixture")
        ax.axhline(points[0]["mc_mean_rmse_floor"], linestyle=":", color="#666666", label="Mean MC RMSE floor")
        ax.set(xlabel="Input norm", ylabel="Conditional mean RMSE", title="Gaussian" if col == 0 else "Sinkhorn K64", yscale="log")
        ax.legend(fontsize=8, frameon=False)
        ax = axes[1, col]
        for i, seed in enumerate(seeds):
            keys = ("analytic", family+"_normal", family+"_broad")
            values = [rows[k, seed]["analytic_curve"][1]["ser"] for k in keys]
            ax.plot(range(3), [v if v > 0 else np.nan for v in values], marker="o", color=seed_colors[i%3], label=str(seed))
        ax.set_xticks(range(3), ("Analytic", "Gaussian inputs", "Broader mixture"))
        ax.set(ylabel="Analytic-channel SER at 6 dB", yscale="log")
        ax.legend(title="AE seed", fontsize=8, title_fontsize=8, frameon=False)
    for pair in axes:
        limits = (min(ax.get_ylim()[0] for ax in pair), max(ax.get_ylim()[1] for ax in pair))
        for ax in pair:
            ax.set_ylim(limits)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SSPA input coverage: fixed query budget, one generator seed", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    for extension in ("png", "pdf"):
        fig.savefig(args.out_dir/f"coverage_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k:v for k,v in audit.items() if k != "checkpoint_hashes"}))


if __name__ == "__main__":
    main()
