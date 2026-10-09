"""Check complete synthetic control grids and export summaries and figures."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, keys, quantities):
    output = []
    for key in dict.fromkeys(tuple(r[k] for k in keys) for r in rows):
        group = [r for r in rows if tuple(r[k] for k in keys) == key]
        row = dict(zip(keys, key)) | {"repeats": len(group)}
        for name in quantities:
            values = [r[name] for r in group]
            row[name] = float(np.mean(values))
            row[name+"_mc_se"] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None
        output.append(row)
    return output


def check_grid(rows, keys, expected):
    actual = [tuple(r[k] for k in keys) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Incomplete/duplicate grid: {keys}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    a = json.loads((args.rotation/"results.json").read_text())
    b = json.loads((args.matched/"results.json").read_text())
    ca, cb = a["manifest"]["config"], b["manifest"]["config"]
    check_grid(a["rows"], ["samples_per_split", "repeat", "omega"],
               {(n, rep, w) for n in ca["rotation_samples"] for rep in range(ca["rotation_repeats"]) for w in (0., 1., 4., 16.)})
    bkeys = ["direction", "eta", "anchor", "repeat", "slope"]
    expected = {(d, eta, anchor, rep, slope) for d in ("development", "unused_direction") for eta in (0., .1)
                for anchor in range(3) for rep in range(cb["matched_repeats"]) for slope in (-4., -1., 0., 1., 4.)}
    check_grid(b["rows"], bkeys, expected)
    names = [f"{group}_{kind}" for group in ("development", "unused_probe") for kind in ("rbf_section", "quadratic", "logistic", "cosine")]
    check_grid(b["losses"], bkeys+["probe"], {key+(name,) for key in expected for name in names})

    rotation_value_spreads = []
    for n in ca["rotation_samples"]:
        for rep in range(ca["rotation_repeats"]):
            group = [r for r in a["rows"] if r["samples_per_split"] == n and r["repeat"] == rep]
            for metric in ("swd", "rbf_mmd", "moments_mmd", "augmented_mmd"):
                rotation_value_spreads.append(float(np.ptp([r[metric] for r in group])))
    if max(rotation_value_spreads) > 1e-10:
        raise ValueError("Rotation changed paired value scores at the centered anchor")
    lookup = {tuple(r[k] for k in bkeys): r for r in b["rows"]}
    violations = {"rbf_section": [], "quadratic": []}
    for row in b["losses"]:
        metric = lookup[tuple(row[k] for k in bkeys)]
        if row["loss"] == "rbf_section":
            bound = metric["population_rbf_embedding_derivative_op"]
        elif row["loss"] == "quadratic":
            offset = .2 if row["probe_split"] == "development" else -.3
            coefficient_norm = np.sqrt(1+2*offset**2)
            bound = coefficient_norm*metric["population_moments_embedding_derivative_op"]
        else:
            continue
        violations[row["loss"]].append(max(0., row["absolute_gradient_error"]-bound))
    if max(max(v) for v in violations.values()) > 1e-7:
        raise ValueError("Population task-class bound failed")

    paired_value_spreads = []
    for group in dict.fromkeys(tuple(r[k] for k in bkeys[:-1]) for r in b["rows"]):
        rows = [r for r in b["rows"] if tuple(r[k] for k in bkeys[:-1]) == group]
        for metric in ("swd", "rbf_mmd", "moments_mmd", "augmented_mmd"):
            spread = np.ptp([r[metric] for r in rows])
            paired_value_spreads.append(float(spread))
            if spread > 1e-10:
                raise ValueError(f"Value metric changed with slope: {group}, {metric}")

    rotation = summary(a["rows"], ["samples_per_split", "omega"],
        ["swd", "linear_self_trace", "linear_cross_trace", "linear_expected_self_trace",
         "rbf_mmd", "rbf_embedding_derivative_op", "rbf_self_trace", "rbf_cross_trace",
         "rbf_estimated_noise_trace", "moments_self_trace", "moments_cross_trace"])
    matched = summary(b["rows"], ["direction", "eta", "anchor", "slope"],
        ["swd", "rbf_mmd", "rbf_embedding_derivative_op", "rbf_self_trace", "rbf_cross_trace",
         "population_rbf_mmd", "population_rbf_embedding_derivative_op", "moments_embedding_derivative_op",
         "population_moments_embedding_derivative_op", "population_mean_jacobian_error"])
    losses = summary(b["losses"], ["direction", "eta", "anchor", "slope", "probe"],
        ["absolute_gradient_error", "sampled_absolute_gradient_error", "gradient_difference_estimation_error"])
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/"rotation_summary.csv", rotation)
    write_csv(out/"matched_summary.csv", matched)
    write_csv(out/"loss_summary.csv", losses)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    colors = ["#287f79", "#b44c59", "#476bb4"]
    for n, color in zip(ca["rotation_samples"], colors):
        rows = [r for r in rotation if r["samples_per_split"] == n]
        axes[0].errorbar([r["omega"] for r in rows], [r["rbf_embedding_derivative_op"] for r in rows],
                         yerr=[r["rbf_embedding_derivative_op_mc_se"] for r in rows], marker="o", color=color, label=f"N={n}", capsize=3)
    axes[0].set(title="Identical channel laws", xlabel="Rotation frequency", ylabel="Empirical RBF derivative norm", yscale="log")
    axes[0].legend(frameon=False)
    rows = [r for r in rotation if r["samples_per_split"] == 512]
    for metric, label, color in (("rbf_self_trace", "Self Gram", "#b44c59"), ("rbf_cross_trace", "Cross Gram", "#287f79")):
        axes[1].errorbar([r["omega"] for r in rows], [r[metric] for r in rows],
                         yerr=[r[metric+"_mc_se"] for r in rows], marker="o", color=color, label=label, capsize=3)
    axes[1].axhline(0, color="black", linewidth=.7, linestyle="--")
    axes[1].set(title="Squared derivative discrepancy, N=512", xlabel="Rotation frequency", ylabel="Derivative Gram trace")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-.05, 20)
        ax.set_xticks([0, 1, 4, 16], ["0", "1", "4", "16"])
        ax.grid(alpha=.2)
    fig.savefig(out/"rotation_null.png", dpi=180)
    fig.savefig(out/"rotation_null.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for eta, color in ((0., "#287f79"), (.1, "#b44c59")):
        rows = [r for r in matched if r["direction"] == "development" and r["anchor"] == 2 and r["eta"] == eta]
        axes[0].plot([r["slope"] for r in rows], [r["population_rbf_mmd"] for r in rows], "o-", color=color, label=f"Mean offset {eta}")
    axes[0].set(title="Output-law error at the input stays fixed", xlabel="Local slope perturbation", ylabel="Population RBF MMD")
    axes[0].legend(frameon=False)
    for kind, color in zip(("quadratic", "logistic", "rbf_section"), colors):
        rows = [r for r in losses if r["direction"] == "development" and r["anchor"] == 2 and r["eta"] == 0 and r["probe"] == f"development_{kind}"]
        axes[1].plot([r["slope"] for r in rows], [r["absolute_gradient_error"] for r in rows], "o-", color=color, label=kind.replace("_", " "))
    axes[1].set(title="Expected-loss gradients change", xlabel="Local slope perturbation", ylabel="Absolute input-gradient error")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks([-4, -1, 0, 1, 4])
        ax.grid(alpha=.2)
    fig.savefig(out/"matched_value_derivative.png", dpi=180)
    fig.savefig(out/"matched_value_derivative.pdf")
    plt.close(fig)
    checks = {"rotation_rows": len(a["rows"]), "matched_rows": len(b["rows"]), "task_rows": len(b["losses"]),
              "max_paired_value_spread": max(paired_value_spreads),
              "max_rotation_value_spread": max(rotation_value_spreads),
              "max_population_bound_violations": {k: max(v) for k, v in violations.items()},
              "max_logistic_quadrature_difference": max(r["quadrature_difference"] for r in b["losses"]),
              "rotation_cost": a["manifest"]["timing_and_queries"], "matched_cost": b["manifest"]["timing_and_queries"],
              "inputs": {"rotation": str(args.rotation), "matched": str(args.matched)}}
    (out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
