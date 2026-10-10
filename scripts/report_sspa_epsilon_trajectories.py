#!/usr/bin/env python3
"""Validate and summarize a complete SSPA trajectory suite."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def load_suite(root):
    manifest = json.loads((root / "manifest.json").read_text())
    keys = [(s, p) for s in manifest["seeds"] for p in manifest["policies"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate manifest tasks")
    expected = set(keys)
    results, histories, traces = {}, {}, {}
    for path in root.glob("*/seed*/result.json"):
        result = json.loads(path.read_text())
        config = result["config"]
        key = config["seed"], config["policy"]
        if key not in expected or key in results:
            raise ValueError(f"Unexpected or duplicate result: {path}")
        configured = next(c for c in manifest["configs"] if (c["seed"], c["policy"]) == key)
        if config != configured:
            raise ValueError(f"Configuration mismatch: {path}")
        status = json.loads((path.parent / "status.json").read_text())
        history = json.loads((path.parent / "trajectory.json").read_text())
        trace = json.loads((path.parent / "training_trace.json").read_text())
        if status["state"] != "complete" or result["final_update"] != manifest["updates"]:
            raise ValueError(f"Incomplete task: {path}")
        if [row["update"] for row in history] != manifest["checkpoints"]:
            raise ValueError(f"Incomplete or duplicate validation checkpoints: {path}")
        if [row["update"] for row in trace] != list(range(100, manifest["updates"] + 1, 100)):
            raise ValueError(f"Incomplete training trace: {path}")
        count = result["counts"]
        updates, batch, samples = manifest["updates"], config["batch_size"], config["samples"]
        expected_counts = dict(training_anchors=updates * batch,
            training_oracle_outputs=updates * batch * samples,
            training_generated_outputs=updates * batch * samples,
            training_reference_outputs=updates * batch * samples,
            calibration_oracle_outputs=2 * config["calibration_anchors"] * samples,
            validation_oracle_outputs=2 * len(history) * config["validation_anchors"] * config["validation_samples"],
            diagnostic_oracle_outputs=len(history) * config["validation_anchors"] * samples)
        if count != expected_counts:
            raise ValueError(f"Sample-count mismatch: {path}")
        selected = min(history, key=lambda row: (row["anchor_swd"], row["update"]))
        if (result["selected_update"] != selected["update"] or
                result["selected_anchor_swd"] != selected["anchor_swd"] or
                result["last_anchor_swd"] != history[-1]["anchor_swd"]):
            raise ValueError(f"Selection mismatch: {path}")
        if history[-1]["counts"] != count:
            raise ValueError(f"Final validation count mismatch: {path}")
        results[key], histories[key], traces[key] = result, history, trace
    missing = expected - results.keys()
    if missing:
        raise ValueError(f"Missing tasks: {sorted(missing)}")
    return manifest, results, histories, traces


def mean_sd(values):
    return f"{statistics.mean(values):.6g} +/- {statistics.stdev(values):.3g}" if len(values) > 1 else f"{values[0]:.6g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, results, histories, traces = load_suite(args.suite_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flat = [dict(seed=s, policy=p, **{k: v for k, v in row.items() if not isinstance(v, (dict, list))})
            for (s, p), rows in histories.items() for row in rows]
    with (args.out_dir / "validation.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    checks = dict(tasks=len(results), checkpoints=len(flat), training_records=sum(map(len, traces.values())),
                  reference_attempts=0, reference_failures=0, max_reference_barycenter_rms=0.,
                  max_postinitial_reference_barycenter_rms=0.,
                  max_training_floor_fraction=0., max_training_row_relative=0.,
                  training_seconds=sum(r["train_seconds"] for r in results.values()),
                  validation_seconds=sum(r["validation_seconds"] for r in results.values()))
    for rows in histories.values():
        for row in rows:
            for ref in row["reference_checks"]:
                checks["reference_attempts"] += 1
                if not ref["converged"]:
                    checks["reference_failures"] += 1
                    if ref["barycenter_rms"] is not None:
                        raise ValueError("Unconverged reference treated as ground truth")
                else:
                    checks["max_reference_barycenter_rms"] = max(checks["max_reference_barycenter_rms"], ref["barycenter_rms"])
                    if row["update"] > 0:
                        checks["max_postinitial_reference_barycenter_rms"] = max(
                            checks["max_postinitial_reference_barycenter_rms"], ref["barycenter_rms"])
    for rows in traces.values():
        for row in rows:
            checks["max_training_floor_fraction"] = max(checks["max_training_floor_fraction"],
                                                         row["cross_floor_fraction"], row["self_floor_fraction"])
            checks["max_training_row_relative"] = max(checks["max_training_row_relative"],
                                                       row["cross_row_relative"], row["self_row_relative"])
    (args.out_dir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    policy_summary = {}
    for policy in manifest["policies"]:
        selected = [results[s, policy]["selected_anchor_swd"] for s in manifest["seeds"]]
        refs = [ref for s in manifest["seeds"] for row in histories[s, policy] for ref in row["reference_checks"]]
        summary = dict(selected_swd_mean=statistics.mean(selected),
                       selected_swd_sd=statistics.stdev(selected) if len(selected) > 1 else None,
                       final_variance_mean=statistics.mean(histories[s, policy][-1]["conditional_variance"]
                                                           for s in manifest["seeds"]),
                       reference_failures=sum(not r["converged"] for r in refs),
                       reference_attempts=len(refs))
        if manifest["updates"] > 30000 and 30000 in manifest["checkpoints"]:
            summary["posthoc_first_saved_swd_over_twice_30k"] = {}
            for s in manifest["seeds"]:
                before = next(r for r in histories[s, policy] if r["update"] == 30000)
                summary["posthoc_first_saved_swd_over_twice_30k"][s] = next(
                    (r["update"] for r in histories[s, policy] if r["update"] > 30000
                     and r["anchor_swd"] > 2 * before["anchor_swd"]), None)
            summary["threshold_scope"] = "Post-hoc description only; not a stopping or selection rule"
        policy_summary[policy] = summary
    (args.out_dir / "policy_summary.json").write_text(json.dumps(policy_summary, indent=2, allow_nan=False) + "\n")
    lines = ["# SSPA epsilon trajectories", "", "Development validation only. Uncertainty below is seed SD, not SE.",
             "", "| Policy | Last conditional SWD | Last conditional GW2 | Last global SWD | Selected updates | Training seconds |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for p in manifest["policies"]:
        last = [histories[s, p][-1] for s in manifest["seeds"]]
        rows = [results[s, p] for s in manifest["seeds"]]
        cells = [p] + [mean_sd([r[m] for r in last]) for m in ("anchor_swd", "anchor_gw2", "global_swd")]
        cells += [", ".join(str(r["selected_update"]) for r in rows), mean_sd([r["train_seconds"] for r in rows])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Selected and final checkpoints by seed (same validation panel; no held-out test):", "",
              "| Policy | Seed | Selected update | Selected SWD | Final SWD | Final variance |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for p in manifest["policies"]:
        for s in manifest["seeds"]:
            result, final = results[s, p], histories[s, p][-1]
            lines.append(f"| {p} | {s} | {result['selected_update']} | {result['selected_anchor_swd']:.6g} | "
                         f"{final['anchor_swd']:.6g} | {final['conditional_variance']:.6g} |")
    first = next(iter(histories.values()))[-1]
    lines += ["", f"Common analytic anchor floors: SWD {first['anchor_swd_floor']:.6g}; GW2 {first['anchor_gw2_floor']:.6g}.",
              "", "```json", json.dumps(checks, indent=2), "```", "",
              f"These trajectories stop at {manifest['updates']:,} updates. Behavior beyond this budget is untested."]
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for p, color in zip(manifest["policies"], colors):
        for j, s in enumerate(manifest["seeds"]):
            rows = histories[s, p]
            x = [r["update"] for r in rows]
            for ax, metric in zip(axes.flat, ("anchor_swd", "anchor_gw2", "conditional_variance", "epsilon_ratio")):
                y = [r["transport"][metric] if metric == "epsilon_ratio" else r[metric] for r in rows]
                ax.plot(x, y, color=color, alpha=.7, marker="o", markersize=3,
                        linestyle=["-", "--", ":"][j % 3], label=p if j == 0 else None)
    for ax, title in zip(axes.flat, ("Conditional SWD", "Conditional Gaussian W2", "Conditional output variance", "Cross / self epsilon")):
        ax.set_title(title)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Training updates")
        ax.grid(alpha=.2)
    for ax, key in zip(axes[0], ("anchor_swd_floor", "anchor_gw2_floor")):
        ax.set_yscale("log")
        ax.axhline(first[key], color="0.45", linewidth=1, linestyle=":", label="Analytic floor")
    axes[1, 0].axhline(manifest["configs"][0]["noise_std"] ** 2 / 2, color="0.45", linestyle=":", label="True variance")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.out_dir / "trajectories.pdf")
    fig.savefig(args.out_dir / "trajectories.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    definitions = (
        ("Detached regression loss", lambda row: row["loss"], "linear"),
        ("Cross / self epsilon", lambda row: row["epsilon_ratio"], "log"),
        ("Raw drift mean norm", lambda row: row["raw_drift_mean_norm"], "linear"),
        ("Drift clipping fraction", lambda row: row["drift_clip_fraction"], "linear"),
        ("Worst raw row-mass relative error", lambda row: max(row["cross_row_relative"], row["self_row_relative"]), "log"),
        ("Largest kernel-floor fraction", lambda row: max(row["cross_floor_fraction"], row["self_floor_fraction"]), "linear"),
    )
    for policy, color in zip(manifest["policies"], colors):
        for j, seed in enumerate(manifest["seeds"]):
            rows = traces[seed, policy]
            for ax, (_, value, _) in zip(axes.flat, definitions):
                ax.plot([r["update"] for r in rows], [value(r) for r in rows], color=color,
                        alpha=.55, linewidth=.65, linestyle=["-", "--", ":"][j % 3],
                        label=policy if j == 0 else None)
    for ax, (title, _, scale) in zip(axes.flat, definitions):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training updates")
        ax.set_xscale("log")
        ax.set_yscale(scale)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.out_dir / "training_traces.pdf")
    fig.savefig(args.out_dir / "training_traces.png", dpi=150)
    plt.close(fig)
    if manifest["updates"] > 30000:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for p, color in zip(manifest["policies"], colors):
            for j, s in enumerate(manifest["seeds"]):
                rows = [r for r in histories[s, p] if r["update"] >= 30000]
                for ax, metric in zip(axes.flat, ("anchor_swd", "anchor_gw2", "conditional_variance", "anchor_mean_l2")):
                    ax.plot([r["update"] / 1000 for r in rows], [r[metric] for r in rows], color=color,
                            marker="o", markersize=3, linestyle=["-", "--", ":"][j % 3],
                            label=p if j == 0 else None)
        for ax, title, floor in zip(axes.flat,
                ("Conditional SWD", "Conditional Gaussian W2", "Conditional output variance", "Conditional mean error"),
                (first["anchor_swd_floor"], first["anchor_gw2_floor"],
                 manifest["configs"][0]["noise_std"] ** 2 / 2, first["anchor_mean_l2_floor"])):
            ax.set_title(title)
            ax.set_xlabel("Training updates (thousands)")
            ax.set_yscale("log")
            ax.axhline(floor, color="0.45", linestyle=":")
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.savefig(args.out_dir / "late_trajectories.pdf")
        fig.savefig(args.out_dir / "late_trajectories.png", dpi=150)
        plt.close(fig)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
