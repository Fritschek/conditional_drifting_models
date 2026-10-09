"""Join completed decoder-free scores to existing task diagnostics, without fitting a score."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-runs", nargs="+", required=True)
    parser.add_argument("--gradient-runs", nargs="+", required=True,
                        help="Later runs replace only overlapping model rows (use higher-sample confirmation last).")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    metrics = [json.loads((Path(p) / "results.json").read_text()) for p in args.metric_runs]
    for run in metrics:
        config = run["manifest"]["config"]
        expected = set()
        for channel in config["channels"].split(","):
            methods = set(config["methods"].split(","))
            if channel == "SSPA" and "fiber_sinkhorn" in methods:
                methods.add("fiber_full")
            for method in methods:
                for mode in config["modes"].split(","):
                    for step in config["steps"]:
                        for rep in range(config["repeats"]):
                            for anchor in range(3 * config["anchor_directions"]):
                                expected.add((channel, method, mode, step, rep, anchor))
        actual = [(r["channel"], r["method"], r["mode"], r["step"], r["repeat"], r["anchor"]) for r in run["rows"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Incomplete or duplicate metric observations; do not report a partial run as complete.")
    tasks = {}
    task_sources = {}
    for path in args.gradient_runs:
        gradient_run = json.loads((Path(path) / "results.json").read_text())
        for task, comparison in gradient_run["comparisons"].items():
            if task in tasks:
                if tasks[task]["codec"]["sha256"] != comparison["codec"]["sha256"] or not math.isclose(tasks[task]["noise_std"], comparison["noise_std"], rel_tol=1e-6):
                    raise ValueError(f"Cannot merge different codec/noise contracts: {task}")
                tasks[task]["methods"].update(comparison["methods"])
            else:
                tasks[task] = comparison
            for method in comparison["methods"]:
                task_sources[(task, method)] = {"gradient_run": path,
                    "gradient_samples_per_message_per_repeat": gradient_run["config"]["gradient_samples"],
                    "reference_samples_per_message_per_repeat": gradient_run["config"]["reference_samples"],
                    "gradient_repeats": gradient_run["config"]["repeats"]}
    joined = []
    for run in metrics:
        n = run["manifest"]["config"]["samples"]
        for row in run["summary"]:
            for task, result in tasks.items():
                if task.split("/")[0] != row["channel"] or row["method"] not in result["methods"]:
                    continue
                grad = result["methods"][row["method"]]
                contract = run["manifest"]["channels"][row["channel"]]
                if not math.isclose(result["noise_std"], contract["noise_std"], rel_tol=1e-6):
                    raise ValueError("Metric/task channel noise mismatch.")
                if row["method"] != "analytic":
                    model_id = f"{row['channel']}/{row['method']}"
                    if run["manifest"]["checkpoints"][model_id]["sha256"] != grad["checkpoint"]["sha256"]:
                        raise ValueError(f"Metric/task checkpoint mismatch: {model_id}")
                joined.append({"channel": row["channel"], "method": row["method"], "codec": task.split("/")[1],
                    "samples": n, "mode": row["mode"], "h": row["step"],
                    **task_sources[(task, row["method"])],
                    **{key: value["anchor_mean"] for key, value in row.items() if isinstance(value, dict)},
                    **{f"encoder_{key}": value for key, value in grad["encoder_alignment"].items()},
                    "analytic_ce_change_small_normalized_step": grad["steps"][0]["ce_delta"]["mean"]})
    if not joined:
        raise ValueError("No matching channel/model diagnostics.")
    with (out / "score_task_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    (out / "inputs.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for i, channel in enumerate(("AWGN", "SSPA")):
        for j, mode in enumerate(("crn", "independent")):
            ax = axes[i, j]
            for color, run in zip(("#007f73", "#c05232"), metrics):
                selected = sorted([r for r in run["summary"] if r["channel"] == channel and r["mode"] == mode and r["method"] == "analytic"], key=lambda r: r["step"])
                h = [r["step"] for r in selected]
                mean = np.array([r["embedding_derivative_op"]["anchor_mean"] for r in selected])
                se = np.array([r["embedding_derivative_op"]["anchor_mean_mc_se"] or 0 for r in selected])
                ax.plot(h, mean, marker="o", color=color, label=f"N={run['manifest']['config']['samples']}")
                ax.fill_between(h, np.maximum(mean - se, 1e-12), mean + se, color=color, alpha=.16)
            ax.set(xscale="log", yscale="log", title=f"{channel}: {'shared noise' if mode == 'crn' else 'independent noise'}",
                   xlabel="Perturbation h", ylabel="Analytic-vs-analytic derivative floor")
            ax.set_xticks([.025, .05, .1], ["0.025", "0.05", "0.1"])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
    fig.suptitle("Finite-sample floor of the full-kernel derivative estimate\nFixed radial anchors; shading is Monte Carlo SE, not a population error bound", fontsize=11)
    fig.savefig(out / "derivative_sampling_floor.png", dpi=160)
    fig.savefig(out / "derivative_sampling_floor.pdf")
    plt.close(fig)

    lines = ["# Decoder-free scores and existing task diagnostics", "",
             "Post-hoc development comparison. Scores use fixed generic inputs and no decoder.",
             "Task gradients use learned codewords and frozen codecs; these are different input panels.",
             "Higher-sample gradient results replace only the models rerun; per-row gradient budgets and sources are in CSV.",
             "No score was fitted, and these known seed-7 cases are not a held-out validation set.", "",
             "![Derivative sampling floor](derivative_sampling_floor.png)", "",
             "## Middle perturbation size (h=0.05), shared noise", "",
             "All step sizes and both noise regimes are retained in the source results and joined CSV.", "",
             "| Channel | N | Model | MMD | Embedding derivative | Mean derivative | Covariance error | Covariance derivative | Encoder relative error, analytic codec | Encoder relative error, Sinkhorn codec |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in metrics:
        for row in run["summary"]:
            if row["mode"] != "crn" or row["step"] != .05:
                continue
            task_errors = []
            for codec in ("analytic", "fiber_sinkhorn"):
                record = tasks.get(f"{row['channel']}/{codec}", {}).get("methods", {}).get(row["method"])
                task_errors.append(f"{record['encoder_alignment']['relative_error']:.4g}" if record else "missing")
            values = [f"{row[k]['anchor_mean']:.4g}" for k in ("mmd", "embedding_derivative_op", "mean_derivative_op", "covariance_fro", "covariance_derivative_op")]
            lines.append(f"| {row['channel']} | {run['manifest']['config']['samples']} | {row['method']} | " + " | ".join(values + task_errors) + " |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(out / "README.md")


if __name__ == "__main__":
    main()
