"""Summarize numerical resolution and join fixed metric scores to task gradients."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_suite(path):
    result = json.loads((path / "results.json").read_text())
    rows = result["rows"]
    cfg = result["manifest"]["config"]
    channels = cfg["channels"].split(",")
    for channel in channels:
        methods = cfg["methods"].split(",")
        if channel == "SSPA" and "fiber_sinkhorn" in methods:
            methods = list(dict.fromkeys(methods + ["fiber_full"]))
        expected = {(m, rep, anchor, h) for m in methods for rep in range(cfg["repeats"])
                    for anchor in range(3 * cfg["anchor_directions"]) for h in [0.] + cfg["steps"]}
        actual = [(r["method"], r["repeat"], r["anchor"], r["step"]) for r in rows if r["channel"] == channel]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"Missing or duplicate metric rows: {path}, {channel}")
    return result


def task_contract(payload, config):
    """Extract task identity, deliberately excluding Monte Carlo budgets.

    Older results may lack split-reference diagnostics or a top-level step
    list; the latter can be recovered from the recorded method steps. Codec
    configuration does not certify the runner's power-normalization semantics:
    legacy files did not record that override, so this checks stored metadata
    only, not unstored protocol/source equivalence.
    """
    codec_sha = payload.get("codec", {}).get("sha256")
    if not codec_sha:
        raise ValueError("Missing codec checkpoint SHA in task contract")
    required = ("noise_std", "rate", "ebno_db", "codec_config")
    for name in required:
        if name not in payload or payload[name] is None:
            raise ValueError(f"Missing {name} in task contract")

    declared_steps = config.get("steps")
    steps = tuple(declared_steps) if declared_steps is not None else None
    for method, row in payload["methods"].items():
        method_steps = row.get("steps")
        if method_steps is None or any("fraction" not in step for step in method_steps):
            raise ValueError(f"Missing recorded step fractions for {method}")
        fractions = tuple(step["fraction"] for step in method_steps)
        if steps is None:
            steps = fractions
        elif fractions != steps:
            raise ValueError(f"Mismatched step fractions within task: {method}")
    if not steps:
        raise ValueError("Missing step fractions in task contract")

    return {"codec_sha256": codec_sha, "steps": steps,
            **{name: payload[name] for name in required}}


def check_task_contract(contracts, key, payload, config):
    """Reject incompatible tasks sharing a (seed, channel, codec-label) key."""
    candidate = task_contract(payload, config)
    previous = contracts.get(key)
    if previous is not None:
        for name in ("codec_sha256", "codec_config", "steps"):
            if candidate[name] != previous[name]:
                raise ValueError(f"Mismatched task contract {name}: {key}")
        for name in ("noise_std", "rate", "ebno_db"):
            if not math.isclose(candidate[name], previous[name], rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(f"Mismatched task contract {name}: {key}")
    else:
        contracts[key] = candidate
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-suite", type=Path, required=True)
    parser.add_argument("--metric-suites", type=Path, nargs="+", required=True)
    parser.add_argument("--gradient-suites", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    resolution = load_suite(args.resolution_suite)
    rows = resolution["rows"]
    references = {(r["channel"], r["method"], r["repeat"], r["anchor"]): r for r in rows if r["estimator"] == "pathwise"}
    convergence = []
    for channel in dict.fromkeys(r["channel"] for r in rows):
        for step in sorted({r["step"] for r in rows if r["step"]}):
            group = [r for r in rows if r["channel"] == channel and r["step"] == step]
            key = "rbf_embedding_derivative_op"
            relative = [abs(r[key] / references[channel, r["method"], r["repeat"], r["anchor"]][key] - 1) for r in group]
            convergence.append({"channel": channel, "step": step, "records": len(group),
                "median_relative_error": float(np.median(relative)), "max_relative_error": max(relative),
                "median_sample_jacobian_relative_error": float(np.median([r["q_jacobian_fd_relative_error"] for r in group])),
                "max_sample_jacobian_relative_error": max(r["q_jacobian_fd_relative_error"] for r in group)})
    save_csv(out / "finite_difference_resolution.csv", convergence)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
    for ax, channel in zip(axes, ("AWGN", "SSPA")):
        group = [r for r in convergence if r["channel"] == channel]
        for key, label, color in (("median_relative_error", "Median", "#237f79"), ("max_relative_error", "Maximum", "#bb4b54")):
            ax.loglog([r["step"] for r in group], [100*r[key] for r in group], "o-", color=color, label=label)
        ax.set(title=channel, xlabel="Input perturbation h", ylabel="Derivative norm error (%)")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(out / "derivative_resolution.png", dpi=180)
    fig.savefig(out / "derivative_resolution.pdf")
    plt.close(fig)

    scores, manifests = {}, {}
    for path in args.metric_suites:
        result = load_suite(path)
        manifest = result["manifest"]
        seed = manifest["config"]["seed"]
        for channel, method in dict.fromkeys((r["channel"], r["method"]) for r in result["rows"]):
            key = (seed, channel, method)
            if key in scores:
                raise ValueError(f"Duplicate metric candidate {key}")
            group = [r for r in result["rows"] if r["channel"] == channel and r["method"] == method and r["estimator"] == "pathwise"]
            score = {"seed": seed, "channel": channel, "method": method, "metric_source": str(path),
                     "metric_samples": manifest["config"]["samples"]}
            names = ["swd"] + [f"{component}_{stat}" for component in ("rbf", "moments", "augmented") for stat in ("mmd", "embedding_derivative_op")]
            for name in names:
                means = [np.mean([r[name] for r in group if r["repeat"] == rep]) for rep in sorted({r["repeat"] for r in group})]
                score[name] = float(np.mean(means))
                score[name + "_mc_se"] = float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None
            scores[key], manifests[key] = score, manifest
    save_csv(out / "metric_summary.csv", list(scores.values()))
    targets, task_contracts = {}, {}
    for path in args.gradient_suites:
        result = json.loads((path / "results.json").read_text())
        seed = result["config"]["seed"]
        for comparison, payload in result["comparisons"].items():
            channel, codec = comparison.split("/")
            contract = check_task_contract(task_contracts, (seed, channel, codec), payload, result["config"])
            for method, row in payload["methods"].items():
                if (seed, channel, method) not in scores:
                    continue
                manifest = manifests[seed, channel, method]
                if not np.isclose(payload["noise_std"], manifest["channels"][channel]["noise_std"], rtol=1e-6):
                    raise ValueError("Mismatched physical noise contract")
                if method != "analytic" and row["checkpoint"]["sha256"] != manifest["checkpoints"][f"{channel}/{method}"]["sha256"]:
                    raise ValueError("Mismatched generator checkpoint")
                target = scores[seed, channel, method] | {"codec": codec, "task_source": str(path),
                    "codec_sha256": contract["codec_sha256"],
                    "gradient_samples": result["config"]["gradient_samples"],
                    "reference_samples": result["config"]["reference_samples"],
                    "encoder_cosine": row["encoder_alignment"]["cosine"],
                    "encoder_norm_ratio": row["encoder_alignment"]["norm_ratio"],
                    "encoder_relative_error": row["encoder_alignment"]["relative_error"],
                    "task_conditional_swd": row["conditional_swd"],
                    "analytic_floor_encoder_cosine": payload["methods"]["analytic"]["encoder_alignment"]["cosine"],
                    "analytic_floor_encoder_relative_error": payload["methods"]["analytic"]["encoder_alignment"]["relative_error"],
                    "reference_split_encoder_cosine": payload.get("reference_encoder_split_alignment", {}).get("cosine"),
                    "first_step_ce_delta": row["steps"][0]["ce_delta"]["mean"],
                    "first_step_ce_delta_mc_se": row["steps"][0]["ce_delta"]["mc_se"]}
                key = (seed, channel, method, codec)
                if key not in targets or target["gradient_samples"] > targets[key]["gradient_samples"]:
                    targets[key] = target
    save_csv(out / "metric_task_comparison.csv", list(targets.values()))
    (out / "inputs.json").write_text(json.dumps({k: [str(p) for p in v] if isinstance(v, list) else str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    print(json.dumps({"metric_rows": len(scores), "task_comparisons": len(targets), "out_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
