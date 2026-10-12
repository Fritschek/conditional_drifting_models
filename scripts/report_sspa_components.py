#!/usr/bin/env python3
"""Audit and plot the completed leave-one-component-out SSPA experiment."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional-drifting-matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from conditional_drifting.sspa_intervention import digest
from conditional_drifting.sspa_trajectory import source_hashes
from report_sspa_interventions import bounded
from run_sspa_solver_replay import write_json

LABELS = {"official64": "Full successful recipe", "k4": "K4 instead of K64",
    "raw_loss64": "No loss normalization", "fixed_epsilon64": "Old physical epsilon",
    "old_optimizer64": "Old optimizer package", "production_k64": "Old trainer, only K/B changed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def report(root, bridge=None):
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    contract = manifest["contract"]
    require(completion["complete"] and not manifest["smoke"] and manifest["horizon"] == 390720, "Incomplete or smoke suite")
    require(contract["historical_sources"] == source_hashes(), "Historical source changed")
    require(digest(ROOT / "results/official_sources_20261010/W-Flow/drift_loss_ot.py") ==
            contract["official_loss_sha256"], "Official loss changed")
    for path, expected in contract["sources"].items():
        require(digest(ROOT / path) == expected, f"Changed source: {path}")
    replay = json.loads((root / "reference_replay.json").read_text())
    require(all(replay[k] for k in ("model", "optimizer", "ema", "rng")), "Failed reference replay")
    reference = ROOT / "results/sspa_interventions_20261010/official64/seed9001"
    require(digest(reference / "update82000.pt") == replay["reference_sha256"], "Changed replay reference")
    reference_result = json.loads((reference / "result390720.json").read_text())
    parent_state = torch.load(reference_result["parent"]["path"], map_location="cpu", weights_only=False)
    parent_row = next(row for row in parent_state["history"] if row["update"] == 80000)
    require(digest(reference_result["parent"]["path"]) == contract["parent_sha256"], "Changed parent")
    series = {"official64": json.loads((reference / "trajectory.json").read_text())}
    records = [("official64", reference_result)]
    for variant in manifest["variants"]:
        folder = root / variant
        result = json.loads((folder / "result.json").read_text())
        require(result["variant"] == variant and result["seed"] == 9001 and result["horizon"] == 390720, "Wrong task identity")
        rows = json.loads((folder / "trajectory.json").read_text())
        series[variant] = rows
        expected_updates = list(range(80000, 200001, 10000)) + [250000, 300000, 350000, 390720]
        require([r["update"] for r in rows] == expected_updates, "Wrong validation schedule")
        state = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
        require(state["contract"] == contract | {"component": variant}, "Changed component contract")
        require(state["update"] == 390720, "Incomplete checkpoint")
        require(state["parent"] == reference_result["parent"], "Unpaired parent")
        require(state["config"]["samples"] == (4 if variant == "k4" else 64), "Wrong K")
        require(state["config"]["batch_size"] == (4096 if variant == "k4" else 256), "Wrong anchor count")
        group = state["optimizer"]["param_groups"][0]
        require(group["betas"] == ((.9, .999) if variant == "old_optimizer64" else (.9, .95)), "Wrong optimizer betas")
        require(group["lr"] == (.001 if variant == "old_optimizer64" else .0002), "Wrong learning rate")
        require(group["weight_decay"] == (0 if variant == "old_optimizer64" else .01), "Wrong weight decay")
        expected_step = 390720 if variant == "old_optimizer64" else 310720
        require(all(v["step"].item() == expected_step for v in state["optimizer"]["state"].values()),
                "Wrong optimizer step count")
        for name in ("oracle", "generated", "reference"):
            key = "training_" + name + "_outputs"
            require(state["counts"][key] - state["parent"]["counts"][key] == 310720 * 16384, "Wrong output exposure")
        require(state["counts"]["training_anchors"] - state["parent"]["counts"]["training_anchors"] ==
                310720 * state["config"]["batch_size"], "Wrong input exposure")
        trace = json.loads((folder / "trace.json").read_text())
        require([r["update"] for r in trace] == list(range(80100, 390701, 100)), "Wrong trace schedule")
        if variant == "fixed_epsilon64":
            require(all(abs(r["physical_epsilon"] - state["fixed_epsilon"]) < 1e-7 for r in trace),
                    "Fixed-epsilon branch changed physical regularization")
        for path, expected in result["checkpoint_sha256"].items():
            require(digest(folder / path) == expected, f"Changed checkpoint {variant}/{path}")
        require(result["late_three_bounded"] == bounded(rows[-3:], parent_row, state["config"]["noise_std"]), "Wrong late gate")
        records.append((variant, result))
    if bridge is not None:
        bm = json.loads((bridge / "manifest.json").read_text())
        result = json.loads((bridge / "result.json").read_text())
        state = torch.load(bridge / "latest.pt", map_location="cpu", weights_only=False)
        bc = bm["contract"]
        require(bc == state["contract"] and bc["historical_sources"] == source_hashes(), "Bridge source/contract mismatch")
        require(bc["parent_sha256"] == contract["parent_sha256"] and state["parent"] == reference_result["parent"], "Bridge parent mismatch")
        for path, expected in bc["sources"].items():
            require(digest(ROOT / path) == expected, f"Changed bridge source {path}")
        require(state["config"] == parent_state["config"] | {"samples": 64, "batch_size": 256}, "Bridge changed more than K/B")
        require(state["branch"] == "control" and state["ema"] is None and state["update"] == 390720, "Wrong bridge trainer")
        group = state["optimizer"]["param_groups"][0]
        require(group == parent_state["optimizer"]["param_groups"][0], "Bridge changed optimizer settings")
        require(all(v["step"].item() == 390720 for v in state["optimizer"]["state"].values()), "Bridge optimizer step mismatch")
        for name in ("oracle", "generated", "reference"):
            key = "training_" + name + "_outputs"
            require(state["counts"][key] - parent_state["counts"][key] == 310720 * 16384, "Bridge exposure mismatch")
        require(state["counts"]["training_anchors"] - parent_state["counts"]["training_anchors"] == 310720 * 256, "Bridge input count mismatch")
        for path, expected in result["checkpoint_sha256"].items():
            require(digest(bridge / path) == expected, f"Changed bridge checkpoint {path}")
        series["production_k64"] = json.loads((bridge / "trajectory.json").read_text())
        require([r["update"] for r in series["production_k64"]] == expected_updates, "Bridge validation schedule mismatch")
        require(result["late_three_bounded"] == bounded(series["production_k64"][-3:], parent_row, state["config"]["noise_std"]), "Bridge gate mismatch")
        records.append(("production_k64", result))
    endpoints = []
    for variant, result in records:
        late = [r for r in series[variant] if r["update"] in (300000, 350000, 390720)]
        for model in ("raw", "ema"):
            if model not in result["independent"]:
                continue
            values = result["independent"][model]
            endpoints.append(dict(variant=variant, model=model,
                late_three_bounded=bounded(late, parent_row, parent_state["config"]["noise_std"], model == "ema"),
                swd=values["anchor_swd"], swd_floor=values["anchor_swd_floor"],
                mean_l2=values["anchor_mean_l2"], variance=values["conditional_variance"],
                gw2=values["anchor_gw2"], train_seconds=result["train_seconds"]))
    with (root / "endpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    audit = dict(complete=True, checkpoints_verified=sum(len(r["checkpoint_sha256"]) for _, r in records[1:]),
        new_updates=310720 * (4 + (bridge is not None)), endpoints=endpoints,
        limitation="One deliberately selected development seed; warm-start component deletions, not independent confirmation")
    if bridge is not None:
        old_path = ROOT / "results/sspa_epsilon_trajectories_full_20261010/fixed_common/seed9001/trajectory.json"
        old_rows = [r for r in json.loads(old_path.read_text()) if r["update"] >= 80000]
        require(old_rows[0]["anchor_swd"] == parent_row["anchor_swd"] and old_rows[-1]["update"] == 390720,
                "Historical K4 trajectory mismatch")
        audit["historical_full_trajectory_sha256"] = digest(old_path)
    write_json(root / "audit.json", audit)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ("#333333", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9")
    for (variant, rows), color in zip(series.items(), colors):
        for ax, key in zip(axes, ("anchor_swd", "conditional_variance")):
            ax.plot([r["update"] / 1000 for r in rows], [r[key] for r in rows],
                label=LABELS[variant], color=color, lw=1.5)
    for ax, key, title in zip(axes, ("anchor_swd", "conditional_variance"),
                              ("Conditional SWD", "Conditional variance")):
        floor = series["official64"][0]["anchor_swd_floor"] if key == "anchor_swd" else parent_state["config"]["noise_std"] ** 2 / 2
        ax.axhline(floor, ls=":", color="0.5", lw=1)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Total updates (thousands)")
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle("SSPA recipe components: seed 9001, raw models")
    fig.tight_layout(rect=(0, .15, 1, .96))
    for extension in ("pdf", "png"):
        fig.savefig(root / f"component_trajectories.{extension}", dpi=160)
    plt.close(fig)
    if bridge is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for rows, color, label in ((old_rows, "#d55e00", "K4: 4096 inputs x 4 samples"),
                (series["production_k64"], "#0072b2", "K64: 256 inputs x 64 samples")):
            for ax, key in zip(axes, ("anchor_swd", "conditional_variance")):
                ax.plot([r["update"] / 1000 for r in rows], [r[key] for r in rows], label=label, color=color)
        for ax, key, title in zip(axes, ("anchor_swd", "conditional_variance"), ("Conditional SWD", "Conditional variance")):
            floor = old_rows[0]["anchor_swd_floor"] if key == "anchor_swd" else parent_state["config"]["noise_std"] ** 2 / 2
            ax.axhline(floor, ls=":", color="0.5", label="Analytic floor / variance")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Total updates (thousands)")
            ax.grid(axis="y", alpha=.15)
        fig.suptitle("Old SSPA trainer: only particle grouping changed (seed 9001)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, .96))
        for extension in ("pdf", "png"):
            fig.savefig(root / f"particle_grouping_only.{extension}", dpi=160)
        plt.close(fig)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--bridge-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.suite_dir, args.bridge_dir), indent=2))
