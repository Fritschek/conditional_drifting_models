#!/usr/bin/env python3
"""Audit and summarize the frozen fresh-start SSPA experiment."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional-drifting-matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from conditional_drifting.sspa_intervention import digest
from conditional_drifting.sspa_trajectory import source_hashes
from run_sspa_interventions import equal_state
from run_sspa_overnight import VARIANTS, SCHEDULE, selected_update, late_gate
from run_sspa_solver_replay import write_json


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    require(summary["state"] == "complete", "Suite is incomplete")
    require(not summary["failures"], "Suite has failed tasks")
    require(summary["completed"] == summary["expected"], "Task count mismatch")
    base = manifest["contract"]
    require(base["historical_sources"] == source_hashes(), "Historical source changed")
    for name, value in base["sources"].items():
        require(digest(ROOT / name) == value, f"Source changed: {name}")
    smoke = base["smoke"]
    schedule = (0, 10, 20) if smoke else SCHEDULE
    gh, ah = manifest["generator_horizon"], manifest["codec_horizon"]
    count, generator_rows, codec_rows, gradient_rows = 0, [], [], []
    for seed in manifest["seeds"]:
        parent = folder / "parents" / f"seed{seed}.pt"
        initial = torch.load(parent, map_location="cpu", weights_only=False)
        require(initial["update"] == 0 and not initial["optimizer"]["state"], "Parent is trained")
        require(initial["config"]["seed"] == seed, "Parent seed mismatch")
        for variant in VARIANTS:
            key = f"{seed}/{variant}"
            result = summary["generators"][key]
            contract = dict(base, variant=variant, seed=seed, parent_sha256=digest(parent))
            require(result["contract"] == contract, "Generator contract mismatch")
            path = folder / f"generator/{variant}/seed{seed}"
            history = json.loads((path / "trajectory.json").read_text())
            require([r["update"] for r in history] == list(schedule), "Missing scheduled validation")
            require(result["checkpoints"]["selected"]["update"] == selected_update(history), "Wrong selection")
            require(result["horizon"] == gh, "Wrong generator horizon")
            if not smoke:
                require(result["late_three_bounded"] == late_gate(history), "Incorrect stability gate")
            for update in schedule:
                state = torch.load(path / f"update{update}.pt", map_location="cpu", weights_only=False)
                require(state["update"] == update and state["contract"] == contract, "Checkpoint contract mismatch")
                require(state["parent"]["sha256"] == digest(parent), "Parent hash mismatch")
                cfg = state["config"]
                require(cfg["samples"] == (4 if variant == "old_k4" else 64), "Wrong K")
                exposure = 64 if smoke else 16384
                require(cfg["batch_size"] * cfg["samples"] == exposure, "Output budget mismatch")
                require(state["counts"]["training_anchors"] == cfg["batch_size"] * update, "Anchor count mismatch")
                for category in ("oracle", "generated", "reference"):
                    require(state["counts"][f"training_{category}_outputs"] == exposure * update, "Output count mismatch")
                if update == 0:
                    require(equal_state(state["model"], initial["model"]), "Initial weights differ")
                    require(equal_state(state["rng"], initial["rng"]), "Initial RNG differs")
                    require(not state["optimizer"]["state"], "Initial optimizer is not empty")
                count += 1
            for stage, checkpoint in result["checkpoints"].items():
                require(digest(checkpoint["path"]) == checkpoint["sha256"], "Generator checkpoint hash mismatch")
                metrics = result["independent"][stage]["raw"]
                generator_rows.append(dict(seed=seed, variant=variant, stage=stage, update=checkpoint["update"],
                    swd=metrics["anchor_swd"], swd_floor=metrics["anchor_swd_floor"],
                    mean_l2=metrics["anchor_mean_l2"], variance=metrics["conditional_variance"],
                    late_three_bounded=result["late_three_bounded"], train_seconds=result["train_seconds"]))
        for offset in manifest["ae_seed_offsets"]:
            ae_seed = seed + offset
            labels_rng = None
            for method in ("analytic",) + tuple(f"{v}_{s}" for v in VARIANTS for s in ("selected", "final")):
                key = f"codec/seed{seed}/ae{ae_seed}/{method}"
                result = summary["codecs"][key]
                require(digest(result["checkpoint"]) == result["checkpoint_sha256"], "Codec hash mismatch")
                state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
                require(state["update"] == ah and state["seed"] == ae_seed, "Codec update/seed mismatch")
                require(state["contract"] == result["contract"], "Codec contract mismatch")
                require(result["train_messages"] == ah * 500, "Codec exposure mismatch")
                for update in ((5, 10) if smoke else (5000, 10000, 20000)):
                    saved = torch.load(folder / key / f"update{update}.pt", map_location="cpu", weights_only=False)
                    require(saved["update"] == update and saved["seed"] == ae_seed, "Intermediate codec identity mismatch")
                    require(saved["contract"] == result["contract"], "Intermediate codec contract mismatch")
                    require(saved["history"][-1] == next(r for r in result["validation"] if r["update"] == update),
                            "Intermediate codec validation mismatch")
                    require(all(int(s["step"].item()) == update for s in saved["optimizer"]["state"].values()),
                            "Codec optimizer step mismatch")
                    count += 1
                if labels_rng is None:
                    labels_rng = state["labels_rng"]
                require(torch.equal(labels_rng, state["labels_rng"]), "Unpaired label RNG state")
                if method != "analytic":
                    variant, stage = method.rsplit("_", 1)
                    generator = summary["generators"][f"{seed}/{variant}"]["checkpoints"][stage]
                    require(state["contract"]["generator"]["sha256"] == generator["sha256"], "Wrong implanted generator")
                require([r["ebno_db"] for r in result["tests"]] == [4., 6., 8.], "Missing SNR point")
                for row in result["tests"]:
                    require(row["trials"] == 64 * (16 if smoke else 32768), "Wrong trial count")
                    require(row["bit_trials"] == 6 * row["trials"], "Wrong bit trial count")
                    require(isinstance(row["errors"], int) and 0 <= row["errors"] <= row["trials"], "Invalid symbol count")
                    require(isinstance(row["bit_errors"], int) and 0 <= row["bit_errors"] <= row["bit_trials"], "Invalid bit count")
                    require(row["ser"] == row["errors"] / row["trials"], "SER mismatch")
                    require(row["ber"] == row["bit_errors"] / row["bit_trials"], "BER mismatch")
                    codec_rows.append(dict(seed=seed, ae_seed=ae_seed, method=method, **row))
        for update in ((5, 10) if smoke else (5000, 20000)):
            key = f"gradients/seed{seed}/update{update}"
            result = summary["gradients"][key]
            require(len(result["methods"]) == 7, "Missing gradient comparator")
            require(digest(result["codec"]) == result["contract"]["codec_sha256"], "Gradient codec hash mismatch")
            require(digest(folder / key / "vectors.pt") == result["vectors_sha256"], "Vector hash mismatch")
            vectors = torch.load(folder / key / "vectors.pt", map_location="cpu", weights_only=False)
            require(tuple(vectors["references"].shape) == (2 if smoke else 4, 64, 8), "Reference shape mismatch")
            for method, row in result["methods"].items():
                if method != "analytic":
                    require(digest(row["checkpoint"]) == result["contract"]["generators"][method], "Gradient generator hash mismatch")
                gradient_rows.append(dict(seed=seed, codec_update=update, method=method,
                    input_resolved=result["reference_input_resolved"], encoder_resolved=result["reference_encoder_resolved"],
                    input_cosine=row["input_alignment"]["cosine"], encoder_cosine=row["encoder_alignment"]["cosine"],
                    encoder_norm_ratio=row["encoder_alignment"]["norm_ratio"],
                    input_absolute_error=row["input_absolute_error"], encoder_absolute_error=row["encoder_absolute_error"],
                    swd=row["conditional_swd"], gw2=row["conditional_gw2"]))
    return manifest, summary, dict(checkpoints_checked=count, status="passed"), generator_rows, codec_rows, gradient_rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(folder, manifest, summary, codecs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(old_k4="#b34b35", old_k64="#167e68", official64="#3269ad", analytic="#222222")
    names = dict(old_k4="Production K4", old_k64="Production K64", official64="Official normalized K64", analytic="Analytic")
    seeds = manifest["seeds"]
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.3 * len(seeds), 3.4), squeeze=False, layout="constrained")
    for seed, ax in zip(seeds, axes[0]):
        for variant in VARIANTS:
            history = json.loads((folder / f"generator/{variant}/seed{seed}/trajectory.json").read_text())
            ax.semilogy([r["update"] / 1000 for r in history], [r["anchor_swd"] for r in history],
                        color=colors[variant], label=names[variant])
            u = summary["generators"][f"{seed}/{variant}"]["checkpoints"]["selected"]["update"]
            row = next(r for r in history if r["update"] == u)
            ax.plot(u / 1000, row["anchor_swd"], "o", color=colors[variant], markersize=4)
        ax.axhline(history[-1]["anchor_swd_floor"], color="#777777", linestyle=":", label="Analytic sampling floor")
        ax.set(title=f"Fresh seed {seed}", xlabel="Generator updates (thousands)", ylabel="Validation conditional SWD")
        ax.grid(alpha=.15)
    axes[0, -1].legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"fresh_trajectories.{extension}", dpi=160, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    for metric in ("ser", "ber"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        for stage, ax in zip(("selected", "final"), axes):
            for variant in ("analytic",) + VARIANTS:
                method = "analytic" if variant == "analytic" else f"{variant}_{stage}"
                values = [[np.mean([r[metric] for r in codecs if r["method"] == method and r["seed"] == seed and r["ebno_db"] == snr])
                           for snr in (4., 6., 8.)] for seed in seeds]
                for row in values:
                    ax.semilogy([4, 6, 8], row, color=colors[variant], alpha=.22, linewidth=.8)
                ax.semilogy([4, 6, 8], np.mean(values, axis=0), "o-", color=colors[variant], label=names[variant], markersize=4)
            ax.set(title=f"{stage.capitalize()} generator; final AE", xlabel="Analytic evaluation Eb/N0 (dB)", ylabel=metric.upper(), xticks=[4, 6, 8])
            ax.grid(alpha=.15)
        axes[-1].legend(fontsize=8)
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"fresh_{metric}.{extension}", dpi=160, bbox_inches="tight", pad_inches=.1)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.suite_dir.resolve()
    manifest, summary, checks, generators, codecs, gradients = audit(folder)
    for name, rows in (("generator_endpoints", generators), ("codec_tests", codecs), ("gradient_comparisons", gradients)):
        write_csv(folder / f"{name}.csv", rows)
    write_json(folder / "audit.json", checks | {"report_sha256": digest(__file__)})
    figures(folder, manifest, summary, codecs)
    lines = ["# Fresh-start SSPA comparison", "", "Smoke run: " + str(manifest["contract"]["smoke"]), "",
        "Identical random initialization per generator seed. Same generated/positive/reference output exposure per update.",
        "Selected means minimum raw validation conditional SWD over the frozen schedule; final means full horizon.",
        "Independent conditional test panel below was not used for selection. EMA is secondary and excluded here.", "",
        "| Seed | Method | Stage | Updates | Conditional SWD | Analytic floor | Variance | Late gate |",
        "|---|---|---|---:|---:|---:|---:|---|"]
    for row in generators:
        lines.append(f"| {row['seed']} | {row['variant']} | {row['stage']} | {row['update']} | {row['swd']:.6g} | {row['swd_floor']:.6g} | {row['variance']:.6g} | {row['late_three_bounded']} |")
    lines += ["", "## Downstream analytic-channel evaluation", "",
        "All AEs use the same full-codebook power constraint, 20000 updates (10 in smoke), and no checkpoint selection.",
        "Two paired AE initializations per generator seed are averaged first. SD below is across the three generator-seed blocks,",
        "not across six independent generators. Raw error counts, trial counts, CE and all SNR points are in codec_tests.csv.",
        "These are development comparisons, with no cross-family superiority or statistical-significance claim.", "",
        "| Method | SER at 8 dB, mean +/- block SD | BER at 8 dB, mean +/- block SD |",
        "|---|---:|---:|"]
    for method in dict.fromkeys(r["method"] for r in codecs):
        entries = []
        for metric in ("ser", "ber"):
            values = [np.mean([r[metric] for r in codecs if r["seed"] == seed and r["method"] == method and r["ebno_db"] == 8.]) for seed in manifest["seeds"]]
            sd = np.std(values, ddof=1) if len(values) > 1 else 0.
            entries.append(f"{np.mean(values):.6g} +/- {sd:.3g}")
        lines.append(f"| {method} | {' | '.join(entries)} |")
    lines += ["", "Off-nominal SNR points test receiver robustness, not off-nominal generator fidelity.", "",
        "## Frozen-codec gradients", "", "Each row uses one common analytic-trained codec. Relative measures are blank when the analytic reference is unresolved.",
        "Four independent Monte Carlo replicates estimate each gradient; these are not additional training seeds.", "",
        "| Seed | AE updates | Method | Encoder cosine | Encoder norm ratio | Absolute encoder error | Conditional SWD |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for row in gradients:
        cosine = "unresolved" if row["encoder_cosine"] is None else f"{row['encoder_cosine']:.4f}"
        ratio = "unresolved" if row["encoder_norm_ratio"] is None else f"{row['encoder_norm_ratio']:.4g}"
        lines.append(f"| {row['seed']} | {row['codec_update']} | {row['method']} | {cosine} | {ratio} | {row['encoder_absolute_error']:.5g} | {row['swd']:.5g} |")
    lines += ["", f"Audit passed: {checks['checkpoints_checked']} generator/codec checkpoints checked, plus gradient vector hashes.",
        "Parent/source contracts, exact output counts, checkpoint selection, paired label RNG and integer BER/SER identities checked.",
        "Historical journal AE normalization and selection differed; historical numbers are not matched comparison rows.", ""]
    (folder / "report.md").write_text("\n".join(lines))
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
