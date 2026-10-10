#!/usr/bin/env python3
"""Extract historical SSPA epoch histories; these are not fidelity trajectories."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = sorted((args.suite_dir / "fiber_sinkhorn").glob("seed*/checkpoints/enhanced_direct_sspa_seed*.pt"))
    records, flat, crossings = [], [], []
    for path in paths:
        state = torch.load(path, weights_only=False, map_location="cpu")
        cfg, history, seed = state["config"], state["history"], int(state["seed"])
        steps = math.ceil(cfg["dataset_size"] / cfg["batch_size"])
        if [r["epoch"] for r in history] != list(range(1, cfg["epochs"] + 1)):
            raise ValueError(f"Missing or duplicate epoch history: {path}")
        records.append(dict(seed=seed, checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            config=cfg, steps_per_epoch=steps, history=history))
        flat.extend(dict(seed=seed, end_update=int(r["epoch"]) * steps, **r) for r in history)
        # Post-hoc descriptive threshold after the initialization epoch. It is
        # neither a prespecified stopping criterion nor an observed SWD failure.
        crossing = next((int(r["epoch"]) * steps for r in history if r["epoch"] >= 3 and r["loss"] > .05), None)
        crossings.append(dict(seed=seed, first_epoch_end_loss_above_0p05_after_epoch2=crossing))
    seeds = sorted(r["seed"] for r in records)
    if seeds != list(range(7, 107)):
        raise ValueError(f"Expected all 100 historical seeds 7..106, got {seeds}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "histories.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    with (args.out_dir / "per_epoch.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    values = [r["first_epoch_end_loss_above_0p05_after_epoch2"] for r in crossings]
    resolved = [v for v in values if v is not None]
    summary = dict(seeds=len(records), epochs=len(flat), descriptive_loss_threshold=.05,
        threshold_excludes_first_two_epochs=True, crossing_is_not_fidelity_failure=True,
        crossing_count=len(resolved), min_crossing=min(resolved) if resolved else None,
        median_crossing=statistics.median(resolved) if resolved else None,
        max_crossing=max(resolved) if resolved else None, per_seed_crossings=crossings,
        missing_epsilon_config_fields=sorted({k for r in records for k in
            ("sinkhorn_epsilon_mode", "sinkhorn_epsilon_samples", "sinkhorn_epsilon_scale") if k not in r["config"]}))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for record in records:
        x = [r["epoch"] * record["steps_per_epoch"] for r in record["history"]]
        for ax, metric in zip(axes, ("loss", "drift_norm")):
            ax.plot(x, [r[metric] for r in record["history"]], color="#0072B2", alpha=.12, linewidth=.6)
    for ax, label in zip(axes, ("Epoch-mean detached regression loss", "Epoch-mean clipped drift norm")):
        ax.set_xlabel("Optimizer updates at epoch end")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("Historical full-budget SSPA: 100 saved training histories")
    fig.savefig(args.out_dir / "historical_training.pdf")
    fig.savefig(args.out_dir / "historical_training.png", dpi=150)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed_crossings"}, indent=2))


if __name__ == "__main__":
    main()
