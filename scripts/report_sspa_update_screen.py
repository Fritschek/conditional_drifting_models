#!/usr/bin/env python3
"""Recompute and audit the repeated-update screen from per-trial records."""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.sspa_trajectory import source_hashes
from conditional_drifting.sspa_update_screen import METHODS
from run_sspa_update_screen import sha, summary


def close(left, right):
    torch.testing.assert_close(torch.tensor(left, dtype=torch.float64), torch.tensor(right, dtype=torch.float64),
                               atol=1e-12, rtol=1e-10)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["smoke"] or (root / "failure.json").exists():
        raise ValueError("Require completed non-smoke screen")
    if manifest["seeds"] != [9001, 9002, 9003] or manifest["repeats"] != 32:
        raise ValueError("Unexpected screen panel")
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("Source mismatch")
    for name, digest in manifest["additional_source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Changed source: {name}")
    rows, digests, cases = [], {}, []
    for seed in manifest["seeds"]:
        folder = root / f"seed{seed}"
        result = json.loads((folder / "result.json").read_text())
        if sha(Path(result["parent"])) != result["parent_sha256"] or sha(folder / "updates.pt") != result["updates_sha256"]:
            raise ValueError("Changed checkpoint or update tensors")
        tensor = torch.load(folder / "updates.pt", map_location="cpu", weights_only=False)
        grouped = {name: [] for name, *_ in METHODS}
        for repeat in range(32):
            path = folder / f"repeat{repeat:02d}.json"
            trial = json.loads(path.read_text())
            digests[str(path)] = sha(path)
            if len(trial) != len(METHODS) or {r["method"] for r in trial} != set(grouped):
                raise ValueError("Incomplete or duplicate methods")
            for row in trial:
                if row["repeat"] != repeat or row["batch_seed"] != 810000 + repeat:
                    raise ValueError("Trial identity mismatch")
                grouped[row["method"]].append(row)
        for name, batch, particles, _ in METHODS:
            trial = grouped[name]
            saved = result["methods"][name]
            if any(r["anchors"] != batch or r["samples"] != particles or r["outputs_per_cloud"] != batch * particles for r in trial):
                raise ValueError("Exposure mismatch")
            for key, values in saved.items():
                if key in ("gradients", "steps"):
                    data = tensor["methods"][name][key].double()
                    if len(data) != 32 or not torch.isfinite(data).all():
                        raise ValueError("Invalid tensor batch")
                    mean = data.mean(0)
                    close(mean.norm().item(), values["mean_vector_norm"])
                    close((data - mean).square().sum(-1).mean().sqrt().item(), values["noise_rms"])
                else:
                    data = ([r["variance_change"] - b["variance_change"] for r, b in zip(trial, grouped["log4"])]
                            if key == "paired_variance_change_vs_log4" else [r[key] for r in trial])
                    computed = summary(data)
                    for stat in values:
                        close(computed[stat], values[stat])
            for index, row in enumerate(trial):
                gradient = tensor["methods"][name]["gradients"][index].double()
                step = tensor["methods"][name]["steps"][index].double()
                vg = tensor["variance_gradient"]
                close(torch.dot(vg, step).item(), row["variance_linear_step"])
                close(-torch.dot(vg, gradient).item(), row["variance_negative_gradient_dot"])
                close(row["variance_linear_step"] + row["variance_remainder"], row["variance_change"])
                close(row["particle_variance"]["first"] + row["particle_variance"]["second"], row["particle_variance"]["total"])
                if not torch.isfinite(gradient).all() or not torch.isfinite(step).all():
                    raise ValueError("Nonfinite gradient/step")
            flat = dict(seed=seed, method=name, anchors=batch, samples=particles,
                        before_variance=result["before_variance"], true_variance=result["true_variance"])
            for key, values in saved.items():
                flat.update({key + "_" + stat: val for stat, val in values.items()})
            rows.append(flat)
        for filename in ("result.json", "updates.pt"):
            digests[str(folder / filename)] = sha(folder / filename)
        cases.append(dict(seed=seed, parent_sha256=result["parent_sha256"],
                          elapsed_seconds=result["elapsed_seconds"], before_variance=result["before_variance"],
                          before_mean_mse=result["before_mean_mse"], master_outputs_per_cloud=result["master_outputs_per_cloud"]))
    return dict(cases=cases, rows=rows, artifacts=digests, trial_records=32 * 3 * len(METHODS),
                scope="Conditional batch uncertainty for three fixed checkpoints, not a long-run training comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(args.suite_dir)
    (args.suite_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.suite_dir / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    print(json.dumps(dict(checkpoints=len(report["cases"]), method_summaries=len(report["rows"]),
                         trial_records=report["trial_records"],
                         elapsed_seconds=sum(c["elapsed_seconds"] for c in report["cases"])), indent=2))


if __name__ == "__main__":
    main()
