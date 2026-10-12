#!/usr/bin/env python3
"""Audit and summarize completed frozen-cloud solver replays."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.sspa_solver_replay import residuals, variance_terms
from conditional_drifting.sspa_trajectory import source_hashes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(suite):
    manifest = json.loads((suite / "manifest.json").read_text())
    if manifest["smoke"] or (suite / "failure.json").exists():
        raise ValueError("A complete, non-smoke replay is required")
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("Numerical source mismatch")
    for path, expected in manifest["added_source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Replay source mismatch: {path}")
    expected_methods = {"production10", "production30", "production100", "production10_exact_row_readout"}
    expected_methods |= {f"log_floor_{dtype}_{steps}" for dtype in ("f32", "f64") for steps in (10, 30, 100)}
    expected_methods |= {f"log_floor_stabilized_{dtype}_10" for dtype in ("f32", "f64")}
    expected_methods |= {"reference_" + kind for kind in
                         ("floor_quantized", "floor_cost32", "unfloored_cost32", "unfloored_cost64")}
    rows, cases, artifacts = [], [], {}
    for policy, updates in manifest["selections"].items():
        for update in updates:
            folder = suite / f"{policy}_u{update}"
            result = json.loads((folder / "result.json").read_text())
            clouds = torch.load(folder / "clouds.pt", map_location="cpu", weights_only=False)
            solves = torch.load(folder / "solves.pt", map_location="cpu", weights_only=False)
            parent = Path(result["parent"])
            if sha(parent) != result["parent_sha256"] or clouds["parent_sha256"] != result["parent_sha256"]:
                raise ValueError(f"Parent hash mismatch: {folder}")
            if (result["config"]["policy"] != policy or result["update"] != update
                    or result["config"]["seed"] != manifest["seed"]):
                raise ValueError(f"Case identity mismatch: {folder}")
            if result["config"] != clouds["config"] or clouds["update"] != update:
                raise ValueError(f"Cloud identity mismatch: {folder}")
            if clouds["batch_label"] != "post_checkpoint_reconstructed_batch" or len(clouds["generated"]) != 4096:
                raise ValueError(f"Unexpected cloud batch: {folder}")
            if set(result["methods"]) != expected_methods or set(solves) != expected_methods:
                raise ValueError(f"Missing or extra methods: {folder}")
            for name, record in result["methods"].items():
                saved = solves[name]
                valid = saved["valid"]
                if int(valid.sum()) != record["valid_anchors"]:
                    raise ValueError(f"Mask/count mismatch: {folder}/{name}")
                for term in ("cross", "self"):
                    solve = saved[term]
                    torch.testing.assert_close(residuals(solve["plan"]), solve["residuals"])
                    if record["reference_solve"]:
                        passed = solve["residuals"].amax(-1) <= manifest["reference_tolerance"]
                        if not torch.equal(passed, solve["converged"]):
                            raise ValueError("Reference residual/convergence mismatch")
                        if not (solve["iterations"][~passed] == manifest["reference_cap"]).all():
                            raise ValueError("Unresolved solve did not reach cap")
                if record["reference_solve"] and not torch.equal(valid, saved["cross"]["converged"] & saved["self"]["converged"]):
                    raise ValueError("Invalid joint reference mask")
                for label in ("raw", "clipped"):
                    terms = variance_terms(clouds["generated"], saved[label + "_field"], result["config"]["drift_scale"])
                    torch.testing.assert_close(terms["actual"], terms["total"], atol=1e-10, rtol=1e-8)
                    for key, value in terms.items():
                        if valid.any():
                            torch.testing.assert_close(value[valid].mean(), torch.tensor(
                                record[label + "_particle_variance"][key]["mean"], dtype=torch.float64))
                neural = record["neural_update"]
                if bool(valid.all()) != (neural is not None):
                    raise ValueError("Neural update computed on unresolved reference")
                if neural is not None:
                    delta = saved["eval_after"]["variance"] - saved["eval_before"]["variance"]
                    torch.testing.assert_close(delta.mean(), torch.tensor(neural["variance_change"]["mean"], dtype=torch.float64))
                rows.append(dict(policy=policy, update=update, method=name,
                    valid_anchors=record["valid_anchors"], anchors=record["anchors"],
                    cross_epsilon=result["cross_epsilon"], self_epsilon=result["self_epsilon"],
                    cross_floor_fraction=result["floor_fraction"]["cross"]["mean"],
                    self_floor_fraction=result["floor_fraction"]["self"]["mean"],
                    max_cross_residual=max(v["max"] for v in record["cross_residuals"].values()),
                    max_self_residual=max(v["max"] for v in record["self_residuals"].values()),
                    field_rms=record["field_rms_vs_production"].get("mean"),
                    particle_first=record["clipped_particle_variance"]["first"].get("mean"),
                    particle_second=record["clipped_particle_variance"]["second"].get("mean"),
                    particle_total=record["clipped_particle_variance"]["total"].get("mean"),
                    gradient_relative_difference=neural["gradient_relative_difference"] if neural else None,
                    step_relative_difference=neural["step_relative_difference"] if neural else None,
                    before_variance=neural["before_variance"] if neural else None,
                    neural_variance_change=neural["variance_change"]["mean"] if neural else None))
            cases.append(dict(case=folder.name, elapsed_seconds=result["elapsed_seconds"],
                              parent_sha256=result["parent_sha256"], comparisons=result["comparisons"]))
            for filename in ("clouds.pt", "solves.pt", "result.json"):
                artifacts[str(folder / filename)] = sha(folder / filename)
    return dict(manifest=manifest, rows=rows, cases=cases, artifacts=artifacts,
                scope=f"Seed {manifest['seed']}; one full reconstructed batch per checkpoint; no new training trajectory or SER/BER")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = report(args.suite_dir)
    (args.suite_dir / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.suite_dir / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps(dict(cases=len(result["cases"]), methods=len(result["rows"]),
        skipped_reference_updates=sum(row["valid_anchors"] != row["anchors"] for row in result["rows"]),
        elapsed_seconds=sum(case["elapsed_seconds"] for case in result["cases"])), indent=2))


if __name__ == "__main__":
    main()
