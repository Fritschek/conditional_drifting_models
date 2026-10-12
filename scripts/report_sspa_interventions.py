#!/usr/bin/env python3
"""Audit the completed continuation matrix without selecting on its test panel."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from conditional_drifting.sspa_intervention import BRANCHES, digest, passes_gate
from conditional_drifting.sspa_trajectory import source_hashes
from run_sspa_interventions import equal_state, PARENT
from run_sspa_solver_replay import write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bounded(rows, parent, noise_std, ema=False):
    # Reuse the frozen numerical gate at explicitly specified later checkpoints.
    if len(rows) != 3:
        return False
    remapped = [dict(row, update=update) for row, update in zip(rows, (180000, 190000, 200000))]
    return passes_gate(remapped, parent, noise_std, ema)


def report(root):
    manifest = json.loads((root / "manifest.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    decision = json.loads((root / "extension_decision.json").read_text())
    require(not manifest["smoke"] and completion["complete"] and not completion["failures"], "Incomplete suite")
    require(manifest["seeds"] == [9001, 9002, 9003] and manifest["branches"] == list(BRANCHES), "Unexpected matrix")
    contract = manifest["contract"]
    require(contract["source_hashes"] == source_hashes(), "Original numerical sources changed")
    for name, sha in contract["child_sources"].items():
        require(digest(ROOT / name) == sha, f"Child source changed: {name}")
    require(digest(ROOT / "results/official_sources_20261010/W-Flow/drift_loss_ot.py") ==
            contract["official_loss_sha256"], "Official loss changed")
    expected_schedule = list(range(80000, 160001, 2000)) + [170000, 180000, 190000, 200000]
    rows, artifacts, gates, extended, references = [], {}, {}, [], []
    control_checks = 0
    for branch in BRANCHES:
        gates[branch] = {}
        for seed in manifest["seeds"]:
            folder = root / branch / f"seed{seed}"
            parent_path = PARENT / f"seed{seed}/update80000.pt"
            require(digest(parent_path) == manifest["parent_hashes"][str(seed)], "Parent changed")
            parent = torch.load(parent_path, map_location="cpu", weights_only=False)
            parent_row = next(r for r in parent["history"] if r["update"] == 80000)
            state200 = torch.load(folder / "update200000.pt", map_location="cpu", weights_only=False)
            require([r["update"] for r in state200["history"]] == expected_schedule, "Missing 200k checkpoint row")
            raw_gate = passes_gate(state200["history"], parent_row, parent["config"]["noise_std"])
            ema_gate = passes_gate(state200["history"], parent_row, parent["config"]["noise_std"], True)
            gates[branch][str(seed)] = dict(raw=raw_gate, ema=ema_gate)
            horizons = [200000] + ([390720] if branch in completion["extended"] else [])
            status = json.loads((folder / "status.json").read_text())
            require(status["state"] == "complete" and status["update"] == horizons[-1], "Task status mismatch")
            for horizon in horizons:
                path = folder / f"result{horizon}.json"
                result = json.loads(path.read_text())
                state = torch.load(folder / f"update{horizon}.pt", map_location="cpu", weights_only=False)
                require(result["horizon"] == horizon and result["branch"] == branch and result["seed"] == seed, "Result identity mismatch")
                require(state["contract"] == contract and state["update"] == horizon and state["branch"] == branch, "Checkpoint contract mismatch")
                require(state["parent"]["sha256"] == manifest["parent_hashes"][str(seed)], "Checkpoint parent mismatch")
                expected_config = dict(parent["config"])
                if branch == "log10_lr01":
                    expected_config["learning_rate"] = .0001
                elif branch == "log10_drift01":
                    expected_config["drift_scale"] = .1
                elif branch == "official64":
                    expected_config.update(batch_size=256, samples=64, learning_rate=.0002)
                require(state["config"] == expected_config, "Intervention configuration mismatch")
                group = state["optimizer"]["param_groups"][0]
                expected_betas = ((.9, .95) if branch == "official64" else
                                  (0., .999) if branch == "log10_beta0" else (.9, .999))
                require(tuple(group["betas"]) == expected_betas and group["lr"] == expected_config["learning_rate"],
                        "Optimizer intervention mismatch")
                require(group["weight_decay"] == (.01 if branch == "official64" else 0.), "Weight decay mismatch")
                expected_step = horizon - 80000 if branch == "official64" else horizon
                require(all(int(value["step"]) == expected_step for value in state["optimizer"]["state"].values()),
                        "Optimizer step history mismatch")
                require((state["ema"] is not None) == (branch == "official64"), "EMA scope mismatch")
                require(result["gate_raw"] == raw_gate and result["gate_ema"] == ema_gate, "Gate mismatch")
                require(result["counts"] == state["counts"], "Result/checkpoint count mismatch")
                for name, sha in result["checkpoint_sha256"].items():
                    require(digest(folder / name) == sha, f"Checkpoint changed: {folder / name}")
                expected = expected_schedule + ([250000, 300000, 350000, 390720] if horizon > 200000 else [])
                require([r["update"] for r in state["history"]] == expected, "Missing validation row")
                require([r["update"] for r in state["trace"]] == list(range(80100, horizon + 1, 100)), "Missing training trace")
                for key in ("oracle", "generated", "reference"):
                    count = "training_" + key + "_outputs"
                    require(state["counts"][count] - parent["counts"][count] == (horizon - 80000) * 16384, "Training exposure mismatch")
                require(state["counts"]["training_anchors"] - parent["counts"]["training_anchors"] ==
                        (horizon - 80000) * (256 if branch == "official64" else 4096), "Anchor count mismatch")
                if branch == "control":
                    for update in expected:
                        old_path = PARENT / f"seed{seed}/update{update}.pt"
                        if not old_path.exists():
                            continue
                        old = torch.load(old_path, map_location="cpu", weights_only=False)
                        child = torch.load(folder / f"update{update}.pt", map_location="cpu", weights_only=False)
                        for key in ("model", "optimizer", "rng"):
                            require(equal_state(child[key], old[key]), f"Control diverged at {seed}/{update}/{key}")
                        control_checks += 1
                for kind in ("raw", "ema") if branch == "official64" else ("raw",):
                    last = state["history"][-1]
                    validation = last if kind == "raw" else last["ema"]
                    independent = result["independent"][kind]
                    for values in (validation, independent):
                        require(all(math.isfinite(values[key]) for key in ("anchor_swd", "anchor_mean_l2", "conditional_variance")), "Nonfinite endpoint")
                    tail = [r for r in state["history"] if r["update"] in (300000, 350000, 390720)]
                    rows.append(dict(branch=branch, seed=seed, horizon=horizon, model=kind,
                        gate_200k=raw_gate if kind == "raw" else ema_gate,
                        late_three_bounded=bounded(tail, parent_row, parent["config"]["noise_std"], kind == "ema") if horizon > 200000 else None,
                        validation_swd=validation["anchor_swd"], validation_mean_l2=validation["anchor_mean_l2"],
                        validation_variance=validation["conditional_variance"],
                        independent_swd=independent["anchor_swd"], independent_swd_floor=independent["anchor_swd_floor"],
                        independent_mean_l2=independent["anchor_mean_l2"], independent_mean_floor=independent["anchor_mean_l2_floor"],
                        independent_variance=independent["conditional_variance"],
                        independent_gw2=independent["anchor_gw2"], independent_gw2_floor=independent["anchor_gw2_floor"],
                        selected_update=result["selected_update"], extension_train_seconds=result["train_seconds"],
                        validation_seconds=result["validation_seconds"]))
                artifacts[str(path.relative_to(root))] = digest(path)
            latest = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
            final_state = torch.load(folder / f"update{horizons[-1]}.pt", map_location="cpu", weights_only=False)
            require(equal_state(latest, final_state), "Latest checkpoint mismatch")
            for row in final_state["history"]:
                for ref in row["transport"].get("reference", []):
                    require(ref["converged"] or ref["barycenter_rms"] is None, "Unresolved reference treated as truth")
                    references.append(dict(branch=branch, seed=seed, update=row["update"], **ref))
            if branch != "official64":
                control = torch.load(root / "control" / f"seed{seed}/update200000.pt", map_location="cpu", weights_only=False)
                require(equal_state(state200["rng"], control["rng"]), "K4 random stream unpaired")
        if any(g["raw"] or g["ema"] for g in gates[branch].values()):
            extended.append(branch)
    require(gates == decision["gates"] and extended == completion["extended"] == decision["extended"], "Extension selection mismatch")
    return dict(rows=rows, gates=gates, extended=extended, control_checkpoint_checks=control_checks,
                references=references, unresolved_references=sum(not r["converged"] for r in references),
                artifacts=artifacts, scope="Three development seeds; Gaussian anchors; no downstream SER/BER or confirmatory selection claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = report(args.suite_dir)
    write_json(args.suite_dir / "audit.json", result)
    with (args.suite_dir / "endpoints.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({key: result[key] for key in ("extended", "control_checkpoint_checks", "unresolved_references")}, indent=2))


if __name__ == "__main__":
    main()
