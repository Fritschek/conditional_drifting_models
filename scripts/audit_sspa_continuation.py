#!/usr/bin/env python3
"""Check inherited evidence, final RNG pairing and cumulative continuation counts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from conditional_drifting.sspa_trajectory import source_hashes
from scripts.report_sspa_epsilon_trajectories import load_suite


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit(root, parent):
    manifest, results, histories, traces = load_suite(root)
    previous, parent_results, parent_histories, parent_traces = load_suite(parent)
    lineage = manifest["continuation"]
    if file_hash(parent / "manifest.json") != lineage["parent_manifest_sha256"]:
        raise ValueError("Parent manifest changed")
    if source_hashes() != manifest["source_hashes"]:
        raise ValueError("Training source changed")
    for relative, expected in lineage["checkpoint_sha256"].items():
        if file_hash(parent / relative) != expected or file_hash(root / relative) != expected:
            raise ValueError(f"Parent or copied checkpoint changed: {relative}")
    task_rows, paired, hashes = [], [], {}
    for seed in manifest["seeds"]:
        initial, final_rng = [], []
        for policy in manifest["policies"]:
            folder = root / policy / f"seed{seed}"
            key = seed, policy
            before, after = parent_results[key], results[key]
            history = histories[key]
            if history[:len(parent_histories[key])] != parent_histories[key]:
                raise ValueError("Inherited validation history changed")
            if traces[key][:len(parent_traces[key])] != parent_traces[key]:
                raise ValueError("Inherited training trace changed")
            state = torch.load(folder / f"update{manifest['updates']}.pt", weights_only=False, map_location="cpu")
            start = torch.load(folder / "update0.pt", weights_only=False, map_location="cpu")
            if (state["history"] != history or state["trace"] != traces[key] or
                    state["counts"] != after["counts"] or state["source_hashes"] != manifest["source_hashes"] or
                    state["update"] != manifest["updates"] or state["config"] != after["config"]):
                raise ValueError("Final checkpoint disagrees with saved records")
            initial.append(tensor_hash(start["model"].values()))
            final_rng.append(tensor_hash([state["rng"]["cpu"], *state["rng"]["cuda"]]))
            if after["fixed_epsilon"] != before["fixed_epsilon"]:
                raise ValueError("Calibration changed")
            for metric in ("anchor_swd_floor", "anchor_gw2_floor", "anchor_mean_l2_floor", "anchor_cov_fro_floor"):
                if any(row[metric] != history[0][metric] for row in history):
                    raise ValueError("Fixed analytic validation floor changed")
            for path in folder.glob("update*.pt"):
                hashes[str(path.relative_to(root))] = file_hash(path)
            row = dict(seed=seed, policy=policy, start_update=previous["updates"],
                final_update=manifest["updates"], selected_update=after["selected_update"],
                start_anchor_swd=parent_histories[key][-1]["anchor_swd"],
                final_anchor_swd=history[-1]["anchor_swd"],
                added_train_seconds=after["train_seconds"] - before["train_seconds"],
                added_validation_seconds=after["validation_seconds"] - before["validation_seconds"],
                added_counts={k: v - before["counts"][k] for k, v in after["counts"].items()})
            if min(row["added_train_seconds"], row["added_validation_seconds"], *row["added_counts"].values()) < 0:
                raise ValueError("Cumulative accounting decreased")
            task_rows.append(row)
        if len(set(initial)) != 1 or len(set(final_rng)) != 1:
            raise ValueError("Paired initialization or final training RNG differs")
        paired.append(dict(seed=seed, initial_model_sha256=initial[0], final_torch_rng_sha256=final_rng[0]))
    return dict(parent_manifest_sha256=lineage["parent_manifest_sha256"],
        inherited_checkpoints_verified=len(lineage["checkpoint_sha256"]),
        checkpoint_sha256=hashes, paired_seeds=paired, tasks=task_rows,
        added_train_seconds=sum(r["added_train_seconds"] for r in task_rows),
        added_validation_seconds=sum(r["added_validation_seconds"] for r in task_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--parent-suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.suite_dir, args.parent_suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"tasks", "checkpoint_sha256", "paired_seeds"}}, indent=2))


if __name__ == "__main__":
    main()
