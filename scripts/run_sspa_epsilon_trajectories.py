#!/usr/bin/env python3
"""Run bounded continuous SSPA development trajectories, with exact resume."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from conditional_drifting.sspa_trajectory import POLICIES, SSPATrajectory, TrajectoryConfig, source_hashes
from scripts.report_sspa_epsilon_trajectories import load_suite


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def continuation_plan(parent, manifest):
    """Validate the complete parent before creating any extended task."""
    previous, results, histories, traces = load_suite(parent)
    for key in ("seeds", "policies", "configs", "source_hashes", "torch_version", "device", "gpu"):
        if previous[key] != manifest[key]:
            raise ValueError(f"Continuation {key} mismatch")
    start = previous["updates"]
    if manifest["updates"] <= start:
        raise ValueError("Continuation must increase the update budget")
    if [u for u in manifest["checkpoints"] if u <= start] != previous["checkpoints"]:
        raise ValueError("Continuation cannot change earlier validation checkpoints")
    hashes = {}
    for seed in previous["seeds"]:
        for policy in previous["policies"]:
            prefix = Path(policy) / f"seed{seed}"
            for update in previous["checkpoints"]:
                path = parent / prefix / f"update{update}.pt"
                hashes[str(prefix / path.name)] = file_hash(path)
            state = torch.load(parent / prefix / f"update{start}.pt", map_location="cpu", weights_only=False)
            result = results[seed, policy]
            if (state["update"] != start or state["config"] != result["config"] or
                    state["source_hashes"] != manifest["source_hashes"] or
                    state["device"] != manifest["device"] or state["torch_version"] != manifest["torch_version"] or
                    state["counts"] != result["counts"] or state["history"] != histories[seed, policy] or
                    state["trace"] != traces[seed, policy]):
                raise ValueError(f"Parent checkpoint/record mismatch: {prefix}")
    return dict(parent_suite=str(parent.resolve()), parent_manifest_sha256=file_hash(parent / "manifest.json"),
                start_update=start, checkpoint_sha256=hashes,
                runner_sha256=file_hash(Path(__file__)))


def initialize_continuation(folder, policy, seed, provenance, trainer):
    parent = Path(provenance["parent_suite"])
    prefix = Path(policy) / f"seed{seed}"
    source = parent / prefix / f"update{provenance['start_update']}.pt"
    # Check before loading and again while copying; never replace parent files.
    for relative, expected in provenance["checkpoint_sha256"].items():
        if Path(relative).parent == prefix and file_hash(parent / relative) != expected:
            raise ValueError(f"Parent checkpoint hash mismatch: {relative}")
    trainer.load(source)
    for relative, expected in provenance["checkpoint_sha256"].items():
        if Path(relative).parent != prefix:
            continue
        destination = folder / Path(relative).name
        if destination.exists() and file_hash(destination) != expected:
            raise ValueError(f"Destination checkpoint conflict: {destination}")
        shutil.copy2(parent / relative, destination)
        if file_hash(destination) != expected:
            raise ValueError(f"Copied checkpoint hash mismatch: {destination}")
    shutil.copy2(source, folder / "latest.pt")
    write_json(folder / "trajectory.json", trainer.history)
    write_json(folder / "training_trace.json", trainer.trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", default="9001,9002,9003")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--updates", type=int, default=4800)
    parser.add_argument("--checkpoints", default="0,100,300,1000,2500,4800")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true",
                        help="Record failed tasks and attempt the rest; exit nonzero if any fail")
    parser.add_argument("--continue-from", type=Path,
                        help="Complete parent suite to extend into a NEW output directory")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("updates must be positive")
    seeds = [int(s) for s in args.seeds.split(",")]
    policies = args.policies.split(",")
    if len(set(seeds)) != len(seeds) or len(set(policies)) != len(policies):
        parser.error("Duplicate seeds or policies")
    if not set(policies) <= set(POLICIES):
        parser.error("Unknown policy")
    checkpoints = sorted({0, args.updates} | {int(s) for s in args.checkpoints.split(",")
                                              if 0 <= int(s) <= args.updates})
    device = torch.device(args.device)
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(seeds=seeds, policies=policies, updates=args.updates, checkpoints=checkpoints,
                    source_hashes=source_hashes(), torch_version=str(torch.__version__),
                    device=str(device), gpu=(torch.cuda.get_device_name(device) if device.type == "cuda" else None),
                    configs=[asdict(TrajectoryConfig(seed=s, policy=p)) for s in seeds for p in policies])
    if args.keep_going:
        manifest["failure_handling"] = "record_failure_continue_other_tasks"
    if args.continue_from:
        if args.out_dir.resolve() == args.continue_from.resolve():
            raise ValueError("Continuation requires a different output directory")
        manifest["continuation"] = continuation_plan(args.continue_from, manifest)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing suite: use --resume with its exact original configuration and source")
    else:
        write_json(manifest_path, manifest)
    failed_tasks = []
    for seed in seeds:
        for policy in policies:
            folder = args.out_dir / policy / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            latest = folder / "latest.pt"
            trainer = SSPATrajectory(TrajectoryConfig(seed=seed, policy=policy), device)
            if latest.exists():
                if not args.resume:
                    raise FileExistsError(latest)
                trainer.load(latest)
            elif args.continue_from:
                initialize_continuation(folder, policy, seed, manifest["continuation"], trainer)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            write_json(folder / "status.json", dict(state="running", update=trainer.update))
            try:
                while True:
                    if trainer.update in checkpoints and not any(
                            row["update"] == trainer.update for row in trainer.history):
                        sync(device)
                        start = time.perf_counter()
                        row = trainer.validate()
                        sync(device)
                        trainer.validation_seconds += time.perf_counter() - start
                        print(json.dumps(dict(seed=seed, policy=policy, update=trainer.update,
                            anchor_swd=row["anchor_swd"], anchor_gw2=row["anchor_gw2"],
                            train_seconds=trainer.train_seconds)), flush=True)
                        trainer.save(folder / f"update{trainer.update}.pt")
                        trainer.save(latest)
                        write_json(folder / "trajectory.json", trainer.history)
                        write_json(folder / "training_trace.json", trainer.trace)
                    if trainer.update >= args.updates:
                        break
                    stop = min(trainer.update + 100, args.updates,
                               min((s for s in checkpoints if s > trainer.update), default=args.updates))
                    sync(device)
                    start = time.perf_counter()
                    while trainer.update < stop:
                        trainer.step(diagnostics=(trainer.update + 1) % 100 == 0)
                    sync(device)
                    trainer.train_seconds += time.perf_counter() - start
                    if trainer.update % 1000 == 0:
                        print(json.dumps(dict(seed=seed, policy=policy, progress_update=trainer.update,
                                              train_seconds=trainer.train_seconds)), flush=True)
                selected = min(trainer.history, key=lambda row: (row["anchor_swd"], row["update"]))
                write_json(folder / "result.json", dict(config=asdict(trainer.cfg), final_update=trainer.update,
                    selected_update=selected["update"], selected_anchor_swd=selected["anchor_swd"],
                    last_anchor_swd=trainer.history[-1]["anchor_swd"], fixed_epsilon=trainer.fixed_epsilon,
                    counts=trainer.counts, train_seconds=trainer.train_seconds,
                    validation_seconds=trainer.validation_seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None))
                write_json(folder / "status.json", dict(state="complete", update=trainer.update))
            except BaseException as exc:
                write_json(folder / "status.json", dict(state="failed", update=trainer.update,
                    exception=type(exc).__name__, message=str(exc),
                    resume_from="latest.pt (last scheduled validated checkpoint)"))
                if args.keep_going and isinstance(exc, Exception):
                    failed_tasks.append(dict(seed=seed, policy=policy, update=trainer.update,
                                             exception=type(exc).__name__, message=str(exc)))
                    write_json(args.out_dir / "failed_tasks.json", failed_tasks)
                    print(json.dumps(dict(failed_task=failed_tasks[-1])), flush=True)
                    continue
                raise
    if failed_tasks:
        raise RuntimeError(f"{len(failed_tasks)} tasks failed; see failed_tasks.json")


if __name__ == "__main__":
    main()
