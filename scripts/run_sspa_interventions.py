#!/usr/bin/env python3
"""Controlled SSPA continuation matrix with explicit intervention provenance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from conditional_drifting.sspa_intervention import BRANCHES, Intervention, digest, passes_gate
from conditional_drifting.sspa_trajectory import rng_state, source_hashes
from conditional_drifting.sspa_update_screen import OFFICIAL_COMMIT, load_official
from run_sspa_solver_replay import write_json

PARENT = ROOT / "results/sspa_epsilon_trajectories_full_20261010/fixed_common"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    if hasattr(left, "shape"):
        return bool((left == right).all())
    return left == right


def control_check(trainer, seed):
    historical = PARENT / f"seed{seed}/update{trainer.update}.pt"
    if not historical.exists():
        return None
    old = torch.load(historical, map_location="cpu", weights_only=False)
    checks = dict(model=equal_state(trainer.model.state_dict(), old["model"]),
                  optimizer=equal_state(trainer.optimizer.state_dict(), old["optimizer"]),
                  rng=equal_state(rng_state(), old["rng"]),
                  training_counts=all(trainer.counts[k] == old["counts"][k] for k in trainer.counts if k.startswith("training_")))
    if not all(checks.values()):
        raise ValueError(f"Control no longer reproduces retained path: {seed}/{trainer.update}: {checks}")
    return checks


def run_task(root, seed, branch, horizon, device, official, contract, resume, smoke=False):
    folder = root / branch / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    trainer = Intervention(PARENT / f"seed{seed}/update80000.pt", branch, device, official, contract)
    latest = folder / "latest.pt"
    if latest.exists():
        if not resume:
            raise FileExistsError(latest)
        trainer.load_child(latest)
    if trainer.update >= horizon and (folder / f"result{horizon}.json").exists():
        return
    checkpoints = ([80000, horizon] if smoke else list(range(80000, 160001, 2000)) +
                   [170000, 180000, 190000, 200000, 250000, 300000, 350000, 390720])
    write_json(folder / "status.json", dict(state="running", update=trainer.update, target=horizon))
    invocation = dict(start=datetime.now(timezone.utc).isoformat(), resume_update=trainer.update, target=horizon)
    with (folder / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps(invocation) + "\n")
    try:
        while True:
            if trainer.update in checkpoints and not any(r["update"] == trainer.update for r in trainer.history):
                sync(device)
                start = time.perf_counter()
                row = trainer.validate_child(reference_check=not smoke and trainer.update in (80000, 200000, 390720))
                if branch == "control":
                    row["control_reproduction"] = control_check(trainer, seed)
                sync(device)
                trainer.validation_seconds += time.perf_counter() - start
                trainer.save_child(folder / f"update{trainer.update}.pt")
                trainer.save_child(latest)
                write_json(folder / "trajectory.json", trainer.history)
                write_json(folder / "trace.json", trainer.trace)
                print(json.dumps(dict(seed=seed, branch=branch, update=trainer.update,
                    swd=row["anchor_swd"], variance=row["conditional_variance"], train_seconds=trainer.train_seconds)), flush=True)
            if trainer.update >= horizon:
                break
            stop = min(trainer.update + 100, horizon, min(u for u in checkpoints if u > trainer.update))
            sync(device)
            start = time.perf_counter()
            while trainer.update < stop:
                trainer.step(diagnostics=(trainer.update + 1) % 100 == 0)
            sync(device)
            trainer.train_seconds += time.perf_counter() - start
            if trainer.update % 5000 == 0:
                write_json(folder / "status.json", dict(state="running", update=trainer.update, target=horizon))
        independent = None
        if not smoke:
            sync(device)
            start = time.perf_counter()
            independent = trainer.independent_panel()
            sync(device)
            trainer.validation_seconds += time.perf_counter() - start
            trainer.save_child(latest)
            trainer.save_child(folder / f"update{horizon}.pt")
        if digest(trainer.parent["path"]) != trainer.parent["sha256"]:
            raise ValueError("Parent changed")
        result = dict(seed=seed, branch=branch, horizon=horizon, parent=trainer.parent,
            gate_raw=passes_gate(trainer.history, trainer.parent_validation, trainer.cfg.noise_std),
            gate_ema=passes_gate(trainer.history, trainer.parent_validation, trainer.cfg.noise_std, ema=True),
            independent=independent, train_seconds=trainer.train_seconds, validation_seconds=trainer.validation_seconds,
            counts=trainer.counts, final=trainer.history[-1], selected_update=min(trainer.history,
                key=lambda row: (row["anchor_swd"], row["update"]))["update"],
            checkpoint_sha256={p.name: digest(p) for p in folder.glob("update*.pt")})
        write_json(folder / f"result{horizon}.json", result)
        write_json(folder / "status.json", dict(state="complete", update=trainer.update, target=horizon))
    except BaseException as exc:
        write_json(folder / "status.json", dict(state="failed", update=trainer.update, target=horizon,
            exception=type(exc).__name__, message=str(exc), resume_from="latest.pt; unsaved interval is replayed"))
        with (folder / "failures.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(update=trainer.update, exception=type(exc).__name__, message=str(exc))) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    repo = ROOT / "results/official_sources_20261010/W-Flow"
    official = load_official(repo)
    seeds = [9001] if args.smoke else [9001, 9002, 9003]
    horizon = 80100 if args.smoke else 200000
    files = ("conditional_drifting/sspa_intervention.py", "conditional_drifting/sspa_update_screen.py",
             "conditional_drifting/sspa_solver_replay.py", "scripts/run_sspa_interventions.py",
             "scripts/run_sspa_solver_replay.py", "Journal_version/sspa_intervention_protocol_20261010.md")
    contract = dict(source_hashes=source_hashes(), child_sources={p: digest(ROOT / p) for p in files},
        official_commit=OFFICIAL_COMMIT, official_loss_sha256=digest(repo / "drift_loss_ot.py"),
        torch_version=str(torch.__version__), device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    manifest = dict(contract=contract, seeds=seeds, branches=BRANCHES, smoke=args.smoke,
        first_horizon=horizon, full_horizon=390720,
        parent_hashes={str(seed): digest(PARENT / f"seed{seed}/update80000.pt") for seed in seeds})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists():
        # JSON turns tuples into lists.
        if not args.resume or json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Existing suite requires exact --resume contract")
    else:
        write_json(manifest_path, manifest)
    failures = []
    for branch in BRANCHES:
        for seed in seeds:
            try:
                run_task(args.out_dir, seed, branch, horizon, device, official, contract, args.resume, args.smoke)
            except Exception as exc:
                failures.append(dict(branch=branch, seed=seed, horizon=horizon, message=str(exc)))
                write_json(args.out_dir / "failed_tasks.json", failures)
    if args.smoke:
        if failures:
            raise RuntimeError(f"Smoke failures: {failures}")
        return
    extended = []
    gates = {}
    for branch in BRANCHES:
        paths = [args.out_dir / branch / f"seed{seed}" / "result200000.json" for seed in seeds]
        if not all(p.exists() for p in paths):
            gates[branch] = "incomplete"
            continue
        values = [json.loads(p.read_text()) for p in paths]
        gates[branch] = {str(v["seed"]): dict(raw=v["gate_raw"], ema=v["gate_ema"]) for v in values}
        if any(v["gate_raw"] or v["gate_ema"] for v in values):
            extended.append(branch)
    write_json(args.out_dir / "extension_decision.json", dict(gates=gates, extended=extended,
        rule="Any seed passing raw or applicable EMA extends all three; independent panel unused"))
    for branch in extended:
        for seed in seeds:
            try:
                run_task(args.out_dir, seed, branch, 390720, device, official, contract, True)
            except Exception as exc:
                failures.append(dict(branch=branch, seed=seed, horizon=390720, message=str(exc)))
                write_json(args.out_dir / "failed_tasks.json", failures)
    write_json(args.out_dir / "completion.json", dict(complete=not failures, extended=extended, failures=failures))
    if failures:
        raise RuntimeError(f"{len(failures)} tasks failed; retained in failed_tasks.json")


if __name__ == "__main__":
    main()
