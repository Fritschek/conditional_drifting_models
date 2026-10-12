#!/usr/bin/env python3
"""Change only K/B in the unchanged production trainer on a known failure seed."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from conditional_drifting.sspa_intervention import Intervention, digest
from conditional_drifting.sspa_trajectory import source_hashes
from run_sspa_interventions import sync
from report_sspa_interventions import bounded
from run_sspa_solver_replay import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda")
    parent = ROOT / "results/sspa_epsilon_trajectories_full_20261010/fixed_common/seed9001/update80000.pt"
    files = ("scripts/run_sspa_particle_bridge.py", "conditional_drifting/sspa_intervention.py",
        "conditional_drifting/sspa_solver_replay.py", "conditional_drifting/sspa_update_screen.py",
        "scripts/run_sspa_interventions.py", "scripts/report_sspa_interventions.py",
        "scripts/run_sspa_solver_replay.py", "Journal_version/sspa_particle_bridge_protocol_20261011.md")
    contract = dict(parent_sha256=digest(parent), historical_sources=source_hashes(),
        sources={p: digest(ROOT / p) for p in files}, component="production_k64",
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(device))
    folder = args.out_dir
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(contract=contract, seed=9001, horizon=390720, samples=64, batch_size=256)
    path = folder / "manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != manifest:
            raise ValueError("Exact --resume contract required")
    else:
        write_json(path, manifest)
    trainer = Intervention(parent, "control", device, None, contract)
    trainer.cfg = replace(trainer.cfg, samples=64, batch_size=256)
    latest = folder / "latest.pt"
    if latest.exists():
        if not args.resume:
            raise FileExistsError(latest)
        trainer.load_child(latest)
    if trainer.update == 390720 and (folder / "result.json").exists():
        return
    schedule = list(range(80000, 200001, 10000)) + [250000, 300000, 350000, 390720]
    try:
        while True:
            if trainer.update in schedule and not any(r["update"] == trainer.update for r in trainer.history):
                sync(device)
                start = time.perf_counter()
                row = trainer.validate_child()
                sync(device)
                trainer.validation_seconds += time.perf_counter() - start
                trainer.save_child(folder / f"update{trainer.update}.pt")
                trainer.save_child(latest)
                write_json(folder / "trajectory.json", trainer.history)
                write_json(folder / "trace.json", trainer.trace)
                print(json.dumps(dict(update=trainer.update, swd=row["anchor_swd"],
                    variance=row["conditional_variance"], train_seconds=trainer.train_seconds)), flush=True)
                write_json(folder / "status.json", dict(state="running", update=trainer.update))
            if trainer.update == 390720:
                break
            stop = min(trainer.update + 100, min(u for u in schedule if u > trainer.update))
            sync(device)
            start = time.perf_counter()
            while trainer.update < stop:
                trainer.step(diagnostics=(trainer.update + 1) % 100 == 0)
            sync(device)
            trainer.train_seconds += time.perf_counter() - start
        sync(device)
        start = time.perf_counter()
        independent = trainer.independent_panel()
        sync(device)
        trainer.validation_seconds += time.perf_counter() - start
        require_parent = digest(parent) == contract["parent_sha256"]
        if not require_parent:
            raise ValueError("Historical parent changed")
        trainer.save_child(latest)
        trainer.save_child(folder / "update390720.pt")
        result = dict(variant="production_k64", seed=9001, horizon=390720, parent=trainer.parent,
            late_three_bounded=bounded(trainer.history[-3:], trainer.parent_validation, trainer.cfg.noise_std),
            independent=independent, final=trainer.history[-1], counts=trainer.counts,
            train_seconds=trainer.train_seconds, validation_seconds=trainer.validation_seconds,
            checkpoint_sha256={p.name: digest(p) for p in folder.glob("update*.pt")})
        write_json(folder / "result.json", result)
        write_json(folder / "status.json", dict(state="complete", update=trainer.update))
    except BaseException as exc:
        write_json(folder / "status.json", dict(state="failed", update=trainer.update, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
