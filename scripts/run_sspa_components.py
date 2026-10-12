#!/usr/bin/env python3
"""Full-horizon, one-seed component deletions from the official W-Flow recipe."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from conditional_drifting.sspa_components import ComponentIntervention, VARIANTS
from conditional_drifting.sspa_intervention import digest
from conditional_drifting.sspa_trajectory import rng_state, source_hashes
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from run_sspa_interventions import equal_state, sync
from run_sspa_solver_replay import write_json
from report_sspa_interventions import bounded


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
    parent = ROOT / "results/sspa_epsilon_trajectories_full_20261010/fixed_common/seed9001/update80000.pt"
    files = ("conditional_drifting/sspa_components.py", "conditional_drifting/sspa_intervention.py",
        "conditional_drifting/sspa_update_screen.py", "conditional_drifting/sspa_solver_replay.py",
        "scripts/run_sspa_components.py", "scripts/run_sspa_interventions.py",
        "scripts/run_sspa_solver_replay.py", "scripts/report_sspa_interventions.py",
        "Journal_version/sspa_component_protocol_20261010.md")
    contract = dict(parent_sha256=digest(parent), historical_sources=source_hashes(),
        sources={p: digest(ROOT / p) for p in files}, official_commit=OFFICIAL_COMMIT,
        official_loss_sha256=digest(repo / "drift_loss_ot.py"), torch_version=str(torch.__version__),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    horizon = 80100 if args.smoke else 390720
    manifest = dict(contract=contract, variants=VARIANTS, horizon=horizon, seed=9001, smoke=args.smoke)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("Exact --resume contract required")
    else:
        write_json(path, manifest)

    # Confirm that refactored loss/step reproduces the saved successful path.
    check = ComponentIntervention(parent, "official64", device, official, contract | {"component": "official64"})
    for _ in range(100 if args.smoke else 2000):
        check.step()
    if not args.smoke:
        reference = ROOT / "results/sspa_interventions_20261010/official64/seed9001/update82000.pt"
        saved = torch.load(reference, map_location="cpu", weights_only=False)
        checks = {name: equal_state(value, saved[name]) for name, value in (
            ("model", check.model.state_dict()), ("optimizer", check.optimizer.state_dict()),
            ("ema", check.ema.state_dict()), ("rng", rng_state()))}
        write_json(args.out_dir / "reference_replay.json", checks | {"reference_sha256": digest(reference)})
        if not all(checks.values()):
            raise ValueError(f"Official reference parity failed: {checks}")
    del check

    schedule = ([80000, horizon] if args.smoke else
                list(range(80000, 200001, 10000)) + [250000, 300000, 350000, 390720])
    results = []
    for variant in VARIANTS:
        folder = args.out_dir / variant
        folder.mkdir(exist_ok=True)
        trainer = ComponentIntervention(parent, variant, device, official, contract | {"component": variant})
        latest = folder / "latest.pt"
        if latest.exists():
            if not args.resume:
                raise FileExistsError(latest)
            trainer.load_child(latest)
        final = folder / "result.json"
        if final.exists() and trainer.update >= horizon:
            results.append(json.loads(final.read_text()))
            continue
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
                    print(json.dumps(dict(variant=variant, update=trainer.update,
                        swd=row["anchor_swd"], variance=row["conditional_variance"],
                        train_seconds=trainer.train_seconds)), flush=True)
                if trainer.update >= horizon:
                    break
                stop = min(trainer.update + 100, min(u for u in schedule if u > trainer.update))
                sync(device)
                start = time.perf_counter()
                while trainer.update < stop:
                    trainer.step(diagnostics=(trainer.update + 1) % 100 == 0)
                sync(device)
                trainer.train_seconds += time.perf_counter() - start
                if trainer.update % 10000 == 0:
                    write_json(folder / "status.json", dict(state="running", update=trainer.update, target=horizon))
            sync(device)
            start = time.perf_counter()
            independent = None if args.smoke else trainer.independent_panel()
            sync(device)
            trainer.validation_seconds += time.perf_counter() - start
            late = [r for r in trainer.history if r["update"] in (300000, 350000, 390720)]
            result = dict(variant=variant, seed=9001, horizon=horizon, parent=trainer.parent,
                late_three_bounded=bounded(late, trainer.parent_validation, trainer.cfg.noise_std, False),
                independent=independent, final=trainer.history[-1], counts=trainer.counts,
                train_seconds=trainer.train_seconds, validation_seconds=trainer.validation_seconds)
            if digest(parent) != contract["parent_sha256"]:
                raise ValueError("Historical parent was modified")
            trainer.save_child(latest)
            trainer.save_child(folder / f"update{horizon}.pt")
            result["checkpoint_sha256"] = {p.name: digest(p) for p in folder.glob("update*.pt")}
            write_json(final, result)
            write_json(folder / "status.json", dict(state="complete", update=trainer.update, target=horizon))
            results.append(result)
        except BaseException as exc:
            write_json(folder / "status.json", dict(state="failed", update=trainer.update,
                message=str(exc), resume_from="latest.pt; unsaved interval is replayed"))
            raise
    write_json(args.out_dir / "completion.json", dict(complete=True, results=results))


if __name__ == "__main__":
    main()
