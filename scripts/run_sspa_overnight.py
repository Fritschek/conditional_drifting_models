#!/usr/bin/env python3
"""Frozen fresh-start SSPA comparison with paired communication experiments."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch

from conditional_drifting.sspa_trajectory import SSPATrajectory, TrajectoryConfig, source_hashes
from conditional_drifting.sspa_intervention import Intervention, digest
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from conditional_drifting.sspa_utility import (CodecTrial, analytic_channel, load_generator,
    gradient_comparison, atomic_save, NOISE_STD)
from run_sspa_solver_replay import write_json
from run_sspa_interventions import sync

VARIANTS = ("old_k4", "old_k64", "official64")
SCHEDULE = (0, 1000, 5000, 10000, 30000, 50000, 80000, 100000,
            150000, 200000, 250000, 300000, 350000, 390720)
EVENT_LOG = None


class BudgetExpired(Exception):
    pass


class Budget:
    def __init__(self, deadline):
        self.stop = datetime.fromisoformat(deadline).timestamp() - 300

    def check(self):
        if time.time() >= self.stop:
            raise BudgetExpired("Five-minute reserve before frozen deadline")


def event(**values):
    line = json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), **values))
    print(line, flush=True)
    if EVENT_LOG is not None:
        with EVENT_LOG.open("a") as handle:
            handle.write(line + "\n")


def checked_result(path, contract):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value["contract"] != contract:
        raise ValueError(f"Result contract mismatch: {path}")
    return value


def child(parent, variant, device, official, contract):
    trainer = Intervention(parent, "official64" if variant == "official64" else "control",
                           device, official, contract)
    if variant == "old_k64":
        trainer.cfg = replace(trainer.cfg, batch_size=trainer.cfg.batch_size * trainer.cfg.samples // 64, samples=64)
    return trainer


def selected_update(history):
    return min((r for r in history if r["update"] > 0), key=lambda r: (r["anchor_swd"], r["update"]))["update"]


def late_gate(history):
    rows = [r for r in history if r["update"] in (300000, 350000, 390720)]
    return len(rows) == 3 and all(r["anchor_swd"] <= 2 * r["anchor_swd_floor"]
        and r["anchor_mean_l2"] <= 2 * r["anchor_mean_l2_floor"]
        and .5 * NOISE_STD ** 2 / 2 <= r["conditional_variance"] <= 2 * NOISE_STD ** 2 / 2 for r in rows)


def train_generator(parent, folder, variant, seed, device, official, base_contract, budget, smoke):
    folder.mkdir(parents=True, exist_ok=True)
    contract = dict(base_contract, variant=variant, seed=seed, parent_sha256=digest(parent))
    if result := checked_result(folder / "result.json", contract):
        for item in result["checkpoints"].values():
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Generator checkpoint changed")
        return result
    trainer = child(parent, variant, device, official, contract)
    latest = folder / "latest.pt"
    if latest.exists():
        trainer.load_child(latest)
    schedule = (0, 10, 20) if smoke else SCHEDULE
    horizon = schedule[-1]
    try:
        while True:
            budget.check()
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
                event(phase="generator", seed=seed, variant=variant, update=trainer.update,
                      swd=row["anchor_swd"], variance=row["conditional_variance"], seconds=trainer.train_seconds)
            # A resume file can precede a failed scheduled checkpoint write.
            if trainer.update in schedule and not (folder / f"update{trainer.update}.pt").exists():
                trainer.save_child(folder / f"update{trainer.update}.pt")
            if trainer.update == horizon:
                break
            stop = min(trainer.update + 100, min(u for u in schedule if u > trainer.update),
                       ((trainer.update // 5000) + 1) * 5000)
            sync(device)
            start = time.perf_counter()
            while trainer.update < stop:
                trainer.step(diagnostics=(trainer.update + 1) % 100 == 0)
            sync(device)
            trainer.train_seconds += time.perf_counter() - start
            if trainer.update % 5000 == 0:
                trainer.save_child(latest)
            write_json(folder / "status.json", dict(state="training", update=trainer.update, horizon=horizon))
    except BaseException:
        trainer.save_child(latest)
        raise
    checkpoints, panels, panel_seconds = {}, {}, {}
    chosen = selected_update(trainer.history)
    for label, update in (("selected", chosen), ("final", horizon)):
        budget.check()
        path = folder / f"update{update}.pt"
        checkpoints[label] = dict(path=str(path), sha256=digest(path), update=update)
        evaluator = child(parent, variant, device, official, contract)
        evaluator.load_child(path)
        sync(device)
        start = time.perf_counter()
        panels[label] = evaluator.independent_panel()
        sync(device)
        panel_seconds[label] = time.perf_counter() - start
    if digest(parent) != contract["parent_sha256"]:
        raise ValueError("Random parent changed")
    result = dict(contract=contract, seed=seed, variant=variant, horizon=horizon,
        checkpoints=checkpoints, independent=panels, late_three_bounded=None if smoke else late_gate(trainer.history),
        final=trainer.history[-1], counts=trainer.counts, train_seconds=trainer.train_seconds,
        validation_seconds=trainer.validation_seconds, independent_panel_seconds=panel_seconds)
    write_json(folder / "result.json", result)
    write_json(folder / "status.json", dict(state="complete", update=horizon))
    return result


def train_codec(folder, seed, generator, device, base_contract, budget, smoke):
    folder.mkdir(parents=True, exist_ok=True)
    contract = dict(base_contract, ae_seed=seed, generator=None if generator is None else
                    dict(path=str(generator), sha256=digest(generator)))
    if result := checked_result(folder / "result.json", contract):
        if digest(result["checkpoint"]) != result["checkpoint_sha256"]:
            raise ValueError("AE checkpoint changed")
        return result
    channel = analytic_channel if generator is None else load_generator(generator, device)
    trainer = CodecTrial(seed, channel, device, contract)
    latest = folder / "latest.pt"
    if latest.exists():
        trainer.load(latest)
    schedule = (5, 10) if smoke else (5000, 10000, 20000)
    horizon = schedule[-1]
    try:
        while True:
            budget.check()
            if trainer.update in schedule and not any(r["update"] == trainer.update for r in trainer.history):
                sync(device)
                start = time.perf_counter()
                stats = trainer.evaluate(16 if smoke else 1024, 3000000 + seed)
                sync(device)
                trainer.eval_seconds += time.perf_counter() - start
                trainer.history.append(dict(update=trainer.update, **stats))
                trainer.save(folder / f"update{trainer.update}.pt")
                trainer.save(latest)
                event(phase="codec", path=str(folder), update=trainer.update, ser=stats["ser"], seconds=trainer.train_seconds)
            if trainer.update in schedule and not (folder / f"update{trainer.update}.pt").exists():
                trainer.save(folder / f"update{trainer.update}.pt")
            if trainer.update == horizon:
                break
            stop = min(trainer.update + 100, min(u for u in schedule if u > trainer.update))
            sync(device)
            start = time.perf_counter()
            while trainer.update < stop:
                trainer.step(trace=(trainer.update + 1) % 100 == 0)
            sync(device)
            trainer.train_seconds += time.perf_counter() - start
            if trainer.update % 1000 == 0:
                trainer.save(latest)
    except BaseException:
        trainer.save(latest)
        raise
    tests = []
    sync(device)
    start = time.perf_counter()
    for ebno in (4., 6., 8.):
        budget.check()
        tests.append(trainer.evaluate(16 if smoke else 32768, 4000000 + seed, ebno))
    sync(device)
    test_seconds = time.perf_counter() - start
    path = folder / f"update{horizon}.pt"
    result = dict(contract=contract, checkpoint=str(path), checkpoint_sha256=digest(path),
        update=trainer.update, train_messages=trainer.update * trainer.batch_size,
        training_oracle_outputs=trainer.update * trainer.batch_size if generator is None else 0,
        training_surrogate_outputs=trainer.update * trainer.batch_size if generator is not None else 0,
        validation_oracle_outputs=sum(row["trials"] for row in trainer.history),
        test_oracle_outputs=sum(row["trials"] for row in tests), tests=tests, validation=trainer.history,
        train_seconds=trainer.train_seconds, validation_seconds=trainer.eval_seconds, test_seconds=test_seconds)
    write_json(folder / "result.json", result)
    event(phase="codec_complete", path=str(folder), tests=tests)
    return result


def main():
    global EVENT_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deadline", default="2026-10-12T04:00:00+00:00")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    budget = Budget(args.deadline)
    budget.check()
    folder = args.out_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    seeds = [9904] if args.smoke else [9004, 9005, 9006]
    files = ("scripts/run_sspa_overnight.py", "conditional_drifting/sspa_utility.py",
        "conditional_drifting/sspa_intervention.py", "conditional_drifting/sspa_update_screen.py",
        "conditional_drifting/sspa_solver_replay.py", "conditional_drifting/symbolic_ae.py",
        "scripts/run_local_gradient_fidelity.py", "scripts/run_sspa_solver_replay.py",
        "scripts/run_sspa_interventions.py", "Journal_version/sspa_fresh_protocol_20261011.md")
    official_dir = ROOT / "results/official_sources_20261010/W-Flow"
    official = load_official(official_dir)
    contract = dict(sources={p: digest(ROOT / p) for p in files}, historical_sources=source_hashes(),
        official_commit=OFFICIAL_COMMIT, torch_version=str(torch.__version__), device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None, smoke=args.smoke)
    manifest = dict(contract=contract, seeds=seeds, variants=list(VARIANTS),
        generator_horizon=20 if args.smoke else 390720, codec_horizon=10 if args.smoke else 20000,
        ae_seed_offsets=[10000, 20000], deadline=args.deadline)
    path = folder / "manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != manifest:
            raise ValueError("Existing suite requires exact --resume manifest")
    else:
        if any(folder.iterdir()):
            raise ValueError("Refusing to initialize a nonempty suite directory")
        write_json(path, manifest)
    EVENT_LOG = folder / "events.jsonl"
    failures, generators, codecs, gradients = [], {}, {}, {}

    def attempt(label, action):
        budget.check()
        write_json(folder / "status.json", dict(state="running", task=label, failures=failures))
        try:
            return action()
        except BudgetExpired:
            raise
        except Exception as exc:
            item = dict(task=label, error=str(exc), traceback=traceback.format_exc())
            failures.append(item)
            write_json(folder / "failures.json", failures)
            event(phase="failed", **item)
            return None

    state = "complete"
    try:
        for seed in seeds:
            budget.check()
            parent = folder / "parents" / f"seed{seed}.pt"
            parent.parent.mkdir(exist_ok=True)
            cfg = TrajectoryConfig(seed=seed, policy="fixed_common")
            if args.smoke:
                cfg = replace(cfg, batch_size=16, calibration_anchors=8,
                    validation_anchors=4, validation_samples=8, validation_projections=4)
            if not parent.exists():
                trainer = SSPATrajectory(cfg, device)
                trainer.validate()
                trainer.save(parent)
            else:
                trainer = SSPATrajectory(cfg, device)
                trainer.load(parent)
            if trainer.update != 0 or trainer.optimizer.state:
                raise ValueError("Fresh run requires an untrained parent and empty optimizer")
            for variant in VARIANTS:
                label = f"generator/{variant}/seed{seed}"
                result = attempt(label, lambda: train_generator(parent, folder / label, variant, seed,
                    device, official, contract, budget, args.smoke))
                if result:
                    generators[f"{seed}/{variant}"] = result
        for seed in seeds:
            paths = {f"{variant}_{stage}": Path(generators[f"{seed}/{variant}"]["checkpoints"][stage]["path"])
                for variant in VARIANTS if f"{seed}/{variant}" in generators for stage in ("selected", "final")}
            for offset in (10000, 20000):
                ae_seed = seed + offset
                for method, checkpoint in {"analytic": None, **paths}.items():
                    label = f"codec/seed{seed}/ae{ae_seed}/{method}"
                    result = attempt(label, lambda: train_codec(folder / label, ae_seed, checkpoint,
                        device, contract, budget, args.smoke))
                    if result:
                        codecs[label] = result
            common = folder / f"codec/seed{seed}/ae{10000 + seed}/analytic"
            for update in ((5, 10) if args.smoke else (5000, 20000)):
                codec = common / f"update{update}.pt"
                if not codec.exists():
                    continue
                label = f"gradients/seed{seed}/update{update}"
                out = folder / label
                out.mkdir(parents=True, exist_ok=True)
                gc = dict(contract, codec_sha256=digest(codec),
                    generators={k: digest(p) for k, p in paths.items()})

                def evaluate_gradients():
                    if existing := checked_result(out / "result.json", gc):
                        if digest(out / "vectors.pt") != existing["vectors_sha256"]:
                            raise ValueError("Gradient vectors changed")
                        return existing
                    values, tensors = gradient_comparison(codec, paths, device, args.smoke, budget.check)
                    atomic_save(tensors, out / "vectors.pt")
                    values.update(contract=gc, vectors_sha256=digest(out / "vectors.pt"))
                    write_json(out / "result.json", values)
                    event(phase="gradients_complete", task=label)
                    return values

                result = attempt(label, evaluate_gradients)
                if result:
                    gradients[label] = result
    except BudgetExpired:
        state = "deadline_incomplete"
    except BaseException as exc:
        write_json(folder / "status.json", dict(state="interrupted", error=str(exc), failures=failures))
        raise
    expected = dict(generators=3 * len(seeds), codecs=14 * len(seeds), gradients=2 * len(seeds))
    completed = dict(generators=len(generators), codecs=len(codecs), gradients=len(gradients))
    if state == "complete" and (failures or completed != expected):
        state = "incomplete"
    write_json(folder / "summary.json", dict(state=state, expected=expected, completed=completed,
        generators=generators, codecs=codecs, gradients=gradients, failures=failures))
    write_json(folder / "status.json", dict(state=state, completed=completed, expected=expected, failures=failures))
    event(phase="finished", state=state, completed=completed, expected=expected)
    if state == "incomplete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
