"""Shared checkpoint, provenance and measurement helpers for local application pilots."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from conditional_drifting.baselines.application_generators import FastConfig, FastTrainer, FastImplant, parameter_count
from conditional_drifting.baselines.fast_evaluation import save, sync, write_json
from conditional_drifting.baselines.official_sources import load_sources, definitions, sha256
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from scripts.run_fast_channel_comparison import ensure_manifest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "Journal_version/application_local_protocol_20261013.md"


def sources(args, scripts):
    official = load_sources(args.source_dir)
    wflow = load_official(args.wflow_source)
    paths = list((ROOT / "conditional_drifting").rglob("*.py"))
    paths += [ROOT / "scripts" / name for name in scripts]
    paths += [Path(__file__), PROTOCOL, ROOT / "scripts/run_fast_channel_comparison.py",
              ROOT / "scripts/run_local_gradient_fidelity.py", ROOT / "scripts/run_symbolic_awgn_benchmark.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}
    return official, wflow, dict(files=hashes, official=official.records,
        wflow=dict(commit=OFFICIAL_COMMIT, sha256=sha256(args.wflow_source / "drift_loss_ot.py")))


def manifest(args, channel, settings, provenance):
    value = dict(schema=1, channel=channel.description(), settings=settings, sources=provenance,
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=args.device,
            hardware=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
            threads=torch.get_num_threads(), precision="float32", compile=False),
        selection="final EMA, development only, no test-SER checkpoint selection")
    value = json.loads(json.dumps(value))
    ensure_manifest(args.out_dir / "manifest.json", value, args.resume)
    return sha256(args.out_dir / "manifest.json")


def train_stage(trainer, path, save_every, resume, kind="standard", sigma_sampler=None, teacher=None):
    history = []
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["kind"] != kind:
            raise ValueError("Training stage mismatch")
        trainer.restore(state["trainer"])
        history = state["history"]
    if trainer.update > trainer.cfg.updates:
        raise ValueError("Checkpoint exceeds update budget")
    while trainer.update < trainer.cfg.updates:
        sync(trainer.device)
        start = time.perf_counter()
        row = trainer.step() if kind == "standard" else trainer.edm_step(kind, sigma_sampler, teacher)
        sync(trainer.device)
        trainer.train_seconds += time.perf_counter()-start
        if trainer.update % 100 == 0 or trainer.update == trainer.cfg.updates:
            history.append(dict(row, seconds=trainer.train_seconds))
        if trainer.update % save_every == 0 or trainer.update == trainer.cfg.updates:
            save(path, dict(kind=kind, trainer=trainer.state(), history=history))
            print(f"[generator] {trainer.cfg.family}/{kind} update={trainer.update} loss={row['loss']:.5g}", flush=True)


def build(name, channel, settings, args, official, wflow):
    cfg = FastConfig(channel=channel.version, family="consistency" if name == "consistency_distilled" else name,
        seed=settings["seed"], parameters=settings["parameters"], batch_size=settings["batch"],
        updates=settings["updates"], gaussian_covariance="full" if channel.n > 8 else "diagonal")
    base = args.out_dir / name
    teacher_cost = None
    if name == "consistency_distilled":
        if cfg.updates % 2:
            raise ValueError("Teacher/student budget must divide equally")
        cfg = replace(cfg, updates=cfg.updates//2)
        ns = dict(vars(official.cm))
        definitions(args.source_dir / "consistency_models/cm/resample.py", ("LogNormalSampler",), ns)
        sigma = ns["LogNormalSampler"]()
        edm = FastTrainer(cfg, args.device, official, channel)
        train_stage(edm, base / "teacher.pt", settings["save_every"], args.resume, "edm", sigma)
        teacher = deepcopy(edm.ema).eval().requires_grad_(False)
        teacher_cost = dict(counts=dict(edm.counts), seconds=edm.train_seconds,
                            parameters=parameter_count(teacher), input_energy_sum=edm.input_energy_sum)
        trainer = FastTrainer(replace(cfg, seed=cfg.seed+10000), args.device, official, channel)
        for model in (trainer.model, trainer.ema, trainer.target):
            model.load_state_dict(teacher.state_dict())
        train_stage(trainer, base / "generator.pt", settings["save_every"], args.resume, "cd", sigma, teacher)
    else:
        trainer = FastTrainer(cfg, args.device, official, channel, wflow)
        train_stage(trainer, base / "generator.pt", settings["save_every"], args.resume)
    total = trainer.counts["oracle_outputs"] + (teacher_cost["counts"]["oracle_outputs"] if teacher_cost else 0)
    if total != settings["updates"]*settings["batch"]:
        raise ValueError("Oracle budget mismatch")
    cost = dict(config=asdict(trainer.cfg), counts=dict(trainer.counts), teacher=teacher_cost,
        oracle_outputs_including_teacher=total,
        train_seconds=trainer.train_seconds + (teacher_cost["seconds"] if teacher_cost else 0),
        parameters=parameter_count(trainer.model), critic_parameters=parameter_count(trainer.critic) if trainer.critic else 0,
        mean_query_energy=trainer.input_energy_sum/trainer.counts["input_anchors"],
        checkpoint_sha256=sha256(base / "generator.pt"))
    return trainer, cost


def cached(path, stamp, resume):
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(path)
    record = json.loads(path.read_text())
    if record["stamp"] != stamp:
        raise ValueError(f"Result provenance mismatch: {path}")
    return record["result"]


def record(path, stamp, result):
    write_json(path, dict(stamp=stamp, result=result))
