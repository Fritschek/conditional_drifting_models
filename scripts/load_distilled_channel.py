"""Load the counted consistency-distillation experiment as a channel implant."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer, FastImplant
from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.sspa_trajectory import isolated_rng

ROOT = Path(__file__).resolve().parents[1]


def load_distilled_channel(suite_dir, source_dir, device="cpu", steps=1):
    suite_dir = Path(suite_dir)
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    for name, digest in manifest["adapter_files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Source changed since distillation: {name}")
    official = load_sources(source_dir)
    if official.records != manifest["sources"]:
        raise ValueError("Official source mismatch")
    checkpoint = torch.load(suite_dir / "cd/checkpoint.pt", map_location="cpu", weights_only=False)
    state = checkpoint["trainer"]
    cfg = FastConfig(**manifest["config"])
    expected = replace(cfg, seed=cfg.seed+10000)
    if checkpoint["kind"] != "cd" or state["config"] != asdict(expected):
        raise ValueError("Distillation stage/configuration mismatch")
    result = json.loads((suite_dir / "cd/result.json").read_text())
    if state["update"] != cfg.updates or result["config"] != state["config"]:
        raise ValueError("Incomplete or incompatible distillation checkpoint")
    with isolated_rng(cfg.seed):
        trainer = FastTrainer(expected, device, official)
        # Inference can move between CPU and CUDA; training RNG states cannot.
        trainer.model.load_state_dict(state["model"])
        trainer.ema.load_state_dict(state["ema"])
        implant = FastImplant(trainer, steps)
    cost = dict(method="consistency_distilled", total_train_seconds=result["total_train_seconds"],
        total_oracle_outputs=result["total_oracle_outputs"], teacher_pretraining_included=True)
    return implant, cost
