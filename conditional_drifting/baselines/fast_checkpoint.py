"""Load a comparison checkpoint as a differentiable, frozen channel implant."""

from pathlib import Path
import json

import torch

from .fast_channels import FastConfig, FastTrainer, FastImplant
from .official_sources import load_sources, sha256
from ..sspa_trajectory import isolated_rng
from ..sspa_update_screen import load_official, OFFICIAL_COMMIT


def load_fast_implant(checkpoint, source_dir, wflow_source=None, device="cpu", steps=1):
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint.parents[3] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    root = Path(__file__).resolve().parents[2]
    for name, digest in manifest["adapter_files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Checkpoint adapter source has changed: {name}")
    official = load_sources(source_dir)
    if official.records != manifest["sources"]:
        raise ValueError("Checkpoint upstream implementation mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["trainer"]
    if state["config"] not in manifest["configs"]:
        raise ValueError("Checkpoint configuration is absent from its suite manifest")
    cfg = FastConfig(**state["config"])
    wflow = None
    if cfg.family == "sinkhorn":
        if wflow_source is None:
            raise ValueError("Provide the pinned W-Flow source directory")
        wflow = load_official(wflow_source)
        if manifest["wflow"] != dict(commit=OFFICIAL_COMMIT,
                loss_sha256=sha256(Path(wflow_source) / "drift_loss_ot.py")):
            raise ValueError("Checkpoint W-Flow loss mismatch")
    with isolated_rng(cfg.seed):
        trainer = FastTrainer(cfg, device, official, wflow)
        trainer.restore(state)
        return FastImplant(trainer, steps)
