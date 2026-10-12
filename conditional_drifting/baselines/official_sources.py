"""Pinned upstream numerical routines; no dependency installation or source rewriting."""

import ast
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

SOURCES = {
    "flow_matching": ("https://github.com/facebookresearch/flow_matching.git", "11568d37f8d5a080e12aa7b5305d9c35ae07d136"),
    "consistency_models": ("https://github.com/openai/consistency_models.git", "e32b69ee436d518377db86fb2127a3972d0d8716"),
    "guided_diffusion": ("https://github.com/openai/guided-diffusion.git", "22e0df8183507e13a7813f8d38d51b072ca1e67c"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checkout(path, commit):
    path = Path(path).resolve()
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"Wrong official revision at {path}: {actual}")
    changed = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changed.strip():
        raise ValueError(f"Modified official checkout: {path}")
    untracked = subprocess.check_output(["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"], text=True)
    if any(p.endswith(".py") for p in untracked.splitlines()):
        raise ValueError(f"Untracked Python source in official checkout: {path}")
    files = subprocess.check_output(["git", "-C", str(path), "ls-files"], text=True).splitlines()
    return {p: sha256(path / p) for p in files if p.endswith(".py") or "LICENSE" in p.upper()}


def definitions(path, names, namespace):
    """Execute exact selected AST definitions, excluding unrelated image/MPI imports."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Missing upstream definitions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def import_package(path, name):
    path = Path(path).resolve()
    sys.path.insert(0, str(path))
    package = importlib.import_module(name)
    if not Path(package.__file__).resolve().is_relative_to(path):
        raise ValueError(f"Another installed {name} shadows the pinned checkout")
    return package


def load_sources(root):
    root = Path(root).resolve()
    records = {name: dict(url=url, commit=commit, files=verify_checkout(root / name, commit))
               for name, (url, commit) in SOURCES.items()}
    import_package(root / "flow_matching", "flow_matching")
    path = importlib.import_module("flow_matching.path").CondOTProbPath()
    import_package(root / "guided_diffusion", "guided_diffusion")
    gd = importlib.import_module("guided_diffusion.gaussian_diffusion")
    respace = importlib.import_module("guided_diffusion.respace")
    embedding = importlib.import_module("guided_diffusion.nn").timestep_embedding

    # L2 consistency training does not require LPIPS, torchvision, MPI or image loaders.
    # The selected class/functions and their decorators are compiled unchanged.
    namespace = {"th": torch, "np": np, "F": F, "nn": torch.nn}
    cm_root = root / "consistency_models/cm"
    definitions(cm_root / "nn.py", ("mean_flat", "append_dims", "append_zero"), namespace)
    definitions(cm_root / "karras_diffusion.py", ("get_weightings", "KarrasDenoiser",
        "get_sigmas_karras", "sample_onestep", "stochastic_iterative_sampler"), namespace)
    definitions(cm_root / "script_util.py", ("create_ema_and_scales_fn",), namespace)
    cm = SimpleNamespace(**{k: v for k, v in namespace.items() if not k.startswith("__")})
    return SimpleNamespace(path=path, gd=gd, respace=respace, embedding=embedding, cm=cm, records=records)
