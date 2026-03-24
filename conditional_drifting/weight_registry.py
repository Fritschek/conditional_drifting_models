from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "weights"
INDEX_PATH = DEFAULT_WEIGHTS_ROOT / "index.json"


def sanitize_name(name: str) -> str:
    value = name.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("._") or "artifact"


def ensure_weights_root(root: str | Path | None = None) -> Path:
    path = Path(root) if root is not None else DEFAULT_WEIGHTS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_index(root: str | Path | None = None) -> dict[str, Any]:
    weights_root = ensure_weights_root(root)
    index_path = weights_root / "index.json"
    if not index_path.exists():
        return {}
    return json.loads(index_path.read_text())


def save_index(index: dict[str, Any], root: str | Path | None = None) -> Path:
    weights_root = ensure_weights_root(root)
    index_path = weights_root / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return index_path


def register_artifact(
    *,
    name: str,
    category: str,
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
    root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    weights_root = ensure_weights_root(root)
    index = load_index(weights_root)
    key = sanitize_name(name)
    entry = {
        "name": key,
        "category": category,
        "checkpoint": str(Path(checkpoint_path).resolve()),
    }
    if metadata_path is not None:
        entry["metadata"] = str(Path(metadata_path).resolve())
    if extra:
        entry.update(extra)
    index[key] = entry
    return save_index(index, weights_root)


def resolve_registered_artifact(name: str, root: str | Path | None = None) -> dict[str, Any]:
    weights_root = ensure_weights_root(root)
    index = load_index(weights_root)
    key = sanitize_name(name)
    if key not in index:
        raise KeyError(f"No registered artifact named {name!r} in {weights_root}.")
    return index[key]


def build_artifact_path(
    *,
    root: str | Path | None,
    category: str,
    group: str,
    name: str,
    suffix: str = ".pt",
) -> Path:
    weights_root = ensure_weights_root(root)
    folder = weights_root / sanitize_name(category) / sanitize_name(group)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{sanitize_name(name)}{suffix}"
