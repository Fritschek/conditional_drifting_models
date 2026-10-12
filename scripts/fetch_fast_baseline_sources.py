#!/usr/bin/env python3
"""Fetch exact official revisions. Never install packages or update existing checkouts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conditional_drifting.baselines.official_sources import SOURCES, verify_checkout
from conditional_drifting.sspa_update_screen import OFFICIAL_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("results/official_fast_sources_20261012"))
    parser.add_argument("--wflow-source", type=Path, default=Path("results/official_sources_20261010/W-Flow"))
    args = parser.parse_args()
    args.source_dir.mkdir(parents=True, exist_ok=True)
    sources = [(name, args.source_dir / name, url, commit) for name, (url, commit) in SOURCES.items()]
    sources.append(("W-Flow", args.wflow_source, "https://github.com/hanjq17/W-Flow.git", OFFICIAL_COMMIT))
    for name, path, url, commit in sources:
        if not path.exists():
            subprocess.run(["git", "init", str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)
            subprocess.run(["git", "-C", str(path), "fetch", "--depth", "1", "origin", commit], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", commit], check=True)
        files = verify_checkout(path, commit)
        print(json.dumps(dict(name=name, commit=commit, verified_files=len(files))))


if __name__ == "__main__":
    main()
