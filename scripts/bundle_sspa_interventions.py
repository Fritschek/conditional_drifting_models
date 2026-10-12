#!/usr/bin/env python3
"""Create a verified compact handover of the completed intervention study."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite_dir.resolve()
    completion = json.loads((suite / "completion.json").read_text())
    if not completion["complete"] or not (suite / "audit.json").exists():
        raise ValueError("Completed, audited suite required")
    files = set()
    for pattern in ("*.json", "*.jsonl", "*.csv", "*.pdf", "*.png"):
        files.update(suite.rglob(pattern))
    for name in ("update80000.pt", "update200000.pt", "update390720.pt"):
        files.update(suite.glob(f"*/seed*/{name}"))
    manifest = json.loads((suite / "manifest.json").read_text())
    files.update(ROOT / p for p in manifest["contract"]["child_sources"])
    files.update(ROOT / "conditional_drifting" / p for p in manifest["contract"]["source_hashes"])
    files.update((ROOT / "conditional_drifting").glob("*.py"))
    for name in ("report_sspa_interventions.py", "plot_sspa_interventions.py", "bundle_sspa_interventions.py",
                 "run_sspa_solver_replay_remaining.py", "report_sspa_solver_replay.py"):
        files.add(ROOT / "scripts" / name)
    files.update((ROOT / "tests").glob("test_sspa*.py"))
    for name in ("sspa_intervention_protocol_20261010.md", "sspa_intervention_results_20261010.md",
                 "sspa_solver_replay_replication_20261010.md", "official_drifting_code_comparison_20261010.md"):
        files.add(ROOT / "Journal_version" / name)
    replay = ROOT / "results/sspa_solver_replay_remaining_20261010"
    for pattern in ("*.json", "*.csv"):
        files.update(replay.rglob(pattern))
    for seed in manifest["seeds"]:
        files.add(ROOT / "results/sspa_epsilon_trajectories_full_20261010/fixed_common" / f"seed{seed}/update80000.pt")
    for name in ("drift_loss_ot.py", "LICENSE"):
        files.add(ROOT / "results/official_sources_20261010/W-Flow" / name)
    entries = {str(p.resolve().relative_to(ROOT)): p.read_bytes() for p in sorted(files)}
    entries["BUNDLE_README.txt"] = (
        "Completed three-seed SSPA intervention study. Start with Journal_version/sspa_intervention_results_20261010.md.\n"
        "Includes all JSON/CSV metrics, figures, source snapshots, tests, 80k parents and 80k/200k/390720 child checkpoints.\n"
        "Intermediate child checkpoints and full replay clouds/solves are retained locally, with hashes; excluded to limit size.\n"
        "A full tensor audit needs those excluded files or their regeneration. No reference accuracy is claimed for capped solves.\n"
        "The pinned W-Flow loss and MIT license are included; reclone its recorded commit for .git-based provenance checks.\n"
        "Training starts from retained checkpoints, not from scratch; optimization interventions are explicitly documented.\n"
    ).encode()
    hashes = {name: sha(data) for name, data in entries.items()}
    entries["BUNDLE_SHA256.json"] = (json.dumps(hashes, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(temporary, "r:gz") as archive:
        for name, expected in hashes.items():
            if sha(archive.extractfile(name).read()) != expected:
                raise ValueError(f"Archive verification failed: {name}")
    temporary.replace(args.out)
    print(json.dumps(dict(path=str(args.out), verified_files=len(hashes), bytes=args.out.stat().st_size,
                          sha256=sha(args.out.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
