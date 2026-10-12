#!/usr/bin/env python3
"""Package the audited overnight comparison, including every scheduled checkpoint."""

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
    summary = json.loads((suite / "summary.json").read_text())
    audit = json.loads((suite / "audit.json").read_text())
    if summary["state"] != "complete" or audit["status"] != "passed":
        raise ValueError("Completed, audited suite required")
    if args.out.exists():
        raise FileExistsError(args.out)
    files = {p for p in suite.rglob("*") if p.is_file() and p.name != "latest.pt" and p.suffix != ".tmp"}
    manifest = json.loads((suite / "manifest.json").read_text())
    files.update(ROOT / p for p in manifest["contract"]["sources"])
    files.update((ROOT / "conditional_drifting").glob("*.py"))
    for name in ("report_sspa_overnight.py", "bundle_sspa_overnight.py", "run_symbolic_awgn_benchmark.py"):
        files.add(ROOT / "scripts" / name)
    files.add(ROOT / "tests/test_sspa_overnight.py")
    for name in ("sspa_fresh_results_20261011.md", "experiment_status_20261011.md",
                 "resubmission_handover.md", "resubmission_cluster_plan.md",
                 "official_drifting_code_comparison_20261010.md"):
        files.add(ROOT / "Journal_version" / name)
    for name in ("drift_loss_ot.py", "LICENSE"):
        files.add(ROOT / "results/official_sources_20261010/W-Flow" / name)
    entries = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(files)}
    entries["BUNDLE_README.txt"] = (
        "Fresh-start SSPA development comparison: three unused seeds, old K4, old K64-only, official normalized K64.\n"
        "Start with Journal_version/sspa_fresh_results_20261011.md and the frozen protocol.\n"
        "Includes every scheduled generator and AE checkpoint, random parents, gradient vectors, counts, metrics and plots.\n"
        "Latest resume duplicates are omitted. The original results tree retains them.\n"
        "Source and checkpoint hashes are included. All paths in result contracts preserve the original workspace paths.\n"
        "This is evidence, not an environment image. Relocate paths or restore the original layout for audits.\n"
        "Pinned W-Flow loss and MIT license included; reclone its recorded commit for the git-provenance check.\n"
        "No cross-family superiority, optimal cloud size, or universal stability is established by this three-seed study.\n"
    ).encode()
    hashes = {name: sha(data) for name, data in entries.items()}
    entries["BUNDLE_SHA256.json"] = (json.dumps(hashes, indent=2) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(temporary, "r:gz") as archive:
        for name, expected in hashes.items():
            if sha(archive.extractfile(name).read()) != expected:
                raise ValueError(f"Archive verification failed: {name}")
    temporary.replace(args.out)
    checksum = sha(args.out.read_bytes())
    args.out.with_suffix(".sha256").write_text(f"{checksum}  {args.out.name}\n")
    print(json.dumps(dict(path=str(args.out), verified_files=len(hashes), bytes=args.out.stat().st_size,
                         sha256=checksum), indent=2))


if __name__ == "__main__":
    main()
