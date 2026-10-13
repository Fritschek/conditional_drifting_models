"""Export compact audited follow-up records, without model checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT = ROOT / "Journal_version/evidence/coverage_followup_20261013"
SUITES = {
    "coverage": ROOT / "results/sspa_coverage_replication_20261012",
    "samplers": ROOT / "results/sspa_saved_sampler_panel_20261012",
}
FILES = {
    "coverage": ("manifest.json", "results.json", "summary.json", "curves.csv",
                 "report.md", "report_audit.json"),
    "samplers": ("manifest.json", "results.json", "report.md", "report_audit.json"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder):
    index = json.loads((folder / "checksums.json").read_text())
    for name, expected in index["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    for kind in FILES:
        suite = folder / kind
        result = json.loads((suite / "results.json").read_text())
        audit = json.loads((suite / "report_audit.json").read_text())
        if audit["source_results_sha256"] != digest(suite / "results.json"):
            raise ValueError(f"Report/result mismatch: {kind}")
        if result["manifest_sha256"] != digest(suite / "manifest.json"):
            raise ValueError(f"Manifest/result mismatch: {kind}")
    print(json.dumps({"files_verified": len(index["files"]),
                      "bytes": sum((folder / f).stat().st_size for f in index["files"]),
                      "scope": "Portable record integrity; no checkpoint replay"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.out_dir)
        return

    from scripts.report_coverage_followup import coverage, samplers

    # The local audit checks numerical sources and checkpoints before export.
    for kind, audit in (("coverage", coverage), ("samplers", samplers)):
        report, _ = audit(SUITES[kind])
        if report != (SUITES[kind] / "report.md").read_text():
            raise ValueError(f"Regenerate the {kind} report before exporting")
    copies = [(SUITES[kind] / name, Path(kind) / name)
              for kind, names in FILES.items() for name in names]
    for source, relative in copies:
        target = args.out_dir / relative
        if target.exists() and digest(target) != digest(source):
            raise FileExistsError(f"Refusing to replace different evidence: {target}")
    hashes = {}
    for source, relative in copies:
        target = args.out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = digest(target)
    index = dict(files=hashes, excluded="Training checkpoints and sampler tensor panels remain in local results/.")
    (args.out_dir / "checksums.json").write_text(json.dumps(index, indent=2) + "\n")
    verify(args.out_dir)


if __name__ == "__main__":
    main()
