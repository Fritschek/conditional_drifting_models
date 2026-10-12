#!/usr/bin/env python3
"""Remaining-seed replication of the frozen 16-method solver replay."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.sspa_trajectory import source_hashes
from run_sspa_solver_replay import run_case, write_json
from run_sspa_update_screen import sha
from report_sspa_solver_replay import report

PAIRS = {
    9002: {"legacy_separate_adaptive": (100000, 110000), "fixed_common": (140000, 150000), "shared_adaptive": (110000, 120000)},
    9003: {"legacy_separate_adaptive": (110000, 120000), "fixed_common": (140000, 150000), "shared_adaptive": (100000, 110000)},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.smoke, args.reference_cap = False, 50000
    torch.set_num_threads(1)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    files = ("scripts/run_sspa_solver_replay_remaining.py", "scripts/run_sspa_solver_replay.py",
             "conditional_drifting/sspa_solver_replay.py", "Journal_version/sspa_intervention_protocol_20261010.md")
    for seed, pairs in PAIRS.items():
        out = args.out_dir / f"seed{seed}"
        out.mkdir()
        selections = {policy: [30000, *pair, 390720] for policy, pair in pairs.items()}
        write_json(out / "manifest.json", dict(selections=selections, seed=seed, smoke=False,
            reference_cap=50000, reference_tolerance=1e-8, device=str(device),
            torch_version=str(torch.__version__), source_hashes=source_hashes(),
            added_source_hashes={p: sha(ROOT / p) for p in files}, gpu=torch.cuda.get_device_name(device)))
        try:
            for policy, updates in selections.items():
                for update in updates:
                    parent = ROOT / f"results/sspa_epsilon_trajectories_full_20261010/{policy}/seed{seed}/update{update}.pt"
                    run_case(parent, out / f"{policy}_u{update}", device, args)
            write_json(out / "audit.json", report(out))
        except BaseException as exc:
            write_json(out / "failure.json", dict(type=type(exc).__name__, message=str(exc)))
            raise


if __name__ == "__main__":
    main()
