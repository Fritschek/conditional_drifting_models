#!/usr/bin/env python3
"""Compare the pinned official W-Flow primitive on our saved SSPA clouds.

This imports only the reviewed Torch loss module, with compilation disabled.
It neither installs the image-generation stack nor reproduces its full loss.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from conditional_drifting.sspa_solver_replay import clipped_field, log_plan, residuals, statistics

PIN = "1ccb34baf57f9f8f4e188460ea3d19b431403293"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, default=Path("results/official_sources_20261010/W-Flow"))
    parser.add_argument("--suite-dir", type=Path, default=Path("results/sspa_solver_replay_20261010"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    commit = subprocess.check_output(["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN or subprocess.check_output(["git", "-C", str(args.official_repo), "status", "--porcelain"], text=True):
        raise ValueError("Expected the unmodified pinned official source")
    os.environ["DRIFT_COMPILE"] = "0"
    source = args.official_repo / "drift_loss_ot.py"
    spec = importlib.util.spec_from_file_location("official_wflow_loss", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.set_num_threads(1)
    audit = json.loads((args.suite_dir / "audit.json").read_text())
    rows = []
    for case in audit["cases"]:
        folder = args.suite_dir / case["case"]
        for filename in ("clouds.pt", "solves.pt"):
            path = folder / filename
            if sha(path) != audit["artifacts"][str(path)]:
                raise ValueError(f"Replay artifact changed: {path}")
        cloud = torch.load(folder / "clouds.pt", map_location="cpu", weights_only=False)
        production = torch.load(folder / "solves.pt", map_location="cpu", weights_only=False)["production10"]
        for dtype, label in ((torch.float32, "f32"), (torch.float64, "f64")):
            g = cloud["generated"].to(dtype)
            for steps in (10, 100):
                record, centers = {}, []
                for term, key in (("cross", "positive"), ("self", "reference")):
                    target = cloud[key].to(dtype)
                    epsilon = torch.tensor(cloud["terms"][term]["epsilon"], dtype=dtype)
                    plan = official._sinkhorn_plan_log_impl(g, target, epsilon, num_iter=steps,
                                                           diag_mask=False, use_quadratic_cost=True)
                    cost = torch.cdist(g, target)
                    cost = .5 * cost * cost
                    control = log_plan(-cost / epsilon, steps)["plan"]
                    torch.testing.assert_close(plan, control, atol=1e-10 if dtype == torch.float64 else 1e-5,
                                               rtol=1e-8 if dtype == torch.float64 else 1e-4)
                    center = torch.bmm(plan, target) / plan.sum(-1, keepdim=True).clamp_min(1e-12)
                    centers.append(center)
                    record[term] = dict(max_plan_difference=(plan - control).abs().max().item(),
                                        marginal_relative_error=statistics(residuals(plan)),
                                        finite=bool(torch.isfinite(plan).all() and torch.isfinite(center).all()))
                raw, clipped = clipped_field(cloud["generated"], *centers)
                rows.append(dict(case=case["case"], dtype=label, iterations=steps, terms=record,
                    raw_field_rms_vs_production=statistics((raw - production["raw_field"]).double().flatten(1).square().mean(1).sqrt()),
                    clipped_field_rms_vs_production=statistics((clipped - production["clipped_field"]).double().flatten(1).square().mean(1).sqrt())))
    payload = dict(repository="https://github.com/hanjq17/W-Flow", commit=commit,
                   source_sha256=sha(source), script_sha256=sha(Path(__file__)), audit_sha256=sha(args.suite_dir / "audit.json"),
                   device="cpu", torch_version=str(torch.__version__), rows=rows,
                   scope="Matched quadratic cost and saved epsilon; no floor/stabilizer; 10/100 fixed iterations. "
                         "No full official normalized-loss training, ImageNet reproduction, or convergence claim.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(comparisons=len(rows), max_plan_difference=max(t["max_plan_difference"]
        for row in rows for t in row["terms"].values())), indent=2))


if __name__ == "__main__":
    main()
