from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import optfib
from conditional_drifting.metrics import (
    conditional_anchor_gaussian_w2,
    conditional_anchor_mean_l2,
    conditional_anchor_swd,
    sliced_wasserstein_distance,
)
from conditional_drifting.training import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare coarse split-step OptFib surrogates against the default K=50 channel.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--true-kstep", type=int, default=50)
    parser.add_argument("--ksteps", type=str, default="1,2,5,10,20,50")
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--swd-projections", type=int, default=128)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--anchor-samples", type=int, default=64)
    parser.add_argument("--anchor-swd-projections", type=int, default=64)
    parser.add_argument("--pn-dbm", type=float, default=-21.3)
    parser.add_argument("--out-dir", type=Path, default=Path("results/optfib_physics_surrogate_sweep"))
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def sample_optfib_batches(
    x: torch.Tensor,
    *,
    kstep: int,
    pn_dbm: float,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device)
        outputs.append(optfib(xb, 0.0, device, Kstep=int(kstep), Pn_dBm=float(pn_dbm)).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def anchor_metrics(
    x_anchor: torch.Tensor,
    *,
    true_kstep: int,
    candidate_kstep: int,
    pn_dbm: float,
    device: torch.device,
    samples_per_anchor: int,
    swd_projections: int,
    seed: int,
) -> dict[str, float]:
    y_true_a = []
    y_true_b = []
    y_candidate = []
    for anchor in x_anchor.to(device):
        x_rep = anchor.unsqueeze(0).repeat(samples_per_anchor, 1)
        y_true_a.append(optfib(x_rep, 0.0, device, Kstep=true_kstep, Pn_dBm=pn_dbm))
        y_true_b.append(optfib(x_rep, 0.0, device, Kstep=true_kstep, Pn_dBm=pn_dbm))
        y_candidate.append(optfib(x_rep, 0.0, device, Kstep=candidate_kstep, Pn_dBm=pn_dbm))
    y_true = torch.stack(y_true_a, dim=0).cpu()
    y_floor = torch.stack(y_true_b, dim=0).cpu()
    y_pred = torch.stack(y_candidate, dim=0).cpu()
    anchor_y_swd = conditional_anchor_swd(y_true, y_pred, num_projections=swd_projections, seed=seed)
    anchor_floor_swd = conditional_anchor_swd(y_true, y_floor, num_projections=swd_projections, seed=seed + 10_000)
    return {
        "anchor_y_swd": anchor_y_swd,
        "anchor_y_floor_swd": anchor_floor_swd,
        "anchor_y_excess_swd": max(anchor_y_swd - anchor_floor_swd, 0.0),
        "anchor_mean_l2": conditional_anchor_mean_l2(y_true, y_pred),
        "anchor_gaussian_w2": conditional_anchor_gaussian_w2(y_true, y_pred),
        "anchor_gaussian_w2_floor": conditional_anchor_gaussian_w2(y_true, y_floor),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    ksteps = parse_int_list(args.ksteps)
    rows: list[dict[str, object]] = []

    x_eval = torch.randn(args.eval_size, 2)
    x_anchor = torch.randn(args.anchor_count, 2)
    y_true = sample_optfib_batches(
        x_eval,
        kstep=args.true_kstep,
        pn_dbm=args.pn_dbm,
        device=device,
        batch_size=args.batch_size,
    )
    y_floor = sample_optfib_batches(
        x_eval,
        kstep=args.true_kstep,
        pn_dbm=args.pn_dbm,
        device=device,
        batch_size=args.batch_size,
    )
    floor_swd = sliced_wasserstein_distance(
        y_true,
        y_floor,
        num_projections=args.swd_projections,
        seed=args.seed + 99,
    )

    for kstep in ksteps:
        start = time.perf_counter()
        y_candidate = sample_optfib_batches(
            x_eval,
            kstep=kstep,
            pn_dbm=args.pn_dbm,
            device=device,
            batch_size=args.batch_size,
        )
        direct_swd = sliced_wasserstein_distance(
            y_true,
            y_candidate,
            num_projections=args.swd_projections,
            seed=args.seed,
        )
        anchor = anchor_metrics(
            x_anchor,
            true_kstep=args.true_kstep,
            candidate_kstep=kstep,
            pn_dbm=args.pn_dbm,
            device=device,
            samples_per_anchor=args.anchor_samples,
            swd_projections=args.anchor_swd_projections,
            seed=args.seed + 1000,
        )
        row = {
            "candidate": f"OptFibK{kstep}",
            "kstep": kstep,
            "true_kstep": args.true_kstep,
            "pn_dbm": args.pn_dbm,
            "direct_swd": direct_swd,
            "direct_floor_swd": floor_swd,
            "direct_excess_swd": max(direct_swd - floor_swd, 0.0),
            "elapsed_seconds": time.perf_counter() - start,
            **anchor,
        }
        rows.append(row)
        write_csv(args.out_dir / "optfib_physics_surrogate_sweep.csv", rows)
        (args.out_dir / "optfib_physics_surrogate_sweep.json").write_text(
            json.dumps({"args": vars(args), "device": str(device), "rows": rows}, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(row, indent=2), flush=True)

    print(
        json.dumps(
            {
                "csv": str((args.out_dir / "optfib_physics_surrogate_sweep.csv").resolve()),
                "summary": str((args.out_dir / "optfib_physics_surrogate_sweep.json").resolve()),
                "num_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
