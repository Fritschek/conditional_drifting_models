from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILES_TO_COPY = [
    "conditional_drifting/__init__.py",
    "conditional_drifting/channels.py",
    "conditional_drifting/losses.py",
    "conditional_drifting/metrics.py",
    "conditional_drifting/model.py",
    "conditional_drifting/paper2309_presets.py",
    "conditional_drifting/progress.py",
    "conditional_drifting/training.py",
    "scripts/run_enhanced_direct_benchmark.py",
]


README_TEXT = """# Enhanced Direct HPC Suite

This bundle contains the minimum code needed to run the enhanced direct drifting benchmark row
for `AWGN`, `Rayleigh`, and `SSPA` without the broader exploratory harness.

## Environment

Python dependencies:

- torch
- numpy
- tqdm

## Single-seed example

```bash
python scripts/run_enhanced_direct_benchmark.py \\
  --device cuda \\
  --seed 7 \\
  --channels AWGN,Rayleigh,SSPA \\
  --drifting-epochs 20 \\
  --eval-size 500000 \\
  --swd-projections 64 \\
  --conditioning-mode joint \\
  --condition-kernel-scale 0.5 \\
  --target-kernel-scale 1.0 \\
  --target-kernel-mode raw \\
  --save-dir results/checkpoints \\
  --out results/enhanced_direct_seed7.json
```

## Seed sweep example

```bash
for seed in 7 8 9 10 11 12 13 14 15 16; do
  python scripts/run_enhanced_direct_benchmark.py \\
    --device cuda \\
    --seed "$seed" \\
    --channels AWGN,Rayleigh,SSPA \\
    --drifting-epochs 20 \\
    --eval-size 500000 \\
    --swd-projections 64 \\
    --conditioning-mode joint \\
    --condition-kernel-scale 0.5 \\
    --target-kernel-scale 1.0 \\
    --target-kernel-mode raw \\
    --save-dir results/checkpoints \\
    --out "results/enhanced_direct_seed${seed}.json"
done
```

Each channel checkpoint is saved immediately after training, before SWD evaluation.
"""


REQUIREMENTS_TEXT = """numpy
tqdm
torch
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a compact HPC bundle for the enhanced direct benchmark.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "dist" / "enhanced_direct_hpc_suite"),
        help="Destination directory for the compact bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for relative in FILES_TO_COPY:
        src = ROOT / relative
        dst = out_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (out_dir / "README.md").write_text(README_TEXT, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(REQUIREMENTS_TEXT, encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
