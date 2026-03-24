# Enhanced Direct HPC Suite

This bundle contains the minimum code needed to run the enhanced direct drifting benchmark row
for `AWGN`, `Rayleigh`, and `SSPA` without the broader exploratory harness.

## Environment

Python dependencies:

- torch
- numpy
- tqdm

## Single-seed example

```bash
python scripts/run_enhanced_direct_benchmark.py \
  --device cuda \
  --seed 7 \
  --channels AWGN,Rayleigh,SSPA \
  --drifting-epochs 20 \
  --eval-size 500000 \
  --swd-projections 64 \
  --conditioning-mode joint \
  --condition-kernel-scale 0.5 \
  --target-kernel-scale 1.0 \
  --target-kernel-mode raw \
  --save-dir results/checkpoints \
  --out results/enhanced_direct_seed7.json
```

## Seed sweep example

```bash
for seed in 7 8 9 10 11 12 13 14 15 16; do
  python scripts/run_enhanced_direct_benchmark.py \
    --device cuda \
    --seed "$seed" \
    --channels AWGN,Rayleigh,SSPA \
    --drifting-epochs 20 \
    --eval-size 500000 \
    --swd-projections 64 \
    --conditioning-mode joint \
    --condition-kernel-scale 0.5 \
    --target-kernel-scale 1.0 \
    --target-kernel-mode raw \
    --save-dir results/checkpoints \
    --out "results/enhanced_direct_seed${seed}.json"
done
```

Each channel checkpoint is saved immediately after training, before SWD evaluation.
