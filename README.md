# Condition-Wise Sinkhorn Drifting: Compact Code Release

This is a compact, readable implementation of the main drifting models from
"Conditional Drifting Models for Learned Channel Simulation".

It is intentionally smaller than the research workspace. It keeps only the
paper channels and the three generator-level drifting variants:

- direct kernel drifting on channel outputs;
- joint Sinkhorn drifting on `(x,y)`;
- condition-wise Sinkhorn drifting on repeated samples of `p(y | x)`.

The constants used by the compact runner are defined once in
`conditional_drifting/config.py`. The implementation avoids experiment-history
switches, cluster helpers, saved results, paper build files, and trained
weights.

## Example Run

```bash
python scripts/run_drifting_benchmark.py \
  --channel Rayleigh \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 30 \
  --dataset-size 1000000 \
  --eval-size 200000 \
  --batch-size 5000 \
  --out results/rayleigh_conditionwise_seed7.json \
  --checkpoint results/rayleigh_conditionwise_seed7.pt
```

For multi-seed experiments, launch the same command over seeds and channels
with your scheduler of choice. This compact release deliberately does not
include the Slurm scripts or the heavier downstream autoencoder pipeline.

## Copyable Runs

The full model/channel matrix can be checked with:

```bash
python scripts/run_quick_model_checks.py \
  --device cuda \
  --seed 7 \
  --epochs 2 \
  --dataset-size 2048 \
  --eval-size 2048 \
  --batch-size 128 \
  --out results/quick_model_checks.json
```

A medium-budget AWGN run can be reproduced with:

```bash
python scripts/run_drifting_benchmark.py \
  --channel AWGN \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 30 \
  --dataset-size 1000000 \
  --eval-size 200000 \
  --batch-size 5000 \
  --out results/awgn_conditionwise_seed7.json
```

For the larger generator-level paper settings, use the same parser knobs with
the following channel-specific budgets:

```bash
# AWGN and Rayleigh
python scripts/run_drifting_benchmark.py \
  --channel AWGN \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 30 \
  --dataset-size 10000000 \
  --eval-size 10000000 \
  --batch-size 5000 \
  --out results/awgn_conditionwise_paper_seed7.json

# SSPA
python scripts/run_drifting_benchmark.py \
  --channel SSPA \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 160 \
  --dataset-size 10000000 \
  --eval-size 10000000 \
  --batch-size 4096 \
  --out results/sspa_conditionwise_paper_seed7.json

# TDL
python scripts/run_drifting_benchmark.py \
  --channel TDL \
  --variant conditionwise_sinkhorn \
  --device cuda \
  --seed 7 \
  --epochs 60 \
  --dataset-size 120000 \
  --eval-size 100000 \
  --batch-size 512 \
  --out results/tdl_conditionwise_paper_seed7.json
```

## AWGN Check

For AWGN, the compact condition-wise Sinkhorn implementation moves toward the
paper-scale direct-output SWD as the training budget is increased. With seed 7
and the paper AWGN preset, we observed:

- 120k training samples, 60 epochs, batch size 512: SWD `1.55e-2`;
- 1M training samples, 30 epochs, batch size 5000: SWD `9.91e-3`;
- 10M training samples, 30 epochs, batch size 5000: SWD `7.95e-3`.

A lightweight downstream check is provided in
`scripts/run_awgn_ser_ber_check.py`. With the medium surrogate budget above and
a 10-epoch symbolic autoencoder check, seed 7 gave analytic-channel evaluation
SER/BER `9.22e-3 / 5.19e-3` after analytic-channel training and
`9.36e-3 / 5.11e-3` after compact-surrogate training. This is only a quick
consistency check; the full paper uses the larger downstream evaluation suite.

## License

MIT License. See `LICENSE`.
