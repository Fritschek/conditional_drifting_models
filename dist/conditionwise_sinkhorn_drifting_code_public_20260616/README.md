# Condition-Wise Sinkhorn Drifting for Learned Channel Simulation

This repository contains the minimal source code release for the paper
"Conditional Drifting Models for Learned Channel Simulation".

The code implements one-shot conditional drifting generators for learned channel
simulation, including the direct drifting baseline, joint Sinkhorn-style
drifting, and condition-wise Sinkhorn drifting. It also includes the reference
WGAN and diffusion implementations used for the paper experiments.

The main channel models used by the journal manuscript are:

- AWGN
- Rayleigh fading without receiver-side channel-state equalization
- SSPA/Rapp nonlinearity
- compact TDL fading channel

The source tree intentionally excludes generated results, logs, trained weights,
paper build artifacts, and cluster-specific Slurm helpers.

## Layout

- `conditional_drifting/`: importable Python package
- `conditional_drifting/channels.py`: analytic channel models
- `conditional_drifting/model.py`: conditional one-shot generator
- `conditional_drifting/losses.py`: drifting and Sinkhorn-style drift losses
- `conditional_drifting/training.py`: training, evaluation, and metric helpers
- `conditional_drifting/baselines/`: WGAN and diffusion baselines
- `conditional_drifting/symbolic_ae.py`: symbolic autoencoder experiments
- `conditional_drifting/e2e_implants.py`: learned channel implants
- `scripts/`: reproducibility and plotting entry points
- `tests/`: smoke and unit tests

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

For GPU runs, install a PyTorch build that matches the local CUDA driver before
installing the package.

## Quick Smoke Test

```bash
python scripts/quickstart_awgn.py \
  --device cpu \
  --dataset-size 2000 \
  --epochs 2 \
  --batch-size 256 \
  --eval-size 1000
```

This writes a small JSON summary and diagnostic figure to `results/`.

Run the test suite with:

```bash
pytest -q
```

## Main Reproduction Entry Points

Train and evaluate W-Flow/drifting variants:

```bash
python scripts/run_enhanced_direct_benchmark.py \
  --device cuda \
  --seed 7 \
  --channels AWGN,Rayleigh,SSPA,TDL \
  --drift-field fiber_sinkhorn \
  --conditioning-mode none \
  --anchor-metrics \
  --out results/example_wflow_seed7.json
```

Aggregate per-seed W-Flow runs:

```bash
python scripts/aggregate_journal_wflow_suite.py \
  --suite-dir results/YOUR_WFLOW_SUITE \
  --seed-start 7 \
  --num-seeds 30 \
  --variants kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn \
  --channels AWGN,Rayleigh,SSPA,TDL
```

Train reference WGAN or diffusion channel implants:

```bash
python scripts/run_journal_baseline_implant_seed_channel.py \
  --device cuda \
  --seed 7 \
  --channel AWGN \
  --variants wgan,diffusion \
  --out-dir results/example_baselines
```

Run downstream symbolic SER/BER evaluation from trained implants:

```bash
python scripts/run_journal_wflow_ser_seed_channel.py \
  --device cuda \
  --seed 7 \
  --channel AWGN \
  --variants analytic,fiber_sinkhorn \
  --wflow-suite-dir results/YOUR_WFLOW_SUITE \
  --out results/example_ser_seed7.json
```

The full paper-scale sweeps are GPU-expensive. The scripts above are intended
to be launched repeatedly over seeds/channels by the user's local scheduler or
cluster tooling.

## Outputs

All scripts write generated artifacts below `results/` unless another output
directory is provided. The release archive does not contain trained checkpoints
or result CSV/JSON files.

## License

This code is released under the MIT License; see `LICENSE`.
