# Conditional Drifting Models for Channel Learning

This repository is a clean standalone implementation of conditional drifting models for learning stochastic communication channels.

The code models the channel residual
`e = y - x`
with a conditional one-shot generator
`e_hat = g_theta(x, z)`
where `x` is the channel input and `z ~ N(0, I)` is latent noise.
Training follows a drifting objective based on attraction-repulsion updates in residual space.

## Included channels

- `AWGN`
- `Rayleigh`
- `SSPA`
- `OptFib`

## Repository layout

- `conditional_drifting/channels.py`: channel models
- `conditional_drifting/model.py`: conditional drifting generator
- `conditional_drifting/losses.py`: kernel drift field and drifting loss
- `conditional_drifting/training.py`: training and evaluation utilities
- `conditional_drifting/metrics.py`: sliced Wasserstein distance
- `conditional_drifting/benchmark.py`: single-channel benchmark entry point
- `conditional_drifting/baselines/`: optional diffusion and GAN reference baselines
- `paper/`: current paper draft snapshot, bibliography, notes, and figure assets
- `scripts/quickstart_awgn.py`: small AWGN demo with figure
- `scripts/run_publication_benchmark.py`: multi-seed benchmark runner
- `scripts/compare_optional_baselines.py`: drifting vs optional baselines on one channel

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

The editable install is recommended on a new machine so `conditional_drifting` and the test suite resolve against this checkout directly.

## Quick start

```bash
python scripts/quickstart_awgn.py --device auto
```

This writes a small figure and JSON summary into `results/`.

## Multi-seed publication run

```bash
python scripts/run_publication_benchmark.py \
  --device cuda:0 \
  --num-seeds 10 \
  --seed-start 7 \
  --channels AWGN,Rayleigh,SSPA,OptFib \
  --dataset-size 120000 \
  --epochs 60 \
  --eval-size 20000 \
  --batch-size 512
```

Outputs:

- `per_seed_results.json`
- `per_seed_results.csv`
- `summary.csv`
- `summary_plot.png`
- `run_config.json`

## HPC / SLURM

For cluster runs, see [hpc/README.md](hpc/README.md). The repo now includes:

- a SLURM array script for one full-budget seed per job,
- a separate aggregation script for combining finished seeds into one suite summary,
- per-seed JSON/log artifacts designed for shared HPC output directories.

## Notes

- The drifting package is the core of the repository; diffusion and GAN are included only as optional reference baselines under `conditional_drifting/baselines/`.
- The diffusion and GAN baseline modules are written as clean standalone ports of the legacy benchmark math so the standalone repo can preserve comparison behavior without depending on the old workspace layout.
- Device handling is explicit and robust: scripts accept `--device auto|cpu|cuda|cuda:0|...`, and the model samples on the actual parameter device.
- The default benchmark configuration matches the recent channel-learning experiments used in this workspace.
- A small GitHub Actions workflow is included under `.github/workflows/ci.yml`.
- Repository-level change notes, including the baseline additions, are documented in `REPO_CHANGES.md`.
