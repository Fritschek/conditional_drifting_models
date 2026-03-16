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
- `scripts/quickstart_awgn.py`: small AWGN demo with figure
- `scripts/run_publication_benchmark.py`: multi-seed benchmark runner
- `scripts/compare_optional_baselines.py`: drifting vs optional baselines on one channel

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

## Notes

- The drifting package is the core of the repository; diffusion and GAN are included only as optional reference baselines under `conditional_drifting/baselines/`.
- Device handling is explicit and robust: scripts accept `--device auto|cpu|cuda|cuda:0|...`, and the model samples on the actual parameter device.
- The default benchmark configuration matches the recent channel-learning experiments used in this workspace.
- A small GitHub Actions workflow is included under `.github/workflows/ci.yml`.
