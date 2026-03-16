# Repository Changes and Baseline Additions

This note documents the structural changes made to the standalone repository and explains how diffusion and GAN baselines are included alongside the conditional drifting model.

## 1. Main Goal of the New Repository

The standalone repository was created to separate the new conditional drifting work from the older mixed codebase.

The design goal is:

- keep the drifting implementation clean and central
- keep the channel models reusable
- keep benchmark scripts runnable on a separate GPU machine
- allow diffusion and GAN comparisons without pulling the old repository structure back in

For that reason, the repository is organized around a **clean drifting core**, with **optional baseline modules** added in a separate namespace.

## 2. Core Repository Structure

The main package is:

- `conditional_drifting/`

The core drifting components are:

- `conditional_drifting/model.py`
  - one-shot conditional residual generator
  - implements `e_hat = g_theta(x, z)`
- `conditional_drifting/losses.py`
  - kernel attraction-repulsion drift field
  - drifting regression objective
- `conditional_drifting/training.py`
  - training loop
  - evaluation loop
  - explicit device selection
  - seed handling
- `conditional_drifting/channels.py`
  - `AWGN`
  - `Rayleigh`
  - `SSPA`
  - `OptFib`
- `conditional_drifting/metrics.py`
  - sliced Wasserstein distance in residual space
- `conditional_drifting/benchmark.py`
  - thin benchmark wrapper for a single channel

This is the part of the repository that should be described as the main proposed method in a new paper.

## 3. Why Diffusion and GAN Were Added Separately

The repository started as a drifting-only standalone project.

To support fair comparisons, diffusion and GAN were added as **optional reference baselines**, but they were deliberately not mixed into the core drifting code. The reason is architectural clarity:

- the main contribution is drifting
- diffusion and GAN are comparison methods
- separating them makes the repository easier to maintain and easier to describe in the paper

The baseline namespace is:

- `conditional_drifting/baselines/`

## 4. Diffusion Baseline Changes

The diffusion baseline was added in:

- `conditional_drifting/baselines/diffusion.py`

This file includes:

- `DiffusionConfig`
  - benchmark and training hyperparameters
- `ConditionalDiffusionMLP`
  - compact conditional epsilon-prediction MLP
  - input: noisy residual, timestep embedding, channel input `x`
- `SinusoidalTimeEmbedding`
  - standard timestep embedding for conditional denoising
- `cosine_beta_schedule`
  - diffusion noise schedule
- `train_conditional_diffusion(...)`
  - diffusion training loop in residual space
- `sample_ddpm(...)`
  - stochastic full-step DDPM sampling
- `sample_ddim(...)`
  - deterministic DDIM-style sampling with configurable trajectory length
- `evaluate_diffusion_model(...)`
  - SWD evaluation in residual space

### Important note

This diffusion module is a **clean reference implementation** written for the standalone repo. It is not a strict bit-for-bit copy of the older diffusion code from the original mixed repository.

That means:

- it is suitable for clean comparisons inside this new repo
- it is suitable for publication figures if we report it honestly as the repo baseline
- it is **not** guaranteed to reproduce exactly the same numbers as the older legacy scripts

If exact historical replication is required, the old diffusion code would need to be ported more literally.

## 5. GAN Baseline Changes

The GAN baseline was added in:

- `conditional_drifting/baselines/gan.py`

This file includes:

- `GANConfig`
  - hyperparameters for generator/discriminator training
- `ConditionalGANGenerator`
  - conditional one-shot generator for residual samples
- `ConditionalGANDiscriminator`
  - discriminator on `(residual, condition)` pairs
- `train_conditional_gan(...)`
  - adversarial training loop with BCE loss
  - includes an L1 stabilization term on residuals
- `evaluate_gan_model(...)`
  - residual-space SWD evaluation

### Important note

As with the diffusion module, this GAN implementation is a **clean standalone reference baseline**. It is not the exact original GAN architecture from the previous repository lineage.

So the same caveat applies:

- good for controlled comparisons in the clean repo
- not a guaranteed exact reproduction of older reported GAN numbers

If we later want a stricter apples-to-apples comparison against a specific historical GAN, we should port that architecture explicitly into a separate baseline file.

## 6. New Comparison Script

A user-facing comparison script was added in:

- `scripts/compare_optional_baselines.py`

This script trains and compares on a single selected channel:

- drifting
- diffusion DDPM
- diffusion DDIM
- GAN

It writes:

- a JSON summary
- a bar plot of residual SWD

This script is mainly intended for:

- smoke checks
- quick local comparisons
- small benchmark figures before the large multi-seed GPU run

## 7. Publication Benchmark Script

The main multi-seed drifting runner remains:

- `scripts/run_publication_benchmark.py`

This is the recommended script for the GPU machine when the main goal is publication-quality drifting results.

It provides:

- explicit `--device` control
- multi-seed sweeps
- per-seed JSON/CSV outputs
- summary CSV/JSON outputs
- confidence intervals via bootstrap
- a summary plot

This script currently focuses on the drifting method itself.

If we later want a full publication benchmark that includes drifting, diffusion, and GAN together across many seeds, the cleanest extension would be:

- add a second script such as `scripts/run_full_multimodel_benchmark.py`
- keep `run_publication_benchmark.py` drifting-only
- avoid making the main drifting workflow depend on all baselines

## 8. Tests and CI Changes

To make the new repo dependable, the following were added:

### Tests

- `tests/test_metrics.py`
  - SWD sanity checks
- `tests/test_channels.py`
  - channel output shape checks
- `tests/test_smoke_training.py`
  - one-epoch drifting train/eval smoke test
- `tests/test_baselines.py`
  - diffusion/GAN baseline shape and schedule smoke tests

### CI

- `.github/workflows/ci.yml`

This installs the package with dev dependencies and runs `pytest`.

## 9. Packaging and Environment Changes

The standalone repo also includes:

- `pyproject.toml`
- `requirements.txt`
- `pytest.ini`
- `.gitignore`

These changes make the repo independent from the old workspace layout.

In particular:

- it can be cloned and installed on another machine directly
- it has its own editable install path
- it has its own test setup
- the scripts set a repo-local `MPLCONFIGDIR` automatically to avoid Matplotlib cache warnings

## 10. Suggested Usage on the GPU Machine

### Drifting-only publication run

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

### Quick drifting vs baseline check on one channel

```bash
python scripts/compare_optional_baselines.py \
  --device cuda:0 \
  --channel AWGN \
  --dataset-size 120000 \
  --epochs 60 \
  --eval-size 20000 \
  --batch-size 512 \
  --num-steps 100 \
  --ddim-steps 20
```

## 11. Recommended Next Step

If the intention is to make this new repository the main project for the paper, then the next recommended structural step is:

- keep `conditional_drifting/` as the method package
- keep `conditional_drifting/baselines/` as comparison code only
- add one dedicated large-scale benchmark script for all methods and all channels
- store final benchmark outputs in a separate `results/publication_*` folder
- freeze benchmark settings before running on the workhorse GPU machine

That gives us a repository that is both paper-friendly and operationally clean.
