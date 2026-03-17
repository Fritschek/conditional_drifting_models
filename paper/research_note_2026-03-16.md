# Research Note: Drifting, Diffusion, GAN, and OptFib Extension

Date: 2026-03-16

Repository: `/Users/rickfritschek/Documents/GitHub/DM_for_learning_channels`

Working branch: `codex/drifting-models-learning-channels`

## 1. Scope of the extension

This note documents the code changes and benchmark infrastructure added on top of the original `DM_for_learning_channels` repository.

The extension has four goals:

1. Add a conditional drifting-model baseline for learned channel simulation.
2. Compare drifting against diffusion under matched residual-space evaluation.
3. Add optical fiber (`OptFib`) and a stronger GAN baseline inspired by `muahkim/DM_OptFib`.
4. Provide a publication-grade multi-seed runner for `T=100` diffusion experiments.

The current benchmarked channel families are:

- `AWGN`
- `Rayleigh`
- `SSPA`
- `OptFib` (simplified memoryless optical-fiber model)

The current benchmarked generative models are:

- `DDPM(T)`
- `DDIM(T)`
- optionally `DDIM(K < T)` for fast sampling
- `Drifting`
- `WGAN` / `GAN_FA`-style conditional GAN baseline


## 2. File-level change log

### Core model and channel additions

- `src/drifting.py`
  - Added conditional drifting generator.
  - Added attraction-repulsion kernel drift objective.
  - Added channel pretraining loop and AE training loop using the drifting generator.

- `src/channel_models.py`
  - Added optional `scipy` fallback handling.
  - Added `ch_OptFib(...)`, a simplified optical-fiber channel compatible with the repo's `(x, noise_std, device)` interface.

- `src/trainer.py`
  - Added/kept fallback handling for environments without `tqdm`.
  - Used as the diffusion training backend for the new benchmarks.

### Experiment and benchmarking scripts

- `examples/drifting_awgn_quickstart.py`
  - Minimal drifting-model quickstart.

- `examples/compare_awgn_drifting_vs_diffusion.py`
  - Early AWGN-only comparison.

- `examples/compare_awgn_diffusion_ddpm_ddim_tuned.py`
  - Tuned AWGN diffusion comparison.

- `examples/compare_awgn_ddpm_ddim_drifting_corrected.py`
  - Corrected AWGN comparison using residual-space drifting.

- `examples/benchmark_awgn_rayleigh_sspa_swd.py`
  - Main benchmark script for channel-level SWD comparison.
  - Originally covered `AWGN`, `Rayleigh`, `SSPA`.
  - Later extended to:
    - add `OptFib`
    - add a stronger GAN baseline
    - add seed control
    - optionally suppress the fast-DDIM bar from figures for cleaner publication plots

- `examples/time_inference_awgn_drifting_vs_ddpm100.py`
  - CPU timing comparison of one-shot drifting vs `DDPM(100)`.

- `examples/publication_benchmark_t100.py`
  - New publication-grade wrapper for multi-seed `T=100` runs.
  - Produces per-seed outputs, aggregate statistics, pairwise win tables, LaTeX tables, and publication-ready figures.

### Documentation and paper material

- `README.md`
  - Added a drifting-model extension section.

- `results/drifting_vs_diffusion_summary.tex`
- `results/drifting_vs_diffusion_summary.pdf`
- `results/references.bib`
- `results/drifting_vs_diffusion_schematic.png`
  - Added a short IEEE-style draft summary and supporting figure.


## 3. Current benchmark architecture

### 3.1 Common evaluation setup

The current comparison is residual-based.

For a channel input `x` and output `y`, we define the residual:

`e = y - x`

All methods are evaluated by comparing the generated residual distribution against the true channel residual distribution using sliced Wasserstein distance (SWD). Smaller is better.

This choice is deliberate:

- it avoids visual ambiguity from scatter-only plots
- it makes AWGN, Rayleigh, SSPA, and OptFib directly comparable
- it aligns diffusion, drifting, and GAN outputs on the same target object

The benchmark code for this is in:

- `examples/benchmark_awgn_rayleigh_sspa_swd.py`


### 3.2 Diffusion model

#### Model form

The diffusion backbone is the conditional fully connected model:

- `src/models.py`
- class: `ConditionalModel_w_Condition`

Architecture:

- input to first conditional layer: concatenation of current diffusion state and channel condition
- hidden width: typically `N = 128` in the benchmark scripts
- three hidden stages:
  - `lin1 = ConditionalLinear_w_Condition(2*n + M, N, n_steps)`
  - `lin2 = ConditionalLinear(N, N, n_steps)`
  - `lin3 = ConditionalLinear(N, N, n_steps)`
- output head:
  - `lin4 = Linear(N, n)`

Nonlinearity:

- `softplus` after each hidden conditional layer

Time conditioning:

- each hidden layer uses an embedding indexed by diffusion step `t`
- that step embedding multiplicatively modulates the linear output

In the benchmark path, the model is used with:

- `M = 0`
- `n = 2`
- `pred_type = "epsilon"`
- residual training enabled (`IS_RES = True`)

That means the model is trained to predict diffusion noise on the residual target rather than on the absolute channel output.

#### Training

Backend:

- `src/trainer.py`
- class: `Trainer_DDM`

Key benchmark settings:

- beta schedule: cosine
- `T = 100` for publication-grade diffusion
- objective: epsilon-prediction
- EMA used after training

Training loop:

1. sample Gaussian channel input `x`
2. pass through the true channel to obtain `y`
3. convert to residual if `IS_RES = True`
4. build `yx = [y_residual, x]`
5. sample a diffusion time step
6. optimize the denoiser with `utils.noise_estimation_loss(...)`

#### Sampling

Two samplers are used:

- `DDPM(T)`:
  - full reverse stochastic chain
- `DDIM(T)`:
  - deterministic full trajectory

An optional fast path also exists:

- `DDIM(K)`
  - uses a reduced step trajectory, e.g. `K = 20`

For publication plots we currently suppress the fast-DDIM bar by default because it dominates axis scaling and makes the main comparison harder to read.


### 3.3 Drifting model

#### Model form

The drifting generator is defined in:

- `src/drifting.py`
- class: `ConditionalDriftingGenerator`

Architecture:

- input: `[condition, latent]`
- latent: Gaussian `z ~ N(0, I)`
- default latent dimension: `16`
- hidden width: `128`
- network:
  - `Linear(condition_dim + latent_dim, hidden_dim)`
  - `SiLU`
  - `Linear(hidden_dim, hidden_dim)`
  - `SiLU`
  - `Linear(hidden_dim, output_dim)`

In the current channel benchmarks:

- `condition_dim = 2`
- `output_dim = 2`
- the drifting generator predicts the residual directly

#### Objective

The drifting target field is implemented in:

- `src/drifting.py`
- `compute_kernel_drift(...)`
- `drifting_loss(...)`

The implemented objective is a practical attraction-repulsion variant:

1. For each generated sample, compute an RBF-weighted barycenter over true samples.
2. Use the vector from generated sample to barycenter as an attractive drift.
3. Add a repulsive correction computed from neighboring generated samples to avoid mode collapse.
4. Form a detached target:
   - `target = generated + drift_scale * V`
5. Optimize:
   - `MSE(generated, target)`

This is not a full reproduction of every drifting-model variant from the paper. It is a practical conditional channel-learning adaptation.

#### Benchmark hyperparameters

The current corrected benchmark uses:

- `drift_scale = 1.0`
- `repulsive_weight = 1.0`
- `min_bandwidth = 0.2`
- `max_drift_norm = 2.0`

These settings are the ones used in the corrected AWGN comparison and the later multi-channel benchmark.

#### Inference

Drifting is one-shot at inference:

- one forward pass through `g_theta(x, z)`
- no reverse-time denoising trajectory

This is the basis for the large inference-time advantage relative to diffusion.


### 3.4 GAN baseline

The original repo already contained WGAN-oriented code, but the final benchmark GAN baseline was upgraded to be closer to the `DM_OptFib` setup.

The current benchmark implementation lives inside:

- `examples/benchmark_awgn_rayleigh_sspa_swd.py`

#### Generator

Class:

- `GeneratorBN`

Structure:

- input: `[x, z]`
- output: `y`
- hidden width: `128`
- one input linear layer
- two batch-normalized hidden layers (`hidden_layers1`)
- one central hidden layer (`hidden_layer2`)
- two more batch-normalized hidden layers (`hidden_layers3`)
- LeakyReLU activations
- residual connection from the condition input `x` to the output

Interpretation:

- the generator learns a residual-style perturbation around the input symbol while still outputting the full channel output

#### Discriminator

Class:

- `DiscriminatorRes`

Structure:

- input: `[y, x]`
- one input linear layer
- three hidden layers
- concatenate the original input back into the hidden representation
- three additional compression layers
- scalar output

This is again close in spirit to the `DM_OptFib` residual discriminator design.

#### GAN training modes

Two modes exist:

- `wgan_gp`
- `gan_fa`

The default and recommended mode is:

- `gan_fa`

`gan_fa` uses:

- `BCEWithLogitsLoss`
- label smoothing on real and fake logits
- delayed generator updates
- an additional `L1` alignment term between generated and real channel outputs

This was introduced because the plain WGAN-GP baseline underperformed badly in the initial channel benchmarks.

The stronger `gan_fa` version materially improved the GAN results, especially on `SSPA`.


### 3.5 Channel models

#### AWGN

- `src/channel_models.py`
- `ch_AWGN(...)`

Output:

- `y = x + n`

#### Rayleigh

- `src/channel_models.py`
- `ch_Rayleigh_AWGN(...)`

Output:

- elementwise Rayleigh fading times `x` plus additive Gaussian noise

#### SSPA

- `src/channel_models.py`
- `ch_SSPA(...)`

Output:

- nonlinear AM/AM compression using the smooth SSPA amplitude law
- plus additive Gaussian noise

#### OptFib

- `src/channel_models.py`
- `ch_OptFib(...)`

This is a simplified memoryless optical-fiber proxy ported into the local benchmark interface.

Mechanism:

1. reshape input into I/Q pairs
2. iterate `Kstep` times
3. rotate each I/Q pair by a power-dependent nonlinear phase
4. inject per-step Gaussian noise

Default parameters:

- `L = 5000`
- `gamma = 1.27`
- `Kstep = 50`
- `Pn_dBm = -21.3`


## 4. Current benchmark and plotting stack

### Main channel benchmark

Primary script:

- `examples/benchmark_awgn_rayleigh_sspa_swd.py`

Capabilities:

- channels:
  - `AWGN`
  - `Rayleigh`
  - `SSPA`
  - `OptFib`
- methods:
  - `DDPM(T)`
  - `DDIM(T)`
  - optional `DDIM(K)`
  - `Drifting`
  - `WGAN`
- flags:
  - `--seed`
  - `--channels`
  - `--plot-fast-ddim`
  - `--gan-mode`

Current plotting default:

- only `DDPM(T)`, `DDIM(T)`, `Drifting`, and `WGAN`
- fast `DDIM(K)` is omitted from the final bar plot unless explicitly requested

### Publication runner

Publication script:

- `examples/publication_benchmark_t100.py`

Capabilities:

- multi-seed execution
- aggregate mean / std / 95% bootstrap CI
- pairwise wins
- CSV / JSON / LaTeX output
- linear and log-scale summary figures
- per-seed boxplots


## 5. Key result artifacts currently in the repo

### Single-run benchmark outputs

- `results/benchmark_channels_swd_gan_t20_ddim10_gan_fa_ge60.json`
- `results/benchmark_channels_swd_gan_t20_ddim10_gan_fa_ge60.png`
- `results/benchmark_channels_swd_gan_t100_ddim20_gan_fa_ge60.json`
- `results/benchmark_channels_swd_gan_t100_ddim20_gan_fa_ge60.png`
- `results/benchmark_channels_swd_gan_t100_ddim20_gan_fa_ge60_no20plot.png`

### Timing output

- `results/timing_ddpm100_vs_drifting.txt`

Recorded CPU timing:

- `DDPM(100)` mean repeat time: `4.261206 s`
- `Drifting(one-shot)` mean repeat time: `0.016339 s`
- measured speedup: `260.80x`

### Sanity and reproducibility outputs

- `results/sanity_repro_run1.json`
- `results/sanity_repro_run2.json`
- `results/sanity_seed_sweep_summary.txt`

### Publication runner smoke output

- `results/publication_t100_multiseed_20260227_161223/`


## 6. Most relevant final benchmark snapshot

The strongest single-run benchmark currently committed is:

- `results/benchmark_channels_swd_gan_t100_ddim20_gan_fa_ge60.json`

Headline SWD values:

- AWGN:
  - `DDPM(100) = 0.0055`
  - `DDIM(100) = 0.0623`
  - `Drifting = 0.0154`
  - `WGAN = 0.0713`

- Rayleigh:
  - `DDPM(100) = 0.0484`
  - `DDIM(100) = 0.1490`
  - `Drifting = 0.0192`
  - `WGAN = 0.1459`

- SSPA:
  - `DDPM(100) = 0.0180`
  - `DDIM(100) = 0.1018`
  - `Drifting = 0.0197`
  - `WGAN = 0.0324`

- OptFib:
  - `DDPM(100) = 0.1662`
  - `DDIM(100) = 0.2754`
  - `Drifting = 0.0808`
  - `WGAN = 0.2957`

Interpretation:

- `DDPM(100)` is strongest on `AWGN` and `SSPA`.
- `Drifting` is strongest on `Rayleigh` and `OptFib` in the current single-run snapshot.
- `DDIM(100)` is consistently worse than `DDPM(100)` here.
- `WGAN` improved substantially after switching to the `GAN_FA` objective, but is not the best model in the final `T=100` snapshot.


## 7. Sanity-check status

The benchmark is in good shape, but the phrase "watertight" should still be interpreted carefully.

What has been verified:

- the benchmark scripts compile
- the SWD implementation satisfies identity, symmetry, and perturbation ordering checks
- repeated identical reruns with fixed seed can produce byte-identical JSON outputs
- a small cross-seed sweep on `AWGN` and `OptFib` showed stable method ordering in that setup
- stored result JSON files were checked for missing keys / NaN / Inf

Current sanity artifacts:

- `results/sanity_repro_run1.json`
- `results/sanity_repro_run2.json`
- `results/sanity_seed_sweep_summary.txt`

Important caveat:

- short-budget smoke tests with `T=100` can produce extremely poor diffusion numbers if the training budget is too small
- this is expected and is not a code bug by itself
- the publication runner should therefore be used with the higher training budgets documented below


## 8. Commands to reproduce or continue

### 8.1 Last full local benchmark command used

This is the last exact command used to launch the improved single-run `T=100` benchmark with `GAN_FA`:

```bash
MPLCONFIGDIR=/Users/rickfritschek/Documents/GitHub/DM_for_learning_channels/.mplcache \
/Users/rickfritschek/Documents/GitHub/turbo_mingru_decoder/.venv/bin/python -u \
/Users/rickfritschek/Documents/GitHub/DM_for_learning_channels/examples/benchmark_awgn_rayleigh_sspa_swd.py \
  --num-steps 100 \
  --ddim-fast-steps 20 \
  --dataset-size 30000 \
  --epochs 20 \
  --gan-epochs 60 \
  --eval-size 5000 \
  --gan-mode gan_fa
```

### 8.2 Recommended long publication run on a better GPU machine

This is the recommended command for a publication-grade run with the new multi-seed wrapper:

```bash
MPLCONFIGDIR=/path/to/.mplcache \
python /Users/rickfritschek/Documents/GitHub/DM_for_learning_channels/examples/publication_benchmark_t100.py \
  --device cuda:0 \
  --num-seeds 12 \
  --seed-start 7 \
  --channels AWGN,Rayleigh,SSPA,OptFib \
  --num-steps 100 \
  --dataset-size 120000 \
  --epochs 60 \
  --gan-epochs 120 \
  --eval-size 20000 \
  --batch-size 512 \
  --gan-mode gan_fa
```

Recommended stronger option if compute allows:

```bash
--num-seeds 20
```

Outputs will be written into:

```text
results/publication_t100_multiseed_<timestamp>/
```

with:

- `run_config.json`
- `per_seed_results.csv`
- `summary_mean_std_ci.csv`
- `pairwise_wins.csv`
- `publication_table.tex`
- `publication_swd_mean_ci_linear.png`
- `publication_swd_mean_log.png`
- `publication_swd_boxplots.png`


## 9. Suggested paper wording anchor

If this work is turned into a new paper, the current codebase supports the following clean narrative:

- We study conditional learned channel simulation in residual space.
- We compare iterative diffusion samplers, one-shot drifting generators, and conditional GAN baselines on `AWGN`, `Rayleigh`, `SSPA`, and simplified optical-fiber channels.
- We evaluate fidelity using SWD between generated and true residual distributions.
- We separately measure inference-time cost and show the expected tradeoff between fidelity and latency.
- We report multi-seed aggregate statistics using the publication runner rather than relying on a single seed.

That is the most defensible story supported by the current implementation.


## 10. Overnight rerun commands

Full 3-seed benchmark suite:

```bash
conda run --no-capture-output -n dl python scripts/run_full_budget_suite.py \
  --device cuda \
  --num-seeds 3 \
  --seed-start 7
```

Run the timing benchmark separately after the suite, not in parallel:

```bash
conda run --no-capture-output -n dl python scripts/run_inference_timing_benchmark.py \
  --device cpu \
  --channel AWGN \
  --prepare-mode train \
  --methods drifting_residual,drifting_direct,ddpm,ddim100,ddim50,ddim10,gan \
  --warmup-repeats 2 \
  --repeats 7 \
  --batch-size 512 \
  --num-batches 40
```
