# Full-Budget Benchmark Protocol

Source suite:

- `results_hpc/20260318_203626/suite_results.json`
- `results_hpc/20260318_203626/paper2309_benchmark_seed7/paper2309_benchmark_summary.json`
- `results_hpc/20260318_203626/optfib_seed7/baseline_compare_optfib.json`

This note records the exact parameter settings used for the final 10-seed full-budget benchmark run that produced the main benchmark tables.

## 1. Suite-Level Settings

- Run identifier: `results_hpc/20260318_203626`
- Device: `cuda`
- Seeds: `7, 8, 9, 10, 11, 12, 13, 14, 15, 16`
- Number of seeds: `10`
- Execution pattern: one seed per SLURM array task, one CUDA device per seed job
- Paper-channel block: `AWGN, Rayleigh, SSPA`
- Separate optional block: `OptFib`

Equivalent per-seed runner:

```bash
python scripts/run_full_budget_seed.py \
  --device cuda \
  --seed <seed> \
  --paper-channels AWGN,Rayleigh,SSPA \
  --paper-eval-size 1000000 \
  --optfib-eval-size 100000 \
  --optfib-dataset-size 120000 \
  --optfib-epochs 60 \
  --optfib-batch-size 512 \
  --optfib-num-steps 100 \
  --suite-dir <suite_dir>
```

## 2. Methods Included In The Final Full Run

### Main Benchmark Channels: `AWGN`, `Rayleigh`, `SSPA`

- Drifting, direct-output mode: `drifting_direct_swd`
- Drifting, residual mode: `drifting_residual_swd`
- Diffusion DDPM: `ddpm_swd`
- Diffusion DDIM-100: `ddim_swd["100"]`
- Diffusion DDIM-50: `ddim_swd["50"]`
- Diffusion DDIM-20: `ddim_swd["20"]`
- Diffusion DDIM-10: `ddim_swd["10"]`
- WGAN: `paper_wgan_swd`

Excluded from the final suite:

- `GAN_FA`
- the later standalone GAN baseline

### OptFib

- Drifting, residual mode: `drifting_residual_swd`
- Drifting, direct-output mode: `drifting_direct_swd`
- Diffusion DDPM: `ddpm_swd`
- Diffusion DDIM-100 only: `ddim_swd`
- WGAN: `paper_wgan_swd`

Excluded from the final suite:

- the later standalone GAN baseline
- DDIM-50 / DDIM-20 / DDIM-10

## 3. Channel Presets (`AWGN`, `Rayleigh`, `SSPA`)

These settings come from `scripts/run_paper2309_benchmark.py`.

| Channel | `n` | `E_b/N_0` | Rate | Noise std | Dataset size | Batch size | Drifting epochs | Diffusion epochs | WGAN epochs | Eval size used | SWD proj. | Diffusion hidden dim | WGAN hidden dim | Diffusion LR schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `AWGN` | `7` | `5 dB` | `4/7` | `0.5260221433` | `10,000,000` | `5,000` | `30` | `30` | `30` | `1,000,000` | `128` | `110` | `128` | `10 @ 1e-3`, then `20 @ 1e-4` |
| `Rayleigh` | `7` | `12 dB` | `4/7` | `0.2349654605` | `10,000,000` | `5,000` | `30` | `30` | `30` | `1,000,000` | `128` | `128` | `256` | `10 @ 1e-3`, then `20 @ 1e-4` |
| `SSPA` | `8` | `8 dB` | `6/8` | `0.3250531436` | `10,000,000` | `4,096` | `160` | `160` | `160` | `1,000,000` | `128` | `110` | `256` | constant `1e-4` |

Notes:

- The preset `eval_size` inside the paper benchmark script is `10,000,000`, but the actual full suite overrode it to `1,000,000`.
- All paper-channel diffusion runs used `num_steps = 100`.
- DDIM evaluation used step counts `100, 50, 20, 10`.

## 4. Shared Drifting Configuration

Drifting was trained through `conditional_drifting.training.DriftingConfig`.

Shared drifting hyperparameters for the AWGN, Rayleigh, and SSPA block:

- Optimizer: `Adam`
- Learning rate: `1e-3`
- Latent dimension: `16`
- Hidden dimension: `128`
- Drift scale: `1.0`
- Bandwidth: `None` (median heuristic per minibatch)
- Minimum bandwidth: `0.2`
- Maximum drift norm: `2.0`
- Repulsive weight: `1.0`
- Output dimension: equals channel dimension `n`
- Architecture: conditional MLP with two hidden layers, `SiLU` activations

Drifting parameterizations included in the final suite:

- Direct mode:
  - target: `y`
  - output of `g_theta`: `\hat y`
  - `is_residual = False`
- Residual mode:
  - target: `e = y - x`
  - output of `g_theta`: `\hat e`
  - reconstruction: `\hat y = x + \hat e`
  - `is_residual = True`

Sampling/data generation:

- Input samples were generated online each minibatch as `x ~ N(0, I_n)`.
- True channel outputs were then obtained by applying the corresponding channel function.

Evaluation on AWGN, Rayleigh, and SSPA:

- Direct drifting was evaluated in output space (`y`-space).
- Residual drifting was evaluated in residual space (`e = y - x`).
- The SWD random-projection seed was set equal to the run seed.

## 5. Shared Diffusion Configuration

The diffusion baseline was trained through `conditional_drifting.baselines.DiffusionConfig`.

Shared diffusion settings on AWGN, Rayleigh, and SSPA:

- Training target: direct-output mode
- `is_residual = False`
- Prediction type: `v`
- Beta schedule: `cosine-zf`
- Number of diffusion steps: `100`
- EMA decay: `0.9`
- SWD projections: `128`
- Evaluation size used in the full suite: `1,000,000`
- `eval_batch_size = None` in the full run, i.e. one evaluation batch

Architecture:

- Conditional diffusion MLP
- Three conditional hidden layers with learned timestep embeddings
- `Softplus` activations
- Hidden dimension:
  - `110` for `AWGN`
  - `128` for `Rayleigh`
  - `110` for `SSPA`

Optimizer:

- `Adam`
- Gradient clipping: norm `1.0`

Learning-rate settings:

- `AWGN`: `10` epochs at `1e-3`, then `20` epochs at `1e-4`
- `Rayleigh`: `10` epochs at `1e-3`, then `20` epochs at `1e-4`
- `SSPA`: constant `1e-4` for `160` epochs

Evaluation:

- DDPM sampling
- DDIM sampling with `100`, `50`, `20`, and `10` steps
- Direct-output diffusion was evaluated in output space (`y`-space)
- For the paper-channel block, the SWD random-projection seed was set equal to the run seed

## 6. Shared WGAN Configuration

The WGAN baseline was trained through `conditional_drifting.baselines.PaperWGANConfig`.

Shared WGAN settings on AWGN, Rayleigh, and SSPA:

- Generator/discriminator hidden dimension:
  - `128` for `AWGN`
  - `256` for `Rayleigh`
  - `256` for `SSPA`
- Generator learning rate: `1e-4`
- Critic learning rate: `1e-4`
- Critic steps per generator step: `5`
- Weight clipping value: `0.01`
- SWD projections: `128`
- Optimizer: `RMSprop`

Architecture:

- Generator:
  - input: concatenation of condition `x` and Gaussian noise of equal dimension
  - two hidden layers with `ReLU`
  - output dimension `n`
- Critic:
  - input: concatenation of sample `y` and condition `x`
  - two hidden layers with `ReLU`
  - scalar output

Condition preprocessing:

- The condition was normalized as `x / std(x)` inside the WGAN code path.

Evaluation:

- The WGAN baseline generated direct channel outputs
- the current implementation reported SWD in residual space, i.e. on `y - x`
- the SWD random-projection seed was set equal to the run seed

## 7. OptFib Block

The OptFib block was run separately through `scripts/compare_optional_baselines.py`.

Top-level OptFib run settings:

- Channel: `OptFib`
- Device: `cuda`
- Dataset size: `120,000`
- Epochs: `60`
- Batch size: `512`
- Evaluation size: `100,000`
- Diffusion steps: `100`
- DDIM steps evaluated: `100` only

### 7.1 OptFib Channel Parameters

OptFib used `conditional_drifting.channels.optfib` with:

- `L = 5000.0`
- `gamma = 1.27`
- `Kstep = 50`
- `Pn_dBm = -21.3`
- `use_noise_std = False`

Important note:

- Because `use_noise_std = False`, the optical-noise term was set by `Pn_dBm` and not by the `noise_std` field stored in the training configs.

### 7.2 OptFib Drifting Parameters

OptFib drifting used `BenchmarkConfig`, which inherits `DriftingConfig`, with the following effective values:

- `n = 2`
- `noise_std = 0.3` in the config object, but not used by the channel because `use_noise_std = False`
- Dataset size: `120,000`
- Epochs: `60`
- Batch size: `512`
- Evaluation size: `100,000`
- Learning rate: `1e-3`
- Latent dimension: `16`
- Hidden dimension: `128`
- Drift scale: `1.0`
- Bandwidth: `None`
- Minimum bandwidth: `0.2`
- Maximum drift norm: `2.0`
- Repulsive weight: `1.0`
- SWD projections: `256`
- Metric seed: fixed `12345`

Both parameterizations were run:

- Direct drifting: `is_residual = False`
- Residual drifting: `is_residual = True`

Evaluation-space note:

- Direct drifting was evaluated in output space.
- Residual drifting was evaluated in residual space.

### 7.3 OptFib Diffusion Parameters

OptFib diffusion used the defaults from `DiffusionConfig`, overridden only by the CLI values passed from the suite runner:

- `n = 2`
- `noise_std = 0.3` in the config object, but not used by the channel because `use_noise_std = False`
- Dataset size: `120,000`
- Epochs: `60`
- Batch size: `512`
- Evaluation size: `100,000`
- Learning rate: `1e-3`
- Hidden dimension: `128`
- Number of diffusion steps: `100`
- EMA decay: `0.995`
- Prediction type: `epsilon`
- Training target: residual mode
- `is_residual = True`
- Beta schedule: `cosine`
- SWD projections: `256`
- DDIM steps evaluated in the final suite: `100`
- Metric seed: fixed `12345`

Architecture:

- the same conditional diffusion MLP family as above
- three hidden layers with learned timestep embeddings
- `Softplus` activations

### 7.4 OptFib WGAN Parameters

OptFib WGAN used `PaperWGANConfig` with:

- `n = 2`
- `noise_std = 0.3` in the config object, but not used by the channel because `use_noise_std = False`
- Dataset size: `120,000`
- Epochs: `60`
- Batch size: `512`
- Evaluation size: `100,000`
- Hidden dimension: `128`
- Generator learning rate: `1e-4`
- Critic learning rate: `1e-4`
- Critic steps: `5`
- Weight clipping: `0.01`
- SWD projections: `256`
- Metric seed: fixed `12345`

Evaluation note:

- the OptFib WGAN path reported residual-space SWD

## 8. Metric Settings

Primary quality metric:

- Sliced Wasserstein Distance (SWD)

AWGN, Rayleigh, and SSPA:

- Projections: `128`
- Evaluation size: `1,000,000`
- `AWGN`, `Rayleigh`, `SSPA`

OptFib block:

- Projections: `256`
- Evaluation size: `100,000`

Evaluation spaces in the final full suite:

- Drifting direct and diffusion on AWGN, Rayleigh, and SSPA: output space `y`
- Drifting residual, OptFib diffusion, and WGAN: residual space `e = y - x`

## 9. Determinism / Seeding

- Global seed was set once per run using `set_seed(seed)`
- Deterministic PyTorch algorithms were enabled when available
- AWGN/Rayleigh/SSPA metric seed: equal to the current run seed
- OptFib metric seed: fixed `12345`

## 10. Final Result Files Produced By This Run

- Suite summary:
  - `results_hpc/20260318_203626/suite_results.json`
- Per-seed paper block summaries:
  - `results_hpc/20260318_203626/paper2309_benchmark_seed<seed>/paper2309_benchmark_summary.json`
- Per-seed OptFib summaries:
  - `results_hpc/20260318_203626/optfib_seed<seed>/baseline_compare_optfib.json`
