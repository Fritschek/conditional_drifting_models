# HPC / SLURM

This directory contains SLURM-ready paths for the full-budget benchmark suite, the partial direct-metric rerun, the enhanced direct-kernel rerun, the journal W-Flow/Sinkhorn rerun, the downstream BER/SER follow-up, and the TurboAE long-block check.

The intended workflow is:

1. launch one seed per SLURM array task,
2. let each task write its per-seed JSON and logs into one shared suite directory,
3. run the aggregation job afterward to produce `suite_results.json`.

## Files

- `check_env.sh`: verify the loaded Python / torch / CUDA / repo setup before long jobs
- `load_env.sh`: shared module / conda bootstrap used by both checks and SLURM jobs
- `slurm_full_suite_array.sh`: one seed per SLURM job or array task
- `slurm_full_suite_aggregate.sh`: aggregate finished seed results into one suite summary
- `slurm_partial_direct_metric_array.sh`: one seed per SLURM task for the partial direct-metric rerun
- `slurm_partial_direct_metric_aggregate.sh`: aggregate the partial direct-metric rerun
- `slurm_enhanced_direct_array.sh`: one seed per SLURM task for the enhanced direct-kernel rerun
- `slurm_enhanced_direct_aggregate.sh`: aggregate the enhanced direct-kernel rerun
- `slurm_journal_wflow_array.sh`: one `(seed, variant)` task for the journal W-Flow/Sinkhorn rerun
- `slurm_journal_wflow_seed_array.sh`: one seed task that loops over journal W-Flow/Sinkhorn variants
- `slurm_journal_wflow_aggregate.sh`: aggregate the journal W-Flow/Sinkhorn rerun
- `slurm_journal_wflow_ser_array.sh`: one `(seed, channel)` task that loops over coding variants
- `slurm_journal_wflow_ser_aggregate.sh`: aggregate symbolic BER/SER follow-up results
- `slurm_turboae_long_block_array.sh`: one seed task for paired analytic/surrogate TurboAE long-block runs
- `slurm_turboae_long_block_aggregate.sh`: aggregate TurboAE long-block runs
- `submit_full_suite.sh`: helper to submit the full benchmark array
- `submit_partial_direct_metric_suite.sh`: helper to submit the partial direct-metric array
- `submit_enhanced_direct_suite.sh`: helper to submit the enhanced direct-kernel array
- `submit_journal_wflow_suite.sh`: helper to submit the journal W-Flow/Sinkhorn array
- `submit_journal_wflow_paper_budget_suite.sh`: helper for the seed-packed paper-budget W-Flow/Sinkhorn array
- `submit_journal_wflow_ser_suite.sh`: helper to submit the symbolic BER/SER follow-up array
- `submit_turboae_long_block_suite.sh`: helper to submit the TurboAE long-block array

## Per-seed runner

The array job calls:

```bash
python -u scripts/run_full_budget_seed.py --suite-dir ...
```

This writes:

- `seed<N>_manifest.json`
- `seed<N>_paper.log`
- `seed<N>_optfib.log`
- `seed<N>_result.json`
- `paper2309_benchmark_seed<N>/...`
- `optfib_seed<N>/...`

into the shared suite directory.

So the HPC path no longer needs to write benchmark artifacts into the repo-level default `results/` tree. You can point `SUITE_DIR` at scratch/workspace storage and keep the entire run self-contained there.

## Aggregation

After all seeds finish, run:

```bash
python -u scripts/aggregate_full_budget_suite.py --suite-dir ...
```

This writes:

- `suite_results.json`

## Partial Direct-Metric Rerun

The partial rerun is intended for the question whether the main metric should be direct output-space SWD rather than residual-space SWD. It reruns only the methods/channels that were previously scored in residual space and reports both metrics:

- residual drifting on `AWGN,Rayleigh,SSPA,OptFib`
- `WGAN` on `AWGN,Rayleigh,SSPA,OptFib`
- `OptFib` diffusion

The per-seed job calls:

```bash
python -u scripts/run_partial_direct_metric_seed.py --suite-dir ...
```

This writes:

- `seed<N>_manifest.json`
- `seed<N>.log`
- `seed<N>_result.json`
- `partial_direct_metric_seed<N>/partial_direct_metric_summary_seed<N>.json`

Aggregation writes:

- `partial_direct_metric_suite_results.json`

## Typical usage

First, sanity-check the environment. For a real SLURM smoke test, use:

```bash
bash hpc/check_env.sh --slurm-smoke
```

If you only want to inspect the currently loaded shell environment without launching `srun`, use:

```bash
bash hpc/check_env.sh
```

Then choose a shared suite directory:

Choose a shared suite directory first:

```bash
export PROJECT_ROOT=$PWD
export SUITE_TAG=$(date -u +%Y%m%d_%H%M%S)
export SUITE_DIR=$PROJECT_ROOT/results/hpc_full_budget_${SUITE_TAG}
export SEED_START=7
export NUM_SEEDS=3
```

Submit the array:

```bash
sbatch --array=0-2 hpc/slurm_full_suite_array.sh
```

Then aggregate after the array completes:

```bash
sbatch --dependency=afterok:<array_job_id> hpc/slurm_full_suite_aggregate.sh
```

or run aggregation manually once all seeds are done:

```bash
sbatch hpc/slurm_full_suite_aggregate.sh
```

## Typical Usage For The Partial Direct-Metric Rerun

Choose a shared suite directory first:

```bash
export PROJECT_ROOT=$PWD
export SUITE_TAG=$(date -u +%Y%m%d_%H%M%S)
export SUITE_DIR=$PROJECT_ROOT/results/partial_direct_metric_hpc_${SUITE_TAG}
export SEED_START=7
export NUM_SEEDS=10
```

Submit the array:

```bash
bash hpc/submit_partial_direct_metric_suite.sh
```

Then aggregate after the array completes:

```bash
sbatch --dependency=afterok:<array_job_id> hpc/slurm_partial_direct_metric_aggregate.sh
```

or aggregate locally after copying the suite directory back:

```bash
python scripts/aggregate_partial_direct_metric_suite.py \
  --suite-dir results/partial_direct_metric_hpc_<tag> \
  --seed-start 7 \
  --num-seeds 10
```

## Enhanced Direct-Kernel Rerun

The enhanced rerun is intended for the added conditioning-aware direct drifting row. It runs only:

- enhanced direct drifting on `AWGN,Rayleigh,SSPA`

The per-seed job calls:

```bash
python -u scripts/run_enhanced_direct_seed.py --suite-dir ...
```

This writes:

- `seed<N>_manifest.json`
- `seed<N>.log`
- `seed<N>_result.json`
- `enhanced_direct_seed<N>/enhanced_direct_summary_seed<N>.json`
- `enhanced_direct_seed<N>/checkpoints/enhanced_direct_<channel>_seed<N>.pt`

Aggregation writes:

- `enhanced_direct_suite_results.json`

Typical usage:

```bash
export PROJECT_ROOT=$PWD
export SUITE_TAG=$(date -u +%Y%m%d_%H%M%S)
export SUITE_DIR=$PROJECT_ROOT/results/enhanced_direct_hpc_${SUITE_TAG}
export SEED_START=7
export NUM_SEEDS=10
# Defaults now match the practical benchmark rerun:
# EVAL_SIZE=1000000 and per-channel drifting epochs from the paper presets.

bash hpc/submit_enhanced_direct_suite.sh
```

## Journal W-Flow / Fiberwise Sinkhorn Rerun

The journal rerun is intended for the drift-field comparison introduced after the W-Flow paper. The submit helpers cap the submitted SLURM array at `MAX_ARRAY_TASKS=32` by default and pack the remaining logical work inside each array task. This matches clusters with a hard 32-element array limit.

Default variants:

- `kernel_target`: standard direct-output drifting with target-space kernel only
- `kernel_joint`: GLOBECOM conditioning-aware direct-output kernel drifting
- `joint_sinkhorn`: naive joint Sinkhorn drift on condition-target features
- `fiber_sinkhorn`: fiberwise conditional Sinkhorn drift with repeated samples per condition

The generic per-variant job calls:

```bash
python -u scripts/run_journal_wflow_task.py --suite-dir ... --seed ... --variant ...
```

Each task writes:

- `<variant>_seed<N>_manifest.json`
- `<variant>_seed<N>_result.json`
- `logs/<variant>_seed<N>.log`
- `<variant>/seed<N>/<variant>_summary_seed<N>.json`
- `<variant>/seed<N>/checkpoints/enhanced_direct_<channel>_seed<N>.pt`

Aggregation writes:

- `journal_wflow_suite_results.json`
- `journal_wflow_per_seed.csv`

Typical 100-seed usage:

```bash
unset CONDA_ENV
unset PYTHON_BIN

bash hpc/submit_journal_wflow_suite.sh
```

The generic helper defaults to `NUM_SEEDS=100`, `MAX_ARRAY_TASKS=32`, `CHANNELS=AWGN,Rayleigh,SSPA,OptFib`, `DATASET_SIZE=120000`, `EVAL_SIZE=100000`, `BATCH_SIZE=512`, `DRIFTING_EPOCHS=60`, and `SWD_PROJECTIONS=128`. Override only the settings that need changing.

For a paper-budget rerun on AWGN/Rayleigh/SSPA/TDL, prefer the seed-packed helper. It runs all variants/channels for each seed and packs seeds into at most 32 SLURM array tasks.

```bash
unset CONDA_ENV
unset PYTHON_BIN

bash hpc/submit_journal_wflow_paper_budget_suite.sh
```

The paper-budget helper defaults to `NUM_SEEDS=100`, `MAX_ARRAY_TASKS=32`, `CHANNELS=AWGN,Rayleigh,SSPA,TDL`, `DATASET_SIZE=-1`, `BATCH_SIZE=-1`, `DRIFTING_EPOCHS=-1`, `SWD_PROJECTIONS=-1`, and `EVAL_SIZE=1000000`. The `-1` values let `scripts/run_enhanced_direct_benchmark.py` use the per-channel paper presets.

To aggregate manually after the array finishes:

```bash
python scripts/aggregate_journal_wflow_suite.py \
  --suite-dir results/journal_wflow_hpc_<tag> \
  --seed-start 7 \
  --num-seeds 100 \
  --variants kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn \
  --channels AWGN,Rayleigh,SSPA,TDL
```

The aggregator reports direct SWD, residual SWD, anchor-conditioned SWD, anchor mean/covariance/Gaussian-W2 excess metrics, training loss/drift norm, elapsed time, and paired deltas versus `kernel_joint`.

## Journal BER / SER Follow-Up

The symbolic coding follow-up consumes checkpoints from a completed W-Flow suite and trains a symbolic block autoencoder through each learned implant. It reports both SER and BER. With the default `message_dim=16`, BER is computed from the 4-bit binary representation of each message index.

The BER/SER submit helper also caps the submitted array at `MAX_ARRAY_TASKS=32` by default. Each array task processes a strided subset of the full `(seed, channel)` grid and loops over the requested variants. If `WFLOW_SUITE_DIR` is unset, the helper auto-detects the newest `results/journal_wflow_paper_hpc_*` or `results/journal_wflow_hpc_*` directory.

```bash
export CHANNELS=AWGN,Rayleigh,SSPA,TDL

bash hpc/submit_journal_wflow_ser_suite.sh
```

For a first coding check, run fewer seeds/channels:

```bash
export WFLOW_SUITE_DIR=$PWD/results/journal_wflow_hpc_<tag>
export NUM_SEEDS=10
export CHANNELS=AWGN,SSPA,TDL
export VARIANTS=analytic,kernel_joint,joint_sinkhorn,fiber_sinkhorn

bash hpc/submit_journal_wflow_ser_suite.sh
```

Aggregation writes:

- `journal_wflow_ser_results.json`
- `journal_wflow_ser_per_seed.csv`

Manual aggregation:

```bash
python scripts/aggregate_journal_wflow_ser_suite.py \
  --suite-dir results/journal_wflow_ser_<tag> \
  --seed-start 7 \
  --num-seeds 100 \
  --channels AWGN,Rayleigh,SSPA,TDL \
  --variants analytic,kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn
```

## TurboAE Long-Block Follow-Up

The TurboAE follow-up trains paired long-block baselines with the matched overnight-style settings from the vendored CNN TurboAE baseline in `external/turbo_mingru_decoder`:

- analytic AWGN training, evaluated on analytic AWGN
- fiber-Sinkhorn AWGN surrogate training, evaluated on analytic AWGN

Each seed task first trains its own `n=2` AWGN fiber-Sinkhorn implant unless `CHANNEL_IMPLANT_CHECKPOINT` is set. The default long-block run uses `L=64`, `300` epochs, `batch_size=500`, `sample_size=50000`, and `eval_num_blocks=50000`.

Typical 30-seed usage:

```bash
unset CONDA_ENV
unset PYTHON_BIN

export SEED_START=7
export NUM_SEEDS=30
export TURBOAE_LENGTHS=64

bash hpc/submit_turboae_long_block_suite.sh
```

Set `TURBO_ROOT=/path/to/turbo_mingru_decoder` only if you explicitly want to use the full external TurboAE repo instead of the vendored minimal CNN baseline.

Aggregation writes:

- `turboae_long_block_results.json`
- `turboae_long_block_per_seed.csv`

Manual aggregation:

```bash
python scripts/aggregate_turboae_long_block_suite.py \
  --suite-dir results/turboae_long_block_hpc_<tag> \
  --seed-start 7 \
  --num-seeds 30 \
  --lengths 64 \
  --modes analytic,checkpoint
```

## Environment setup

The SLURM scripts support two common cluster patterns:

- module-based Python / PyTorch
- conda activation via `CONDA_ENV`

The shared environment bootstrap lives in:

```bash
hpc/load_env.sh
```

That is the single file to edit if the cluster uses different module names.

If your cluster uses conda, export:

```bash
export CONDA_ENV=dl
```

If it uses a custom Python binary, export:

```bash
export PYTHON_BIN=/path/to/python
```

The module lines in `slurm_full_suite_array.sh` are examples copied from the previous working cluster job and should be adjusted to the target HPC environment.
