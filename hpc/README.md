# HPC / SLURM

This directory contains SLURM-ready paths for the full-budget benchmark suite, the partial direct-metric rerun, the enhanced direct-kernel rerun, and the journal W-Flow/Sinkhorn rerun.

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
- `slurm_journal_wflow_aggregate.sh`: aggregate the journal W-Flow/Sinkhorn rerun
- `submit_full_suite.sh`: helper to submit the full benchmark array
- `submit_partial_direct_metric_suite.sh`: helper to submit the partial direct-metric array
- `submit_enhanced_direct_suite.sh`: helper to submit the enhanced direct-kernel array
- `submit_journal_wflow_suite.sh`: helper to submit the journal W-Flow/Sinkhorn array

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

The journal rerun is intended for the drift-field comparison introduced after the W-Flow paper. It flattens the grid into one SLURM array where each task runs one `(seed, variant)` pair.

Default variants:

- `kernel_target`: standard direct-output drifting with target-space kernel only
- `kernel_joint`: GLOBECOM conditioning-aware direct-output kernel drifting
- `joint_sinkhorn`: naive joint Sinkhorn drift on condition-target features
- `fiber_sinkhorn`: fiberwise conditional Sinkhorn drift with repeated samples per condition

The per-task job calls:

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
export PROJECT_ROOT=$PWD
export SUITE_TAG=$(date -u +%Y%m%d_%H%M%S)
export SUITE_DIR=$PROJECT_ROOT/results/journal_wflow_hpc_${SUITE_TAG}
export SEED_START=7
export NUM_SEEDS=100
export VARIANTS=kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn
export CHANNELS=AWGN,Rayleigh,SSPA,OptFib

# Conservative shared budget used for the first journal-scale W-Flow run.
export DATASET_SIZE=120000
export EVAL_SIZE=100000
export BATCH_SIZE=512
export DRIFTING_EPOCHS=60
export SWD_PROJECTIONS=128

# Avoid flooding the scheduler if needed.
export MAX_PARALLEL=32

bash hpc/submit_journal_wflow_suite.sh
```

For a paper-budget rerun on AWGN/Rayleigh/SSPA, set `DATASET_SIZE=-1`, `BATCH_SIZE=-1`, `DRIFTING_EPOCHS=-1`, `EVAL_SIZE=1000000`, and `CHANNELS=AWGN,Rayleigh,SSPA`. The `-1` values let `scripts/run_enhanced_direct_benchmark.py` use the per-channel paper presets.

To aggregate manually after the array finishes:

```bash
python scripts/aggregate_journal_wflow_suite.py \
  --suite-dir results/journal_wflow_hpc_<tag> \
  --seed-start 7 \
  --num-seeds 100 \
  --variants kernel_target,kernel_joint,joint_sinkhorn,fiber_sinkhorn \
  --channels AWGN,Rayleigh,SSPA,OptFib
```

The aggregator reports direct SWD, residual SWD, anchor-conditioned SWD, anchor mean/covariance/Gaussian-W2 excess metrics, training loss/drift norm, elapsed time, and paired deltas versus `kernel_joint`.

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
