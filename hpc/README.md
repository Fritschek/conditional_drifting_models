# HPC / SLURM

This directory contains a SLURM-ready path for the full-budget benchmark suite.

The intended workflow is:

1. launch one seed per SLURM array task,
2. let each task write its per-seed JSON and logs into one shared suite directory,
3. run the aggregation job afterward to produce `suite_results.json`.

## Files

- `check_env.sh`: verify the loaded Python / torch / CUDA / repo setup before long jobs
- `load_env.sh`: shared module / conda bootstrap used by both checks and SLURM jobs
- `slurm_full_suite_array.sh`: one seed per SLURM job or array task
- `slurm_full_suite_aggregate.sh`: aggregate finished seed results into one suite summary

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

into the shared suite directory.

## Aggregation

After all seeds finish, run:

```bash
python -u scripts/aggregate_full_budget_suite.py --suite-dir ...
```

This writes:

- `suite_results.json`

## Typical usage

First, sanity-check the environment inside the HPC setup you intend to use:

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
