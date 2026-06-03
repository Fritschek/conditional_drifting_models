# Timing Audit

The timing table in `timing_table_cuda.tex` uses local CUDA measurements on an NVIDIA GeForce RTX 5060 Ti.
All rows use the same timing protocol: train on 2% of the paper training workload and extrapolate by the effective step fraction.

Source summaries:

- Baselines: `results/timing_suite_baselines_20260601_1755/timing_suite_summary.json`
- W-Flow ablation rows: `results/timing_suite_wflow_20260601_1715/timing_suite_summary.json`
- Merged convenience summary: `results/timing_suite_local_cuda_20260601_1755/timing_suite_summary.json`

Projected training hours used in the table:

| Method | AWGN | Rayleigh | SSPA |
|---|---:|---:|---:|
| Drifting (res.) | 1.309 | 1.300 | 5.656 |
| Drifting (dir.) | 1.299 | 1.299 | 5.656 |
| Joint-kernel drift | 1.863 | 1.864 | 8.112 |
| Joint Sinkhorn | 0.917 | 0.919 | 3.956 |
| Condition-wise Sinkhorn | 0.048 | 0.048 | 0.296 |
| WGAN | 0.064 | 0.085 | 0.478 |
| DDPM / DDIM shared training | 0.020 | 0.020 | 0.124 |

Important implementation note:

During the local timing pass, the condition-wise Sinkhorn row initially caused a CUDA out-of-memory error.
The issue was not the small per-condition Sinkhorn solve itself.
The batched epsilon estimator in `_batched_sinkhorn_barycentric_projection` flattened all condition fibers and computed a global cross-condition `cdist`.
That created an unnecessary quadratic matrix over unrelated conditions.
The implementation now estimates epsilon from within-condition batched costs only.

This changes the condition-wise Sinkhorn training field relative to runs produced before this fix.
The timing table reflects the corrected condition-wise implementation.
