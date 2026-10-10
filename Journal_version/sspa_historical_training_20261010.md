# Historical SSPA training histories recovered

Read-only extraction on 10 October 2026, while the new 30k trajectories run.
Source: all 100 final SSPA checkpoints (seeds 7--106) in
`results/journal_wflow_fiber_fixed_paper_hpc_20260601_161729/fiber_sinkhorn/`.
Every checkpoint contains 160 epoch-mean loss/drift records, despite lacking
intermediate model weights or intermediate conditional fidelity measurements.

## What the records establish

The full preset makes 2,442 optimizer updates per epoch, 390,720 in total. At
epoch 12 (29,304 updates), every seed still has a similar regression loss:
0.028201--0.028440. Loss later increases substantially in every history.

As a **post-hoc description only**, consider the first epoch after the first two
whose average detached-regression loss exceeds 0.05. All 100 seeds cross:

| First crossing, optimizer updates at epoch end | Value |
| --- | ---: |
| Minimum | 78,144 |
| Median | 114,774 |
| Maximum | 161,172 |

This threshold was not an original stopping rule or the frozen rule for the new
runs. It locates a training-loss transition approximately, at epoch resolution.
It is **not** an observed onset time for SWD/SER failure. A detached-target loss
can change with the vector field, so its magnitude alone does not establish
channel quality or a mechanism.

For historical seed 7, loss is 0.02831 at 97,680 updates, 0.08518 at 122,100,
0.21409 at 146,520, and 0.36631 at the final 390,720 updates. These are epoch
averages, not losses from individual saved model states.

## Implication for the continuation

Stable behavior through 30,000 updates is compatible with the historical loss
records. A direct investigation of the late failure needs to extend past roughly
100k--160k updates and retain actual conditional metrics and model checkpoints
through the transition. Do not treat a successful 30k screen as resolution of the
long-run stability objection.

The inspected old full/short seed-7 configs differ in dataset size and evaluation
size. The older full checkpoint also lacks the subsequently serialized epsilon
mode, subsample-count and scale fields. These are **absent keys**, not recorded
null settings. The archived final config therefore does not establish all old
solver defaults or the exact source revision. Preserve this provenance limit.

## Reproduction

```bash
python scripts/report_historical_sspa_training.py \
  --suite-dir results/journal_wflow_fiber_fixed_paper_hpc_20260601_161729 \
  --out-dir results/sspa_historical_training_20261010
```

Outputs include all 16,000 epoch records, full saved configurations, checkpoint
SHA-256 digests, per-seed descriptive crossing locations and a two-panel plot.
The extraction does not rerun or modify the old training. Historical model
weights remain local; the extracted records are included in the next transfer
archive with the new 30k evidence.
