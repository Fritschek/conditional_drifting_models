# Repeated-update SSPA results

10 October 2026. Completed the [frozen update-screen protocol](sspa_update_screen_protocol_20261010.md)
after the official W-Flow implementation comparison.

## Completed panel

Three fixed-common models at 100000 updates (seeds 9001--9003), 32 fresh paired
batches each, nine methods: **864 cloned single-update trials**. No update was
carried into another trial. Historical checkpoints, training code and optimizer
states were not modified. The two-repeat smoke is separate.

The audit verifies all 96 trial files, 27 method summaries, parent/source hashes,
stored gradient/step summaries, particle variance identities and neural variance
linearizations. All values are finite. The focused test suite passes 86 tests,
including five update-screen tests and a direct official-loss scaling check.

This measures local update behavior, not trained-model performance. There are no
new long trajectories, SWD curves, downstream codecs or BER/SER results.

## Average variance change

Values below are mean +/- batch SE, multiplied by **100000** for readability.
SE describes fresh-batch variability conditional on each fixed model, Adam state
and evaluation panel. It is not seed uncertainty or a multiple-comparison test.

| Single-update method | Seed 9001 | Seed 9002 | Seed 9003 |
| --- | ---: | ---: | ---: |
| Production, 4096 inputs x K4 | 8.52 +/- 0.35 | -6.03 +/- 0.34 | -4.71 +/- 0.30 |
| Official log solver, 4096 x K4 | 8.52 +/- 0.35 | -6.03 +/- 0.34 | -4.71 +/- 0.30 |
| Log solver, 1024 x K16 | 7.23 +/- 0.28 | -6.79 +/- 0.33 | -5.41 +/- 0.28 |
| Log solver, 256 x K64 | 6.42 +/- 0.27 | -7.67 +/- 0.26 | -6.92 +/- 0.27 |
| Official normalization, preserved epsilon | 12.16 +/- 4.96 | 20.31 +/- 5.05 | -17.91 +/- 4.24 |
| Official normalization, R=0.05 | 20.38 +/- 4.57 | 18.94 +/- 4.50 | -12.54 +/- 4.65 |
| Log K4, zeroed Adam first moment | -0.56 +/- 0.35 | 0.95 +/- 0.34 | -1.99 +/- 0.30 |

The true conditional variance is 0.052829773. Starting fresh-panel variances are
0.05276862, 0.05266509 and 0.05256100. Thus shrinking variance is not inherently an
improvement. The complete results include changes in squared aggregate variance
bias and conditional mean MSE, rather than ranking methods by variance change.

## What the comparisons show

### Solver arithmetic remains a small effect here

Across all 96 paired batches, the maximum absolute difference in neural variance
change between production and the official log solver is 3.14e-10. This extends
the earlier one-batch observation to three checkpoints and repeated batches.
It does not rule out sensitivity during later unsaved transition intervals.

### More particles need more exposure to reduce gradient noise

At a fixed 256 inputs, gradient-noise RMS falls from approximately 0.057--0.059
with K4, to 0.028--0.029 with K16, to 0.0145--0.0152 with K64. This spends 16 times
as many outputs per cloud from K4 to K64. Noise is the empirical RMS norm of the
raw parameter gradient minus its 32-batch mean.

At a fixed 16384 outputs per cloud, using fewer inputs and larger clouds leaves
gradient-noise RMS around 0.0142--0.0152. It changes the mean update and reduces
variance-change variability somewhat, but does not produce a comparable drop
in total parameter-gradient noise. Particle count also changes the finite-cloud
estimator; it is not a solver-accuracy knob alone.

### Optimizer history changes the local direction

Zeroing only Adam's first moment changes the average variance direction in seeds
9001 and 9002. Second moments, step counters, learning rates and bias corrections
remain inherited. This is not a newly initialized optimizer or a long-run
beta1=0 experiment.

For seed 9001, the average negative raw-gradient dot product with the panel
variance gradient is -5.84e-5, whereas the actual Adam step's linear variance
contribution is +8.25e-5. The corresponding actual change is +8.52e-5, leaving
a remainder of +2.69e-6. For seed 9002, the raw descent direction and Adam step
likewise have opposing variance effects, with the signs reversed. These
projections are empirical panel derivatives, not population or downstream-loss
gradients.

The first-moment intervention also improves conditional mean MSE more in this
single-step panel: mean changes are -3.06e-5, -2.55e-5 and -4.68e-5, compared
with -3.23e-6, +1.35e-5 and -2.37e-5 for the inherited log-K4 update. This is a
reason to test optimizer controls, not proof that momentum causes late failure.

### A sudden normalization change is not an official-training reproduction

Transplanting the official normalized loss at the same physical epsilon increases
raw gradient-noise RMS to roughly 0.269--0.275. Neural variance-change SD becomes
about 14--15 times the unnormalized value. Conditional mean MSE increases by
0.00199--0.00321 after one update, versus starting mean MSE of 0.000342--0.000385.

These large steps use the old optimizer's moments and learning rate, accumulated
under a differently scaled loss. The nonlinear remainder in the neural variance
change is about 1.2e-4, much larger than the unnormalized remainder. This does not
show that the official normalization is intrinsically unstable. It shows that
this abrupt transplant is not a drop-in repair.

Using released R=0.05 gives physical epsilon 0.03835--0.03892, compared with our
fixed 0.38870695. With K4 and 10 iterations, the mean across batches of the
**maximum** relative marginal residual is 0.56--0.68. This is a worst-anchor
statistic, not a typical residual or a failed-convergence fraction. Do not call
these couplings converged. The image recipe uses K64 and different optimization;
this screen did not train that complete recipe or test normalized K64.

## Next bounded experiment

The screen does not identify a stability fix. It makes an optimizer-aware paired
continuation more informative than another epsilon-only sweep:

1. Preserve an unchanged continuation control and an official-log-solver control.
2. Separate a sustained first-moment change from a learning-rate reduction. A
   single reset is transient and should not be conflated with a different beta1.
3. Use a new intervention manifest and frequent validation through the known
   transition region. Measure conditional mean/covariance/SWD together, including
   the analytic sampling floor; shrinking variance alone is not a success rule.
4. Evaluate an official-normalization recipe separately with a calibrated
   optimizer/step size and K64. Do not attribute a multi-setting rescue to one
   component, and check coupling residual distributions at its sharper epsilon.

No continuation was automatically started from these one-step results. Repeat
or extend the panel as needed before interpreting a mechanism. The previously
completed nine long trajectories remain the evidence for late deterioration.

## Reproduction and transfer

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_sspa_update_screen.py \
  --out-dir results/sspa_update_screen_repeat --device cuda

/home/rick/.local/share/mamba/envs/ml/bin/python scripts/report_sspa_update_screen.py \
  --suite-dir results/sspa_update_screen_20261010
```

Suite: `results/sspa_update_screen_20261010/`. Each seed has 32 trial JSON files,
`result.json`, and `updates.pt` containing raw gradients, actual parameter steps,
the variance gradient and evaluation panel. Root exports are `audit.json` and
`summary.csv`. Master generation consumes 8388608 outputs from each of the
positive/generated/reference clouds per seed, including unused prefixes. Branch
loss exposure is recorded separately and is not a wall-clock cost comparison.

Portable archive: `Journal_version/evidence/sspa_update_screen_evidence_20261010.tar.gz`.
It includes all scalar records, source/protocol/test snapshots and the reviewed
official loss with its MIT license. It excludes the approximately 144 MB of
`updates.pt` tensors, retained locally with hashes. Full tensor-audit regeneration
requires those tensors or a rerun; parent checkpoints are in the prior full-budget
archive. The pinned official repository must be available for execution.
