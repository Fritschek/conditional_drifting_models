# SSPA full-budget stability results

All nine trajectories completed 390,720 updates on 10 October 2026, under the
[frozen protocol](sspa_full_budget_protocol_20261010.md). The intentional pause
and resume are documented there. No training settings changed during the run.

## Main finding

**All three epsilon policies deteriorate late in all three development seeds.**
Fixed-common epsilon delays deterioration in these paired runs and has less
severe final errors, but does not prevent it. The nearly tied 30k endpoints were
not evidence of long-run stability. This experiment removes the earlier
short-versus-long-run data-budget confound by following continuous trajectories.
It does not prove the precise mechanism or retrospectively recover the exact
source used in the historical HPC experiment.

Conditional SWD below uses the fixed validation panel. Uncertainty is the sample
SD across three development seeds, not SE or a confirmatory confidence interval.

| Epsilon policy | Selected-checkpoint SWD | Final SWD | Final conditional variance |
| --- | ---: | ---: | ---: |
| Legacy separate adaptive | 0.038892 +/- 0.000210 | 1.94482 +/- 0.178 | 24.1925 |
| Fixed common | 0.038448 +/- 0.000300 | 1.22944 +/- 0.0854 | 9.33972 |
| Shared adaptive | 0.038517 +/- 0.000229 | 1.53435 +/- 0.189 | 17.2520 |

The analytic-versus-analytic SWD floor is 0.0358438. The true per-coordinate
conditional variance is 0.0528298. Variance entries above are means over seeds,
anchors and coordinates. The final model clouds therefore have dramatically
excess variance. Selected-checkpoint scores reuse the selection panel and must
not be presented as independent test performance. No downstream AE, SER or BER
training/evaluation was performed in this stage.

The selected update counts for seeds 9001, 9002, 9003 are respectively:

- Legacy: 80k, 80k, 50k.
- Fixed common: 80k, 100k, 110k.
- Shared adaptive: 90k, 100k, 90k.

These are retrospective selections under the frozen minimum-validation-SWD
rule, not an online early-stopping experiment. All trajectories ran to the full
budget. Small differences between good checkpoints are not a validated ranking.

## When deterioration becomes visible

For a compact **post-hoc description**, the first saved checkpoint above twice
that trajectory's 30k SWD is listed below. This threshold was not a selection,
stopping or preregistered hypothesis-test rule. Checkpoint spacing limits onset
resolution, and the preceding checkpoint is only the last saved low-error state.

| Policy | Seed 9001 | Seed 9002 | Seed 9003 |
| --- | ---: | ---: | ---: |
| Legacy | 90k | 110k | 120k |
| Fixed common | 110k | 150k | 150k |
| Shared adaptive | 100k | 120k | 110k |

The [late-trajectory figure](../results/sspa_epsilon_full_report_20261010/late_trajectories.png)
shows SWD, Gaussian W2, variance and mean error. Solid, dashed and dotted lines
denote seeds 9001, 9002 and 9003. Gray lines are analytic sampling floors, except
in the variance panel where the line is the exact variance.

## Transport and derivative evidence

The practical solver is accurate on the earlier healthy clouds, but becomes
poorly behaved on the later generated clouds. Across the complete retained
training trace, the largest kernel-floor fraction is 0.3863 and the largest raw
row-mass relative error is 0.9987. These maxima span policies, seeds and updates;
they do not describe a typical batch or isolate the onset mechanism.

Of 3,744 scheduled small float64 log-domain reference solves, **145 failed** the
50,000-iteration cap at the prescribed tolerance (41 legacy, 44 fixed, 60 shared).
Their barycenter errors remain null, not zero. Among converged references, the
largest practical/reference barycenter RMS error is 7.7999. This combines
iteration, kernel-floor and denominator effects; it does not isolate any one of
them. No training task crashed or was excluded.

In legacy seed 9001, sampled training summaries show self row-mass error growing
from about 0.0013 at 80k to 0.0132 at 81k and 0.0960 at 82k, before the large
recorded regression-loss rise at 83k--84k. The distribution was not validated
at every one of those updates. This ordering is suggestive, not evidence that
solver error preceded all changes in the generated law.

The additional [checkpoint analysis](../results/sspa_checkpoint_derivatives_20261010/checkpoint_derivatives.png)
was specified after the first legacy failure became visible. It is exploratory,
read-only, and did not affect selection. It evaluates all 234 saved models at
128 fresh Gaussian anchors with 256 fixed latent samples per anchor, seed 800201.
It compares pathwise conditional-moment derivatives against the exact SSPA mean
Jacobian and input-independent analytic covariance.

| Policy | Mean-Jacobian relative error, 30k -> final | Covariance-derivative norm, 30k -> final | Parameter norm, 30k -> final |
| --- | ---: | ---: | ---: |
| Legacy | 0.0721 -> 0.1364 | 0.0901 -> 38.46 | 34.22 -> 92.37 |
| Fixed common | 0.0717 -> 0.0969 | 0.0885 -> 8.91 | 33.87 -> 84.73 |
| Shared adaptive | 0.0724 -> 0.1803 | 0.0874 -> 49.08 | 33.87 -> 91.18 |

Entries are means across seeds. Mean-Jacobian relative error is RMS Frobenius
error divided by the analytic Jacobian's RMS Frobenius norm. The covariance
derivative is the RMS Frobenius norm of the three-index derivative tensor; its
analytic value is zero. Parameter norm is total Euclidean norm, including biases.
Layer Frobenius and operator norms are also recorded in the CSV.

The covariance derivative grows much more than the conditional-mean derivative.
Parameter norms grow smoothly even before the visible failure. Neither fact
proves that parameter growth or covariance sensitivity causes instability.
Finite-latent-sample estimates have substantial variability after degradation;
both half-panel results are retained, not used as confidence intervals. These
are conditional-moment derivatives, **not downstream expected-loss gradients**,
and they do not validate a new checkpoint-selection metric.

## Verification and costs

- All 9 tasks, 234 scheduled checkpoints and 35,163 sampled training records pass
  strict aggregation. No partial tasks are treated as completed.
- All 72 inherited checkpoint hashes and parent histories remain unchanged.
- Paired initial weights and final Torch CPU/CUDA RNG states match across policies
  for each seed. Final checkpoint records agree with JSON histories and counts.
- All 234 derivative-analysis checkpoint hashes match the continuation audit.
  All 702 full/half-panel derivative records are unique and finite.
- 75 focused regression tests pass, including finite-difference checks of the
  moment derivatives. The new plots were visually inspected.
- Retained cumulative training time: 11,542.95 s. Extension-only training:
  10,608.41 s (2.947 h). Cumulative validation: 198.28 s, of which 189.41 s is new.
  These do not include all process/serialization overhead or discarded replay.

Each retained trajectory consumes 1,600,389,120 training anchors and
6,401,556,480 outputs each for positive/oracle, generated, and independent
reference clouds. It records 32,768 calibration oracle outputs, 851,968 validation
oracle outputs, and 13,312 solver-check oracle outputs. The protocol separately
accounts for loader calibration draws, replayed work and the completed-task
peak-memory reporting limitation after resume. Do not interpret these counters
as the entire physical execution cost.

## Next bounded experiment

The epsilon-only stability hypothesis is not supported. Do not extend this same
comparison to more seeds or claim common epsilon fixes the failure.

1. Replay fixed clouds from pre-transition and degraded checkpoints. Compare the
   practical solver with log-domain solvers for both its floored kernel and the
   original unfloored cost. Vary iteration count separately and report unresolved
   reference solves. This separates denominator stabilization, floor distortion
   and iteration error without retraining.
2. After that audit, fork a pre-transition fixed-common checkpoint into an
   unchanged control and the justified numerical correction, with matched RNG,
   optimizer, data and budget. Retain long-enough control trajectories to cross
   the observed failure region. A single improved endpoint cannot identify cause.
3. If a numerically accurate field still deteriorates, test drift scale/optimizer
   step or finite-cloud effects separately. Do not change solver, learning rate,
   particle count and stopping rule together.

These interventions are proposals, not executed experiments. Fair modern fast
baselines and the larger structured channel remain open work. The submitted
paper, public repository and HPC scripts were not changed.

## Reproduction and transfer

Raw suite: `results/sspa_epsilon_trajectories_full_20261010`.
Reports: `results/sspa_epsilon_full_report_20261010`.
Derivative records: `results/sspa_checkpoint_derivatives_20261010`.

```bash
python scripts/report_sspa_epsilon_trajectories.py \
  --suite-dir results/sspa_epsilon_trajectories_full_20261010 \
  --out-dir results/sspa_epsilon_full_report_20261010
python scripts/audit_sspa_continuation.py \
  --suite-dir results/sspa_epsilon_trajectories_full_20261010 \
  --parent-suite results/sspa_epsilon_trajectories_30k_20261010 \
  --out results/sspa_epsilon_full_report_20261010/continuation_audit.json
python scripts/analyze_sspa_checkpoint_derivatives.py \
  --suite-dir results/sspa_epsilon_trajectories_full_20261010 \
  --out-dir results/sspa_checkpoint_derivatives_recheck --device cuda
```

The derivative runner requires a new output directory. Its full-panel input
draws are generated on CPU; numerical results need not be bit-identical across
GPU architectures. See [the evidence index](evidence/README.md) for the complete
portable archive and parent-archive dependencies. No new model training is
needed to regenerate the report.
