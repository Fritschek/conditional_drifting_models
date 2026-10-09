# Exact-law rotation and matched-value derivative controls

Completed locally on 9 October 2026 under the
[frozen protocol](metric_controls_protocol_20261009.md), following the
[independent assessment](metric_experiment_assessment_20261009.md).
No training, checkpoint tuning, dependency installation, HPC jobs or manuscript
edits. These are synthetic mechanism/estimator controls, not new SER results.

## Conclusions

1. **Candidate-dependent estimator variance is a real confound.** The ordinary
   derivative norm can rise roughly eightfold while the complete conditional
   channel law stays exactly unchanged. The analytic-versus-analytic floor is
   not a valid universal correction or uncertainty estimate.
2. **Derivative information is useful in the controlled same-input setting.**
   Matching the distributions at the evaluated inputs does not determine their
   local input derivatives or expected-loss gradients. The population derivative
   score detects differences that pointwise SWD/MMD cannot see in this example.
3. **This does not establish a superior kernel metric.** Simple mean/Jacobian
   information also resolves the deliberately Gaussian mean-perturbation
   example. Independent-batch cross estimates remove a squared-trace bias in
   expectation, but they are noisy and do not provide a calibrated operator-norm
   confidence bound or a universal task ranking.

## A: same channel law, different derivative-estimator variance

Use a two-dimensional Gaussian channel with f(x)=x and sigma=0.5. The surrogate
rotates its standard Gaussian latent vector through angle
`omega*(x[0]-a[0])`, with anchor a=(0.6,-0.8). Rotation preserves the entire
conditional Gaussian law for every input. Centering at a also makes the
sampled output values identical across frequencies within each repetition.
Only their pathwise input derivatives change.

N=128,512,2048 per independent batch; omega=0,1,4,16; sixteen independent MC
repetitions. Each record uses two independent P/Q batches. The ordinary
statistic averages the two self Gram matrices; it does not pool their mean
embeddings. Cross matrices use the independent batches and remain signed.

| Samples per batch | Ordinary RBF derivative norm, omega=0 | Ordinary norm, omega=16 | Cross derivative trace at omega=16, mean +/- MC SE |
|---:|---:|---:|---:|
| 128 | 0.22973 | 1.79859 | 0.01627 +/- 0.17562 |
| 512 | 0.11401 | 0.92764 | 0.03327 +/- 0.04080 |
| 2048 | 0.05950 | 0.45787 | 0.00647 +/- 0.01009 |

Every population value/derivative discrepancy in this table is exactly zero.
The ordinary norm includes model-specific Monte Carlo variance. Its positive
expectation must not be called population channel mismatch. The independent
cross trace targets squared Hilbert--Schmidt discrepancy, not operator norm.
Its mean is within the observed Monte Carlo uncertainty of zero in this grid.
That is a finite control check, not a confidence-coverage theorem. Negative
individual cross traces and eigenvalues are retained.

The unnormalized linear feature has a closed-form variance reference:

    E ||D_hat||_F^2 = 2*sigma^2*omega^2/N.

At omega=16:

| N | Exact expected squared error | Observed, mean +/- MC SE |
|---:|---:|---:|
| 128 | 1.00000 | 1.06045 +/- 0.13850 |
| 512 | 0.25000 | 0.22136 +/- 0.02964 |
| 2048 | 0.06250 | 0.06116 +/- 0.00994 |

The error scales quadratically with frequency and inversely with sample count.
Shared draws make the measured frequency scaling exactly quadratic for this
linear feature within a sample-size/repetition cell. The three N values use
independent draws. This does not imply independence between frequency contrasts.

![Exact-law rotation control](../results/metric_controls_summary_20261009_v2/rotation_null.png)

**Implication for earlier checkpoints:** their above-floor derivative scores
cannot be read as pure population derivative error without candidate-specific
variance assessment. This does not invalidate their measured task-gradient
disagreements or value discrepancies. It limits the interpretation of the
new feature-derivative estimator. Reparameterization-dependent gradient noise
could matter for SGD efficiency even with exact conditional laws, but that
is a separate optimization property and was not tested by training here.

## B: identical input locations, matched value errors, varied derivatives

Three fixed inputs: (-1,0), (0,1), (1,0). Smooth disjoint bumps are constant
near each input. At anchor a the surrogate mean and Jacobian are

    mean_Q(a) = a + eta*u,
    J_Q(a) = I + lambda*u*v^T.

Eta is 0 or 0.1; lambda is -4,-1,0,1,4. Covariance stays sigma^2 I. Both the
development and unused directions have unit u,v. Eight MC repetitions use
512 samples per independent batch. Values are paired across lambda, with
exact Gaussian population embedding calculations as an independent reference.

**All four empirical value scores (SWD, RBF, moments, augmented) are identical
across lambda within each paired cell.** Maximum recorded difference: zero.
The population RBF MMD is zero at eta=0 and 0.0615672 at eta=0.1, independently
of lambda, input location and declared direction. Conditional distributions
match across lambda at the three anchors, not throughout their neighborhoods.

Example: development directions u=v=(1,0), anchor (1,0), eta=0. The fixed
quadratic probe is `.5*((0.8,0.6)^T y-0.2)^2`. Its true expected input gradient
is (0.48,0.36), of norm 0.6.

| lambda | Population MMD | Population RBF derivative norm | Absolute expected-gradient error | Gradient cosine | First-order common-SGD progress ratio |
|---:|---:|---:|---:|---:|---:|
| -4 | 0 | 2.46577 | 1.92 | -0.631 | -1.56 |
| -1 | 0 | 0.61644 | 0.48 | 0.600 | 0.36 |
| 0 | 0 | 0 | 0 | 1.000 | 1.00 |
| 1 | 0 | 0.61644 | 0.48 | 0.960 | 1.64 |
| 4 | 0 | 2.46577 | 1.92 | 0.880 | 3.56 |

These are exact population/expected-gradient quantities, not noisy rankings.
The last column is `dot(g,g_Q)/||g||^2`, a first-order prediction for sufficiently
small plain-SGD steps. No actual update or Adam trajectory was run in this
control. Positive and negative slopes can have the same error norm but quite
different direction/progress; the derivative norm was not designed to encode
that sign. The simple mean-Jacobian error is |lambda| and already distinguishes
the mismatches here.

![Matched-value control](../results/metric_controls_summary_20261009_v2/matched_value_derivative.png)

Eight fixed loss probes were evaluated, with the second member of each pair
reserved as an unused-probe check. Two direction settings include an unused
direction. Neither group tunes feature/kernel weights. The probes cover
mixture-kernel sections, quadratic projections, binary-logistic losses, and
bounded cosine functions.

- Kernel sections satisfy `absolute gradient error <= population RBF d1`
  in every tested cell, including unused probes/directions. Their RKHS norm is
  one. This checks the specified function-class bound, not universal tightness.
- Quadratic errors satisfy the polynomial-feature bound in every cell. With
  unit w and offset b, its coefficient norm for our normalized moment features
  is `sqrt(1+2*b^2)`; the irrelevant constant cancels.
- Logistic and cosine probes are Gaussian-integrable; no controlled RBF-RKHS
  norm is asserted for them. The maximum 64-versus-128-node logistic quadrature
  difference is 4.97e-16.
- Ordinary empirical norms still have finite-sample floors. Cross traces and
  exact population quantities are kept alongside them, rather than subtracting
  an analytic scalar floor or taking square roots of signed estimates.

There are 480 metric records (60 conditions, eight repetitions) and 3,840
loss records (eight probes each). These are not independent trained models.

## Execution, coverage and evidence

The initial CUDA rotation run was stopped for throughput reasons after saving
the N=128/512 panels. CPU timing showed these small float64 kernels were much
faster there. Both complete studies were rerun on CPU with four Torch threads,
unchanged frequency/sample/anchor/probe settings and no score tuning. CPU and
CUDA random streams differ; the partial GPU data are not pooled into any table.
No failure result was removed by this hardware change.

| Complete run | Loop wall time | Shared preparation | Candidate and metric work | Distinct Gaussian vectors | Logical output evaluations |
|---|---:|---:|---:|---:|---:|
| Rotation | 138.34 s | 0.013 s | 138.30 s | 172,032 | 430,080 |
| Matched-value | 46.29 s | 0.011 s | 46.05 s | 196,608 | 589,824 |

The loop timer includes saves, shared preparation, candidate evaluations and
metrics, but excludes interpreter imports and initial manifest/source copying.
Preparation includes shared P/Q latent draws and reference outputs. Logical
outputs count reused samples evaluated at each candidate setting; they are
not distinct oracle queries. The synthetic formulas have negligible oracle
cost, so these numbers are not representative channel-simulator timing claims.

All expected rotation/condition/probe/repetition cells are present, without
duplicates. The feature, gradient, report-contract and new-control suites have
**37 passing tests**. Source snapshots, parameters, hashes, per-repeat Gram
matrices, loss gradients and timing/query counts are retained.

Artifacts:

- `results/metric_rotation_control_cpu_20261009/`: complete A, 192 records.
- `results/metric_matched_control_cpu_20261009/`: complete B.
- `results/metric_controls_summary_20261009_v2/`: final checked CSVs/figures.
- `results/metric_rotation_control_20261009/`: partial CUDA run, not primary.
- `results/metric_controls_summary_20261009/`: first summary, before additional
  value-invariance and quadratic-bound checks; v2 is the final summary.
- `scripts/run_channel_metric_controls.py`, `scripts/report_channel_metric_controls.py`,
  `conditional_drifting/feature_metrics.py`, `tests/test_metric_controls.py`.

A compact [October evidence archive](evidence/metric_research_evidence_20261009.tar.gz)
contains the completed earlier pilots and these controls, their raw JSON/CSV,
saved gradient tensors, figures and source snapshots, plus the research notes.
It excludes generator/codec training checkpoints, intermediate summaries and
the partial CUDA control. It is an evidence handoff, not a standalone training
repository. The archive is stored outside the ignored results directory so it
can be committed and transferred through Git. See [extraction instructions](evidence/README.md).

Reproduce from the repository root, choosing new output directories:

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/run_channel_metric_controls.py \
  --experiment rotation --device cpu --out-dir results/rotation_control_repeat
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/run_channel_metric_controls.py \
  --experiment matched --device cpu --out-dir results/matched_control_repeat
```

## Next decision

Do not start another large ranking campaign or replace checkpoint selection.
The next learned-model check must separate **population discrepancy**, **its
estimation uncertainty**, and **stochastic optimization noise** at identical
inputs. Use candidate-specific split/repetition controls, absolute expected
gradient errors, and simple moment/value/Jacobian comparators. A Gaussian
mean/covariance baseline remains important on AWGN/SSPA.

These experiments support the need to examine derivatives beyond pointwise
distribution matching. They do not yet prove useful kernel-metric selection
on unknown models, robustness to passive measurements, or a Sinkhorn advantage.
