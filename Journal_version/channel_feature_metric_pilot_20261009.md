# Decoder-independent channel metric: local pilot results

Completed 9 October 2026 on the RTX 5060 Ti using existing seed-7 checkpoints.
No generator training, downstream training, dependency installation, cluster
job, or manuscript change. This implements the small experiment proposed in
[the metric note](theory_preoptimization_channel_metric.md), following the
[recorded development protocol](metric_pilot_protocol_20261009.md).

## What was computed

The score runner loads no encoder or decoder. It queries each channel at nine
fixed radial inputs and all coordinate perturbations, using h=0.025, 0.05,
0.1 and three Monte Carlo repeats. Inputs, noise calibration, kernel scales,
and aggregation are fixed across candidate generators.

For each input it computes:

- Full RBF-kernel mean-embedding error (empirical MMD).
- The operator norm of its centered finite input-difference map, computed
  through a signed embedding Gram matrix in float64.
- Conditional SWD, conditional mean and covariance errors, and finite input
  derivatives of those moment errors.

Kernels average three bandwidths fixed from physical noise. There is no
learned feature bank, random-feature approximation, decoder loss, optimized
critic, or fitted composite score. The estimates retain V-statistic diagonal
terms and are compared with independent analytic-versus-analytic floors.
Floors are not subtracted to manufacture a corrected metric.

Two sampling regimes are kept separate: independent draws across perturbed
inputs, and common random numbers (CRN) within each simulator. True and
surrogate noise streams remain independent. CRN is an additional access
assumption supported by the current code, not by arbitrary measured data.

The 128-sample run covers analytic, kernel-target, joint Sinkhorn, selected
condition-wise Sinkhorn, WGAN, DDIM-10/100, plus full-budget Sinkhorn on SSPA.
The predeclared 512-sample resolution check covers analytic, selected
Sinkhorn and DDIM-10/100; the runner also retains the degraded SSPA control.
There are 2,430 and 1,458 complete per-anchor/method/regime/step/repeat records,
respectively. These are not independent trained-model replications.

## Main results

The following uses the middle declared step h=0.05 and N=512, with shared
noise. Entries are averages over the nine anchors and three repeats, not
population suprema. All scales, maxima, and repeat-level values remain in JSON.

| Channel | Model | MMD | Embedding derivative norm | Mean derivative error | Covariance error | Covariance derivative error |
|---|---|---:|---:|---:|---:|---:|
| AWGN | Analytic floor | 0.0460 | 0.0632 | <1e-6 | 0.1307 | <1e-6 |
| AWGN | Condition-wise Sinkhorn | 0.0484 | 0.0656 | 0.0279 | 0.1285 | 0.0193 |
| AWGN | DDIM-10 | 0.0854 | 0.1101 | 0.0111 | 0.2211 | 0.0039 |
| AWGN | DDIM-100 | 0.0473 | 0.0652 | 0.0096 | 0.1301 | 0.0039 |
| SSPA | Analytic floor | 0.0472 | 0.3128 | <1e-6 | 0.0281 | <1e-6 |
| SSPA | Condition-wise Sinkhorn | 0.1018 | 0.5615 | 0.3999 | 0.0498 | 0.0713 |
| SSPA | DDIM-10 | 0.1541 | 1.0989 | 0.3800 | 0.0705 | 0.0075 |
| SSPA | DDIM-100 | 0.0603 | 0.4332 | 0.3589 | 0.0285 | 0.0095 |

Different columns have different units and scaling; do not sum them or compare
their absolute sizes across columns. The near-zero analytic moment-derivative
floor under CRN follows from the additive, input-independent noise: its
sample-mean offset and sample covariance are constant across perturbations.
The kernel features are nonlinear, so their derivative floor is not zero.

### 1. Sampling access materially changes feasibility

At h=0.05 and N=512, the analytic embedding-derivative floor is 0.0632 with CRN
versus 0.9031 with independent draws on AWGN. On SSPA it is 0.3128 versus
0.9221. On SSPA the selected Sinkhorn independent-draw score is 0.9934, only
modestly above the 0.9221 floor, whereas the CRN score is 0.5615 against 0.3128.

Increasing N from 128 to 512 roughly halves these floors. This is consistent
with Monte Carlo scaling, not a certified asymptotic-rate experiment. The
independent-draw estimate becomes noisier at smaller h. A passive-data or
black-box application without CRN would require more samples or another
derivative estimator before subtle errors can be resolved reliably.

![Sampling floors across perturbation scales](../results/channel_feature_metric_comparison_20261009_v3/derivative_sampling_floor.png)

### 2. The score detects a gap, but added predictive value is not established

AWGN Sinkhorn and DDIM-100 remain close to the finite-sample kernel floor.
SSPA Sinkhorn separates from it under CRN. The mean-derivative and
covariance-derivative checks also identify errors, and MMD alone distinguishes
the selected SSPA models. This panel does not demonstrate that the derivative
embedding score improves selection beyond those simpler controls.

At the previously tested analytic-trained SSPA codec, encoder relative
gradient errors were 0.614 for Sinkhorn, 0.954 for DDIM-10, and 0.159 for
DDIM-100. The new kernel scores have that qualitative ordering. However, MMD
and SWD also have it, and ranking by gradient **direction** instead gives
DDIM-10 an advantage over Sinkhorn. Direction and total gradient error are
different validation targets. No correlation significance or final BER
prediction is claimed from these already known seed-7 cases.

The metric uses generic radial inputs, while the gradient experiment uses
learned codewords. It is therefore not a same-input proof of a connection.
Next validation needs explicit input-coverage and held-out-task controls.

### 3. The bounded kernel misses the severity of a large-variance failure

At N=128, h=0.05, CRN, the degraded full-budget SSPA model has MMD 0.425 and
embedding-derivative norm 1.873. Kernel-target drifting scores 0.697 and 3.335,
respectively. Thus the kernel ranks the full-budget failure as less severe,
although its covariance error is 68.69 versus 0.1475 for kernel-target and
its previously measured encoder-gradient relative error exceeds 2,000.
The high-sample check still finds covariance error 66.74 for the full-budget
model. This is not a noise-floor artifact.

A bounded kernel distance is not required to order large variance errors or
an unbounded decoder cross-entropy correctly. The loss-class limitation in
the theory note matters in practice. Retain moment/tail checks; do not promote
this kernel norm alone to a universal channel-utility score.

### 4. Step-size dependence remains relevant

On SSPA, derivative magnitudes change across h=0.025, 0.05, 0.1. The tested
scales do not establish a zero-step derivative limit or a bound on truncation
bias. Label the results finite-difference estimates. Do not pick the scale
whose model ranking looks most favorable after seeing task results.

## Decision and next work

The implementation passes estimator checks and supplies useful decoder-free
measurements. It has **not passed the gate for replacing the current selector**
or claiming a better predictor than conditional SWD/MMD.

The next small experiment should:

1. Check smaller-step convergence or exact pathwise derivatives on a limited
   simulator-access panel, separating truncation error from sampling noise.
2. Keep moment/tail information alongside the bounded embedding and compare
   against moments plus derivatives alone. Freeze any augmentation before
   evaluating new models; do not retrofit a scalar to this table.
3. Freeze score choices, then evaluate unused seeds/families and separately
   declared receiver tasks. Test total gradient error, direction, and actual
   common-learning-rate progress separately, with query/compute costs.

Shared-noise results must not be marketed as a passive single-observation
solution. No measured-data claim, novelty clearance, fair across-family
training comparison, or validated BER predictor follows from this pilot.

## Checks, artifacts, and reproduction

Seven new tests cover signed derivative layout, explicit kernel sums, exact
cloud zero, Gaussian kernel integrals, label reversal, the fixed-power
oscillatory construction, moment-matched laws, and shared-noise moment
cancellation (some are combined in one test). Together with the six existing
gradient tests, all 13 pass. Report generation rejects incomplete/duplicate
observations and checks matching checkpoint hashes and channel noise before
joining the task results. Both source scripts and score data are preserved.

- [Metric implementation](../conditional_drifting/feature_metrics.py)
- [Runner](../scripts/run_channel_feature_metric_pilot.py)
- [Tests](../tests/test_feature_metrics.py)
- [128-sample records](../results/channel_feature_metric_panel_20261009/README.md)
- [512-sample records](../results/channel_feature_metric_resolution_20261009/README.md)
- [Checked score/task comparison, CSV and figure](../results/channel_feature_metric_comparison_20261009_v3/README.md)

The comparison retains lower-sample task-gradient rows for models absent from
the higher-sample gradient confirmation; source paths and per-row sample
budgets are explicit in its CSV. It does not silently treat those gradients as
equally precise. Result directories are ignored by git and require separate
transfer when handing the raw data to another system.

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_channel_feature_metric_pilot.py \
  --out-dir results/channel_feature_metric_repeat

/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_channel_feature_metric_pilot.py \
  --methods analytic,fiber_sinkhorn,ddim10,ddim100 --samples 512 \
  --out-dir results/channel_feature_metric_resolution_repeat
```

Run from the repository root and use new output directories. The generated
README and JSON report all declared steps and sampling regimes; the compact
table above uses the middle step for readability, not checkpoint selection.
