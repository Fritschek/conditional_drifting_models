# Channel-metric resolution and additional-seed checks

9 October 2026. Local RTX 5060 Ti, existing checkpoints only. No generator or
codec training, dependency installation, HPC submission, or manuscript edits.
This follows the [frozen protocol](metric_resolution_protocol_20261009.md)
and extends the [first feature-metric pilot](channel_feature_metric_pilot_20261009.md).

## Decision

The input derivative can be measured consistently on these differentiable
simulators, and adding explicit moments exposes the large-variance failure
that the bounded kernel understates. These are useful implementation and
model-checking results. They do **not** establish a metric that predicts
downstream optimization better than conditional SWD.

Keep value and derivative errors separate. Do not replace checkpoint selection
or launch a large metric campaign yet. A useful next controlled test should
hold distributional value error approximately fixed while changing input
derivative error, then measure expected-loss gradients at identical inputs.
That would test added information more directly than another ranking of the
same historical model families.

## What changed

- Added exact pathwise derivatives of the finite empirical RBF mean embedding.
  The calculation contracts mixed kernel derivatives with sampled output
  Jacobians. It does not compare unrelated samplewise simulator Jacobians.
- Compared common-random-number (CRN) differences at h = 0.05, 0.0125,
  0.003125, 0.00078125 against that reference, with identical noise draws.
- Added the fixed feature vector `(y/sqrt(d), vec(yy^T)/d)` alongside the RBF
  feature map. Here d is real output dimension. Report the components and
  their direct sum. These are raw second moments, not covariances.
- Kept the nine radial input anchors, physical noise convention, three RBF
  bandwidths, and 64 SWD projections fixed. No decoder determines these choices.
- Evaluated N=128, two MC repetitions on seed 7 for numerical resolution, then
  N=512 pathwise scores on AWGN/SSPA seed 7 and SSPA seeds 8 and 9. All historical
  methods are retained, including the full-budget SSPA control.
- Separately evaluated frozen-codec gradients on seeds 8 and 9, using both
  analytic-trained and Sinkhorn-trained codecs, four MC repetitions, 2048
  gradient samples per message, 16384 reference/evaluation samples per message,
  and 1024 conditional-SWD samples per message. This is the existing gradient
  protocol, with equal-norm encoder steps and fixed full-codebook normalization.

The metric uses no encoder/decoder. The task check necessarily does. Generic
metric anchors and learned codewords are different sets; this is not a
same-input causal validation. Historical training budgets and architectures
are still unmatched. Seeds 8 and 9 are additional exploratory checkpoint
checks, not a publication-level confirmatory study.

## Finite-difference score norms approach pathwise empirical score norms

Relative discrepancy in the **RBF derivative operator norm**, against
pathwise differentiation of the same sampled embedding. Median and maximum
are across methods, anchors and two MC repetitions, not confidence intervals.

| h | AWGN median | AWGN maximum | SSPA median | SSPA maximum |
|---:|---:|---:|---:|---:|
| 0.05 | 0.212% | 0.432% | 4.40% | 37.4% |
| 0.0125 | 0.0134% | 0.134% | 0.370% | 25.5% |
| 0.003125 | 0.00112% | 0.0849% | 0.0233% | 4.66% |
| 0.00078125 | 0.00234% | 0.0374% | 0.00342% | 0.321% |

![Finite-difference convergence](../results/channel_metric_resolution_comparison_20261009_v2/derivative_resolution.png)

The worst SSPA case at every step is the full-budget model at anchor 4,
MC repetition 0. Small-step float32 roundoff is visible on AWGN. The samplewise
Jacobian checks also improve, with slower convergence possible at WGAN ReLU
boundaries. There is no reason to reduce h indefinitely in float32.

This checks numerical resolution of the **reported scalar derivative norm**.
It does not compute the norm of the difference between the full finite-difference
and pathwise derivative operators, nor establish their orientation agreement.
The samplewise Jacobian checks are additional, distinct evidence. It also does
not remove the population sampling floor. For example, the N=512
pathwise RBF derivative floors are 0.06327 on AWGN and 0.35605 on SSPA. The
previous SSPA h=0.05 CRN floor was 0.3128 using three repetitions: step bias and
the repetition count differ, so these are not identical estimates.

Autograd needs differentiable simulator access. CRN needs reproducible noise
across perturbations. Neither is a remedy for passive single-observation data.

The analytic-versus-analytic floor is a particular null control, not a universal
error bar for every candidate. Each generator can have different pathwise
feature-derivative variance, even when its conditional laws are exactly right.
The [independent assessment](metric_experiment_assessment_20261009.md) gives
the variance decomposition and an exact-law rotation control to test this.

## Moments expose the variance failure across the three seeds

N=512, nine-anchor/two-repeat means on SSPA. Values in different columns use
different feature normalizations and are not directly interchangeable.

| Seed | Model | Conditional SWD | RBF value | RBF derivative | Moment value | Augmented derivative |
|---:|---|---:|---:|---:|---:|---:|
| 7 | Selected Sinkhorn | 0.03887 | 0.10148 | 0.61958 | 0.06979 | 0.66334 |
| 7 | Full-budget Sinkhorn | 1.70535 | 0.41488 | 1.93133 | 8.19176 | 4.72630 |
| 8 | Selected Sinkhorn | 0.03826 | 0.10678 | 0.66375 | 0.06691 | 0.70212 |
| 8 | Full-budget Sinkhorn | 1.69760 | 0.36050 | 1.84501 | 11.49770 | 2.48434 |
| 9 | Selected Sinkhorn | 0.03895 | 0.11099 | 0.66827 | 0.06230 | 0.71237 |
| 9 | Full-budget Sinkhorn | 1.60939 | 0.40428 | 1.95410 | 8.23720 | 5.23602 |

The common analytic floors are SWD 0.01820, RBF value 0.04679, RBF derivative
0.35605, moment value 0.02530, and augmented derivative 0.35677. The same
reference and query noise streams are reused across checkpoint seeds. Repeated
floor numbers are therefore the same control, not three independent floors.

The full-budget moment-value error is more than two orders of magnitude above
the selected model for each seed. This makes its failure conspicuous even
when RBF scores are smaller than those of kernel-target/joint models. However,
**SWD already detects this failure**. Moment augmentation has not added a new
downstream prediction result here.

The augmented derivative alone still does not consistently rank this failure
worst. For seed 8, its value is 2.484 versus 3.348 for kernel-target. Combining
feature components is not equivalent to combining value and derivative errors,
and neither yields a universally valid scalar model ranking.

## Task validation and uncertainty

On the seed-8 Sinkhorn-trained frozen codec, selected Sinkhorn has encoder
gradient cosine 0.425 versus 0.949 for DDIM-10, although its conditional SWD is
lower (0.0322 versus 0.0492 at those codewords). DDIM-10's encoder-gradient norm
ratio is only 0.0547, so its good direction still has a severe magnitude error.
The analytic gradient floor cosine is 0.986 for this codec. This extends the
earlier observation that distributional error and gradient direction can rank
models differently.

At the seed-8 analytic-trained codec, however, the independent analytic
estimate itself has cosine 0.836 and relative gradient error 0.682. Splitting
the higher-sample reference gives cosine 0.900. Fine model rankings at this
operating point are not resolved by this sample budget. All comparisons must
retain those floors rather than treating the reference gradient as exact.

The generic-panel RBF/augmented derivative scores still prefer selected
Sinkhorn to DDIM-10. A worst-case feature discrepancy does not directly predict
the direction of a particular decoder's loss gradient. Bounds for a specified
loss class remain mathematically useful, but their practical tightness is open.

The seed-9 Sinkhorn-trained codec gives the following additional comparison:

| Method | Codeword conditional SWD | Encoder cosine | Encoder norm ratio | Relative encoder-gradient error |
|---|---:|---:|---:|---:|
| Analytic sampling floor | 0.01293 | 0.972 | 0.910 | 0.244 |
| Selected Sinkhorn | 0.03281 | 0.480 | 0.0508 | 0.977 |
| WGAN | 0.04477 | 0.857 | 0.468 | 0.645 |
| DDIM-10 | 0.04925 | 0.764 | 0.0395 | 0.970 |
| DDIM-100 | 0.01628 | 0.976 | 0.834 | 0.260 |

WGAN's expected-loss gradient is better aligned than selected Sinkhorn's here,
despite worse conditional SWD. At the smallest equal-norm encoder step, the
true-channel CE changes are `-7.50e-6 +/- 0.13e-6` for WGAN and
`-4.37e-6 +/- 0.06e-6` for Sinkhorn (MC standard errors, not training-seed
uncertainty). Both directions are useful. Generic-panel RBF derivative norms
are 0.962 and 0.668, respectively, and do not recover that task ordering.

The seed-9 full-budget model has a negative encoder-gradient cosine (-0.265)
at this codec, with a positive paired CE change `+2.60e-6 +/- 0.14e-6` for
the same smallest normalized step. This is an observed harmful direction,
not a finding that every full-budget checkpoint has a reversed gradient.
Do not infer an instability mechanism from these terminal checkpoints: the
old training budgets/data differ, and no controlled long trajectory was run.

The seed-9 analytic-trained codec again has a noisier floor (cosine 0.874).
New runs use the initial pilot budget, whereas the retained seed-7 results
for selected methods use the higher-sample confirmation. The joined CSV
records sample budgets and reference floors explicitly; do not pool their
uncertainties or average all codec cases as independent generator seeds.

## Artifacts and checks

- `scripts/run_channel_metric_resolution.py`: numerical/pathwise experiments.
- `conditional_drifting/feature_metrics.py`: feature calculations and references.
- `scripts/report_channel_metric_resolution.py`: completeness checks, metric
  summaries, convergence plot and joins to task gradients. The join verifies
  generator hashes and physical noise calibration.
- `results/channel_metric_pathwise_seed7_20261009/`: N=128 step comparison.
- `results/channel_metric_pathwise_n512_seed{7,8,9}_20261009/`: pathwise panels.
- `results/gradient_fidelity_validation_seed{8,9}_20261009/`: task validation.
- `results/channel_metric_resolution_comparison_20261009_v2/`: final summaries,
  completeness-checked convergence figure and 62 metric/task comparisons.
  The unversioned comparison directory is a partial first reporting attempt
  that encountered an absent split-reference field in an older result schema.

There are 1,350 finite-difference/pathwise records in the N=128 comparison
and 558 N=512 pathwise records across 31 channel/model/seed cases. These are
anchor/MC observations, not 558 independent generator runs.

Example reproduction (a new output directory is required):

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/run_channel_metric_resolution.py \
  --channels SSPA --seed 8 --samples 512 --repeats 2 --steps \
  --methods analytic,kernel_target,joint_sinkhorn,fiber_sinkhorn,wgan,ddim10,ddim100 \
  --out-dir results/channel_metric_resolution_repeat_seed8
```

An empty `--steps` selects pathwise evaluation only. Omit that argument to
repeat the four-step finite-difference comparison. This script evaluates
existing checkpoints and never trains an autoencoder.

Source snapshots, checkpoint hashes/configurations, anchors and raw per-anchor
records are saved in the experiment directories. Finite-feature tests include
the mixed-kernel derivative formula, unequal empirical sample counts,
polynomial normalization, identity cancellation and variance-growth detection.
The existing moment-matched/oscillatory counterexamples remain in the test set.
The combined feature/gradient suite has 18 passing tests.

No novelty or final BER-prediction claim follows from these checks. Results
directories are git-ignored, so share the small reports/data files explicitly
when handing this work to another system.
