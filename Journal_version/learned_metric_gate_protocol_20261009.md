# Same-input SSPA derivative-metric development test

Frozen before running the new panel, 9 October 2026. This is a bounded development
test, not a trained-model population comparison or a validated checkpoint selector.

## Inputs and estimands

- Seed 7: selected condition-wise Sinkhorn, WGAN, DDIM-10, and independent analytic
  samples as a null control. Check all checkpoint hashes against the previous panel.
- Copy eight-dimensional SSPA anchors 3, 4, 5 and bandwidths from
  `results/channel_metric_pathwise_n512_seed7_20261009/results.json` verbatim.
  Ambient derivatives, no input renormalization. Physical component noise standard
  deviation is the stored `noise_std / sqrt(2)`.
- Ordered Gram-Schmidt on the first two anchors defines v1, v2. Use directions
  .8 v1 + .6 v2 and -.6 v1 + .8 v2, offsets .2 and -.3, and kernel centers
  .3 v1 - .4 v2 and -.7 v1 + .6 v2. For each direction use the quadratic
  .5(w'y-b)^2, softplus(w'y-b), 1-cos(w'y-b), and mixture-RBF section at its
  center. Both probe groups are development probes.
- Reference expected gradients use the actual SSPA mean Jacobian and conditional
  isotropic Gaussian integration. Logistic integration uses 128 Hermite nodes,
  checked against 64. All other reference probe gradients are closed form.
- Target: absolute error of the expected input-loss gradient. Save the vector
  estimates, pooled absolute error, its repeat-level spread, and independent-split
  cross squared error. The latter is signed and unbiased under unbiased independent
  pathwise estimates; its square root is not used as an unbiased norm estimate.

## Sampling and comparators

N=512 samples per split, two independent splits, eight repetitions. Use four
disjoint stream roles (reference score, candidate score, candidate task, reference
task control), distinct model/anchor/repeat/split seeds, and fresh seeds at N=2048.
Reference score draws are shared across candidates and counted once. The eight
loss probes share their task draws, not the score draws. Analytic target gradients
are deterministic; the analytic null uses independent noisy candidate estimates.
All models receive the same sample access. Each sampled output includes its full
8-by-8 input Jacobian. Preserve output/Jacobian samples to permit reanalysis.

Record SWD (64 fixed projections shared across methods), RBF MMD, ordinary
derivative norms, both self matrices, signed cross matrix, and self-minus-cross
variance matrix. Also record mean/covariance values and derivatives, fixed
polynomial features [y/sqrt(8), vec(yy')/8], and projected fourth raw moments and
their derivatives in the two directions. Covariance and covariance derivative
estimates use the unbiased sample covariance formula. Simple moment cross products
are saved without clipping. Fourth-moment uncertainty requires finite eighth
output moments and finite second moments of cubic-output-weighted Jacobians.
No moment subtraction from a kernel norm, no analytic-floor subtraction.

## Predeclared comparisons and refinement

For each anchor and probe compare all three unordered learned-model pairs. Work
with paired repetitions, since the reference score draws are shared. Report means
and Monte Carlo SEs of the signed cross squared target-error differences and
score differences. A contrast is *resolved for this development screen* when its
absolute mean exceeds three MC SEs plus 1e-12. This conservative descriptive rule
is neither a calibrated confidence interval nor a multiple-testing guarantee.
Its purpose is to avoid spending a new-seed campaign on an ordering hidden by
Monte Carlo noise. There is no practical-equivalence claim from a small score.

Refine an entire anchor (all models and analytic control) to N=2048 if any of its
predeclared target contrasts is unresolved, or if a resolved target contrast has
a resolved matching RBF derivative ordering without a resolved matching ordering
from at least one of the cheaper checks. The cheaper checks are SWD, RBF value
cross score, mean derivative cross score, covariance derivative cross score,
polynomial derivative cross score, and projected-fourth derivative cross score.
The second trigger deliberately includes ambiguous cases, not only successes.
Save the decisions from N=512 before generating N=2048 samples. Stop at N=2048,
even if still unresolved. N levels are reported separately, never pooled.

At the final available N for each anchor, mark a *candidate added-information
contrast* only when a resolved target ordering agrees with the RBF derivative
ordering and no cheaper comparator has a resolved agreeing ordering. Report
contrary orderings, unresolved checks, all negative results, and the total number
of contrasts. Such a contrast only warrants independent confirmation; it does
not validate a selector. Its absence stops the better-selector claim at this
scale. Quadratic losses calibrate the polynomial baseline. No pooled correlation,
no treating anchors/probes/repetitions as independently trained models.

## Execution and provenance

Local GPU for learned forward/backward sampling, CPU float64 for kernel/moment
algebra. No training, decoder fitting, dependencies, HPC jobs, or paper edits.
Record exact inputs, probes, checkpoint configs/hashes, code/protocol hashes,
git revision, device/software, per-stage measured time and sampled-output counts.
Separate model loading, reference sampling, candidate score/task sampling, metric
algebra, and report overhead. GPU times synchronize before and after sampling.
Missing inputs produce an explicit failure manifest. A reduced preflight run is
for implementation checking only and is not pooled with the frozen main panel.
