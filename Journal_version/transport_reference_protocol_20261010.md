# Small transport-reference protocol

10 October 2026, specified before the numerical panel. This is P1a, not an SSPA
training trajectory or an explanation of the historical long-run degradation.

## Reference and production comparison

Use squared Euclidean cost divided by two, uniform probability masses, and
entropy epsilon times KL(plan || source_mass x target_mass). Implement a CPU
float64 log-domain reference independent of the production scaling routines.
Stop only when both maximum relative marginal residuals are at most 1e-8.
Check every ten iterations, cap at 50,000. Record failures explicitly and exclude
unconverged references from accuracy claims; do not silently increase the cap.

Production projections remain unchanged by default. Optional diagnostics expose
the actual cost, resolved epsilon, floored kernel, raw coupling and final row
weights for both unbatched and batched functions. Audit both before and after
row normalization. A normalized row is not evidence of a correct target marginal.

Deterministic cases: identical clouds, unequal counts, near duplicates,
concentrated Gaussians, broad Gaussians, separated clouds, unequal cluster
occupancy, and collapsed symmetric sources. Test epsilon 0.001, 0.05, 1 and
production iteration counts 10, 30, 100. Retain all cases and nonconvergence.
Measure barycenter error, plan L1 error, marginal residuals, floor activation,
and differences between batched/unbatched/default/diagnostic code paths.

To distinguish mechanisms, also solve the production's floored kernel in
float64 log space to the same tolerance: this defines a modified cost, not the
original OT problem. Its discrepancy isolates persistent kernel-floor distortion
from insufficient iterations and additive denominator stabilization. Do not claim
that increasing iterations alone repairs a changed kernel.

## Gradients and equilibrium checks

At fixed epsilon, raw coordinates, no clipping, and identical empirical self
clouds, differentiate the debiased objective using fresh converged plans and the
full two-sided self derivative. Compare every parameter coordinate with centered
finite differences (step 1e-5, tolerance 1e-6 absolute plus 1e-4 relative).
Use a small affine generator and verify the detached coordinate-mean MSE gradient
equals (2 * drift_scale / output_dimension) times the objective gradient.
This is an instantaneous finite-batch identity, not a flow convergence guarantee.

Separately compare independent-reference drift to the same-batch field; a
difference is an estimator/reference change, not evidence of numerical failure.
Check the collapsed q=delta_0, p=(delta_-1+delta_1)/2 example against
S_epsilon = epsilon/2 * log(cosh(1/epsilon)): zero velocity can coexist with
positive divergence. Also check identical empirical laws give zero drift.

## SSPA-shaped clouds and epsilon policies

Use eight Gaussian-distributed eight-real-coordinate anchors, K_g=K_p=K_r=4,
the actual SSPA mean map, and component noise 0.3250531435997416/sqrt(2).
Synthetic generated clouds have a fixed mean perturbation and 1.4 times the
physical noise standard deviation, with an independent reference cloud. This
tests numerical behavior at plausible scales, not a trained generator.

Compare fixed_common, shared_adaptive, and legacy_separate_adaptive policies on
identical clouds. Freeze fixed epsilon using an independent analytic-only pilot
cloud. For shared adaptive, pool strictly positive cross/self within-anchor
costs and take one median before either solve. Legacy uses the two existing
within-anchor medians separately. Record both actual epsilons, drift changes,
mass residuals and practical-versus-reference errors at 10/30/100 iterations.
Also record the older global/marginal scale as a labeled historical comparator.
Only fixed_common is a fixed-objective test; adaptive scales are held fixed per
numerical solve, not differentiated or promoted to a fixed-objective guarantee.

No large training, HPC jobs, dependency installation, or submitted-paper edits.
Save source hashes, inputs, plans, tolerances, failures, measured runtimes, complete
tables, and a compact Git-visible evidence archive. Long-trajectory preparation
remains a subsequent task after reviewing these checks.
