# Frozen-cloud SSPA solver replay

Frozen before the main replay. This follows the independent full-budget
assessment. No original checkpoints, trajectory source files, selection rules
or historical records are modified. This is a mechanism-development experiment,
not evidence of a repaired training method.

## Panel

Seed 9001 only initially, four saved checkpoints per policy:

- Legacy: 30k, 80k, 90k, 390720.
- Fixed common: 30k, 100k, 110k, 390720.
- Shared adaptive: 30k, 90k, 100k, 390720.

Reconstruct the next full batch from each checkpoint RNG: 4096 Gaussian inputs,
four source/positive/independent-reference samples per input, in original draw
order. Save inputs, latents, outputs, before/after RNG, epsilon and production
cost/kernel arrays. This is a post-checkpoint batch, not the unsaved batch that
initiated historical deterioration. Epsilon is resolved from the complete batch
and then held fixed across every numerical comparison. Do not select favorable
anchors. All 4096 anchors enter each solve.

## Separate comparisons

1. Production at 10, 30 and 100 iterations.
2. Production-10 with exact row normalization at readout, removing only its final
   additive denominator (the internal solver stays unchanged).
3. Log-domain updates on the exact production float32 floored kernel, promoted
   to float32 or float64 before taking logs, at 10/30/100 iterations. Unit initial
   scalings and production barycenter readout are retained.
4. Log-domain-10 with the internal additive denominator reinstated, in both
   dtypes, to check the numerical reformulation independently of removing it.
5. Float64 reference solves, each with both marginal-relative errors <=1e-8,
   checked every 10 iterations and capped at 50000. Freeze converged anchors
   individually and retain every unresolved anchor. Compare the quantized
   production kernel, ideal flooring of the float32 scaled cost, unfloored
   float32 scaled cost, and recomputed unfloored float64 quadratic cost. This
   separates kernel exp quantization, flooring, and cost/scale quantization.

Save plans, per-anchor residuals, iteration counts and convergence masks. Compare
plans, raw/clipped fields and barycenters only on intersecting valid reference
anchors. Report counts, means, quantiles and maxima. Compare common production,
exact-row and uniform-mass readouts explicitly; a nonconverged plan's uniform-mass
barycenter is not interchangeable with its row-normalized readout.

## Consequences of the field

For raw and clipped fields, report the signed first-order and nonnegative
second-order terms in the exact empirical particle-variance identity at drift
scale 1. Check their sum against direct recomputation. This is not the neural
update.

For each finite-iteration method, clone model and Adam state, compute the original
detached coordinate-mean MSE on the complete saved batch, clip the parameter
gradient at 1, and perform exactly one Adam step. Save raw gradients and actual
parameter steps, and differences against production-10. For reference methods,
do this only if both terms converge for every anchor; never silently substitute
a partially converged field or a reduced gradient batch.

Measure before/after conditional variance on the same fresh 128-input by
128-latent panel, seed 800301. Use common latents and float64 reductions. Also
record mean-output changes. This paired single-step check is not an expected
loss-gradient measurement or a long-run stability test. One reconstructed batch
per checkpoint is exploratory, without significance tests or population claims.

## Gates and scope

Before launch: CPU unit tests for log/primal agreement, scalar/batched reference
agreement, explicit cap failures, readout distinctions and the variance identity.
Then a separately labeled 32-anchor GPU smoke replay, not included in results.

Interpret whether corrections change late healthy fields/updates or only failed
endpoints. More iterations do not restore information removed by flooring. A
large error on a failed endpoint is insufficient evidence of prevention. Repeat
informative contrasts on other seeds before any general mechanism claim.
Only then consider a separately recorded paired continuation from fixed-common
80k to 200k; no automatic training launch is part of this runner.
