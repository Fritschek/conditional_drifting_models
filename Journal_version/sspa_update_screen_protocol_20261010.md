# Repeated-update SSPA screen

Frozen before execution on 10 October 2026. Follow-up to the solver replay and
official implementation comparison. Historical training code stays unchanged.

## Fixed panel

Fixed-common checkpoints at update 100000, seeds 9001, 9002, 9003. Each is healthy
in the existing validation history. Use 32 independent batch seeds 810000--810031,
shared across models and methods. This is a fixed-step development panel, not a
new validation selection rule. A separate two-repeat, seed-9001 smoke is excluded.

Each repeat draws 4096 Gaussian inputs and master clouds of 64 positive,
generated and independent-reference samples per input. Nested prefixes couple
comparisons across particle and input counts. Restore model and original Adam
state before every method. No update carries over to the next trial. The master
clouds are fresh probes, not historical training batches.

## Nine methods

1. Production solver, 4096 inputs x 4 particles per cloud, saved fixed epsilon.
2. Official log solver, same data, epsilon, 10 iterations, original radial cap.
3. Log solver, 1024 x 16, same epsilon and cap.
4. Log solver, 256 x 64, same epsilon and cap.
5. Log solver, 256 x 4, same epsilon and cap.
6. Log solver, 256 x 16, same epsilon and cap.
7. Official normalized loss, 4096 x 4, with its R converted to preserve the
   saved physical-space epsilon. This bundles feature and force normalization
   and replaces the radial cap, so it does not isolate these ingredients.
8. Official normalized loss, 4096 x 4, released R=0.05, quadratic cost,
   independent reference, diagonal mask disabled, no classifier-free guidance.
   Comparing 7 and 8 isolates regularization within this normalized recipe.
9. Method 2 with only Adam's first moment zeroed. Keep second moment, step counter,
   learning rate and bias correction. This is an optimizer-state intervention,
   not a fresh Adam optimizer or a proposed training schedule.

Methods 2--4 consume 16384 samples from each cloud per update, while changing
the number of distinct inputs. Methods 5, 6, 4 fix 256 inputs and spend more
samples as K increases. These two comparisons separate exposure from input
coverage; neither is a matched wall-clock benchmark. Probe master generation
actually consumes 262144 oracle outputs per repeat, recorded separately.

## Measurements and interpretation

Use the same fresh 128-input x 256-latent panel, seed 800401, for before/after
conditional variance and mean-error measurements. Use double reductions.
Compute the parameter gradient of panel variance at each starting checkpoint.
For each Adam step, save its dot product with that gradient (linear contribution)
and the actual variance change minus that contribution (remainder). These are
empirical panel derivatives, not exact population derivatives.

Retain raw parameter gradients, actual parameter steps, coupling residuals,
physical epsilon, normalization scales, clipping, particle-variance terms, and
fresh-panel variance/mean-error changes. The normalized loss's physical target
displacement is recovered from its output gradient and coordinate scale; its
regression also includes a 1/scale^2 multiplier. Do not interpret particle and
neural variance changes interchangeably.

Summarize within-checkpoint means, batch SD and SE, and paired method-minus-log4
differences. Batch SE is conditional on the fixed model, optimizer and evaluation
panel; it is not cross-seed uncertainty or a significance claim. Report all three
models separately. Mean-gradient norm and gradient-noise RMS are distinct from
the mean of gradient norms. Do not select a method solely for shrinking variance:
the starting variance can already be below truth, and mean fidelity can worsen.

The official module is pinned to W-Flow commit
`1ccb34baf57f9f8f4e188460ea3d19b431403293`, compilation disabled, no dependency
installation. Parent bytes and numerical-source hashes must remain unchanged.
Retain failure records, all methods and all batch seeds. Do not automatically
launch a long continuation from a favorable one-step result.
