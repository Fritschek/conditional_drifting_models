# Remaining SSPA experiments: frozen intervention protocol

10 October 2026, before launching the remaining experiments. Requested completion
of the outstanding solver replications, continuations, optimizer/scaling changes,
official-style K64 test and full-budget checks. No submitted manuscript changes.

## Remaining solver replay

Run the existing 16-method replay without numerical changes on seeds 9002/9003.
Each has 30k and 390720 checkpoints plus these healthy/degraded pairs:

| Seed | Separate adaptive | Fixed common | Shared adaptive |
| --- | --- | --- | --- |
| 9002 | 100k / 110k | 140k / 150k | 110k / 120k |
| 9003 | 110k / 120k | 140k / 150k | 100k / 110k |

Same full-batch reconstruction, reference caps and unresolved-case rules as the
seed-9001 replay. No new reference accuracy claim for capped solves.

## Continuation matrix

All branches start from fixed-common update 80000 for seeds 9001--9003. Verify
the parent source/configuration/Torch/device hashes and preserve parent bytes.
Store intervention hashes separately; never relax the original loader.

| Branch | Change from inherited training |
| --- | --- |
| control | None; original trainer step |
| log_floor10 | Log-domain scaling of the original float32 floored kernel; remove internal additive denominator, keep original readout |
| log10 | Official unfloored log solver, 10 iterations, row-mass-clamped readout |
| log100 | As log10, 100 iterations |
| log10_beta0 | As log10, sustained Adam beta1=0; inherit second moment and step counter |
| log10_lr01 | As log10, learning rate 0.0001 instead of 0.001 |
| log10_drift01 | As log10, target drift scale 0.1 instead of 1 |
| official64 | K64, 256 inputs, normalized official quadratic loss at R=0.05; fresh AdamW, beta=(0.9,0.95), weight decay 0.01, 5000-step linear warmup from 1e-6 to 0.0002; parameter gradient cap 2; EMA=0.999 |

Except official64, retain K4, 4096 inputs, fixed epsilon, independent reference,
gradient cap 1 and original optimizer state. Radial drift cap 2 is retained on
the unnormalized branches. All branches consume 16384 oracle/generated/reference
outputs per update. Official64 has fewer independent inputs and different random
draw grouping; it is an exposure-controlled package, not a single-factor test or
a reproduction of image-feature training. Its raw model is primary, EMA is an
explicit secondary series. No learned feature encoder or classifier-free guidance.

The unfloored log10/log100 branches are fixed-iteration solvers, not certified
converged transport. Retain residual distributions. Warmup/optimizer reset are
intentional interventions, not exact resumes of the old objective.

## Schedule and gates

Run every branch through 200000 total updates. Validate/save at 80000, every
2000 updates to 160000, then 170000,180000,190000,200000. Training traces every
100 updates. Use the existing isolated validation panel and analytic floors.
Control model/optimizer/training-RNG states must reproduce the historical saved
states at matching checkpoints; added isolated validations must not perturb them.

A seed passes the bounded gate only if all three final validation points
(180k,190k,200k) satisfy:

- conditional SWD <= twice the larger of its parent's SWD and analytic floor;
- average conditional variance in [0.5,2] times analytic variance;
- conditional mean L2 <= twice the larger of parent's mean L2 and its floor.

For official64, record raw and EMA gates separately. Extend a branch if ANY seed
passes either applicable gate, then extend ALL THREE seeds of that branch to
390720 (additional saves at 250k,300k,350k,390720). This permissive development
gate prevents excluding an inconsistent candidate; it is not statistical
evidence of a successful method. Report all failures and seed differences.
Branches with no passing seeds remain explicit 200k failures, not full-budget
tests. Already-established historical full-budget control results remain available.

At completed 200k/final endpoints, use a separate Gaussian panel with seed 800501,
256 inputs, 512 outputs each and 128 SWD projections. Do not select branches
using that panel. Report raw and EMA separately. A full-budget stability statement
requires all seeds and final validation points to pass and independent-panel
fidelity to remain acceptable; never equate delayed failure with stabilization.

Numerical checks use the actual field on an isolated 128-input batch at each
saved validation. Record both marginal errors (max, median, p99 and fraction
above 1e-3), floor activation, effective epsilon and force scales. At 80k, 200k
and 390720, compare the first four anchors with float64 reference solves at
tolerance 1e-8, cap 10000; preserve unresolved status. This smaller reference
panel/cap differs from the full frozen-cloud replay and is explicitly labeled.

## Execution integrity

CPU and CUDA tests must cover unchanged control reproduction, intervention
resume, parent immutability, wrong-source rejection and normalization arithmetic.
Run a separately labeled short GPU preflight/profile before the matrix. Record
complete GPU training time, validation time, sample counts and invocation history.
Resume only saved atomic child checkpoints with matching intervention manifest.
An interrupted partial interval may be replayed and must be reported, not
silently counted as retained trajectory work. Do not stop finite but poor runs.
Record exceptions and continue other branches; do not aggregate failed tasks as
completed. No HPC jobs, dependency installs, commits or pushes are part of this work.
