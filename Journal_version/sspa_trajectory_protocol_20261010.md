# Bounded SSPA epsilon-policy trajectories

Frozen before the local comparison on 10 October 2026. This implements the first
4,800 updates of P1b, not its proposed 30k screen or 390,720-update reproduction.
These are development experiments, not confirmatory paper results.

## Design

- Seeds 9001, 9002, 9003. No existing result-directory names using these seeds
  were found before launch.
- Three policies: historical separately adaptive within-anchor epsilon;
  fixed common epsilon; shared adaptive epsilon. No other training changes.
- Direct-output SSPA, eight real coordinates, independent standard-normal
  input coordinates with no power rescaling. Rapp gain 5, saturation 1.5,
  smoothness 3; noise_std 0.3250531435997416, per-real-coordinate standard
  deviation noise_std/sqrt(2). These are the retained short-run settings.
- 4,096 inputs per update, four generated/positive/independent reference
  outputs per input. Hidden width 128, latent dimension 16, existing SiLU MLP,
  Adam 0.001, drift scale 1, drift norm cap 2, parameter-gradient norm cap 1.
  Ten practical Sinkhorn iterations, epsilon lower bound 0.001, kernel floor
  and denominator stabilizer 1e-8. No scheduler or parameter averaging.
- Fixed epsilon: median positive half-squared distance between two independent
  analytic clouds at each of 4,096 training-law anchors (four outputs each),
  pooled across anchors. Calibration seed 800001. This uses only the training
  distribution, not validation, a trained model, or downstream performance.
- Shared adaptive epsilon: median positive half-squared distance pooled over
  cross and independent-reference costs at the current training inputs. Apply
  that same scalar to both solves, with no gradient through its selection.
- Same initialization and training RNG ordering across policies. Preserve the
  old coupled training RNG sequence to test exact legacy equivalence; these
  same-architecture comparisons consume identical draws. This is not a general
  independent-stream interface for differing model families.

## Validation and logging

Continuous checkpoints at 0, 100, 300, 1000, 2500, 4800. Select minimum mean
conditional SWD, earliest checkpoint on a tie. Retain every checkpoint and last
state. No held-out test or SER/BER claim from this run.

Fixed validation panel: 128 Gaussian inputs, 128 outputs per input, 64 SWD
directions. Anchor, first/second analytic cloud, first/second latent cloud and
projection seeds are distinct offsets from 800101. Every operation restores
the training RNG. The analytic floors compare the two independent analytic
clouds at the same inputs. Report conditional SWD/GW2, mean/covariance errors,
global SWD, covariance eigenvalues, output norm and latent-pair output RMS.

Every 100 updates log both epsilons, raw coupling marginal errors, normalized
column error, floor activation, drift norms/clipping and gradient norm. At each
validation checkpoint, use an independent small repeated-input batch to compare
eight cross and eight self barycenters against the float64 reference (1e-8
marginal tolerance, 50k iteration cap). Unconverged references stay explicit.

Record training anchors and each generated/oracle/reference output separately;
count calibration, validation and solver-check oracle calls outside training.
Synchronize GPU timings per 100-update block. Training timing includes periodic
scalar diagnostics; validation/reference evaluation and checkpoint IO are
outside it. This is not a precision timing benchmark.

## Execution checks and resource bound

First verify exact CPU/GPU resume, validation isolation, count totals, rejected
source/config mismatches, and exact legacy four-update equivalence. Then profile
100 updates, without using fidelity scores to choose the budget. Run all nine
4,800-update trajectories if projected training fits two local GPU-hours.
Otherwise retain the profile and report the resource limitation before scaling.
The profile is a separate artifact and is not pooled into the comparison.

Use `scripts/run_sspa_epsilon_trajectories.py`. Resume requires an identical
manifest, configuration, source hashes, Torch version and device. Each saved
checkpoint includes Adam state and Python/NumPy/CPU/CUDA RNG states. Earlier
checkpoints remain available if a run fails between scheduled saves.
