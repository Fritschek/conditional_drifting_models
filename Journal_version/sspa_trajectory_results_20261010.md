# SSPA epsilon-policy trajectories: local results

Completed 10 October 2026 under the [frozen bounded protocol](sspa_trajectory_protocol_20261010.md).
Nine newly trained generators: three policies, development seeds 9001--9003,
4,800 continuous updates each. This is the short operating budget, not the
historical 390,720-update degradation experiment. Submitted manuscripts and
historical results were not edited.

**Subsequent execution:** the [30k continuation](sspa_30k_results_20261010.md)
now extends all nine saved states with verified provenance. The future-work
paragraph below records the decision at the end of the short experiment; the
continuation interface and 30k screen are no longer unimplemented work.

## Main findings

Mean over three training seeds, with seed SD after +/-:

| Epsilon policy | Conditional SWD at 1,000 updates | Conditional SWD at 4,800 | Conditional GW2 at 4,800 | Conditional variance at 1,000 / 4,800 |
| --- | ---: | ---: | ---: | ---: |
| Separate adaptive (legacy) | 0.208623 | 0.045646 +/- 0.000273 | 0.190657 +/- 0.001400 | 0.003215 / 0.046169 |
| Fixed common | 0.162763 | 0.045481 +/- 0.000200 | 0.190563 +/- 0.000918 | 0.037804 / 0.050213 |
| Shared adaptive | 0.162540 | 0.045536 +/- 0.000439 | 0.189917 +/- 0.001300 | 0.038218 / 0.050376 |

The true per-coordinate conditional variance is **0.052830**. Common epsilon
substantially reduces the early variance deficit. At 1,000 updates the legacy
cross/self epsilon ratio is about 13--18, versus exactly one for common epsilon.
By 4,800 updates the legacy ratio is about 1.12--1.15 and its variance recovers.
This controlled intervention shows a policy-dependent training transient. It
does not establish that epsilon mismatch caused the historical late degradation.

All nine validation-SWD trajectories improve at every saved checkpoint and
select their final 4,800-update checkpoint. There is no deterioration at those
checkpoints. The final policy differences are small; these three development
seeds do not establish a meaningful final-fidelity winner.

The common analytic-vs-analytic validation floors are SWD **0.035844** and GW2
**0.144683**, from independent analytic clouds at the same inputs. They are
finite-sample references, not quantities to subtract mechanically from errors.
Final conditional SWD is about 1.27 times this floor for all policies. The
validation inputs are fixed across seeds, so seed SD does not represent input-
population or independent validation-panel uncertainty.

Global SWD at the final checkpoint is 0.010383, 0.009736 and 0.009972 respectively.
Here it uses 128 fixed inputs with 128 repeated outputs each. It is not directly
comparable to the manuscript's million-sample global-SWD protocol. No downstream
AE, SER/BER evaluation or held-out test was run in this experiment.

## Numerical behavior

All 864 scheduled float64 reference solves converged. After update zero, the
largest practical-versus-reference barycenter RMS error among the first eight
validation transport inputs is **8.25e-7**. These eight-input checks do not cover
every training input. Across the recorded full training batches, worst relative
raw row-mass errors reach 0.0581 (legacy), 0.0232 (fixed) and 0.0301 (shared).
Column normalization must not be mistaken for a fully converged balanced plan.
The largest recorded training kernel-floor fraction is 3.05e-5, in legacy.

There is a distinct initialization problem for the pooled-median shared policy:
the cross and self cost distributions are initially far apart. The lower median
of their equally sized pooled lists lies near the high end of the self costs.
On the fixed validation batch its epsilon is 0.0775--0.0884, below the fixed
training-law calibration of **0.388707**. Initial barycenter RMS error reaches
**0.04038**, and maximum raw row error reaches about 0.099. This transient does
not persist in the subsequent scheduled small-panel reference checks.

A **post-hoc numerical probe**, without changing or retraining any trajectory,
separates iteration error from the floored-kernel distortion on those initial
clouds. For the worst case (seed 9001, input 5), error is 0.04038 at 10 iterations,
0.03439 at 30 and 0.03232 at 100. Fully converging the floored kernel still leaves
0.03184 error against the unfloored reference. Two of its 16 kernel entries are
floored. All 72 original/floored probe comparisons converged. Thus more iterations
alone do not remove this initialization error. This concerns the tested pooled
statistic, not every possible shared-adaptive scale rule.

## Execution checks

- Five trajectory contract tests pass on CPU and CUDA: exact interrupted/resumed
  training for all three policies; validation leaves future updates unchanged;
  exact four-update equivalence to the original legacy trainer; exact raw-field
  equivalence/common-epsilon enforcement; source/config mismatch rejection.
- Three report tests reject missing tasks, wrong counts/selection and duplicate
  validation checkpoints. The combined focused suite passes 65 tests.
- Independent checkpoint audit confirms identical initial weights across policies
  for each seed and identical final CPU/CUDA RNG states. All 54 checkpoints have
  recorded SHA-256 digests. Source hashes match the run manifest.
- Complete records: nine tasks, 54 validation checkpoints, 432 training summaries.
  Per task: 19,660,800 training inputs, 78,643,200 analytic outputs and that many
  generated outputs and independent generated references. Additional analytic
  outputs: 32,768 calibration, 196,608 validation, 3,072 solver-check outputs.
- Actual accumulated training time is 164.5 seconds, plus 6.7 seconds validation
  and solver checks on an RTX 5060 Ti. Checkpoint IO is excluded. These timings
  include logging, desktop use and concurrent CPU tests; they are development
  accounting, not an isolated latency benchmark or a replacement timing table.

## Artifacts and reproduction

- Runner: `scripts/run_sspa_epsilon_trajectories.py`
- Restricted trainer: `conditional_drifting/sspa_trajectory.py`
- Complete suite: `results/sspa_epsilon_trajectories_20261010/`
- Validated CSV, table and figures: `results/sspa_epsilon_report_20261010/`
- Matched-checkpoint audit and initial-cloud numerical probe:
  `results/sspa_epsilon_checkpoint_audit_20261010/`
- Separate, unpooled 100-update resource profile:
  `results/sspa_epsilon_profile_20261010/`

From the repository root, with the existing Torch environment:

```bash
python scripts/run_sspa_epsilon_trajectories.py --device cuda --out-dir results/sspa_epsilon_trajectories_20261010
python scripts/report_sspa_epsilon_trajectories.py --suite-dir results/sspa_epsilon_trajectories_20261010 --out-dir results/sspa_epsilon_report_20261010
python scripts/audit_sspa_trajectory_checkpoints.py --suite-dir results/sspa_epsilon_trajectories_20261010 --out-dir results/sspa_epsilon_checkpoint_audit_20261010
```

For an existing suite the first command requires `--resume` with an unchanged
manifest/config/source. Use a new output directory for a new experiment. A
portable evidence archive, including all saved training states, is under
`Journal_version/evidence/`; its README records the checksum and restore command.

## Next decision

Keep the fixed-common rule as a clean reference, but do not declare a selected
winner from nearly identical endpoints. Extend the same controlled trajectories
to the planned 30k screen, followed by a predeclared long-budget comparison if
needed to reproduce the historical failure. Implement explicit continuation
provenance first: the current CLI deliberately rejects changing an existing
suite's update budget. It is not yet an extension interface.

Use saved checkpoints to examine variance recovery and input sensitivity if
longer training separates policies. The early shared-median failure warrants a
numerically safe alternative only as a separately labeled development ablation.
Fair Gaussian/flow/low-step baselines remain the next independent workstream.
The negative RBF-selector gate remains closed.
