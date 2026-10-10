# SSPA continuation through 30,000 updates

Completed on 10 October 2026 under the [frozen continuation protocol](sspa_30k_protocol_20261010.md).
All nine existing trajectories continued from update 4,800 to update 30,000.
No generator was restarted, no policy was dropped, and the numerical training
module and validation panel were unchanged. This is development evidence.

**Later status:** all nine trajectories have since [completed 390,720 updates](sspa_full_budget_results_20261010.md) and deteriorated. The final section below preserves the decision at 30k; it is no longer an instruction to launch that continuation. See the [independent assessment](sspa_full_budget_assessment_20261010.md) for the next experiment.

## Results

Mean +/- training-seed SD for seeds 9001--9003:

| Policy | Conditional SWD | Conditional GW2 | Conditional variance | Global SWD |
| --- | ---: | ---: | ---: | ---: |
| Separate adaptive (legacy) | 0.039382 +/- 0.000443 | 0.162028 +/- 0.001990 | 0.050820 | 0.009258 +/- 0.000456 |
| Fixed common | 0.039311 +/- 0.000378 | 0.160732 +/- 0.001298 | 0.052126 | 0.008814 +/- 0.000787 |
| Shared adaptive | 0.039110 +/- 0.000335 | 0.160481 +/- 0.001770 | 0.052165 | 0.008740 +/- 0.000536 |

The true conditional variance is 0.052830. Analytic-vs-analytic validation floors
remain 0.035844 (conditional SWD) and 0.144683 (GW2), with identical independent
analytic clouds at every checkpoint. Final SWD is about 1.09--1.10 times that
finite-sample reference. A floor ratio is not a confidence interval or a proof
of correct conditional laws.

Every trajectory improves in validation conditional SWD at every scheduled
checkpoint, including the added 10k and 30k points. All nine select the final
30k checkpoint under the unchanged minimum-SWD/earliest-tie rule. No collapse
appears in the saved conditional metrics, and the regression/drift traces remain
bounded. The earlier difference in variance recovery is much smaller by 30k.

Shared adaptive has the numerically lowest mean SWD, but the small differences
and three development seeds do not justify a general fidelity ranking. Fixed
common remains useful as the reference with a constant common entropy scale.
The shared-median initialization issue found earlier is inherited evidence, not
fixed by this continuation.

The global SWD column uses the repeated-input validation cloud, not the paper's
million-independent-input evaluation. It should not be compared numerically
against the old main table. No downstream AE, SER/BER or held-out test was run.
Repeated validation inputs across seeds also mean seed SD does not account for
uncertainty over a newly drawn input panel.

## Numerical and execution audit

- Complete: nine tasks, 72 saved validation checkpoints, 2,700 training summaries.
  The 54 inherited checkpoints are byte-identical to the parent files; parent
  files and manifest were not changed. All inherited metrics/traces are intact.
- Initial weights agree across policies within each seed, and final CPU/CUDA
  RNG-state hashes agree after 30,000 updates. Each final checkpoint agrees with
  the saved configuration, history, optimizer-update count and sample counters.
- All 1,152 scheduled float64 reference solves converged, including 864 inherited
  solves. The maximum barycenter RMS error on the **new** 10k/30k small panels is
  1.01e-7. The earlier shared-policy initialization error remains in the complete
  record. Do not interpret eight sampled transport inputs as an exhaustive check.
- Across the newly recorded full training batches, maximum relative raw row-mass
  errors are 0.00350 (legacy), 0.01487 (fixed) and 0.00334 (shared). Occasional
  poorly converged individual couplings are compatible with the small errors
  on the limited reference panels. Ten iterations are not an exact-solver claim.
- The focused suite passes 69 CPU tests. The CUDA run passes all 12 SSPA tests
  (nine trajectory/continuation contracts plus three report checks), including
  exact continuation, validation isolation, unchanged parents, tamper rejection
  and end-to-end continuation auditing.

Per-trajectory totals: 122,880,000 training inputs; 491,520,000 analytic outputs,
the same number of generated outputs and independent generated references;
32,768 calibration, 262,144 validation and 4,096 solver-check analytic outputs.
These totals include the parent stage. The loader repeats 32,768 calibration
samples before restoring each saved state; those 294,912 extra analytic samples
across nine initializations are outside the trajectory counters. They do not
change the restored calibration or training RNG. Tests use separate samples.

Added measured training time: **770.0 seconds** (12.8 minutes); added validation
and reference evaluation: 2.13 seconds. Cumulative training is 934.5 seconds,
including the earlier 164.5 seconds. These are local development timings on an
RTX 5060 Ti, not isolated timing benchmarks; serialization and loader/preflight
costs are excluded, and CPU checks ran during training.

## New historical evidence

The [historical training-history audit](sspa_historical_training_20261010.md)
recovers 16,000 epoch records from all 100 old full-budget checkpoints. The
regression loss rises substantially only later: a post-hoc 0.05 threshold after
initialization is crossed at 78,144--161,172 updates (median 114,774).

This is evidence about training loss, not intermediate SWD or the cause of
failure. Nevertheless, it explains why success at 30k cannot resolve the old
stability concern. Three entropy-policy fields are absent from the historical
full checkpoint configs, so exact old implementation provenance remains limited.

## Artifacts and next step

- Suite: `results/sspa_epsilon_trajectories_30k_20261010/`
- Tables, validation/solver trace plots, CSV and continuation audit:
  `results/sspa_epsilon_30k_report_20261010/`
- Historical records and figures: `results/sspa_historical_training_20261010/`
- Transfer archive: `Journal_version/evidence/sspa_30k_evidence_20261010.tar.gz`.
  The evidence README records its checksum and parent-archive dependency.

Rebuild the reports with:

```bash
python scripts/report_sspa_epsilon_trajectories.py --suite-dir results/sspa_epsilon_trajectories_30k_20261010 --out-dir results/sspa_epsilon_30k_report_20261010
python scripts/audit_sspa_continuation.py --suite-dir results/sspa_epsilon_trajectories_30k_20261010 --parent-suite results/sspa_epsilon_trajectories_20261010 --out results/sspa_epsilon_30k_report_20261010/continuation_audit.json
```

The next stability experiment should continue the paired trajectories through
the historical transition and toward the original 390,720-update endpoint,
saving conditional metrics and model states around 100k--160k. Freeze that
expanded checkpoint schedule before launch. No such long-budget continuation
has been launched yet. At current local throughput, all nine full-budget
extensions would require roughly three GPU-hours, not another few-minute check.
Do not select a new epsilon rule from nearly tied 30k endpoints or reopen the
closed RBF-selector sweep. Fair modern fast baselines remain a separate priority.
