# Research evidence

## Independent full-budget assessment

`sspa_full_budget_independent_audit_20261010.json` records the independent report
regeneration, numerical-source hashes, checkpoint/parent-byte verification,
policy summaries, variance ratios and post-hoc transition brackets. It is a
saved-evidence audit; GPU training, internal Torch checkpoint tensors and the
execution host's tests were not rerun. See the
[assessment](../sspa_full_budget_assessment_20261010.md) for scope and next steps.

## Full-budget SSPA stability experiment

`sspa_full_budget_evidence_20261010.tar.gz` contains all nine completed
390,720-update trajectories: 234 scheduled checkpoints, latest-state copies,
complete training/validation records, strict aggregation, continuation audit,
all 702 exploratory derivative records, figures, numerical source/test snapshots,
and the frozen protocol and result note. Degraded models and unconverged
reference checks are retained. This is research evidence, not a standalone
training repository or a public-code release.

Size: 97,451,901 bytes (about 93 MiB); all 315 files verified byte-for-byte against
their local inputs. The archive is below GitHub's 100 MiB per-file limit.

SHA-256: `e093ff6aaa82e8b7eacc7d9040d4746aedc36a6b3aec5b722d85e4f8c9782ad9`

```bash
tar -xzf Journal_version/evidence/sspa_full_budget_evidence_20261010.tar.gz --keep-old-files results/
```

Restore only `results/` to avoid replacing current code with archived snapshots.
The full-suite report is self-contained in this archive. To rerun the
parent-comparison audit, also restore `sspa_30k_evidence_20261010.tar.gz`; the
earlier short archive is needed only to audit the 30k suite against its own parent
or reconstruct the complete earlier lineage. Existing files are preserved by
`--keep-old-files`, with tar reporting a nonzero exit status for skipped files.

See [the full-budget findings](../sspa_full_budget_results_20261010.md). All nine
trajectories deteriorate late; common epsilon alone is not a stability fix. The
next solver interventions described there are proposals, not completed work.

## SSPA continuation to 30k and historical histories

`sspa_30k_evidence_20261010.tar.gz` contains all nine extended trajectories,
72 scheduled checkpoints (including inherited states), latest-state copies,
validation/training traces, the continuation audit, reports/figures, source and
test snapshots, and the frozen protocol/result notes. It also includes the
16,000 historical epoch records extracted from the 100 old full-budget states.
Historical model weights are not duplicated in this archive.

Size: 21,926,669 bytes (about 22 MB); all 148 files verified against local inputs.

SHA-256: `8883046f5ce162e9abf57dbe8478a75849ed77e46e45f9e426d7dd14ef9031d6`

```bash
tar -xzf Journal_version/evidence/sspa_30k_evidence_20261010.tar.gz --keep-old-files results/
```

The new report is reproducible from this archive. For the parent-comparison
audit or exact continuation/resume, also restore the original
`sspa_trajectory_evidence_20261010.tar.gz` described below, which supplies the
parent manifest, records and original checkpoint paths. Restore only `results/`
to avoid replacing current source with snapshots. Exact resume remains strict
about configuration, numerical source, Torch version/device and recorded paths.

See the [30k findings](../sspa_30k_results_20261010.md) and
[historical-history interpretation](../sspa_historical_training_20261010.md).
The 30k archive remains a historical snapshot. Its completed 390,720-update
continuation is documented in the [full-budget report](../sspa_full_budget_results_20261010.md).

## Bounded SSPA training trajectories

`sspa_trajectory_evidence_20261010.tar.gz` contains all nine 4,800-update runs,
54 scheduled training checkpoints plus latest-state copies, validation and
training records, the paired-checkpoint audit, initial-cloud numerical probe,
plots, source snapshots, tests, protocol and result note. The separate 100-update
resource profile is included under its own directory, not pooled with the runs.
Size: 15,579,619 bytes (about 15 MB); 129 files verified against the local inputs.

SHA-256: `35607b48186a2c41396310ca052830331dfb48dfe69e5aac2b2562cb7cb9eafd`

```bash
tar -xzf Journal_version/evidence/sspa_trajectory_evidence_20261010.tar.gz --keep-old-files results/
```

This restores evidence without replacing current source. Unlike the older metric
archives, this one includes the new training states needed for continuation.
It supplements this repository rather than constituting a standalone codebase.
See [the findings and reproduction commands](../sspa_trajectory_results_20261010.md).
The short trajectories do not establish long-budget stability or SER/BER quality.

## Transport-reference audit

`transport_reference_evidence_20261010.tar.gz` contains the final small P1a
transport panel, all input clouds and saved couplings, gradient/equilibrium checks,
policy tables, reference source, optional production diagnostics, tests and notes.
The superseded first development run is excluded. The archive is about 335 KB.

SHA-256: `d01cd4052d420ab5681b169f55c7ff4deebd2872e0057e1f9c286b88d262e5b1`

```bash
tar -xzf Journal_version/evidence/transport_reference_evidence_20261010.tar.gz --keep-old-files results/
```

This restores result files without replacing current code. See the
[result note](../transport_reference_results_20261010.md) for the explicit
nonconverged reference case, scope limitations and reproduction commands.
No training checkpoints are needed for this small synthetic audit.

## Independent learned-gate audit, 10 October

`learned_metric_gate_independent_audit_20261010.json` records independent NumPy
checks of the saved sufficient statistics, summary CSVs and frozen contrasts,
plus explicitly post-hoc kernel-section and operator summaries. It is separate
from the original experiment archive and does not alter the frozen protocol.
After restoring that archive, regenerate it with a NumPy-enabled Python:

```bash
python Journal_version/figures_src/audit_learned_metric_gate_records.py
```

This does not rerun simulator sampling or the Torch tests. See the
[independent assessment](../learned_metric_gate_assessment_20261010.md) for scope.

## Same-input learned-model test

`learned_metric_gate_evidence_20261009.tar.gz` contains the completed SSPA
N=512/2048 panel's per-repeat scores, split matrices, task-gradient vectors,
manifest, all report tables/contrasts, source snapshots, protocol, and result note.
It is about 3.7 MB and is outside the ignored results tree.

SHA-256: `84a437a2007eacf7265b8b41da8215a671c398f18730ed735e96526e6e689835`

The archive reproduces report aggregation. Large individual-output/Jacobian `.pt`
files and pretrained checkpoints are excluded and remain on the execution host.
It is not a standalone training repository. Restore only the result directories
without overwriting current code or existing results:

```bash
tar -xzf Journal_version/evidence/learned_metric_gate_evidence_20261009.tar.gz --keep-old-files results/
```

See [the result note](../learned_metric_gate_results_20261009.md) for the frozen
decision rule, complete negative finding, scope limitations, and reproduction.

## Earlier controls and pilots

`metric_research_evidence_20261009.tar.gz` contains the completed October
metric experiments, raw records, gradient tensors, figures and source snapshots.
Training checkpoints are excluded. The archive is not covered by `.gitignore`.

SHA-256: `8eea91e2a70329e74f5930df174ca070270eef23cd05ce704eaceed54bef4c9a`

From the repository root, restore only the result files with:

```bash
tar -xzf Journal_version/evidence/metric_research_evidence_20261009.tar.gz --keep-old-files results/
```

This leaves current source files and research notes untouched. Existing result
files are preserved; tar reports a nonzero exit status for files it skips.
The archived notes and source snapshots describe the state when the experiments
ran. Use the current notes under `Journal_version` for subsequent updates.
