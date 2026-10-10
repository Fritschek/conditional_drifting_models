# Research evidence

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
