# Research evidence

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
