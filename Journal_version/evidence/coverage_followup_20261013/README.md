# Coverage and sampler follow-up evidence

Read the [completed-results note](../../coverage_followup_results_20261013.md)
and [frozen protocol](../../coverage_followup_protocol_20261012.md).

- `coverage/`: all 12 generator records, 45 AE outcomes, 135 curve points,
  per-message counts, paired physical-noise discordances, manifest and reports.
- `samplers/`: all six saved-checkpoint resolutions, conditional metrics,
  gradient comparisons, moment checks, raw timing repeats and manifest.
- `checksums.json`: SHA-256 of every exported data/report file.

Only compact JSON, CSV and Markdown are included. Generator/AE checkpoints,
sampler output/Jacobian tensors and the reference-gradient tensors remain in
the ignored local `results/` directories. The export is sufficient to inspect
the reported comparisons, but is not a checkpoint backup or full replay package.

Local report generation verified source hashes and checkpoint/tensor hashes
before export. Portable verification checks record integrity, manifest linkage
and report/result linkage. It does not independently rerun those checkpoint
checks, regenerate simulations, or establish scientific correctness.

From the repository root:

```bash
python3 scripts/export_coverage_followup.py --verify-only
```

The nominal SSPA SNR labels retain the frozen software convention. See the
[noise-label audit](../../sspa_noise_label_audit_20261012.md) before external
comparisons. Do not treat smoke runs or a finest-step sampler as ground truth.
