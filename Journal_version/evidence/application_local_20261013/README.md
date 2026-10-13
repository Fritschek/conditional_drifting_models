# Local application evidence

Read [the results](../../application_local_results_20261013.md) and
[readiness note](../../application_local_readiness_20261013.md) first.
These are development results, not final multi-seed rankings.

- `sspa/`: six common-input generators, nine AEs and 27 physical curve points.
- `ofdm_clean/`: five generators, six trained receivers and a conventional
  reference, giving 35 physical curve points.
- `ofdm_interference/`: the same panel with hidden block-wise interference,
  also 35 physical curve points, including active/inactive counts.
- `receivers/`: preliminary CPU physical-channel receiver comparisons. The
  plain MLP was tested before choosing the frequency-structured receiver.

Main suite folders contain manifests, results, reports, local audit records
and SHA-256 checksums. Audits include execution-host checkpoint verification;
weights remain in ignored `results/` and are not included in this export.
The old CPU MLP references retain source hashes in their result JSON rather
than a separate runner manifest. CPU and CUDA receiver outcomes are separate
runs; their samples and trained weights are not identical.

`sspa/codebook_inspection.json` is a later, post-hoc evaluation of the frozen
codebooks with fresh CPU physical noise and exact nearest-mean decoding.
It is additional to the original suite checksums, has its own source and
parent hashes, and has SHA-256
`3b591b0268f0c3c67f70c2b8321f3aaad5fc6a8ccd422c4ef7b2aeb4a9cfa3eb`.
It is not a replacement of the primary evaluation or a checkpoint selector.

Reproduce full local checkpoint/source audits with:

```bash
python scripts/report_application_local.py --kind sspa --suite-dir results/common_coverage_9301
python scripts/report_application_local.py --kind ofdm --suite-dir results/ofdm_frequency_clean_9401
python scripts/report_application_local.py --kind ofdm --suite-dir results/ofdm_frequency_interference_9401
```

No HPC job or package installation was performed. The GPU preflight script
is an environment/execution check only; scientific campaign settings are not
yet frozen.
