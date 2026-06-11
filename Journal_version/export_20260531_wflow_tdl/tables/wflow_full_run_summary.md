# Journal W-Flow Artifact Summary

Generated from:
- `results/journal_wflow_paper_hpc_20260519_064227/journal_wflow_per_seed.csv`
- `results/journal_wflow_paper_hpc_20260531_143208/journal_wflow_per_seed.csv`
- `results/journal_wflow_ser_20260526_082924/journal_wflow_ser_per_seed.csv`
- `results/journal_wflow_ser_tdl_20260531_151126/journal_wflow_ser_per_seed.csv`

## Best learned variants

| Channel | Direct SWD | Anchor SWD | BER | SER |
|---|---|---|---|---|
| AWGN | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn |
| Rayleigh | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn |
| SSPA | Kernel target | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn |
| OptFib | Kernel target | Joint Sinkhorn | Joint Sinkhorn | Joint Sinkhorn |
| TDL | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn | Fiberwise Sinkhorn |

Interpretation:
- Fiberwise Sinkhorn is the best learned coding surrogate on AWGN, Rayleigh, SSPA, and TDL.
- Joint Sinkhorn is the best learned coding surrogate on OptFib.
- Direct SWD favors kernel-target OptFib, while anchor metrics and coding favor joint Sinkhorn; this is the clearest metric-mismatch case.
