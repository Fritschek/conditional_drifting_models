# Journal W-Flow Artifact Summary

Generated from:
- `results/journal_wflow_paper_hpc_20260519_064227/journal_wflow_per_seed.csv`
- `results/journal_wflow_paper_hpc_20260531_143208/journal_wflow_per_seed.csv`
- `results/journal_wflow_fiber_fixed_paper_hpc_20260601_161729/journal_wflow_per_seed.csv`
- `results/journal_wflow_sspa_budget_screen_20260602_071106/journal_wflow_per_seed.csv`
- `results/journal_wflow_ser_20260526_082924/journal_wflow_ser_per_seed.csv`
- `results/journal_wflow_ser_tdl_20260531_151126/journal_wflow_ser_per_seed.csv`
- `results/journal_wflow_ser_fiber_fixed_20260601_201405/journal_wflow_ser_per_seed.csv`
- `results/journal_wflow_ser_sspa_budget_screen_20260602_073353/journal_wflow_ser_per_seed.csv`

If repeated `(seed, channel, variant)` rows are present, later CSVs replace earlier rows.

## Best learned variants

| Channel | Direct SWD | Anchor SWD | BER | SER |
|---|---|---|---|---|
| AWGN | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn |
| Rayleigh | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn |
| SSPA | Kernel target | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn |
| TDL | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn | Condition-wise Sinkhorn |

Interpretation:
- Condition-wise Sinkhorn is the best learned coding surrogate on AWGN, Rayleigh, SSPA, and TDL under the reported channel-specific coding setups.
- The SSPA condition-wise row uses the corrected compact-budget M_msg=64 screen; the older full-budget corrected run was an over-optimization failure of the sharp field, not the reported SSPA model.
- Direct SWD and downstream coding do not always agree, so the paper reports both global and condition-wise diagnostics.
