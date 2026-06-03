# W-Flow Outlier Audit

Date: 2026-06-01

This note records the robustness check behind the asterisk in Table III.
The audit compares means, medians, maxima, and downstream BER/SER for the completed W-Flow runs.

## Sources

- `results/journal_wflow_paper_hpc_20260519_064227/journal_wflow_per_seed.csv`
- `results/journal_wflow_paper_hpc_20260531_143208/journal_wflow_per_seed.csv`
- `results/journal_wflow_ser_20260526_082924/journal_wflow_ser_per_seed.csv`
- `results/journal_wflow_ser_tdl_20260531_151126/journal_wflow_ser_per_seed.csv`
- `results/journal_wflow_curves_20260601_074712/journal_wflow_curve_per_seed.csv`

## Finding

Only the SSPA condition-wise Sinkhorn generator-level metrics show a severe heavy tail.
The all-seed mean in Table III is correct, but it is dominated by four unstable generator-level seeds.
No other W-Flow channel/variant combination in the completed AWGN, Rayleigh, SSPA, or TDL data shows a comparable mean-vs-median instability.

## SSPA Condition-Wise Sinkhorn

Direct SWD over 100 seeds:

| Statistic | Value |
|---|---:|
| Mean | 0.0440 |
| SEM | 0.0200 |
| Median | 0.00450 |
| IQR | [0.00377, 0.00499] |
| Mean without four unstable seeds | 0.00442 |

Unstable generator-level seeds:

| Seed | Direct SWD | Anchor SWD | Anchor GW2 |
|---:|---:|---:|---:|
| 82 | 1.3078 | 1.4232 | 23.555 |
| 25 | 0.9890 | 1.0834 | 21.415 |
| 99 | 0.9887 | 1.1978 | 19.245 |
| 85 | 0.6905 | 0.8097 | 13.658 |

The 30-seed SSPA BER/SER curve subset still contains seed 25, so reducing Table III to 30 seeds would not remove the issue.
At 8 dB, condition-wise Sinkhorn remains the best learned SSPA surrogate in the M_msg=64 curve:

| Variant | BER mean | SER mean |
|---|---:|---:|
| Analytic | 7.83e-06 | 1.57e-05 |
| Kernel target | 4.38e-03 | 8.48e-03 |
| Kernel joint | 8.50e-03 | 1.81e-02 |
| Joint Sinkhorn | 1.52e-02 | 3.08e-02 |
| Condition-wise Sinkhorn | 3.02e-05 | 6.00e-05 |

## Other W-Flow Generator Metrics

The remaining W-Flow generator-level results are stable by the same mean/median/max check.

| Channel | Variant | Direct SWD mean | Direct SWD median | Direct SWD max |
|---|---|---:|---:|---:|
| AWGN | Kernel target | 0.01065 | 0.01057 | 0.01454 |
| AWGN | Kernel joint | 0.02902 | 0.02852 | 0.03976 |
| AWGN | Joint Sinkhorn | 0.01481 | 0.01509 | 0.02001 |
| AWGN | Condition-wise Sinkhorn | 0.00676 | 0.00674 | 0.00896 |
| Rayleigh | Kernel target | 0.00990 | 0.00978 | 0.01300 |
| Rayleigh | Kernel joint | 0.04921 | 0.04915 | 0.05269 |
| Rayleigh | Joint Sinkhorn | 0.01519 | 0.01505 | 0.02254 |
| Rayleigh | Condition-wise Sinkhorn | 0.00894 | 0.00859 | 0.01258 |
| SSPA | Kernel target | 0.00701 | 0.00697 | 0.00918 |
| SSPA | Kernel joint | 0.00738 | 0.00736 | 0.00982 |
| SSPA | Joint Sinkhorn | 0.01167 | 0.01161 | 0.01581 |
| TDL | Kernel target | 0.04128 | 0.04134 | 0.04938 |
| TDL | Kernel joint | 0.02894 | 0.02877 | 0.03719 |
| TDL | Joint Sinkhorn | 0.09657 | 0.09636 | 0.10902 |
| TDL | Condition-wise Sinkhorn | 0.01276 | 0.01265 | 0.01779 |

## Downstream Checks

The downstream AWGN, Rayleigh, and TDL BER/SER tables do not show comparable catastrophic outliers.
The SSPA M_msg=64 curve has ordinary seed variation at 8 dB; condition-wise Sinkhorn has maximum SER 1.5e-04 over the 30 seeds, still far below the other learned SSPA surrogates.

## Older Enhanced-Direct Summaries

The available ten local summaries under `results/enhanced_direct_hpc_20260324_202013` also do not show a comparable heavy tail:

| Channel | Mean SWD | Median SWD | Max SWD |
|---|---:|---:|---:|
| AWGN | 0.02776 | 0.02729 | 0.03286 |
| Rayleigh | 0.04901 | 0.04903 | 0.05155 |
| SSPA | 0.00725 | 0.00716 | 0.00820 |

## Interpretation

The asterisk should stay only on the SSPA condition-wise Sinkhorn global-SWD value in Table III.
The effect is a real generator-level stability caveat under random Gaussian SSPA inputs, not an aggregation typo.
It does not invalidate the downstream SSPA result, because the symbolic autoencoder probes a learned finite codebook region where condition-wise Sinkhorn remains the strongest learned surrogate.
