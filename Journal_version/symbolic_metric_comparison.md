# Symbolic Conditional Metric Comparison

Source files:
- `results/symbolic_awgn_overnight_20260322_233425/anchor_moment_recompute.json`
- `results/symbolic_sspa_overnight_20260323_073903/anchor_moment_recompute.json`

Rows labeled `Residual drifting (best)` and `Direct drifting (best)` are channel-specific:
- `AWGN residual best`: `drifting_residual_product_whitened_cond0p25_adapt_emb8_to_analytic`
- `AWGN direct best`: `drifting_direct_product_cond0p5_rawplusres_to_analytic`
- `SSPA residual best`: `drifting_residual_joint_cond0p25_to_analytic`
- `SSPA direct best`: `drifting_direct_joint_cond0p5_to_analytic`

## Compact Table

| Method | AWGN SER | AWGN MeanEx | AWGN CovEx | AWGN GW2Ex | SSPA SER | SSPA MeanEx | SSPA CovEx | SSPA GW2Ex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Analytic | 0.00565 | 0.01245 | 0.01723 | 0.01973 | 0.00009 | 0.03153 | 0.00000 | 0.00975 |
| WGAN | 0.00620 | 0.03837 | 0.02099 | 0.04023 | 0.00039 | 0.17277 | 0.12807 | 0.27921 |
| Residual drifting (best) | 0.00652 | 0.00000 | 0.00000 | 0.00000 | 0.00335 | 0.99712 | 0.44087 | 1.01324 |
| Direct drifting (best) | 0.02836 | 0.18622 | 0.12978 | 0.28776 | 0.01249 | 0.77339 | 0.16546 | 0.89194 |
| Diffusion residual | 0.02153 | 0.41608 | 4.76432 | 1.85823 | 0.23518 | 2.44196 | 5.38568 | 3.33928 |
| Diffusion direct | 0.69884 | 0.00000 | 0.02369 | 0.00357 | 0.61046 | 0.03016 | 0.02378 | 0.04735 |

Metric definitions:
- `MeanEx`: excess per-anchor conditional mean L2 error over the analytic-vs-analytic floor
- `CovEx`: excess per-anchor conditional covariance Frobenius error over the analytic-vs-analytic floor
- `GW2Ex`: excess per-anchor Gaussian conditional Wasserstein-2 distance over the analytic-vs-analytic floor

Interpretation notes:
- On `AWGN`, the best residual drifting variant is competitive with `WGAN` in SER and looks strongest under the moment-based conditional metrics.
- On `SSPA`, `WGAN` remains clearly best among learned surrogates; the residual drifting variants degrade much more strongly than on `AWGN`.
- `Diffusion direct` remains anomalous: its decoder-agnostic conditional metrics can look deceptively small while downstream SER is catastrophic. It should not be used as a trusted reference point yet.
