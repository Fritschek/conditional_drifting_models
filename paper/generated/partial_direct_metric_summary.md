# Direct-vs-Residual Metric Comparison

Source: `results_hpc/partial_direct_metric_20260319_120917`
Seeds: `[7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`

## Drifting (Residual) and WGAN in Both Metric Spaces

| Channel | Method | Direct `y`-space SWD | Residual-space SWD |
|---|---|---:|---:|
| AWGN | Drifting (res.) | 0.0670 $\pm$ 0.0138 | 0.0103 $\pm$ 0.0010 |
| AWGN | WGAN | 0.0198 $\pm$ 0.0050 | 0.0195 $\pm$ 0.0051 |
| Rayleigh | Drifting (res.) | 0.3681 $\pm$ 0.0031 | 0.0063 $\pm$ 0.0011 |
| Rayleigh | WGAN | 0.0171 $\pm$ 0.0050 | 0.0170 $\pm$ 0.0050 |
| SSPA | Drifting (res.) | 0.4812 $\pm$ 0.0022 | 0.0128 $\pm$ 0.0007 |
| SSPA | WGAN | 0.0287 $\pm$ 0.0156 | 0.0269 $\pm$ 0.0162 |
| OptFib | Drifting (res.) | 0.3864 $\pm$ 0.0211 | 0.0477 $\pm$ 0.0059 |
| OptFib | WGAN | 0.0414 $\pm$ 0.0179 | 0.0424 $\pm$ 0.0181 |
