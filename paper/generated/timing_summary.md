# Timing Summary

Source: `/home/entropy/GitHub/conditional_drifting_models/results/timing_suite_20260319_110813`
Train fraction: `0.02`

## CUDA

### AWGN

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 0.733 | 0.405 us | 0.734 |
| Drifting (dir.) | 0.724 | 0.381 us | 0.725 |
| DDPM | 0.049 | 0.132 ms | 0.416 |
| DDIM-100 | 0.049 | 0.114 ms | 0.366 |
| DDIM-50 | 0.049 | 0.057 ms | 0.208 |
| DDIM-20 | 0.049 | 0.023 ms | 0.113 |
| DDIM-10 | 0.049 | 0.012 ms | 0.081 |
| Paper WGAN | 0.206 | 0.393 us | 0.207 |

### Rayleigh

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 0.730 | 0.407 us | 0.731 |
| Drifting (dir.) | 0.725 | 0.382 us | 0.726 |
| DDPM | 0.050 | 0.134 ms | 0.422 |
| DDIM-100 | 0.050 | 0.115 ms | 0.371 |
| DDIM-50 | 0.050 | 0.058 ms | 0.211 |
| DDIM-20 | 0.050 | 0.023 ms | 0.115 |
| DDIM-10 | 0.050 | 0.012 ms | 0.083 |
| Paper WGAN | 0.202 | 0.386 us | 0.203 |

### SSPA

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 3.197 | 0.424 us | 3.198 |
| Drifting (dir.) | 3.183 | 0.396 us | 3.184 |
| DDPM | 0.333 | 0.139 ms | 0.719 |
| DDIM-100 | 0.333 | 0.120 ms | 0.666 |
| DDIM-50 | 0.333 | 0.060 ms | 0.500 |
| DDIM-20 | 0.333 | 0.024 ms | 0.400 |
| DDIM-10 | 0.333 | 0.012 ms | 0.367 |
| Paper WGAN | 1.352 | 0.408 us | 1.354 |

### OptFib

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 0.054 | 0.410 us | 0.054 |
| Drifting (dir.) | 0.049 | 0.382 us | 0.049 |
| DDPM | 0.048 | 0.112 ms | 0.051 |
| DDIM-100 | 0.048 | 0.107 ms | 0.051 |
| DDIM-50 | 0.048 | 0.054 ms | 0.050 |
| DDIM-20 | 0.048 | 0.022 ms | 0.049 |
| DDIM-10 | 0.048 | 0.011 ms | 0.048 |
| Paper WGAN | 0.236 | 0.382 us | 0.236 |

