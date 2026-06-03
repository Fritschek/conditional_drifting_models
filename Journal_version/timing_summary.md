# Timing Summary

Source: `results/timing_suite_local_cuda_20260601_1755`
Train fraction: `0.02`

## CUDA

### AWGN

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 1.309 | 0.131 us | 1.310 |
| Drifting (dir.) | 1.299 | 0.114 us | 1.300 |
| Joint-kernel drift | 1.863 | 0.116 us | 1.864 |
| Joint Sinkhorn | 0.917 | 0.116 us | 0.918 |
| Condition-wise Sinkhorn | 0.048 | 0.115 us | 0.048 |
| WGAN | 0.064 | 0.108 us | 0.064 |
| DDPM | 0.020 | 0.046 ms | 0.147 |
| DDIM-100 | 0.020 | 0.039 ms | 0.128 |
| DDIM-50 | 0.020 | 0.019 ms | 0.073 |
| DDIM-20 | 0.020 | 0.008 ms | 0.042 |
| DDIM-10 | 0.020 | 0.004 ms | 0.031 |

### Rayleigh

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 1.300 | 0.123 us | 1.300 |
| Drifting (dir.) | 1.299 | 0.117 us | 1.300 |
| Joint-kernel drift | 1.864 | 0.116 us | 1.864 |
| Joint Sinkhorn | 0.919 | 0.115 us | 0.920 |
| Condition-wise Sinkhorn | 0.048 | 0.115 us | 0.049 |
| WGAN | 0.085 | 0.108 us | 0.086 |
| DDPM | 0.020 | 0.045 ms | 0.145 |
| DDIM-100 | 0.020 | 0.039 ms | 0.128 |
| DDIM-50 | 0.020 | 0.019 ms | 0.073 |
| DDIM-20 | 0.020 | 0.008 ms | 0.042 |
| DDIM-10 | 0.020 | 0.004 ms | 0.031 |

### SSPA

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 5.656 | 0.121 us | 5.656 |
| Drifting (dir.) | 5.656 | 0.118 us | 5.656 |
| Joint-kernel drift | 8.112 | 0.119 us | 8.112 |
| Joint Sinkhorn | 3.956 | 0.117 us | 3.956 |
| Condition-wise Sinkhorn | 0.296 | 0.117 us | 0.297 |
| WGAN | 0.478 | 0.107 us | 0.478 |
| DDPM | 0.124 | 0.047 ms | 0.253 |
| DDIM-100 | 0.124 | 0.039 ms | 0.231 |
| DDIM-50 | 0.124 | 0.019 ms | 0.177 |
| DDIM-20 | 0.124 | 0.008 ms | 0.145 |
| DDIM-10 | 0.124 | 0.004 ms | 0.135 |

### TDL

| Method | Train [h] | Inference | Total [h] |
|---|---:|---:|---:|
| Drifting (res.) | 0.013 | 0.124 us | 0.013 |
| Drifting (dir.) | 0.007 | 0.116 us | 0.007 |
| Joint-kernel drift | 0.008 | 0.115 us | 0.008 |
| Joint Sinkhorn | 0.010 | 0.116 us | 0.010 |
| Condition-wise Sinkhorn | 0.009 | 0.115 us | 0.009 |
| WGAN | 0.022 | 0.108 us | 0.022 |
| DDPM | 0.005 | 0.045 ms | 0.007 |
| DDIM-100 | 0.005 | 0.039 ms | 0.006 |
| DDIM-50 | 0.005 | 0.019 ms | 0.006 |
| DDIM-20 | 0.005 | 0.008 ms | 0.006 |
| DDIM-10 | 0.005 | 0.004 ms | 0.005 |

