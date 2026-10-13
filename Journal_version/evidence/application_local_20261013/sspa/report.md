# Local sspa development

Profile: development; seed: 9301.

Development outcomes only. Final confirmation and model selection remain separate.

## Generator cost

| Family | Parameters | Oracle outputs (including teacher) | Training seconds | Mean input energy per anchor |
|---|---:|---:|---:|---:|
| gaussian | 59872 | 40960000 | 45.00 | 48.69 |
| sinkhorn | 60235 | 40960000 | 139.38 | 48.79 |
| wgan | 60235 | 40960000 | 69.32 | 48.69 |
| flow | 60176 | 40960000 | 45.45 | 48.69 |
| diffusion | 60176 | 40960000 | 50.77 | 48.69 |
| consistency_distilled | 60176 | 40960000 | 72.71 | 48.67 |

Physical input-referenced Eb/N0 is shown. These retain the previous physical
variance, with labels 3.0103 dB above the historical software parameter.

| Method | Physical Eb/N0 (dB) | SER | BER | AE training seconds |
|---|---:|---:|---:|---:|
| analytic | 7.0103 | 0.00576496 | 0.00284004 | 32.10 |
| gaussian_1 | 7.0103 | 0.00506878 | 0.00254599 | 36.81 |
| sinkhorn_1 | 7.0103 | 0.00615692 | 0.00309118 | 34.68 |
| wgan_1 | 7.0103 | 0.0303926 | 0.0160219 | 36.36 |
| flow_32 | 7.0103 | 0.00920486 | 0.00445827 | 244.36 |
| flow_128 | 7.0103 | 0.0131254 | 0.00656255 | 912.07 |
| diffusion_20 | 7.0103 | 0.00605774 | 0.00306813 | 391.42 |
| diffusion_50 | 7.0103 | 0.00523758 | 0.00257794 | 916.74 |
| consistency_distilled_1 | 7.0103 | 0.0082798 | 0.00421858 | 45.38 |
| analytic | 9.0103 | 0.000424385 | 0.000212193 | 32.10 |
| gaussian_1 | 9.0103 | 0.000291824 | 0.000144641 | 36.81 |
| sinkhorn_1 | 9.0103 | 0.00052166 | 0.000257174 | 34.68 |
| wgan_1 | 9.0103 | 0.010994 | 0.0062569 | 36.36 |
| flow_32 | 9.0103 | 0.00106239 | 0.000478268 | 244.36 |
| flow_128 | 9.0103 | 0.00228119 | 0.00116952 | 912.07 |
| diffusion_20 | 9.0103 | 0.000380516 | 0.000199477 | 391.42 |
| diffusion_50 | 9.0103 | 0.000287056 | 0.000132402 | 916.74 |
| consistency_distilled_1 | 9.0103 | 0.000810623 | 0.000407219 | 45.38 |
| analytic | 11.0103 | 1.14441e-05 | 4.29153e-06 | 32.10 |
| gaussian_1 | 11.0103 | 5.72205e-06 | 2.22524e-06 | 36.81 |
| sinkhorn_1 | 11.0103 | 1.23978e-05 | 6.51677e-06 | 34.68 |
| wgan_1 | 11.0103 | 0.00379848 | 0.0023934 | 36.36 |
| flow_32 | 11.0103 | 6.67572e-05 | 2.54313e-05 | 244.36 |
| flow_128 | 11.0103 | 0.000295639 | 0.000151634 | 912.07 |
| diffusion_20 | 11.0103 | 4.76837e-06 | 2.70208e-06 | 391.42 |
| diffusion_50 | 11.0103 | 2.86102e-06 | 1.27157e-06 | 916.74 |
| consistency_distilled_1 | 11.0103 | 3.52859e-05 | 1.7643e-05 | 45.38 |

| Method | Ordinary-input conditional SWD/floor |
|---|---:|
| gaussian_1 | 1.068 |
| sinkhorn_1 | 1.523 |
| wgan_1 | 12.14 |
| flow_32 | 1.207 |
| flow_128 | 1.172 |
| diffusion_20 | 1.51 |
| diffusion_50 | 1.321 |
| consistency_distilled_1 | 1.617 |
