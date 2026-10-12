# SSPA input-coverage results

One generator seed and paired AE initializations. No checkpoint selection.
All downstream curves use analytic SSPA. Normal and broad generators have identical
architectures, output-query counts and optimization settings; only training input law changes.

| Variant | AE seed | SER 4 dB | SER 6 dB | SER 8 dB | Surrogate SER 8 dB | Maximum input norm | Own mean RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic | 209111 | 0.00509453 | 0.000305176 | 5.72205e-06 | 7.62939e-06 | 3.4765 | 0.014362 |
| analytic | 209112 | 0.0110359 | 0.00094223 | 2.67029e-05 | 2.57492e-05 | 3.5522 | 0.014362 |
| analytic | 209113 | 0.00737667 | 0.000610352 | 1.14441e-05 | 1.62125e-05 | 3.3785 | 0.014362 |
| gaussian_normal | 209111 | 0.017437 | 0.00650597 | 0.0031004 | 5.91278e-05 | 10.272 | 0.082893 |
| gaussian_normal | 209112 | 0.0174685 | 0.00381756 | 0.00083065 | 2.95639e-05 | 8.5239 | 0.072245 |
| gaussian_normal | 209113 | 0.0185728 | 0.00748634 | 0.00452518 | 5.43594e-05 | 8.8433 | 0.072101 |
| gaussian_broad | 209111 | 0.00765991 | 0.0007267 | 4.1008e-05 | 3.43323e-05 | 4.1106 | 0.015923 |
| gaussian_broad | 209112 | 0.0130081 | 0.001544 | 9.53674e-05 | 0.000102997 | 5.313 | 0.015894 |
| gaussian_broad | 209113 | 0.00805855 | 0.00100231 | 0.00011158 | 8.67844e-05 | 3.8229 | 0.0162 |
| sinkhorn_normal | 209111 | 0.00887966 | 0.00112057 | 7.72476e-05 | 1.04904e-05 | 6.9189 | 0.039654 |
| sinkhorn_normal | 209112 | 0.0100365 | 0.000958443 | 3.43323e-05 | 1.04904e-05 | 5.186 | 0.026115 |
| sinkhorn_normal | 209113 | 0.0121803 | 0.00239563 | 0.00044632 | 5.72205e-06 | 7.8395 | 0.061997 |
| sinkhorn_broad | 209111 | 0.0106392 | 0.00130844 | 9.91821e-05 | 9.82285e-05 | 4.0969 | 0.029726 |
| sinkhorn_broad | 209112 | 0.0120182 | 0.00112247 | 4.00543e-05 | 2.57492e-05 | 4.2553 | 0.030293 |
| sinkhorn_broad | 209113 | 0.011075 | 0.00136852 | 9.44138e-05 | 0.000114441 | 4.3196 | 0.030849 |

## Generator accuracy and cost

| Variant | Random anchor SWD / floor | Mean RMSE on old Gaussian codebook | Generator seconds | Broad inputs / total inputs |
|---|---:|---:|---:|---|
| gaussian_normal | 1.0973 | 0.084089 | 42.80 | 0 / 40960000 |
| gaussian_broad | 1.0609 | 0.016919 | 46.37 | 10236313 / 40960000 |
| sinkhorn_normal | 1.2004 | 0.083415 | 136.64 | 0 / 640000 |
| sinkhorn_broad | 1.4548 | 0.033176 | 142.72 | 159630 / 640000 |

## Radial accuracy

| Variant | Input norm | Mean RMSE | Variance / analytic |
|---|---:|---:|---:|
| gaussian_normal | 2.8284 | 0.019216 | 1.0066 |
| gaussian_normal | 6 | 0.083946 | 0.98523 |
| gaussian_normal | 10 | 0.3491 | 0.61341 |
| gaussian_normal | 16 | 0.8701 | 0.1833 |
| gaussian_normal | 22.627 | 1.4921 | 0.040871 |
| gaussian_broad | 2.8284 | 0.017281 | 1.0066 |
| gaussian_broad | 6 | 0.02113 | 1.0163 |
| gaussian_broad | 10 | 0.026252 | 1.018 |
| gaussian_broad | 16 | 0.035634 | 1.0405 |
| gaussian_broad | 22.627 | 0.050235 | 1.0714 |
| sinkhorn_normal | 2.8284 | 0.025471 | 0.98649 |
| sinkhorn_normal | 6 | 0.11347 | 0.80151 |
| sinkhorn_normal | 10 | 0.45777 | 0.6646 |
| sinkhorn_normal | 16 | 1.1095 | 0.58921 |
| sinkhorn_normal | 22.627 | 1.872 | 0.55729 |
| sinkhorn_broad | 2.8284 | 0.037081 | 1.0002 |
| sinkhorn_broad | 6 | 0.053099 | 0.96362 |
| sinkhorn_broad | 10 | 0.071066 | 0.84351 |
| sinkhorn_broad | 16 | 0.082331 | 0.68918 |
| sinkhorn_broad | 22.627 | 0.13472 | 0.58729 |

## Limits

Three AE initializations do not replace independent generator seeds.
The broad-law fraction and radius were fixed before training. No conclusions about the
optimal mixture or universal applicability follow. Ordinary-input error may increase.
Mean-error measurements include finite-sample noise, with floors retained in raw JSON.
Zero observed errors are kept as zero in tables and omitted from log axes, not floored.
No peak-power constraint, extra physical output restriction or downstream feedback was added.
Expected training-query squared norm increases from 8 to 48.6667; equal oracle counts
are not equal query energy. The physical noise and AE average-power constraint are unchanged.
Source: `results/sspa_input_coverage_20261012`. Audit passed for 15 rows and unchanged parent artifacts.
