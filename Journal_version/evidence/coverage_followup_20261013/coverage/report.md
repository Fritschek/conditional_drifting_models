# Coverage replication

Profile: paper. 12 generators, 45 AEs.

Primary: broad-minus-normal physical SER at 8 dB, averaged within generator seed.
Negative is improvement. Generator seeds, not individual AEs, are the replication units.

| Family | Generator seed | Normal mean SER | Broad mean SER | Difference |
|---|---:|---:|---:|---:|
| gaussian | 9201 | 0.00111898 | 3.30607e-05 | -0.00108592 |
| gaussian | 9202 | 0.000207265 | 1.30335e-05 | -0.000194232 |
| gaussian | 9203 | 0.000598272 | 1.30335e-05 | -0.000585238 |
| sinkhorn | 9201 | 5.40415e-05 | 4.22796e-05 | -1.1762e-05 |
| sinkhorn | 9202 | 0.000221252 | 7.94729e-06 | -0.000213305 |
| sinkhorn | 9203 | 0.000187238 | 6.38962e-05 | -0.000123342 |

## All nominal-SNR pairs

| Family | Generator seed | AE seed | Normal SER | Broad SER | Difference |
|---|---:|---:|---:|---:|---:|
| gaussian | 9201 | 392010 | 0.000716209 | 3.24249e-05 | -0.000683784 |
| sinkhorn | 9201 | 392010 | 3.62396e-05 | 3.33786e-05 | -2.86102e-06 |
| gaussian | 9201 | 392011 | 0.000890732 | 2.57492e-05 | -0.000864983 |
| sinkhorn | 9201 | 392011 | 3.33786e-05 | 6.00815e-05 | +2.67029e-05 |
| gaussian | 9201 | 392012 | 0.00174999 | 4.1008e-05 | -0.00170898 |
| sinkhorn | 9201 | 392012 | 9.25064e-05 | 3.33786e-05 | -5.91278e-05 |
| gaussian | 9202 | 392020 | 1.23978e-05 | 2.28882e-05 | +1.04904e-05 |
| sinkhorn | 9202 | 392020 | 0.000344276 | 3.8147e-06 | -0.000340462 |
| gaussian | 9202 | 392021 | 0.000593185 | 1.43051e-05 | -0.00057888 |
| sinkhorn | 9202 | 392021 | 0.000290871 | 9.53674e-06 | -0.000281334 |
| gaussian | 9202 | 392022 | 1.62125e-05 | 1.90735e-06 | -1.43051e-05 |
| sinkhorn | 9202 | 392022 | 2.86102e-05 | 1.04904e-05 | -1.81198e-05 |
| gaussian | 9203 | 392030 | 0.00128746 | 6.67572e-06 | -0.00128078 |
| sinkhorn | 9203 | 392030 | 1.90735e-05 | 6.00815e-05 | +4.1008e-05 |
| gaussian | 9203 | 392031 | 7.43866e-05 | 1.71661e-05 | -5.72205e-05 |
| sinkhorn | 9203 | 392031 | 0.000464439 | 6.10352e-05 | -0.000403404 |
| gaussian | 9203 | 392032 | 0.000432968 | 1.52588e-05 | -0.000417709 |
| sinkhorn | 9203 | 392032 | 7.82013e-05 | 7.05719e-05 | -7.62939e-06 |

## Conditional fidelity and cost

| Seed | Variant | Ordinary SWD/floor | Radius-10 mean RMSE | Max-radius variance/truth | Mean query energy | Train seconds |
|---:|---|---:|---:|---:|---:|---:|
| 9201 | gaussian_normal | 1.136 | 0.4235 | 0.03241 | 8 | 42.33 |
| 9201 | gaussian_broad | 1.055 | 0.02936 | 1.076 | 48.66 | 45.95 |
| 9201 | sinkhorn_normal | 1.276 | 0.4853 | 0.54 | 7.997 | 134.75 |
| 9201 | sinkhorn_broad | 1.471 | 0.07414 | 0.5653 | 48.5 | 138.27 |
| 9202 | gaussian_normal | 1.139 | 0.424 | 0.02928 | 8 | 41.32 |
| 9202 | gaussian_broad | 1.055 | 0.02848 | 1.077 | 48.64 | 45.27 |
| 9202 | sinkhorn_normal | 1.215 | 0.4589 | 0.5765 | 8.006 | 135.11 |
| 9202 | sinkhorn_broad | 1.511 | 0.07868 | 0.562 | 48.52 | 138.29 |
| 9203 | gaussian_normal | 1.123 | 0.4206 | 0.03308 | 8.001 | 41.06 |
| 9203 | gaussian_broad | 1.061 | 0.02906 | 1.078 | 48.65 | 44.98 |
| 9203 | sinkhorn_normal | 1.208 | 0.4257 | 0.5361 | 8.003 | 135.65 |
| 9203 | sinkhorn_broad | 1.43 | 0.06432 | 0.5912 | 48.44 | 138.04 |

## Transfer optimism and codewords

| Seed | AE seed | Variant | Physical SER | Surrogate SER | Maximum codeword norm | AE seconds |
|---:|---:|---|---:|---:|---:|---:|
| 9201 | 392010 | analytic | 2.67029e-05 | 1.81198e-05 | 3.536 | 32.01 |
| 9201 | 392010 | gaussian_normal | 0.000716209 | 7.62939e-06 | 8.433 | 36.51 |
| 9201 | 392010 | gaussian_broad | 3.24249e-05 | 2.09808e-05 | 4.359 | 36.13 |
| 9201 | 392010 | sinkhorn_normal | 3.62396e-05 | 6.67572e-06 | 5.31 | 35.85 |
| 9201 | 392010 | sinkhorn_broad | 3.33786e-05 | 1.52588e-05 | 4.169 | 35.24 |
| 9201 | 392011 | analytic | 3.43323e-05 | 4.29153e-05 | 3.862 | 31.82 |
| 9201 | 392011 | gaussian_normal | 0.000890732 | 2.86102e-05 | 7.716 | 36.54 |
| 9201 | 392011 | gaussian_broad | 2.57492e-05 | 2.38419e-05 | 3.853 | 35.81 |
| 9201 | 392011 | sinkhorn_normal | 3.33786e-05 | 1.62125e-05 | 4.678 | 35.72 |
| 9201 | 392011 | sinkhorn_broad | 6.00815e-05 | 2.86102e-05 | 4.272 | 35.41 |
| 9201 | 392012 | analytic | 3.62396e-05 | 2.76566e-05 | 3.63 | 31.99 |
| 9201 | 392012 | gaussian_normal | 0.00174999 | 1.71661e-05 | 9.337 | 36.27 |
| 9201 | 392012 | gaussian_broad | 4.1008e-05 | 4.673e-05 | 4.159 | 36.97 |
| 9201 | 392012 | sinkhorn_normal | 9.25064e-05 | 2.57492e-05 | 4.615 | 35.39 |
| 9201 | 392012 | sinkhorn_broad | 3.33786e-05 | 1.71661e-05 | 4.342 | 36.17 |
| 9202 | 392020 | analytic | 6.67572e-06 | 6.67572e-06 | 3.479 | 31.12 |
| 9202 | 392020 | gaussian_normal | 1.23978e-05 | 6.67572e-06 | 7.151 | 35.25 |
| 9202 | 392020 | gaussian_broad | 2.28882e-05 | 2.19345e-05 | 4.357 | 37.01 |
| 9202 | 392020 | sinkhorn_normal | 0.000344276 | 8.58307e-06 | 6.986 | 35.38 |
| 9202 | 392020 | sinkhorn_broad | 3.8147e-06 | 1.90735e-06 | 4.554 | 35.86 |
| 9202 | 392021 | analytic | 4.95911e-05 | 3.52859e-05 | 3.764 | 31.88 |
| 9202 | 392021 | gaussian_normal | 0.000593185 | 3.62396e-05 | 8.746 | 36.97 |
| 9202 | 392021 | gaussian_broad | 1.43051e-05 | 6.67572e-06 | 3.696 | 36.82 |
| 9202 | 392021 | sinkhorn_normal | 0.000290871 | 7.82013e-05 | 5.614 | 35.47 |
| 9202 | 392021 | sinkhorn_broad | 9.53674e-06 | 9.53674e-06 | 5.201 | 35.84 |
| 9202 | 392022 | analytic | 2.86102e-06 | 3.8147e-06 | 3.46 | 32.06 |
| 9202 | 392022 | gaussian_normal | 1.62125e-05 | 9.53674e-06 | 7.127 | 35.59 |
| 9202 | 392022 | gaussian_broad | 1.90735e-06 | 2.86102e-06 | 3.689 | 36.97 |
| 9202 | 392022 | sinkhorn_normal | 2.86102e-05 | 1.43051e-05 | 4.288 | 35.67 |
| 9202 | 392022 | sinkhorn_broad | 1.04904e-05 | 7.62939e-06 | 3.802 | 34.47 |
| 9203 | 392030 | analytic | 3.8147e-06 | 9.53674e-07 | 3.665 | 31.99 |
| 9203 | 392030 | gaussian_normal | 0.00128746 | 4.76837e-06 | 9.305 | 35.47 |
| 9203 | 392030 | gaussian_broad | 6.67572e-06 | 4.76837e-06 | 4.417 | 36.57 |
| 9203 | 392030 | sinkhorn_normal | 1.90735e-05 | 0 | 6.543 | 35.34 |
| 9203 | 392030 | sinkhorn_broad | 6.00815e-05 | 2.67029e-05 | 5.409 | 36.54 |
| 9203 | 392031 | analytic | 4.76837e-06 | 5.72205e-06 | 3.956 | 31.68 |
| 9203 | 392031 | gaussian_normal | 7.43866e-05 | 7.62939e-06 | 9.325 | 36.52 |
| 9203 | 392031 | gaussian_broad | 1.71661e-05 | 9.53674e-06 | 3.8 | 36.03 |
| 9203 | 392031 | sinkhorn_normal | 0.000464439 | 2.28882e-05 | 6.725 | 35.88 |
| 9203 | 392031 | sinkhorn_broad | 6.10352e-05 | 2.67029e-05 | 5.079 | 36.14 |
| 9203 | 392032 | analytic | 1.04904e-05 | 1.04904e-05 | 3.544 | 31.43 |
| 9203 | 392032 | gaussian_normal | 0.000432968 | 6.67572e-06 | 9.131 | 36.17 |
| 9203 | 392032 | gaussian_broad | 1.52588e-05 | 1.04904e-05 | 4.495 | 36.26 |
| 9203 | 392032 | sinkhorn_normal | 7.82013e-05 | 8.58307e-06 | 5.246 | 35.48 |
| 9203 | 392032 | sinkhorn_broad | 7.05719e-05 | 4.48227e-05 | 4.546 | 35.68 |

Surrogate/physical evaluations here use independent noise. Paired physical comparisons
use common noise and retain discordance counts in results.json. Expected query energy
is 6.0833x higher under the broader law; no claim of equal RF measurement energy.
