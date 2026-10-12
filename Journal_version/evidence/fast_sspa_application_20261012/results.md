# SSPA fast-generator application pilot

Development seed 9111. Fixed final EMA checkpoints, no downstream selection.
Each learned method receives 40,960,000 analytic training outputs. This is not matched compute.
The consistency model includes learned EDM teacher pretraining. All downstream evaluation
uses the analytic channel, after fresh AE training through each frozen surrogate.

## Communication results

| Method | Eb/N0 (dB) | SER | Symbol errors / trials | Natural-label BER | Bit errors |
|---|---:|---:|---|---:|---:|
| Analytic | 4 | 0.00509453 | 5342 / 1048576 | 0.00262817 | 16535 |
| Analytic | 6 | 0.000305176 | 320 / 1048576 | 0.000153701 | 967 |
| Analytic | 8 | 5.72205e-06 | 6 / 1048576 | 2.70208e-06 | 17 |
| Gaussian | 4 | 0.017437 | 18284 / 1048576 | 0.00915384 | 57591 |
| Gaussian | 6 | 0.00650597 | 6822 / 1048576 | 0.0035216 | 22156 |
| Gaussian | 8 | 0.0031004 | 3251 / 1048576 | 0.00172567 | 10857 |
| WGAN | 4 | 0.00132656 | 1391 / 1048576 | 0.000687281 | 4324 |
| WGAN | 6 | 6.48499e-05 | 68 / 1048576 | 3.25839e-05 | 205 |
| WGAN | 8 | 1.90735e-06 | 2 / 1048576 | 7.94729e-07 | 5 |
| DDIM-20 | 4 | 0.00849056 | 8903 / 1048576 | 0.00405709 | 25525 |
| DDIM-20 | 6 | 0.000886917 | 930 / 1048576 | 0.000371933 | 2340 |
| DDIM-20 | 8 | 6.77109e-05 | 71 / 1048576 | 2.06629e-05 | 130 |
| Flow-8 | 4 | 0.0149593 | 15686 / 1048576 | 0.00763226 | 48018 |
| Flow-8 | 6 | 0.00178814 | 1875 / 1048576 | 0.000913779 | 5749 |
| Flow-8 | 8 | 0.000149727 | 157 / 1048576 | 7.9155e-05 | 498 |
| Sinkhorn K64 | 4 | 0.00887966 | 9311 / 1048576 | 0.00459162 | 28888 |
| Sinkhorn K64 | 6 | 0.00112057 | 1175 / 1048576 | 0.000572522 | 3602 |
| Sinkhorn K64 | 8 | 7.72476e-05 | 81 / 1048576 | 3.78291e-05 | 238 |
| Consistency distillation | 4 | 0.010067 | 10556 / 1048576 | 0.00550079 | 34608 |
| Consistency distillation | 6 | 0.00185966 | 1950 / 1048576 | 0.00118192 | 7436 |
| Consistency distillation | 8 | 0.000436783 | 458 / 1048576 | 0.000304222 | 1914 |

## Distribution and moments

Conditional metrics are output-space distances at fixed inputs. Floors use two independent
analytic output clouds. Codebook rows use the final common analytic-trained encoder.

| Method | Global SWD | Global floor | Anchor SWD / floor | Codebook SWD / floor | Mean RMSE | Variance / truth |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 0.0069261 | 0.0072921 | 1.1288 | 1.0406 | 0.021423 | 1.0042 |
| WGAN | 0.024892 | 0.0072921 | 11.245 | 6.974 | 0.29533 | 4.083 |
| DDIM-20 | 0.019147 | 0.0072921 | 1.7402 | 1.6052 | 0.046191 | 0.83834 |
| Flow-8 | 0.01094 | 0.0072921 | 1.9865 | 1.8952 | 0.023723 | 0.59165 |
| Sinkhorn K64 | 0.0072413 | 0.0072921 | 1.2495 | 1.1203 | 0.026572 | 0.98857 |
| Consistency distillation | 0.0074244 | 0.0072921 | 1.5761 | 1.2411 | 0.03705 | 1.0758 |

## Common-codec gradients

Mean-vector comparisons include every method. Interpret direction comparisons only when
the analytic reference is resolved; the flag is a Monte Carlo heuristic, not a guarantee.

| Method | Codec | Space | Cosine | Relative error | Norm ratio | Ref signal / MC SE | Split cosine | Resolved |
|---|---|---|---:|---:|---:|---:|---:|---|
| Gaussian | early | code | 0.97085 | 0.26702 | 1.0885 | 13.724 | 0.9933 | True |
| Gaussian | early | encoder | 0.99107 | 0.16513 | 1.0884 | 28.214 | 0.99768 | True |
| Gaussian | final | code | 0.84466 | 0.55613 | 0.99542 | 2.2673 | 0.73283 | False |
| Gaussian | final | encoder | 0.90965 | 0.42674 | 0.81182 | 3.2958 | 0.88104 | False |
| WGAN | early | code | 0.034478 | 178.34 | 178.37 | 13.724 | 0.9933 | True |
| WGAN | early | encoder | 0.62671 | 146.67 | 147.3 | 28.214 | 0.99768 | True |
| WGAN | final | code | 0.13462 | 1451.3 | 1451.5 | 2.2673 | 0.73283 | False |
| WGAN | final | encoder | 0.26797 | 1961.1 | 1961.4 | 3.2958 | 0.88104 | False |
| DDIM-20 | early | code | 0.78596 | 0.62364 | 0.86761 | 13.724 | 0.9933 | True |
| DDIM-20 | early | encoder | 0.96048 | 0.40015 | 0.67301 | 28.214 | 0.99768 | True |
| DDIM-20 | final | code | 0.57641 | 0.90623 | 0.18461 | 2.2673 | 0.73283 | False |
| DDIM-20 | final | encoder | 0.70862 | 0.89121 | 0.16419 | 3.2958 | 0.88104 | False |
| Flow-8 | early | code | 0.90682 | 0.63911 | 0.42642 | 13.724 | 0.9933 | True |
| Flow-8 | early | encoder | 0.97237 | 0.62888 | 0.38842 | 28.214 | 0.99768 | True |
| Flow-8 | final | code | 0.76086 | 0.97215 | 0.036992 | 2.2673 | 0.73283 | False |
| Flow-8 | final | encoder | 0.83513 | 0.97508 | 0.030001 | 3.2958 | 0.88104 | False |
| Sinkhorn K64 | early | code | 0.9366 | 0.38253 | 1.09 | 13.724 | 0.9933 | True |
| Sinkhorn K64 | early | encoder | 0.99174 | 0.13573 | 1.0361 | 28.214 | 0.99768 | True |
| Sinkhorn K64 | final | code | 0.6694 | 0.74339 | 0.6964 | 2.2673 | 0.73283 | False |
| Sinkhorn K64 | final | encoder | 0.84555 | 0.56912 | 0.64843 | 3.2958 | 0.88104 | False |
| Consistency distillation | early | code | 0.84934 | 0.68361 | 1.2837 | 13.724 | 0.9933 | True |
| Consistency distillation | early | encoder | 0.97187 | 0.24694 | 1.0461 | 28.214 | 0.99768 | True |
| Consistency distillation | final | code | 0.81832 | 0.66597 | 1.1547 | 2.2673 | 0.73283 | False |
| Consistency distillation | final | encoder | 0.89058 | 0.482 | 1.0501 | 3.2958 | 0.88104 | False |

## Measured costs

Seconds cover synchronized training updates, including optimizer/EMA work, excluding
initialization, evaluation and checkpoint I/O. No extrapolation. Latency is sampling only.

| Method | Parameters | NFE | Generator train (s) | AE train (s) | Batch-1 (ms) | Batch-500 amortized (ms/sample) |
|---|---:|---:|---:|---:|---:|---:|
| Analytic | - | - | - | 35.52 | - | - |
| Gaussian | 59872 | 1 | 42.80 | 37.48 | 0.21833 | 0.00040885 |
| WGAN | 60235 | 1 | 68.32 | 36.21 | 0.18435 | 0.00039423 |
| DDIM-20 | 60176 | 20 | 51.13 | 362.86 | 12.902 | 0.02542 |
| Flow-8 | 60176 | 8 | 42.90 | 88.70 | 1.7562 | 0.0036047 |
| Sinkhorn K64 | 60235 | 1 | 136.64 | 37.20 | 0.18204 | 0.00037452 |
| Consistency distillation | 60176 | 1 | 81.59 | 41.77 | 0.36439 | 0.00074481 |

## Own-codebook transfer check (post hoc)

The same final surrogate-trained codec is evaluated through both channels at nominal SNR.
These checks did not affect training or selection. Common-codec and own-codebook metrics
answer different questions; they should not be pooled.

| Method | Own anchor SWD / floor | Mean RMSE | Variance / truth | Surrogate SER | Analytic SER | Analytic errors / trials |
|---|---:|---:|---:|---:|---:|---|
| Gaussian | 2.1331 | 0.081776 | 0.98397 | 3.62396e-05 | 0.00297546 | 1560 / 524288 |
| WGAN | 12.173 | 0.10994 | 2.9918 | 0.00288963 | 0 | 0 / 524288 |
| DDIM-20 | 3.1776 | 0.051375 | 0.7937 | 0 | 5.91278e-05 | 31 / 524288 |
| Flow-8 | 4.1486 | 0.051868 | 0.58716 | 0 | 0.0001297 | 68 / 524288 |
| Sinkhorn K64 | 1.8421 | 0.036768 | 0.98209 | 1.14441e-05 | 9.15527e-05 | 48 / 524288 |
| Consistency distillation | 2.2589 | 0.045573 | 1.0307 | 1.90735e-06 | 0.000421524 | 221 / 524288 |

## Input coverage and error localization (post hoc)

Every codebook has mean squared norm eight. The power constraint does not bound each
individual codeword. Training uses iid standard-normal coordinates; the radial tail
is computed from chi-square(8). Error counts exactly replay the preceding evaluation.

| Method | Maximum input norm | Training-law probability above that norm | Errors at two largest-norm words / all errors |
|---|---:|---:|---|
| Gaussian | 10.272 | 3.1817e-19 | 1532 / 1560 |
| WGAN | 3.0992 | 0.29383 | 0 / 0 |
| DDIM-20 | 6.1849 | 6.7608e-06 | 8 / 31 |
| Flow-8 | 7.5399 | 1.9263e-09 | 42 / 68 |
| Sinkhorn K64 | 6.9189 | 1.0455e-07 | 32 / 48 |
| Consistency distillation | 8.9861 | 3.4532e-14 | 217 / 221 |

## Scope

One development seed does not establish a significant model-family ranking.
The oracle observations are equal but distinct-input coverage and training compute differ.
The analytic SSPA conditional law is Gaussian; this channel alone cannot establish a need
for a flexible non-Gaussian generator. Fixed NFEs and recipes are not per-family optima.
BER uses natural binary labels; there is no claim of optimized bit labeling.

Source suite: `results/fast_sspa_application_20261012`. See the frozen protocol for all budgets and settings.
