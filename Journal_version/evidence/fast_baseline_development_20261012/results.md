# AWGN baseline development results

All final EMA one-step results are retained, including failed recipes. These are development
checks, not downstream SER/BER results or a final across-family ranking.

| Recipe | Seed | SWD / floor | Mean RMSE | Variance / truth | Gate | Training seconds | Oracle outputs |
|---|---:|---:|---:|---:|---|---:|---:|
| ct_default | 9101 | 12.7038 | 0.86256 | 0.0642 | fail | 31.87 | 20480000 |
| ct_coarse | 9101 | 1.2375 | 0.02254 | 0.7478 | fail | 31.84 | 20480000 |
| ct_coarse_lr | 9101 | 1.2600 | 0.02197 | 0.7407 | fail | 31.77 | 20480000 |
| ct_coarse_lr_scale | 9101 | 1.2588 | 0.02296 | 0.7403 | fail | 32.07 | 20480000 |
| wgan_default | 9101 | 1.1108 | 0.04146 | 0.9485 | pass | 32.20 | 20480000 |
| wgan_low_lr | 9101 | 1.1395 | 0.04840 | 0.9469 | pass | 31.66 | 20480000 |
| wgan_wide_clip | 9101 | 2.1880 | 0.05665 | 0.5270 | fail | 32.04 | 20480000 |
| wgan_wide_clip_low_lr | 9101 | 1.5298 | 0.05772 | 0.7341 | fail | 32.08 | 20480000 |
| consistency_seed9102 | 9102 | 1.3081 | 0.02297 | 0.7460 | fail | 32.00 | 20480000 |
| consistency_seed9103 | 9103 | 1.2508 | 0.02285 | 0.7448 | fail | 31.95 | 20480000 |
| wgan_seed9102 | 9102 | 1.1359 | 0.04179 | 0.9441 | pass | 32.19 | 20480000 |
| wgan_seed9103 | 9103 | 1.1099 | 0.03780 | 0.9541 | pass | 31.98 | 20480000 |
| EDM + CT | 19101 | 4.5991 | 0.02313 | 0.1184 | fail | 55.71 | 40960000 |
| EDM + CD | 19101 | 1.0161 | 0.02501 | 1.0288 | pass | 68.03 | 40960000 |
| EDM + CT | 19104 | 4.6351 | 0.02098 | 0.1217 | fail | 56.29 | 40960000 |
| EDM + CD | 19104 | 1.0052 | 0.02437 | 1.0347 | pass | 69.26 | 40960000 |
| EDM + CT | 19105 | 4.6276 | 0.02130 | 0.1268 | fail | 54.44 | 40960000 |
| EDM + CD | 19105 | 1.0159 | 0.02470 | 1.0363 | pass | 67.68 | 40960000 |

Warm-start student seeds equal their pretraining seed plus 10000. Both student rows
include the complete teacher/pretraining cost. The teacher is shared only in execution.

The development gate requires conditional SWD/floor <= 1.5, conditional mean RMSE <= .1,
and variance/truth in [.8,1.2]. These thresholds are not confidence intervals.

## WGAN EMA lag

| Outer updates | Generator updates | Weights | SWD / floor | Mean RMSE | Variance / truth |
|---:|---:|---|---:|---:|---:|
| 5000 | 1000 | ema | 7.5269 | 0.50508 | 0.3058 |
| 5000 | 1000 | raw | 1.4173 | 0.07611 | 0.9892 |
| 10000 | 2000 | ema | 3.0941 | 0.20026 | 0.6779 |
| 10000 | 2000 | raw | 1.2429 | 0.05538 | 1.0178 |
| 15000 | 3000 | ema | 1.5031 | 0.07931 | 0.8698 |
| 15000 | 3000 | raw | 1.1466 | 0.04778 | 1.0048 |
| 20000 | 4000 | ema | 1.1108 | 0.04146 | 0.9485 |
| 20000 | 4000 | raw | 1.1499 | 0.05239 | 1.0049 |

EMA decay .999 is applied per generator update. Initial-weight contributions are
36.770% after 1000 generator updates and 1.828% after 4000.
This explains a substantial part of the short pilot's weakness, without changing the
generator sampling mechanism or claiming adversarial training universally succeeds.

## Sources

- /home/rick/GitHub/conditional_drifting_models/results/fast_baseline_development_20261012/results.json
- results/consistency_warmstart_development_20261012/results.json
- results/consistency_warmstart_development_20261012_seed9104/results.json
- results/consistency_warmstart_development_20261012_seed9105/results.json
