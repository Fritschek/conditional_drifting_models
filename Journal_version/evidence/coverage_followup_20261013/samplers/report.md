# Saved-checkpoint sampler panel

Profile: paper. No retraining or NFE selection.

| Family | NFE | Conditional SWD/floor | Variance/truth | Sampling ms/batch | Forward+input backward ms/batch | Encoder cosine | Reference resolved |
|---|---:|---:|---:|---:|---:|---:|---|
| flow | 8 | 1.911 | 0.6012 | 1.832 | 3.055 | 0.976 | True |
| flow | 32 | 1.239 | 0.9004 | 7.005 | 10.88 | 0.9952 | True |
| flow | 128 | 1.206 | 0.9832 | 28.01 | 42.2 | 0.9937 | True |
| diffusion | 20 | 1.819 | 0.8526 | 12.8 | 16.67 | 0.9599 | True |
| diffusion | 50 | 1.723 | 0.9812 | 32.04 | 40.82 | 0.9225 | True |
| diffusion | 100 | 1.805 | 1.045 | 64.14 | 81.23 | 0.8848 | True |

Timing batch: 500. These are batch times, not per-sample latency.
Batch-one measurements and full raw timing repeats are retained in results.json.

| Family | NFE | Output RMSE vs finest | Pathwise Jacobian RMSE vs finest |
|---|---:|---:|---:|
| flow | 8 | 0.0488457 | 0.00753297 |
| flow | 32 | 0.0097063 | 0.00145554 |
| flow | 128 | 0 | 0 |
| diffusion | 20 | 0.0377293 | 0.0285145 |
| diffusion | 50 | 0.0147492 | 0.0118185 |
| diffusion | 100 | 0 | 0 |

The finest sampler is not an analytic reference. Differences test finite-grid
self-consistency only. Resolved flags qualify MC reference precision, not model correctness.
