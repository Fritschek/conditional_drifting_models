# Local application development results

This note tracks the [new local protocol](application_local_protocol_20261013.md)
and the [receiver-gate addendum](ofdm_receiver_gate_20261013.md). It is distinct
from the completed coverage replication and saved-sampler assessment. No
cluster job has been submitted.

## Calibration and implementation

The independent SSPA energy/noise tests pass. The common-input comparison keeps
the historical physical noise and uses explicit input-referenced physical SNR
labels. Both clean and intermittent-interference OFDM conditions pass empirical
noise/covariance checks and the shared block-activity check. All 24 profile
delays/powers match the pinned Sionna v1.2.2 JSON.

The application tests pass on CPU and CUDA (ten each), including the
frequency receiver. The original
CUDA smoke suites exercise every supported generator family, including full
Gaussian mixtures and learned-constellation gradients, and their completed
resumes leave result records unchanged. The frequency-receiver version also
passes its all-family CPU smoke and completed-resume check. Smoke runs are
excluded from scientific tables.

## The receiver needed a frequency-structured architecture

The original plain MLP receiver was a bottleneck even with true-channel training.
Before running surrogate pilots, the receiver-only reference was tested with
the same 5000 updates and batch 256 using a frequency residual network. It has
25,604 parameters and receives I/Q plus a fixed pilot-position mask, without
channel covariance, fading realizations or activity labels.

Clean-channel results on 16,384 physical test blocks per point:

| Es/N0 (dB) | Plain MLP BER | Frequency receiver BER | Conventional pilot-LMMSE BER |
|---|---:|---:|---:|
| 0 | 0.29162 | 0.26589 | 0.25255 |
| 5 | 0.20234 | 0.14440 | 0.14176 |
| 10 | 0.14492 | 0.06349 | 0.06455 |
| 15 | 0.11787 | 0.02750 | 0.02566 |
| 20 | 0.10802 | 0.01557 | 0.00970 |

The nominal-point frequency receiver is numerically close to the conventional
reference. It still loses at high SNR, so the pilot does not establish universal
receiver superiority or full convergence. The conventional receiver knows
second-order channel statistics and test noise power. The neural receiver is
trained only at 10 dB and receives no explicit test-SNR input.

For the interference condition, the frequency receiver BER is
0.26743/0.15795/0.08415/0.05115/0.04020 over the same SNR grid, compared with
0.29682/0.21315/0.16012/0.13542/0.12642 for the plain MLP. Activity-stratified
counts and block-error rates remain in the records. Both comparisons use one
receiver initialization; these are development gates, not final reliability
estimates. The pilot-assisted residual architecture was implemented but not
trained, because the frequency receiver already provided a useful reference.

The full plain-MLP surrogate campaign was not launched. The separate frequency
pilot applies the same receiver to every implant and increases the common
generator budget to 10,000 updates so it extends beyond Sinkhorn's warmup.
The physical law, test protocol and receiver budget remain unchanged.

## Common-input SSPA

The full one-seed run completed at `results/common_coverage_9301/`. Its audit
verifies 27 curve points, 16 checkpoints and 43 frozen source/protocol files.
All six generators used 40.96 million physical outputs, including the teacher
for distillation, the same broad input law and approximately 60k parameters.
Each operating point trained its own AE for 20k updates, batch 500.

At physical input Eb/N0 = 11.0103 dB (historical software label 8 dB), the
paired evaluation contains 1,048,576 symbols per method:

| Method | Symbol errors | SER | AE training seconds |
|---|---:|---:|---:|
| Physical channel | 12 | 1.1444e-5 | 32.10 |
| Gaussian | 6 | 5.7220e-6 | 36.81 |
| Sinkhorn | 13 | 1.2398e-5 | 34.68 |
| WGAN | 3983 | 3.7985e-3 | 36.36 |
| Flow-32 | 70 | 6.6757e-5 | 244.36 |
| Flow-128 | 310 | 2.9564e-4 | 912.07 |
| DDIM-20 | 5 | 4.7684e-6 | 391.42 |
| DDIM-50 | 3 | 2.8610e-6 | 916.74 |
| Distilled consistency | 37 | 3.5286e-5 | 45.38 |

These are one generator seed and one AE seed. Small differences in very low
error counts do not establish population rankings. The full curves and raw
paired counts are in [the compact records](evidence/application_local_20261013/sspa/).
Generator training took 422.62 seconds across the six families; the nine AEs
took 2649.92 seconds in total. These figures exclude evaluation and overhead.

Flow-128 improves ordinary-input conditional SWD/floor from 1.207 to 1.172,
yet its trained AE has more physical errors than Flow-32. WGAN remains a weak
baseline with this broad-input recipe and needs bounded validation before a
final comparison. All maximum codeword norms are below 4.8, so the earlier
extreme-norm extrapolation explanation does not automatically apply here.

### Saved-codebook check

An additional, explicitly post-hoc CPU evaluation uses new paired physical
noise and each frozen codebook. Equal priors and isotropic additive SSPA noise
make nearest physical output mean the exact maximum-likelihood decoder.
This does not retrain a receiver or change any primary result.

| Method | Saved neural decoder errors | Exact physical ML errors |
|---|---:|---:|
| Physical channel | 16 | 1 |
| Gaussian | 8 | 1 |
| Sinkhorn | 22 | 0 |
| WGAN | 3957 | 4 |
| Flow-32 | 69 | 1 |
| Flow-128 | 289 | 0 |
| DDIM-20 | 8 | 0 |
| DDIM-50 | 7 | 0 |
| Distilled consistency | 33 | 0 |

Again, each row has 1,048,576 physical symbols. This localizes much of the
observed error to receiver transfer/training, rather than poor physical
separation of the learned codewords. It does not uniquely identify which
training dynamic caused the discrepancy. Zero observed errors are not zero
error probabilities. The fresh CPU draws differ from the primary GPU draws.
At the learned codewords, WGAN's estimated variance is 3.21 times physical
variance and its mean RMSE is 0.224. Flow-32/128 variance ratios are 0.883/0.959,
so correcting this moment alone does not ensure decoder transfer. The moment
panel uses 512 samples per message and has analytic mean-RMSE floor 0.01016.

## Structured surrogate pilots

The clean frequency-receiver pilot completed and passed its 35-point,
12-checkpoint, 41-source audit. GPU experiments are serialized for timing.
Five generator families use about 600k parameters, 10k updates and 2.56 million
physical outputs each; all receivers use 5k updates, batch 256. Distillation
includes teacher training in its common observation budget.

At the nominal 10 dB Es/N0, physical BER is 0.06349 for physical-channel
receiver training and 0.06414 for conventional pilot-LMMSE. Learned implants
give 0.15461 (full Gaussian), 0.13591 (full mixture), 0.10843 (Sinkhorn),
0.10465 (Flow-32), and 0.11264 (distilled consistency). Sinkhorn has the best
conditional SWD/floor and pilot/data covariance-error/floor among these learned
models (1.524 and 1.367), but does not have the lowest numerical BER.

The full Gaussian/mixture heads are expensive parameterizations of a structured
covariance law and may be underfit. Their weakness is not evidence that the
Gaussian family is intrinsically unsuitable: the clean law is exactly Gaussian.
All learned models leave a material receiver-transfer gap. A learned structured
covariance baseline and a bounded convergence check are gates before scaling.
The interference pilot also completed and passed its 35-point, 12-checkpoint,
41-source audit. Nominal-point results for both pilots are:

| Receiver training channel | Clean BER | Interference BER |
|---|---:|---:|
| Physical channel | 0.06349 | 0.08478 |
| Learned full Gaussian | 0.15461 | 0.15034 |
| Learned full mixture | 0.13591 | 0.22239 |
| Sinkhorn | 0.10843 | 0.12825 |
| Flow-32 | 0.10465 | 0.12232 |
| Distilled consistency | 0.11264 | 0.13231 |
| Conventional pilot-LMMSE | 0.06414 | 0.08623 |

Each point uses 16,384 physical blocks, 112 bits per block. In the interference
case, physical-channel receiver training gives inactive/active BER
0.06353/0.27369. Sinkhorn gives 0.11121/0.27978 and Flow gives 0.10470/0.27891.
Their overall gap therefore includes a substantial inactive-block loss, not
only missed rare interference. Sinkhorn's covariance-error/floor is 1.288,
versus Flow's 3.974, but its numerical BER is worse. Matching this covariance
summary alone does not establish downstream sufficiency.

The dense learned mixture is especially weak under interference despite the
true law lying in the unrestricted two-Gaussian family. It needs a stronger
parameterization/convergence check, not a claim that mixtures cannot work.
The five generator training times sum to 191.26 seconds clean and 185.96 seconds
with interference. Timing includes the training steps and physical queries,
but excludes checkpoint writing, receiver training and evaluation.

Full learned-constellation performance is deferred; its gradient smoke test
passes. All runs in this note have finished. The code and environment preflight
are prepared, but final cluster campaign settings are not frozen.
