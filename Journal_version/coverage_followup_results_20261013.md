# Coverage replication and saved-sampler results

Completed locally on an RTX 5060 Ti. This report supersedes the execution status
in the [frozen protocol](coverage_followup_protocol_20261012.md), whose contents
remain unchanged. Neither experiment was retuned after inspecting its results.

## Completed work

- Coverage: 12 fresh generators, 36 surrogate-trained AEs, nine independently
  trained analytic-channel references, and 135 physical-channel curve points.
- Three generator seeds (9201--9203), three paired AE initializations per seed.
  Each generator received 40,000 updates and each AE 20,000 updates.
- Every physical curve point uses 1,048,576 symbols, balanced over 64 messages.
  The five AEs in each paired block share physical noise for comparison.
- Saved samplers: one existing Flow checkpoint at 8/32/128 steps and one existing
  diffusion checkpoint at DDIM 20/50/100 steps. No retraining or step selection.
- Strict aggregation and source/checkpoint audits passed. All 68 focused CPU
  tests and seven new CUDA tests passed during preparation. Both CUDA smoke
  suites resumed with byte-identical result records. No simulation is running.

The [compact evidence](evidence/coverage_followup_20261013/README.md) contains
complete result records and audited summaries. Weights remain local.

## Broader inputs improve transfer across fresh generators

The intervention replaces normal training inputs with a fixed mixture of 75%
normal inputs and 25% radial inputs covering the feasible codeword-radius bound.
It changes generator training inputs, with no codeword clipping or downstream
optimization changes. Final EMA checkpoints are used throughout.

The primary endpoint is broad-minus-normal physical SER at nominal 8 dB,
averaged over the three paired AEs within each generator seed.

| Family | Normal-input mean SER | Broader-input mean SER | Ratio of means | Numerically improved AE pairs |
|---|---:|---:|---:|---:|
| Learned Gaussian | 6.4150e-4 | 1.9709e-5 | 32.55 | 8/9 |
| Condition-wise Sinkhorn | 1.5418e-4 | 3.8041e-5 | 4.05 | 7/9 |

Both families improve in all three generator-seed means. These are three
independent generator replicas, not nine. The ratios are descriptive, not
significance tests or typical per-seed improvements. Gaussian's mean paired
change is -6.2180e-4 (between-generator SD 4.4697e-4); Sinkhorn's is -1.1614e-4
(SD 1.0096e-4). The analytic-reference mean is 1.9497e-5. Numerical proximity
to that mean does not establish equivalence.

All individual outcomes, including the one Gaussian and two Sinkhorn worsened
pairs, are retained in the [paired tables](evidence/coverage_followup_20261013/coverage/report.md).
The complete 4/6/8 dB curves and per-message error/discordance counts are in the
records. Final evaluation noise is common within each physical paired block;
surrogate-versus-physical error estimates use independent noise.

### What changes in the conditional law?

For Gaussian generators, radius-10 conditional mean RMSE falls from about
0.42 to 0.029 across the three seeds. At the largest tested radius, predicted
variance relative to truth changes from 0.029--0.033 to 1.076--1.078.

For Sinkhorn, radius-10 mean RMSE falls from 0.426--0.485 to 0.064--0.079.
Variance at the largest radius remains low (0.562--0.591 of truth). Ordinary
normal-input conditional SWD/floor gets worse in every replica, from
1.208--1.276 to 1.430--1.511, despite improved mean downstream SER.
This directly limits the use of an ordinary-input average metric for choosing
an implant that an encoder can optimize against.

Broader coverage also reduces the largest learned codeword norms and severe
surrogate optimism. This supports coverage as a useful intervention, without
isolating whether mean accuracy, variance accuracy, or altered optimization
paths account for each downstream improvement.

The intervention has equal output-query counts, but its expected training-input
energy is 6.0833 times higher. It is not an equal-RF-energy measurement protocol.
SSPA has a Gaussian conditional law, so these results do not establish a need
for flexible non-Gaussian generators or a general model-family ranking.

## Sampling resolution matters, but more steps are not always better

All rows below use saved seed-9111 EMA weights, common initial latent streams,
64 distributional anchors with 256 samples each, and a common early analytic
codec. The gradient reference uses four repeats of 8192 samples per message;
surrogate gradients use two repeats. All reference-resolution checks passed.

| Sampler | Conditional SWD/floor | Variance/truth | Encoder-gradient cosine | Sampling ms/batch of 500 |
|---|---:|---:|---:|---:|
| Flow 8 | 1.911 | 0.6012 | 0.9760 | 1.832 |
| Flow 32 | 1.239 | 0.9004 | 0.9952 | 7.005 |
| Flow 128 | 1.206 | 0.9832 | 0.9937 | 28.01 |
| DDIM 20 | 1.819 | 0.8526 | 0.9599 | 12.80 |
| DDIM 50 | 1.723 | 0.9812 | 0.9225 | 32.04 |
| DDIM 100 | 1.805 | 1.0450 | 0.8848 | 64.14 |

These are batch times, not individual-call latency. Batch-one measurements,
raw timing repeats and input-backward costs are retained in the
[full panel](evidence/coverage_followup_20261013/samplers/report.md).
Gradient cosines concern one intermediate codec, not final-AE SER predictions.

Flow's low-step variance deficit substantially recedes at finer integration.
An [ideal Gaussian flow calculation](gaussian_flow_discretization_20261012.md)
predicts variance ratios 0.553/0.866/0.965 for Euler 8/32/128 even with an exact
velocity. This explains a possible source of underdispersion without attributing
all learned-model error to discretization.

DDIM has its closest variance and lowest SWD here at 50 steps, while the
encoder-gradient cosine decreases with more steps. A finer sampler therefore
does not universally repair this checkpoint. Fixed-latent output and Jacobian
differences shrink relative to the finest sampler, but that finest sampler is
not a physical-channel oracle. Historical Flow-8/DDIM-20 downstream results
remain unchanged. No new downstream NFE comparison has been run.

## Noise-label convention needs resolution before scaling

The [separate calibration audit](sspa_noise_label_audit_20261012.md) found that
the current SSPA software label is 3.0103 dB below input-referenced Eb/N0 under
the stated real-coordinate normalization and conventional N0/2 real noise.
Nominal 8 dB corresponds to real noise variance 0.05282977 and input-referenced
Eb/N0 of 11.0103 dB. All arms here retain identical frozen physical settings.
This does not invalidate within-setting contrasts. It does affect external
comparisons and future figure labels. No code or frozen data was silently changed.

## Next decisions

1. Declare the energy reference and noise convention before a new paper campaign.
   Add an independent physical-normalization test rather than only checking
   agreement with the existing simulator helper.
2. Assess these complete results before choosing a shared input-coverage protocol
   for all competing generators. Do not compare broader-input Sinkhorn against
   normal-input competitors as an isolated generative-mechanism advantage.
3. Include sampler resolution in a bounded, predeclared baseline validation
   allowance. Flow-8 alone is insufficient to characterize the trained field;
   DDIM-100 alone is not automatically its best operating point either.
4. Define a genuinely structured/non-Gaussian channel experiment with meaningful
   communication metrics. The larger structured-channel experiment and optional
   optical-fiber revisit remain unexecuted. Repeating SSPA alone cannot resolve
   the journal's practical-scope objection.

## Reproduce the reports

From the repository root, using the existing Torch environment and local results:

```bash
python scripts/report_sspa_coverage_replication.py --suite-dir results/sspa_coverage_replication_20261012
python scripts/report_coverage_followup.py --kind coverage --suite-dir results/sspa_coverage_replication_20261012
python scripts/report_coverage_followup.py --kind samplers --suite-dir results/sspa_saved_sampler_panel_20261012
python scripts/export_coverage_followup.py
```

The portable copy can be integrity-checked without Torch or local checkpoints:

```bash
python3 scripts/export_coverage_followup.py --verify-only
```
