# Fresh-start SSPA: stability, gradients and communication performance

Completed 11 October 2026, 18:15 UTC. Read this before the older warm-start
component report. The [protocol](sspa_fresh_protocol_20261011.md) was fixed
before execution. All local tasks completed without numerical or execution
failures, well before the overnight deadline. No cluster was needed.

## What ran

- Three unused development seeds, 9004--9006. Identical random generator weights
  and starting RNG within each seed, with empty optimizer states.
- Old K4/B4096, old K64/B256, and official normalized K64/B256, each through
  390720 updates at 16384 generated/positive/reference outputs per update.
  This reallocates exposure across inputs; distinct input counts differ.
- Old K64 changes only K/B, retaining the production solver, fixed calibrated
  epsilon, raw loss, Adam, learning rate and clipping. Official K64 uses the
  pinned authors' W-Flow loss and its recorded optimizer recipe. This is not
  the original Drifting JAX code. Raw weights are primary; no EMA is implanted.
- Minimum conditional validation SWD over the frozen schedule selects one
  checkpoint per generator. Final checkpoints are retained separately. An
  independent panel evaluates both without selecting them.
- 42 paired AE trials: three generator seeds, two AE initializations per seed,
  seven implants. All train for 20000 updates with the same full-codebook power
  constraint, optimizer and independently paired message streams. Final AE
  weights are used, with no AE selection.
- Analytic SSPA evaluation at 4/6/8 dB, 2097152 messages per AE per SNR.
  Integer symbol/bit errors, denominators and cross-entropy are retained.
- Six common analytic-trained codecs, at 5000 and 20000 AE updates, for
  expected-input and encoder-gradient comparisons of all seven implants.

Measured training times were 10255.99 s for generators and 1503.98 s for AEs.
The complete batch, including validation, test evaluation and gradient checks,
ran from about 14:58 to 18:15 UTC, approximately 3 h 18 min. Smoke runs and
test setup are separate. No timing extrapolation is used here.

## 1. The late K4 failure reproduces from scratch

All three K4 trajectories are initially healthy and deteriorate between the
100k and 150k saved checks. Both K64 variants stay bounded through the full
horizon and pass the frozen late stability gate on all three seeds.

| Full-horizon generator | Stable seeds | Mean independent conditional SWD | Conditional variance range |
| --- | ---: | ---: | ---: |
| Old K4 | 0/3 | 1.27721 | 9.6743--9.8646 |
| Old K64-only | 3/3 | 0.0208845 | 0.052574--0.052778 |
| Official normalized K64 | 3/3 | 0.0200636 | 0.052744--0.052879 |

Analytic-vs-analytic SWD floor: 0.0183507. True per-real-coordinate variance:
0.0528298. Validation SWD values around 0.037 use a smaller sample panel than
the independent test; they must not be compared as identical estimators.

Selected K4 checkpoints are at 80k, 50k and 100k updates, with independent SWD
0.0222343, 0.0226085 and 0.0225074. The failure is not an inability to fit SSPA
initially. This matched fresh-start comparison strengthens the evidence that
changing per-input grouping prevents the observed long-horizon instability.
It does not prove universal K4 failure, identify minimal sufficient K, or
establish a unique mathematical cause.

## 2. The full official recipe improves gradient fidelity further

All six analytic gradient references pass the frozen resolution threshold.
Analytic directional finite-difference checks pass. Equal-sample analytic
Monte Carlo estimates remain important uncertainty references near convergence.

| Generator/estimator | Encoder cosine range across six common codecs | Encoder norm ratio range |
| --- | ---: | ---: |
| Analytic equal-sample estimate | 0.967--1.000 | 0.920--1.326 |
| Selected K4 | 0.680--0.970 | 0.712--1.317 |
| Final K4 | -0.362--0.460 | 301--13684 |
| Selected old K64 | 0.754--0.992 | 1.025--2.387 |
| Final old K64 | 0.688--0.985 | 1.073--3.832 |
| Selected official K64 | 0.969--0.998 | 0.823--1.058 |
| Final official K64 | 0.952--0.999 | 0.797--1.058 |

The K/B change suffices for the observed stability repair. The full official
recipe gives more faithful expected encoder gradients in these checks. Do not
attribute that extra improvement to normalization alone: solver representation,
entropic scaling, optimizer, learning-rate schedule and clipping also differ.
The previous deletion study tested law fidelity from a warm start, not this
fresh-start gradient comparison.

Two codec stages within a generator seed are correlated. Four Monte Carlo
replicates are estimator repetitions, not training seeds. Raw vectors, absolute
errors, reference splits and conditional SWD/GW2 floors are retained.

## 3. Communication performance is useful, but healthy methods are mixed

At 8 dB, two AE initializations are averaged within each generator seed first.
SD below is across the three seed blocks. Evaluation always uses analytic
SSPA. Natural-label BER is not a separately coded BER.

| Training implant | SER, mean +/- block SD | BER, mean +/- block SD |
| --- | ---: | ---: |
| Analytic | 3.51e-5 +/- 4.70e-5 | 1.53e-5 +/- 1.99e-5 |
| Selected K4 | 1.96e-5 +/- 4.65e-6 | 1.00e-5 +/- 1.82e-6 |
| Final K4 | 1.34e-2 +/- 1.07e-2 | 6.47e-3 +/- 5.13e-3 |
| Selected old K64 | 3.54e-5 +/- 8.87e-6 | 1.74e-5 +/- 3.79e-6 |
| Final old K64 | 3.25e-5 +/- 1.55e-5 | 1.62e-5 +/- 7.37e-6 |
| Selected official K64 | 3.34e-5 +/- 3.09e-5 | 1.95e-5 +/- 2.23e-5 |
| Final official K64 | 3.00e-5 +/- 3.29e-5 | 1.86e-5 +/- 2.28e-5 |

Final K4 damages downstream performance by orders of magnitude. Stable K64
methods produce useful SER/BER in the same numerical range as analytic training
under this protocol. However, selected K4 has the lowest mean SER at 8 dB here,
and healthy-method ordering varies with seed and SNR. Better SWD or closer
expected gradients at a frozen codec do not establish a universal ordering of
final AE performance. This batch does not validate a new metric for selecting
the best trained communication system.

The analytic row is a finite stochastic training reference, not optimized
achievable physical-channel performance; its variability is substantial.
Three generator seeds cannot support a strong performance-ranking claim.
Off-nominal SNR tests measure receiver robustness, not generator fidelity at
untrained SNRs. Historical journal normalization/selection differed, so its
results should not be merged into this table as matched observations.

## Next decisions

The fresh-start/unused-seed and basic communication-utility questions are now
answered for this SSPA development setting. Another broad SSPA knob sweep is
not required before assessment. Use official K64 as a candidate for controlled
comparisons and retain K64-only as the simple mechanism control. Preserve
early-selected K4 as a competitive healthy reference.

Journal work still pending: matched joint-versus-condition-wise geometry, fair
modern fast baselines, the larger structured channel, and any measured-data
claim. Optical fiber stays on the backlog. No new conditional-generation
machinery follows from these repairs; prior conditional drifting/W-Flow work
must still be credited. Spare overnight time was not used to expand experiments.

## Artifacts and checks

- Suite: `results/sspa_fresh_overnight_20261011/`.
- `summary.json`, `manifest.json`, `events.jsonl`, `audit.json`, `report.md`.
- `generator_endpoints.csv`, `codec_tests.csv`, `gradient_comparisons.csv`.
- `fresh_trajectories`, `fresh_ser`, `fresh_ber`, each as PDF and PNG.
  Thin communication-curve lines are seed-block means, not confidence bounds.
- All random parents, scheduled generator/AE checkpoints, exact-resume latest
  states, and gradient vectors retained.
- 252 scheduled checkpoints audited, plus parent/source hashes, exposure,
  validation selection, paired labels, optimizer steps, error-count identities
  and gradient-vector hashes.
- Five CPU and five CUDA unit checks passed. Two separate end-to-end smoke runs
  each completed 3 generator, 14 AE and 2 gradient tasks; those data are excluded.

Runner: `scripts/run_sspa_overnight.py`; report: `scripts/report_sspa_overnight.py`;
packager: `scripts/bundle_sspa_overnight.py`. A verified transfer bundle is
prepared outside ignored results under `Journal_version/evidence/`.
No historical numerical source, manuscript, public repository, dependency,
HPC job, commit or push was changed by this experiment.
