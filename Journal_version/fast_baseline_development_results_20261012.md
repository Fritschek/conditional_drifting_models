# Fast-baseline development: completed AWGN round

## Decision

Use the validated clipped-WGAN recipe and **consistency distillation** as the
next application-comparison candidates. Do not use the weak short-pilot WGAN
or randomly initialized consistency model to claim a Sinkhorn advantage.
Standalone consistency training remains under-dispersed at the tested budgets.

The round completed 21 training stages: eight initial recipes, four confirmation
runs, and three independent EDM/CT/CD sequences. All models use the existing
AWGN law with n=7, 5 dB, rate 4/7 and approximately 60k inference parameters.
No downstream AE, SSPA or final multi-seed paper comparison was run here.

See the [frozen development protocol](fast_baseline_development_20261012.md),
[complete numerical tables](evidence/fast_baseline_development_20261012/results.md)
and [training trajectories](evidence/fast_baseline_development_20261012/awgn_development.pdf).
The numerical tables retain every failed recipe and both warm-start controls.

## What changed

### WGAN: the short EMA evaluation was misleading

The original recipe, RMSprop 2e-4, critic clipping .01, five critic updates per
generator update, passes the predeclared quality check on seeds 9101, 9102 and
9103 after 20k outer updates with batch 1024. Final conditional SWD/floor is
1.1108, 1.1359 and 1.1099; variance/truth is .9485, .9441 and .9541. The selected
learning rate and clipping did not change. Broader clipping was worse here.

On seed 9101 at only 5000 outer updates, raw weights already give SWD/floor
1.4173 and variance/truth .9892. EMA gives 7.5269 and .3058 at the same checkpoint.
The model EMA is updated only when the generator updates. Decay .999 therefore
retains 36.77% of its initialization after 1000 generator updates, versus 1.83%
after 4000. More learning and EMA burn-in both occur, but the raw-versus-EMA
comparison isolates a substantial averaging effect. This is not evidence that
adversarial training is generally easy or that raw weights are always preferable.

### Standalone consistency: curriculum helps but does not fix variance

Reducing the final scale parameter from 150 to 20 in the official coupled
noise-grid/target-EMA schedule substantially improves the conditional mean.
The selected coarse recipe has mean RMSE .0225--.0230 on three development seeds,
close to the .02325 finite-Monte-Carlo mean RMSE scale for this sampling panel.
However, variance/truth remains .7448--.7478. It fails the predeclared [.8,1.2]
variance check despite SWD/floor of 1.2375--1.3081. Increasing learning rate or
changing sigma_data did not resolve this in the bounded grid.

This change adjusts both interval spacing and the adaptive target EMA. It is
not an isolated proof that one of those components causes the improvement.
Neither near-floor SWD nor a small consistency loss is enough to pass the check.

### Learned-score distillation: a successful one-step reference

The authors' published [launch examples](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/scripts/launch.sh)
provide score-model checkpoints, and their
[training entry point](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/scripts/cm_train.py)
copies those parameters into the student even for consistency training. Our
earlier random-start CT was a supported standalone route, not a reproduction of
that initialization.

We trained an EDM denoiser for 20k updates using the exact official weighted L2
loss and lognormal noise sampler, then separately tested 20k warm-started CT
updates and 20k consistency-distillation updates. No analytic teacher was used.
The distillation loss uses the authors' learned-teacher Heun targets, 40 fixed
scales and target EMA .95. The inference map remains one network evaluation.

| Pretraining seed | Student seed | Distilled SWD/floor | Mean RMSE | Variance/truth | Check |
|---:|---:|---:|---:|---:|---|
| 9101 | 19101 | 1.0161 | .02501 | 1.0288 | Pass |
| 9104 | 19104 | 1.0052 | .02437 | 1.0347 | Pass |
| 9105 | 19105 | 1.0159 | .02470 | 1.0363 | Pass |

All are final EMA one-step results. The independent confirmation recipes and
budgets were unchanged. Warm-starting the original standalone CT curriculum
alone still fails: variance/truth .1184--.1268. The successful result is
**distillation**, not a claim that all consistency-training routes work here.

The student uses the same ~60k architecture as the other timed models. Each
distilled row counts 40.96 million oracle outputs and the complete score-model
pretraining, totaling 67.68--69.26 measured training seconds on the local RTX
5060 Ti. WGAN uses 20.48 million oracle outputs and about 32 training seconds.
These are unequal development budgets, not a fair cross-family ranking.
Measured training time excludes evaluation, checkpoint I/O and initialization.
Development trials, unused controls and teacher fitting must remain visible in
any later compute accounting.

## Artifacts and checks

- First round: `results/fast_baseline_development_20261012/`.
- Score-pretraining sequences: `results/consistency_warmstart_development_20261012/`,
  plus the corresponding `_seed9104/` and `_seed9105/` directories.
- Each sequence saves separate `edm/checkpoint.pt`, `ct/checkpoint.pt` and
  `cd/checkpoint.pt`, including optimizer, random state, progress and history.
- `scripts/load_distilled_channel.py` loads the verified CD checkpoint as a
  frozen differentiable implant and returns its teacher-inclusive cost metadata.
  CUDA-trained weights were loaded on CPU and input gradients checked. Training
  RNG states are not restored for inference, since CPU/CUDA generator states
  are incompatible.
- Eleven CPU unit tests pass, including exact EDM/CD optimizer restart,
  frozen teacher gradients, correct teacher-call counts and known-AWGN moments,
  in addition to the existing sampler/input-gradient tests. All experiment
  training stages ran on CUDA. Completed-run resume was also checked.
- No packages were installed or updated. Historical trainers and upstream
  numerical sources were unchanged. No manuscript tables were replaced.

The general `run_fast_channel_comparison.py` family named `consistency` still
means **standalone CT**, not the newly tested distilled baseline. Use the
separate warm-start runner/loader for CD; do not silently relabel old results.
Its present development runner is AWGN-specific and must be generalized before
claiming SSPA or other-channel distillation results.

## Next round

1. Carry the successful recipes into SSPA with the same careful mean/variance
   and conditional-law checks. Declare any further tuning allowance in advance.
2. Generalize the CD runner to the shared channel protocol and include teacher
   cost in matched-query and measured-compute comparisons.
3. Train downstream AEs through the frozen generators and evaluate on the
   analytic channels; measure expected-loss gradients at shared learned codebooks.
4. Only then freeze final settings and seeds for the broader campaign.

AWGN success is necessary evidence for a credible baseline, not proof of
communication utility on nonlinear or structured channels. Conditioning-novelty
and joint-geometry ablations remain secondary to that application question.
