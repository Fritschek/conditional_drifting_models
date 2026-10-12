# SSPA application pilot, frozen before execution

Entry point: `scripts/run_fast_channel_application.py`. This is one development
seed, 9111, not final paper data or a statistical model-family ranking.

## Generator comparison

SSPA uses the unchanged n=8, M=64, 8 dB, rate 6/8 Rapp channel with gain 5,
saturation 1.5 and smoothness 3. Per-real-coordinate noise variance remains
`ebno_to_noise(8,6/8)^2 / 2`. Training inputs are iid standard-normal real
coordinates. Every method generates direct outputs and has approximately 60k
inference parameters. There is no input-SNR conditioning.

Fixed methods and inference evaluations:
- Learned diagonal Gaussian: one evaluation.
- Clipped WGAN: one evaluation; original validated clipping .01 and RMSprop 2e-4.
- Diffusion: DDIM-20 from 1000-step cosine noise training.
- Conditional flow matching: eight Euler evaluations.
- Normalized condition-wise Sinkhorn: one evaluation, K64, ten Sinkhorn iterations,
  R=.05, AdamW and the established warmup.
- Consistency distillation: one evaluation, 40-scale learned-teacher Heun targets,
  target EMA .95. Exact pinned upstream score-training and distillation losses.

All receive **40.96 million oracle output observations**. The first five use
40k outer updates at batch 1024. Distillation uses 20k EDM-pretraining updates
plus 20k student updates at batch 1024. WGAN has 40k critic updates and 8k
generator updates. Sinkhorn has 16 distinct conditioning inputs per update,
with 64 positive, generated and reference samples per input. Equal observation
counts therefore do not give identical distinct-input coverage or compute.

Use the fixed AWGN development settings, final EMA weights and the stated NFEs.
No checkpoint, NFE or hyperparameter selection on SSPA SER or gradient results.
Student RNG seed is 19111; pretraining and other generators use 9111. The
learned teacher's entire cost is included in each distilled-model cost. Capacity,
oracle queries, teacher/critic work and measured training time are reported.

## Distribution and application evaluation

Generator metrics use 16,384 global samples, 64 anchors with 256 outputs each,
and 128 fixed random projections, shared across methods. Analytic-versus-analytic
floors use independent output sets at the same anchors. Additional mean and
variance checks use the exact deterministic Rapp output and known noise variance.

Each method gets a fresh, paired symbolic AE initialization, hidden width 16,
NAdam .001, batch 500 and 20k updates. Uniform message sampling and full-codebook
centering/unit mean-square normalization are unchanged. Training runs through
the frozen surrogate, including its input derivative. All final communication
tests run on the **analytic SSPA channel**, at 4, 6 and 8 dB. Each point has
16,384 draws per message, or 1,048,576 symbol trials. Raw error counts and natural-
label BER are retained; zero errors are not replaced with a fabricated BER/SER.

The analytic training reference uses the same AE settings. Its 2000-update and
20k-update checkpoints provide shared intermediate/final codecs for conditional
accuracy and expected-loss gradient comparisons. These are fixed checkpoints,
not chosen by achieved SER. The final analytic codec continues the early one.

At each shared codec, compute four independent analytic reference gradients and
two independent surrogate gradients, each with 8192 draws per message. Report
codeword and encoder-parameter gradient cosine, relative error, norm ratio,
and Monte Carlo variation. An analytic reference is provisionally resolved only
when its mean-vector norm / MC-standard-error norm is at least five and the
cosine between the two half-panel means is at least .95. This is a practical MC
check, not a theorem or a statistical significance test. Do not interpret an
unresolved final-codec comparison as established population gradient mismatch.

The analytic gradient reference is common to all methods at each checkpoint.
No gradient result is used for model selection. Conditional metrics are also
reported at those actual codebooks, since Gaussian-input accuracy need not
transfer to encoder-selected codewords.

Timing includes generator training and separate downstream training, with all
sampler steps and input derivatives. No extrapolation is used. Sampling and
input-backward timing use batches 1 and 500, three warmups and ten repeats.

## Execution and limits

Use the existing Torch environment, with no dependency changes:

```bash
python scripts/run_fast_channel_application.py \
  --device cuda --channel SSPA --seed 9111 \
  --out-dir results/fast_sspa_application_20261012
```

Resume the identical command with `--resume`. The manifest rejects changed
configuration or numerical sources. Generator and AE checkpoints include
optimizer/RNG state; completed results are reused. No historical trained
models or manuscripts are overwritten. The old AWGN-only development runner
remains frozen; this new runner supports AWGN and SSPA explicitly.

The pilot tests whether the stronger fast baselines transfer to a nonlinear
channel and whether their distribution/gradient/communication metrics agree.
One seed cannot establish a significant SER ranking. This is a matched-query
development comparison, not a matched-wall-clock study or passive measured-data
experiment. The larger jointly generated wireless block remains separate work.

## Post-hoc explanatory check

After the Gaussian row completed, its near-floor random-anchor error and much
higher downstream SER motivated an additional evaluation. This addition was
not part of the frozen comparison above. `scripts/check_fast_application_codebooks.py`
evaluates every learned method at its own final surrogate-trained codebook.
Use 1024 outputs per codeword for conditional metrics and exact-mean comparisons,
and 8192 draws per message for paired surrogate/analytic loss evaluation at
the nominal SNR. Retain per-message mean errors, codewords and I/Q radii.
This is explanatory, post-hoc work with no retraining, checkpoint selection or
changes to the primary comparison. All six methods are included.

The own-codebook results then motivated a per-message replay in
`scripts/check_fast_application_support.py`. It uses the same analytic evaluation
draws, verifies identical total errors and localizes them by input norm. For the
Gaussian model it also compares its exact predicted means against the exact Rapp
means. This is post-hoc localization, not an independent replication or causal
intervention.
