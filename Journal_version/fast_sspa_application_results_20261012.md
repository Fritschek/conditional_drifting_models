# Fast channel generators: SSPA application pilot

Completed locally on the RTX 5060 Ti. All six generator families, the analytic
reference, seven AE runs and all planned evaluations finished. This is development
seed 9111, not confirmatory paper data. No methods or failed results were dropped.

Protocol: [fixed comparison](fast_sspa_application_protocol_20261012.md).
Full tables: [numerical report](evidence/fast_sspa_application_20261012/results.md).
Curves: [SER/BER](evidence/fast_sspa_application_20261012/communication_curves.png).
Raw suite: `results/fast_sspa_application_20261012/`.

## Comparison

All generators have approximately 60k inference parameters and receive 40.96m
analytic output observations. Distillation includes 20k EDM-pretraining plus
20k student updates; the other methods use 40k outer updates, all at batch 1024.
WGAN has 8k generator and 40k critic updates. Sinkhorn has K64 and 16 distinct
inputs per update. These are matched output-query budgets, not identical input
coverage, optimizer steps or compute. Fixed final EMA checkpoints are used.

Each AE has 20k updates, identical initial weights and message-label stream.
All SER/BER below are evaluated on analytic SSPA, not the learned channel.
Each point contains 1,048,576 symbol trials. Each method has only one trained AE.

| Training channel | SER at 4 dB | SER at 6 dB | SER at 8 dB | AE training seconds |
|---|---:|---:|---:|---:|
| Analytic | 5.09e-3 | 3.05e-4 | 5.72e-6 | 35.52 |
| Learned Gaussian | 1.74e-2 | 6.51e-3 | 3.10e-3 | 37.48 |
| WGAN | 1.33e-3 | 6.48e-5 | 1.91e-6 | 36.21 |
| DDIM-20 | 8.49e-3 | 8.87e-4 | 6.77e-5 | 362.86 |
| Flow-8 | 1.50e-2 | 1.79e-3 | 1.50e-4 | 88.70 |
| Sinkhorn K64 | 8.88e-3 | 1.12e-3 | 7.72e-5 | 37.20 |
| Consistency distillation | 1.01e-2 | 1.86e-3 | 4.37e-4 | 41.77 |

Sinkhorn and DDIM-20 have numerically similar SER in this seed, with substantially
different downstream training costs. WGAN has the lowest numerical SER but poor
conditional-law fidelity. At 8 dB, WGAN and analytic have only two and six symbol
errors respectively. Neither that difference nor the single-seed rankings
establish a statistically supported training-method advantage.

The complete synchronized generator training costs are 42.80, 68.32, 51.13,
42.90, 136.64 and 81.59 seconds in the learned-method order above. The last value
includes the EDM teacher. Total generator training is 423.39 seconds; all AE
training totals 639.74 seconds. These exclude initialization, evaluation and
checkpoint I/O. No timing extrapolation is used. Batch-one latency, batch-500
throughput and input-backward measurements are retained separately.

## Distribution and gradient evidence

On shared random inputs, anchor SWD / analytic sampling floor is 1.129 for
Gaussian, 11.245 for WGAN, 1.740 for DDIM-20, 1.986 for Flow-8, 1.250 for
Sinkhorn and 1.576 for distilled consistency. Corresponding conditional variance
ratios are 1.004, 4.083, .838, .592, .989 and 1.076. WGAN's good communication
result therefore does not establish an accurate channel law. Flow-8's variance
deficit may include finite-step integration error; this pilot does not isolate it.

At the common 2000-update analytic codec, the analytic expected-loss gradients
pass the predeclared Monte Carlo resolution check. Encoder-gradient cosine is
.991 for Gaussian, .627 for WGAN, .960 for DDIM-20, .972 for Flow-8, .992 for
Sinkhorn and .972 for distilled consistency. Gaussian and Sinkhorn also have
the smallest relative encoder-gradient errors, .165 and .136. WGAN has a large
norm mismatch. Codeword-gradient and parameter-gradient results are distinct
and both are retained in the full report.

The common 20k-update analytic codec's gradients **fail** the resolution check.
Its encoder-gradient signal/MC-SE ratio is 3.30 and split-panel cosine is .881.
Do not use those final-codec cosines as established population-gradient claims.
The intermediate checkpoint alone also does not explain an entire AE trajectory.

## Input coverage explains an important part of the gap

Two post-hoc checks, with no retraining or selection, evaluated each model at its
own final AE codewords. They were added after observing the Gaussian row's good
random-input fidelity and poor transfer. This chronology is explicit in the protocol.

The shared full-codebook normalization enforces average squared norm eight. It
does not enforce an individual codeword norm. Most Gaussian-trained codewords
have norms around 2.2, but messages 4 and 31 have norms 10.27 and 10.05. The
analytic-trained codebook's maximum is 3.48. Under the generator's iid normal
training law, the probability of an input norm above 10.27 is about 3.18e-19
(chi-square with eight degrees of freedom). These inputs lie far into the tail
of the training distribution, despite satisfying the average-power constraint.

At those two codewords, the Gaussian model's exact conditional-mean RMSEs are
.490 and .428. It predicts some I/Q-pair mean amplitudes as high as 2.78 and 2.44,
where the analytic Rapp mean saturates at 1.5. The same saved AE gives surrogate
SER 3.62e-5 versus analytic SER 2.98e-3 on the post-hoc panel. Messages 4 and 31
account for 1532 of 1560 analytic errors (98.2%). The per-message replay exactly
reproduces the preceding aggregate error count.

The pattern is not exclusive to Gaussian. Distilled consistency has a codeword
with norm 8.99 and 214 of its 221 analytic errors occur on that message. Sinkhorn
reaches norm 6.92, with 32 of 48 errors on its two highest-norm codewords. WGAN's
codewords stay between norms 2.46 and 3.10, while its conditional variance on
that codebook is roughly three times the analytic variance. This is compatible
with a robustness effect, but it does not prove why WGAN found the better code.

The evidence localizes extrapolation and optimistic surrogate losses at
encoder-selected inputs. A controlled intervention is still needed to establish
how much input coverage, output constraints or gradient behavior causes the gap.
Do not call this a universally solved SWD-to-SER relationship.

## Next bounded experiments

1. Test input coverage prospectively. Compare the unchanged Gaussian-input
   training law with a declared broader radial input law at equal oracle budget,
   retaining the same physical channel and AE power constraint. Apply the same
   coverage rule across the chosen model families; do not tune it against final
   SER. Include Gaussian and Sinkhorn, and retain distilled consistency if feasible.
2. Separately test an individual-codeword power cap across analytic and learned
   channels. It changes the communication optimization problem, so it must not
   be mixed with the existing average-power results as if only training changed.
3. Check sampler convergence for the saved flow/diffusion models before treating
   their fixed-NFE outcomes as competitive optima. Preserve the original rows.
4. Repeat AE initialization on the frozen generators, then use fresh generator
   seeds after a shared development protocol is settled. Seeds 9101--9105 and
   9111 have already been used for development and must not be relabeled as
   untouched confirmation seeds.

The larger joint wireless block and optical-fiber follow-up remain pending.
No new conditioning-novelty claim follows from this work.

## Verification and artifacts

- 13 focused CPU tests and nine CUDA tests passed.
- Tiny all-family end-to-end run and identical-config resume passed.
- Exact AE continuation and SSPA mean/noise calibration tests passed.
- All seven final AE checkpoints reached 20k updates. All generator/teacher
  budgets, SER/BER integer counts and 35 frozen source hashes passed the audit.
- Own-codebook checks cover all six methods, with generator/codec hashes.
- Per-message error replay matches every saved analytic total exactly.
- SER/BER figure inspected for layout and labels. All weights remain local.

New runners are `scripts/run_fast_channel_application.py`,
`scripts/check_fast_application_codebooks.py`,
`scripts/check_fast_application_support.py` and
`scripts/report_fast_channel_application.py`. Numerical training sources remain
frozen. `--resume` reuses completed generator evaluations as well as checkpoints.
No packages were installed, no historical results overwritten, and no HPC jobs
submitted. No experiment process remains running after completion.
