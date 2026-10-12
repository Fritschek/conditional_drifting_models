# Overnight fresh-start SSPA and communication checks

Specified before execution on 11 October 2026. This is a bounded development
confirmation, not a final cross-family journal benchmark.

## Generator comparison

Unused development seeds 9004, 9005, 9006. For each seed save one randomly
initialized update-zero parent, including its empty Adam state and training
RNG. Reuse the identical weights and starting RNG across these three arms:

1. `old_k4`: unchanged production step, 4096 anchors x 4 samples.
2. `old_k64`: same production step, 256 anchors x 64 samples.
3. `official64`: authors' pinned normalized W-Flow loss and the previously
   successful AdamW/warmup recipe, 256 anchors x 64 samples.

Every arm draws 16384 generated, true-positive and independent-reference outputs
per update. Counts are equal but the number of distinct inputs differs. Keep
SSPA n=8, Gaussian input anchors without per-batch power normalization, hidden
128, latent16 and noise_std=0.3250531435997416 (noise per real coordinate divided
by sqrt(2)). This corresponds to the retained M64/n8/rate .75, Eb/N0=8dB setup.
All old settings and the frozen training-only K4 epsilon calibration are
preserved. Both old arms use the same calibrated epsilon, Adam lr .001,
beta(.9,.999), radial field cap2 and parameter-gradient cap1. Official64 uses
R=.05, 10 log iterations, fresh AdamW beta(.9,.95), decay .01, lr warmup from
1e-6 to .0002 over5000 updates, parameter-gradient cap2, and EMA .999. Raw model
weights are primary in every downstream test; EMA is reported only as secondary
generator evidence. No architecture/physics change and no warm-start training.

Run all nine arms through 390720. Validation/checkpoints at 0,1000,5000,10000,
30000,50000,80000,100000,150000,200000,250000,300000,350000,390720. Save a resume
checkpoint at least every5000 updates, trace every100. Use unchanged isolated
128-anchor/128-output/64-projection validation panel. Select the minimum raw
conditional validation SWD, earliest update on ties, excluding update0.
Retain the final checkpoint even when worse. Run the separate 256-anchor/
512-output/128-projection test panel at the selected and final checkpoints.

Late stability means all 300000/350000/390720 checks have SWD <= twice their
analytic-vs-analytic floor, mean L2 <= twice its analytic floor, and variance
between .5 and2 times the known true variance. Unlike the warm-start gate,
this criterion is not relative to a randomly initialized parent.

## Downstream utility

For each generator seed, train two paired AE initializations, seeds10000+seed
and20000+seed. For each initialization compare analytic training and all six
learned arms (three generator variants, selected and final checkpoints): 42 AE
trials total. Use M64, n8, two hidden layers of width16, no output activation,
NAdam lr .001, gradient cap1, batch500 and exactly20000 updates. No AE checkpoint
selection: the final AE is primary. Save AE checkpoints at5000,10000,20000.

Normalize the complete codebook by global centering and unit per-real-coordinate
RMS before indexing messages. This makes a message's codeword independent of
the random minibatch composition. The old journal used a different minibatch
normalization, so its BER/SER values are historical context, not matched rows.
Keep an independently seeded message-label generator, paired across methods;
channel/latent draws use separate state from labels. Seeds alone do not imply
identical channel-noise realizations across analytic and neural methods.

Freeze all surrogate weights while differentiating through input x. Train at
8dB only. Evaluate every final AE on the analytic channel at4,6,8dB with32768
draws per message (2097152 message trials per SNR). Report integer symbol/bit
errors and denominators, natural-label BER, CE, raw rates and timing. No
zero-error clipping. These off-nominal SNR tests evaluate receiver robustness,
not surrogate fidelity at untrained SNRs. During training, small analytic
validation panels at5000/10000/20000 are logged but never select the AE.

## Gradient fidelity

Use the first paired analytic-trained AE for each generator seed, at5000 and
20000 AE updates. These six frozen reference codecs are common to all methods
within their comparison. Compare analytic sampling and all six learned arms at
the same codewords and decoder. Reuse the existing expected-loss and encoder
chain-rule routines, including differentiation through codebook normalization.

Four independent repetitions, 8192 gradient draws per message and65536 analytic
reference draws per message. Record vectors, absolute error, cosine, norm ratio,
MC variability and equal-sample analytic floors. Suppress relative metrics when
reference norm <= five times the estimated norm of the MC standard error.
Retain the reference split check and an analytic directional finite-difference
check. Use4096 output samples per codeword,64 projections and two analytic
clouds for conditional SWD/Gaussian-W2 floors. No surrogate-specific decoder
optimization occurs before these comparisons. This batch evaluates expected
gradients and full AE training, not a new pre-optimization selector or a theorem.

## Execution and reporting

No dependency installation or HPC submission. Keep all historical sources and
results immutable. Record source/config/parent/checkpoint hashes, RNG, optimizer,
oracle exposure, Torch/GPU and actual timings. Test deterministic resume,
validation isolation, paired messages, full-codebook power and integer counts.
Use separate smoke directories and configurations; smoke outcomes are not data.

Hard deadline: 12 October 2026,04:00 UTC. Stop training at a resumable boundary
five minutes before that deadline, leaving time for status export. Failures are
recorded and other independent tasks may proceed. A numerical task failure is
not a fidelity-based exclusion. Unfinished tasks remain explicitly incomplete.
Do not use spare time to silently expand the hypothesis/tuning space. Optical
fiber and modern matched baselines remain separately planned follow-ups.
