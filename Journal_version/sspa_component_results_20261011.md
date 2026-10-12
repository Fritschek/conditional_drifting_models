# What made the SSPA W-Flow recipe work?

11 October 2026. The four component deletions and the K64-only bridge to the
original production trainer are complete and audited.
These are development-seed experiments, not new manuscript results.

## Attribution

`official64` directly calls the unmodified authors' W-Flow PyTorch
`drift_loss_ot` at commit `1ccb34baf57f9f8f4e188460ea3d19b431403293`.
This is Han et al.'s Ermon/Candes paper implementation, distinct from Deng
et al.'s original Drifting JAX release. K64, normalized feature/force loss and
the AdamW recipe were taken from the pinned public release. Our experiment
uses our raw-output SSPA generator and a pretrained checkpoint, not their
image feature extractor, guidance setup or complete image training procedure.
Per-condition grouping is shared with prior work, not a new invention here.

## Controlled component deletions

All branches start from the same fixed-common seed-9001 checkpoint at update
80000 and finish at 390720. Seed 9001 was deliberately selected because it
exposed the delayed failure of the earlier lower-learning-rate intervention.
No branch was stopped based on intermediate fidelity.

All branches draw 16384 generated, positive and independent-reference outputs
per update. K4 uses 4096 input anchors; K64 uses 256. Thus K64 reallocates the
same output exposure into larger conditional clouds. It does not use 16 times
as many total output samples. Changing K changes both within-condition
estimation and the number of distinct inputs per update.

Final independent-panel values, raw models:

| Change from successful recipe | Conditional SWD | Conditional variance | Late validation bounds |
| --- | ---: | ---: | --- |
| None: normalized K64 reference | 0.01962653 | 0.0526603 | Pass |
| Return to K4 only | 0.13261548 | 4.8805523 | Fail |
| Remove loss normalization only | 0.01962660 | 0.0526613 | Pass |
| Restore old physical epsilon only | 0.01958877 | 0.0527112 | Pass |
| Restore old optimizer package only | 0.02142193 | 0.0525201 | Pass |

Analytic-versus-analytic SWD floor: 0.01835069. True per-coordinate conditional
variance: 0.052829773. This panel has 256 anchors, 512 samples and 128 projections.
Its SWD floor differs from the trajectory panel's 128 samples per anchor.
Do not compare the two panels as a time-series improvement.

The late bounds require SWD, conditional mean error and variance to remain
within the pre-existing bounds at all three checkpoints 300000, 350000 and
390720. EMA has the same pass/fail classification, so smoothing does not
explain the raw model's stability. Small differences among the passing rows
are numerical comparisons on one seed, not statistically established rankings.

The K4 branch begins deviating around 110k. Its variance reaches 0.151 at 120k
and 4.622 on the final validation panel. On the independent panel, the largest
conditional covariance eigenvalue is 931.0, compared with average coordinate
variance 4.881. This is more than a small SWD fluctuation, and the retained
mean/covariance metrics show that SWD alone understates the severity.

## What normalization changes

The unnormalized-loss deletion keeps exactly the same scaled transport costs
and uses direct-space regression toward g plus the physical cross-minus-self
velocity. It does not restore the old epsilon or introduce a field cap.

For the same batch and transport field v, let s be the detached coordinate
scale and f = sqrt(max(mean((v/s)^2), 1e-8)). With a single regularization
scale, the normalized and raw regression gradients satisfy

    grad_theta L_raw = s^2 f * grad_theta L_normalized.

Thus loss normalization alone changes a batch-wide scalar, not the gradient
direction. Adam largely cancels a constant gradient rescaling through its
first/second moments; varying scales, epsilon, clipping and inherited moments
can break exact equivalence. Here the endpoints are nearly identical.
For example at update 180100, normalized parameter-gradient norm is 0.41166
and raw norm is 0.01570, while both trained laws remain accurate. The gradient
relation is also tested directly on CPU and CUDA. This is distinct from the
feature-scaled epsilon, which changes the transport geometry.

## Interpretation

The component deletions identify small per-input clouds as the strongest
observed stability factor in this package. Normalization, the new epsilon and
the new optimizer are each dispensable individually in the K64 recipe on this
trajectory. The optimizer reversion changes learning rate, schedule, moments,
betas and weight decay together, so it cannot identify their individual effects.

Deleting one component at a time does not establish that K64 alone is
sufficient: several other changes remain active in each passing deletion.
The separately specified K64-only bridge therefore tested the unchanged old
trainer, altering only K4/B4096 to K64/B256. It also passed the full horizon,
without loss normalization, log-domain solver, new epsilon, optimizer reset,
learning-rate change, weight decay, warmup or EMA.

Its independent-panel conditional SWD is **0.02068247**, mean L2 is 0.04950375
and variance is **0.05240189**, against true variance 0.05282977. On the same
fixed validation panel at update 390720, the historical old K4 trajectory has
SWD **1.30415058** and variance **9.94802475**, whereas changing only grouping
to K64 gives SWD **0.03714138** and variance **0.05238461**. These comparisons
use identical evaluation settings. The independent and validation SWDs should
not be mixed because their finite-sample floors differ.

Thus larger per-input clouds were sufficient to rescue this known failing
warm-start trajectory at matched output exposure. This supports particle
grouping as the decisive practical change here, rather than attributing the
rescue to the full normalized package or improved solver arithmetic.

A plausible explanation is that four samples provide a poor finite-cloud
estimate of the conditional shape in eight output dimensions, whereas larger
clouds provide more useful shape-restoring interactions. This is a hypothesis,
not a proved instability mechanism. The earlier equal-output one-step screen
did not show a simple reduction of total parameter-gradient noise with K.
Rank deficiency of an individual four-particle cloud also does not imply
population non-identifiability across repeated batches. Further K16/K32 tests
could locate a practical threshold, but no minimal or optimal K is established.

The full normalized recipe previously passed on three development seeds.
Component deletions here use only one selected seed and a healthy warm start.
They do not establish fresh-start convergence, held-out-seed reliability,
downstream SER/BER, gradient fidelity or generalization to another channel.

## Reproducibility

- Protocol: `sspa_component_protocol_20261010.md`.
- Runner: `scripts/run_sspa_components.py`; exact `--resume` supported.
- Results: `results/sspa_components_20261011/`.
- K64-only bridge: `results/sspa_particle_bridge_20261011/`, with runner
  `scripts/run_sspa_particle_bridge.py` and separately frozen protocol
  `sspa_particle_bridge_protocol_20261011.md`.
- Auditor/plotter: `scripts/report_sspa_components.py`.
- Five new paths, 1553600 new updates, 85 verified saved checkpoints.
- Successful-reference replay matches model, optimizer, EMA and RNG bitwise
  after the first 2000 updates. Historical source and parent hashes are intact.
- CPU/CUDA tests cover loss scaling, epsilon conversion, intervention settings,
  exact resume, validation RNG preservation and the K64-only bridge.
- All 22 focused CPU tests and all four component CUDA tests passed.
- Measured synchronized training time totals 5688.02 seconds (1.58 hours),
  excluding the reused parent, reference replay, validation and file I/O.

The CSV includes both raw and EMA endpoints. The two-panel trajectory figure
uses the fixed validation panel throughout. `particle_grouping_only.pdf/png`
compares old K4 and old K64 directly. Original historical sources and the
prior intervention protocol remain unchanged.
