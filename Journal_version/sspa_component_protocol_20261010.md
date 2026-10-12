# SSPA recipe component study

Frozen before training. This is an exploratory, single-seed mechanism study,
not an independent confirmation or a model-selection experiment.

## Question and attribution

The successful `official64` intervention directly calls the pinned authors'
W-Flow PyTorch loss (Han et al., commit
`1ccb34baf57f9f8f4e188460ea3d19b431403293`). It adapts their normalized loss,
K64 and optimizer recipe to our raw-output SSPA generator. It is not a
reproduction of their image experiments or the original Drifting JAX release.

The existing three-seed full-budget result establishes package-level stability.
It does not identify which components are necessary or sufficient.

## Fixed design

Use seed 9001, explicitly chosen because the lower-learning-rate intervention
failed late on this seed. Resume the same untouched fixed-common update-80000
parent used previously. Run every branch to update 390720 irrespective of its
200k performance. Reuse the successful `official64` reference trajectory;
verify the new implementation against it in an exact checkpoint replay.

Each branch changes one component/group of components of `official64`:

| Name | Change | Unchanged |
| --- | --- | --- |
| k4 | K64 to K4; 256 to 4096 distinct inputs | 16384 outputs per generated/positive/reference batch, normalized loss and optimizer |
| raw_loss64 | Remove feature-coordinate regression and force-RMS normalization | Same feature-scaled transport, epsilon, K64, optimizer; no radial field cap |
| fixed_epsilon64 | Restore parent physical epsilon | Normalized loss, K64, optimizer; convert physical epsilon to normalized R each batch |
| old_optimizer64 | Inherit parent Adam, moments, lr .001, beta(.9,.999), no weight decay or warmup | Official normalized loss, K64, R .05, parameter gradient cap 2, EMA |

Other branches use fresh AdamW, lr 1e-6 to .0002 over 5000 child updates,
beta(.9,.95), weight decay .01. All retain EMA .999 for secondary reporting.
Raw-model law metrics are primary. The optimizer branch tests an optimizer
package, not beta2 or learning rate individually. The K branch changes particle
grouping and input coverage at equal output exposure, not equal anchor count.
Loss normalization and adaptive epsilon are separated: removing normalization
keeps the identical transport geometry and uses MSE toward g plus the physical
cross-minus-self velocity. No clipping of that velocity is introduced.

Use the prior fixed validation panel (128 anchors, 128 samples, 64 projections),
every 10000 updates through 200000, then at 250000, 300000, 350000 and 390720
updates. Save restartable checkpoints at those points.
Trace every 100 updates. The independent final panel uses 256 anchors, 512
samples, 128 projections and seed 800501, as before. It is not used for selection.
Assess the original SWD/mean/variance bounds jointly at 300k, 350k and 390720.
All results, including failures, remain reportable. No new seed is substituted.

Keep historical source files and artifacts unchanged. Record parent, upstream,
source and protocol hashes, runtime, output counts, device and Torch version.
Test component isolation, normalized-loss parity, raw-gradient equivalence,
fixed-epsilon conversion, exact resume, and validation RNG preservation.

## Interpretation limits

A failing deletion identifies a needed ingredient in this particular package,
seed and horizon. A passing deletion shows that ingredient was not necessary
here. Interactions, independent seeds, cold-start training, downstream SER and
expected-loss gradient fidelity remain separate questions. Do not label this
as proof of a unique instability mechanism.
