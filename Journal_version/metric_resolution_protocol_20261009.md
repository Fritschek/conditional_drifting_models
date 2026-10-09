# Derivative resolution and moment augmentation

Frozen before running this extension on 9 October 2026. Seed 7 is development
data already examined. Seeds 8 and 9, if evaluated, are additional checkpoint
checks, not a new independent confirmatory study of the historical paper.

## Questions

1. Do CRN finite differences approach the derivative of the same finite empirical
   embedding? Compare h = 0.05, 0.0125, 0.003125, 0.00078125 with pathwise
   autograd derivatives. Keep the random draws identical across h and autograd.
2. Can explicit moments expose the full-budget SSPA variance failure understated
   by a bounded RBF kernel? Add features y/sqrt(d), vec(yy^T)/d to the existing
   equally weighted three-bandwidth RBF embedding. These are raw second moments,
   not covariances. Weights and normalization are fixed, not fit to task outcomes.
3. Does the pattern persist for additional existing checkpoint seeds?

Use the previous fixed 9-anchor panel (three directions, three radii), fixed
physical noise and RBF bandwidths, N=128, two independent MC repetitions.
Report RBF and moment components separately, as well as their direct sum.
Use N=512 to check the sampling floor after the derivative implementation agrees.
No retraining, decoder loading, or downstream-loss fitting is used for these scores.

After the seed-7 numerical checks, run N=512 pathwise estimates on seeds 7, 8,
and 9, keeping all nine anchors and two MC repetitions. Evaluate all historical
methods, including the full-budget SSPA control. Include the same 64-projection
conditional SWD at these anchors, with projection seeds 81001 + anchor index.
For seeds 8 and 9, use the existing frozen-codec gradient test separately as a
validation target: SSPA, both analytic-trained and Sinkhorn-trained codecs,
four MC repeats, 2048 gradient samples/message, 16384 reference/evaluation
samples/message, 1024 SWD samples/message, equal-normalized encoder steps
1e-4, 3e-4, 1e-3. Codec and generator checkpoint seeds both follow the run seed.
Do not fit a score or pool the two codecs as independent training seeds.
The generic metric panel and learned codewords differ; comparative prediction
claims must retain that limitation. These small additional-seed checks remain
exploratory, not a publication-level confirmatory test.

## Derivative reference

For k(y,z) = exp(-||y-z||^2/(2 l^2)), the mixed derivative is

    D_y D_z k = k (I/l^2 - (y-z)(y-z)^T/l^4).

Contract with each sampled output's input Jacobian and signed empirical weights
to obtain the Gram matrix of the derivative of the P-Q mean embedding. Its
largest eigenvalue's square root is the operator norm. This differentiates
distributional features, not a samplewise comparison of unrelated Jacobians.
It is exact for these empirical, fixed-noise maps up to numerical precision.
It does not remove Monte Carlo error or prove population derivative accuracy.

This calculation requires differentiable simulators and batch-independent
outputs. Check the implementations against samplewise CRN differences. It is
a reference calculation, not a solution for passive single-observation datasets.
Polynomial features require corresponding integrable moments (fourth moments
for finite variance of empirical raw-second-moment estimates). Derivative
estimation additionally requires integrability of the feature Jacobians.

## Decision rule

Report paired finite-difference/reference discrepancies and analytic floors.
Do not change weights, discard anchors or rank a model by a preferred step.
Even agreement with autograd and recognition of a variance failure establish
only numerical resolution and failure detection. Added prediction of task
gradients or downstream SER requires a separate, frozen validation protocol.
