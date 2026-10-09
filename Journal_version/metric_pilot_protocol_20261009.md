# Decoder-independent metric pilot: development protocol

Specified 9 October 2026 before computing the feature scores. This is an
estimator-feasibility study on known seed-7 checkpoints, not held-out validation
or another training campaign. The existing codec-gradient results are already
known and must not be presented as an unseen prediction target.

## Frozen choices

- AWGN n=7 at 5 dB, rate 4/7; SSPA n=8 at 8 dB, rate 3/4.
- Three seeded Gaussian directions, each at radii 0.5, 1, and 1.5 times sqrt(n),
  giving nine fixed anchors per channel. No encoder or decoder supplies inputs.
  This is a radial input panel, not an integration estimate under a Gaussian law
  or a covering net of the full admissible codebook domain.
- Full Gaussian RBF kernel averaged equally over lengths 0.5, 1, and 2 times
  the physical per-real-component noise standard deviation times sqrt(n).
  No bandwidth fitting against models, learned codewords, or downstream scores.
- At each anchor, query the base and all coordinate perturbations +/-h, using
  h=0.025, 0.05, 0.1. Do not renormalize individual perturbations: the target is
  the ambient conditional-law derivative, not a tangent derivative on a sphere.
- Initially 128 outputs per query and three Monte Carlo repeats. Report mean
  and maximum over this finite anchor panel, with repeat-level sampling error.
- Two access regimes: independent draws at different inputs, and common random
  numbers (CRN) within each simulator across perturbed inputs. P and Q use
  independent random streams. CRN is possible in the present simulation code;
  repeated observations alone do not guarantee that access in measured data.
- Compare empirical MMD and its embedding-derivative operator norm, conditional
  SWD, mean/covariance errors, and derivatives of means/covariances. Do not fit
  a scalar combination or subtract a noise floor and call it a certificate.
- Retain kernel V-statistic diagonal terms: scores are norms of empirical
  embeddings. Report the independent analytic-versus-analytic floor in the
  same sampling regime. Means/covariances use unbiased sample covariances.

The kernel derivative norm is computed from the signed finite-difference
embedding Gram matrix; its largest eigenvalue gives the squared operator norm.
All Gram arithmetic is float64. It is a full-kernel calculation, not a
random-feature or fitted-critic approximation. Finite differences remain
biased, and finite-panel maxima are not the population suprema d0 and d1.

## Controls and escalation

Before checkpoint measurements, test layout/sign/scaling, exact-cloud zero,
exact isotropic Gaussian kernel integrals, label reversal, the fixed-power
oscillatory example, and moment-matched laws distinguished by the kernel.
Check shared-noise mean/covariance derivative cancellation for the identity
channel. The oscillatory example checks estimator algebra, not real-network
performance or finite-resolution guarantees for arbitrary oscillation rates.

After a small execution check, run the 128-sample panel on the existing analytic,
kernel-target, joint Sinkhorn, condition-wise Sinkhorn, WGAN, DDIM-10/100 models,
plus the degraded full-budget SSPA checkpoint. If the initial sampling floor
limits interpretation, increase to 512 samples with the same anchors, steps,
kernel, and repeats for analytic, selected Sinkhorn and DDIM-10/100. This is a
predeclared sample-resolution check, not kernel/score selection.

Join the already computed codec-gradient outcomes only after feature scoring.
Keep direction error and gradient magnitude error separate. Compare relative
gradient error as well; a good direction with a very weak norm is not faithful
in the latter sense. Do not rank final BER from this small development panel.
The score must demonstrate added value over moments plus derivatives before
an expanded campaign. If it fails, retain and report the negative finding.

No new dependencies, channel training, cluster jobs, or manuscript edits are
required. Save checkpoint and source hashes, input panel, all scores, sampling
regime, draw counts and runtime. Existing checkpoints retain unequal training
budgets; these measurements cannot establish a fair model-family ranking.
