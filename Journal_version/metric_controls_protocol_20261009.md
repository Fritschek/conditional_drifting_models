# Frozen protocol: exact-law and same-input metric controls

Prepared before execution on 9 October 2026, following
`metric_experiment_assessment_20261009.md`. No checkpoint ranking or training.
The purpose is estimator calibration and a controlled added-information test,
not validation of a general-purpose model selector.

## Common settings

- Synthetic two-real-dimensional AWGN, mean f(x)=x, per-coordinate sigma=0.5.
  This is a mathematical control, not a new coding operating point.
- Float64 outputs, Jacobians and kernels. RBF length scales are
  (0.5,1,2)*sigma*sqrt(2), equally weighted, as in the earlier normalization.
- Moment features remain y/sqrt(2), vec(yy^T)/2. No fitted weights.
- Two independent batches per candidate/reference. Within each batch P and Q
  are independent. Reuse draws across candidate settings for paired contrasts.
- Report ordinary empirical value/derivative norms and the symmetric
  independent-batch cross Gram. Keep negative traces/eigenvalues; never clip
  the cross estimate or label its square root unbiased.
- Record per-repeat matrices, candidate-specific MC variability, exact
  population values, full runtime, reference preparation time, candidate/metric
  time, logical evaluations and distinct Gaussian sample vectors.
- New scripts and protocol are copied into the result directory with hashes.

## A: identical laws, varied reparameterization

Anchor a=(0.6,-0.8), rotation in the two output coordinates, input coordinate 0.

    P(x,Z) = x + sigma Z
    Q_omega(x,Z) = x + sigma R(omega*(x[0]-a[0])) Z

Centering the rotation at the anchor makes even the finite output samples
identical across omega, while retaining the derivative-variance effect. For
every input all candidates have the exact same Gaussian conditional law.

- omega = 0,1,4,16; N = 128,512,2048 **per independent batch**.
- Sixteen independent repetitions. No post-hoc sample-size selection.
- Check raw, unnormalized linear-feature squared derivative error against
  its exact expectation 2*sigma^2*omega^2/N. Its cross-batch expectation is zero.
- All population RBF, moment and augmented discrepancies are exactly zero.
- Finite-difference/autograd unit tests resolve the fastest rotation in
  float64, rather than using a large input step that aliases it.

The analytic omega=0 null does not estimate the sampling variance at omega>0.
An unbiased signed cross trace is a diagnostic; 16 repetitions do not create
an operator-norm confidence guarantee. Optimization gradient variance may be
useful separately, but it must not be called conditional-law mismatch.

## B: fixed value error, independently varied local derivatives

Anchors (-1,0), (0,1), (1,0). Smooth disjoint bumps have support radius 0.6 and
equal one within radius 0.15. The conditional mean is

    m_Q(x) = x + sum_a bump_a(x)*(eta + lambda*v^T*(x-a))*u.

At anchor a, m_Q=a+eta*u and J_Q=I+lambda*u*v^T. Sigma is unchanged.

- eta = 0,0.1; lambda = -4,-1,0,1,4.
- Development directions u=v=(1,0).
- Unused-direction check u=(1,1)/sqrt(2), v=(1,-1)/sqrt(2).
- N=512 per batch, eight independent repetitions. Both batches use fresh
  P/Q streams, shared across lambda within each anchor/direction/eta cell.
- Value metrics must be identical across lambda on the paired samples.
- Compute exact Gaussian RBF embedding values/derivative Gram and exact
  first/raw-second-moment counterparts, not just noisy empirical scores.
- Fixed task probes at the **same anchors**: two mixture-kernel sections
  centered at (0.3,-0.4) and (-0.7,0.6); quadratic, binary-logistic and bounded
  cosine losses with projection/offset ((0.8,0.6),0.2) and ((-0.6,0.8),-0.3).
  The second probe in each pair is an unused-probe check. No weights are tuned
  on either group. The quadratic is .5*(w^T y-b)^2; logistic is softplus(w^T y-b)
  (label zero); bounded loss is 1-cos(w^T y-b).
- Exact Gaussian expectations where available. Logistic expectations use
  Gauss-Hermite quadrature with a 64-versus-128-node convergence check.
- Primary target: absolute error of expected input-loss gradients. Also
  record cosine, norm ratio, total relative error (where nonzero), and
  first-order normalized-step and plain-SGD predictions as distinct quantities.
  No claim of actual finite-step progress is made in this experiment.

Mixture-kernel sections have RKHS norm one, so their exact gradient discrepancy
is bounded by the exact RBF derivative operator norm. Quadratic losses are
covered by the polynomial augmentation up to an irrelevant constant. Logistic
and bounded cosine probes are integrable under these Gaussian laws but are
not asserted to belong to the RBF RKHS with controlled norm.

## Decision and interpretation

Run A before B. If candidate-dependent bias appears, report it and use the
known population reference in B to distinguish useful derivative information
from finite-sample variance. B may proceed as an exact synthetic mechanism
test, not as evidence that the empirical estimator has been calibrated for
unknown learned models. Keep simple mean-Jacobian errors as a comparator.
For Gaussian mean perturbations they may be sufficient, and that outcome
does not establish a need for a kernel metric. Do not extrapolate these
controls to final SER/BER, passive measured data or new model-family rankings.
