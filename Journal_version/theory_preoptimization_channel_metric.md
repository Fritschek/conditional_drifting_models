# A channel-surrogate metric before downstream optimization

Research direction, 9 October 2026. Extends [the SWD/gradient theory note](theory_swd_downstream_gradient_fidelity.md).

**Author's intended target:** evaluate whether a learned channel will be useful for communication-system design without training a new downstream encoder/decoder for every candidate surrogate. Measuring gradients through an already trained decoder is a diagnostic toward that goal, not the final metric.

**Proposed direction:** compare conditional expectations of a fixed function class and their derivatives with respect to transmitted inputs. Kernel mean embeddings make the supremum over a large function class computable by norms rather than fitting downstream models. Below are precise population guarantees, a finite-sample version, and the limitations that prevent calling the candidate a validated predictor of final BER.

**Latest empirical decision:** the [same-input learned-model test](learned_metric_gate_results_20261009.md)
completed the bounded gate proposed after the synthetic controls. The RBF
derivative score had no resolved added ordering information beyond cheaper
comparators in the seed-7 SSPA panel. Covariance/moment checks performed better
there, without becoming validated selectors themselves. Retain the population
bounds and gradient-fidelity distinction; stop expansion of this particular
kernel-selector candidate. Do not treat the proposed experiments later in this
note as overriding that completed negative test.

The [independent 10 October assessment](learned_metric_gate_assessment_20261010.md#3-why-a-valid-fidelity-bound-can-fail-as-a-task-ranker)
checks the results and supplies the exact ranking distinction: uniform
feature-loss gradient ordering requires positive-semidefinite ordering of the
loss-side operators \(DD^*\), which neither an operator norm nor a
Hilbert--Schmidt norm implies. DDIM-10 also outperforms Sinkhorn on all six
tested kernel-section losses, so loss-class membership alone does not explain
the empirical reversal. The bound remains valid; this selector is unvalidated.

The elementary bounds are proved here. The proposed metric's empirical usefulness, adequate loss-class coverage, useful numerical constants, and publication novelty are open. A full-kernel finite-difference estimator has now been implemented and tested in the [local metric pilot](channel_feature_metric_pilot_20261009.md). It has not demonstrated a predictive advantage over simpler controls; no new channel training was performed.

**Evidence update, 9 October.** The [local gradient-fidelity study](gradient_fidelity_local_pilot_20261008.md)
was run separately and is now part of the development evidence. It used trained
decoders and therefore did not compute the score proposed here. Section 8
below incorporates its results into the next experiment without treating that
task-dependent test as validation of a decoder-free metric.

**Subsequent metric pilot.** The [128/512-sample experiment](channel_feature_metric_pilot_20261009.md)
computes decoder-free scores at fixed radial inputs. CRN substantially reduces
the derivative sampling floor, SSPA mismatch remains visible, and moment
checks expose a large-variance failure whose severity the bounded kernel
does not rank correctly. Added value over ordinary MMD or derivative-aware
moments remains unestablished. Keep this as development evidence, not a
validated selector or final BER predictor.

## 1. Specify which downstream outcome is being predicted

There are at least three different targets:

1. **Receiver transfer:** how differently the same receiver behaves on true and generated channel outputs.
2. **Attainable performance:** how the best achievable receiver or communication system differs between channels.
3. **Optimization usefulness:** whether training through the surrogate follows useful directions and reaches a good solution within a budget.

A metric can bound one without predicting the others. A replacement-channel metric primarily measures **fidelity**, not the intrinsic difficulty of the true channel. An exact surrogate of a very noisy channel has perfect fidelity and can still yield poor absolute BER. Comparing final BER across different true channels requires a target-channel difficulty measure or reference as well.

For the paper, use target 3 as the main intended application, with formal guarantees first for loss values and local gradients. A claim to predict the endpoint of arbitrary nonconvex optimization would additionally need assumptions on the optimizer, initialization, objective geometry, and stochastic-gradient distribution.

## 2. Decision theory already supplies an ideal reference

Let \(P_x,Q_x\) be conditional channel laws. With \(\mathrm{TV}(P,Q)=\sup_A|P(A)-Q(A)|\),

\[
d_{\rm TV}(P,Q)=\sup_x\mathrm{TV}(P_x,Q_x)
\]

bounds the discrepancy of every fixed measurable output loss in \([0,1]\), hence every fixed receiver's SER, uniformly over transmitted inputs. This follows from the layer-cake representation \(f(y)=\int_0^1\mathbf1\{f(y)>t\}\,dt\). Thus a task-independent bounded-risk discrepancy is not mathematically impossible.

Its empirical version is problematic: two independent finite point clouds from the same nonatomic law have disjoint supports almost surely, and their empirical TV is one. Useful estimation needs structure, smoothing, or density modeling. Moreover, value discrepancy alone does not control input derivatives, as the previous note proves.

There is also a classical theory of comparing statistical experiments through **Blackwell ordering and Le Cam deficiency**. Define the directional quantity

\[
\delta(P\to Q)=\inf_K\sup_x\mathrm{TV}(K P_x,Q_x),
\]

where one common Markov postprocessing kernel \(K\) maps true-channel observations to surrogate-channel observations without knowing \(x\). If a particular \(K\) achieves error at most \(\epsilon\), any receiver using \(Q\) can be transferred to \(P\) by applying \(K\) before that receiver, with bounded-loss discrepancy at most \(\epsilon\). This direction follows immediately from the TV bound. Taking an infimum gives the same statement up to arbitrarily small slack if no minimizing kernel exists. The broader randomization theory relates such comparisons to decision problems. [Le Cam randomization criterion, research treatment](https://www.eurandom.tue.nl/reports/2002/017-report.pdf).

This concerns information available after an allowed transformation, not use of the same receiver unchanged or gradient-based training through a generator. For example, \(P_x=\mathcal N(x,\sigma^2)\) and \(Q_x=\mathcal N(-x,\sigma^2)\) have zero deficiency in both directions through \(y\mapsto-y\), despite a receiver trained for one failing if deployed unchanged on the other. Computing the infimum over transformations is also an optimization problem. It is a conceptual benchmark, not the inexpensive screening score proposed below.

## 3. Candidate: conditional feature values and input derivatives

Fix a compact admissible input set \(\mathcal X\subset\mathbb R^k\) and an open neighborhood on which derivatives will be defined. Choose a real separable Hilbert space \(\mathcal H\) and a fixed measurable feature map \(\Phi:\mathbb R^d\to\mathcal H\). Assume its Bochner means exist and are continuously Fréchet differentiable on that neighborhood. For example, integrability of \(\|\Phi(Y)\|\), with suitable measurability, ensures existence of the means; differentiability is an additional assumption.

Define

\[
\mu_P(x)=E_{P_x}\Phi(Y),\quad \mu_Q(x)=E_{Q_x}\Phi(Y),\quad \Delta(x)=\mu_P(x)-\mu_Q(x),
\]

\[
d_0=\sup_{x\in\mathcal X}\|\Delta(x)\|_{\mathcal H},\qquad
d_1=\sup_{x\in\mathcal X}\|D_x\Delta(x)\|_{\mathrm{op}(\mathbb R^k,\mathcal H)}.
\tag{1}
\]

Report \((d_0,d_1)\) separately. If one scalar is required, use \(D_\tau=\max\{d_0,\tau d_1\}\), with the input scale \(\tau>0\) declared in advance. Rescaling feature amplitudes or input units changes the numerical score and its associated loss-class bound; normalize and freeze these choices.

For finite features this is a **pseudometric**: distinct channel laws can have score zero. For a characteristic output kernel, exact equality of all conditional mean embeddings identifies the conditional laws on the evaluated domain. Neither statement grants a finite-sample certificate automatically.

**Proposition — simultaneous loss and gradient control without fitting a task.** Consider every collection of losses of the form

\[
\ell_m(y)=c_m+\langle w_m,\Phi(y)\rangle_{\mathcal H},\qquad \|w_m\|_{\mathcal H}\le R,
\]

where the coefficients are held fixed as \(x\) varies. For any finite codebook \(x_m\in\mathcal X\) and priors \(\pi_m\),

\[
|J_P-J_Q|\le R d_0,\qquad
\sup_m\|\nabla_x E_{P_x}\ell_m-\nabla_x E_{Q_x}\ell_m\|\le R d_1.
\tag{2}
\]

The second supremum includes all \(x\in\mathcal X\). For a differentiable encoder with
\(B^2=\sum_m\pi_m\|D_\phi e_\phi(m)\|_{\rm op}^2\), the encoder-gradient discrepancy is at most \(BRd_1\).

**Proof.** The loss difference at an input is \(\langle w_m,\Delta(x)\rangle\), bounded by \(R\|\Delta(x)\|\). Its gradient is \(D_x\Delta(x)^*w_m\), bounded by \(R\|D_x\Delta(x)\|_{\rm op}\). Average the first bound; apply the encoder chain rule and weighted Cauchy–Schwarz to the second. \(\square\)

If decoder parameters enter through \(w_m=w_{\psi,m}\), and \(\|D_\psi w_{\psi,m}\|_{\rm op}\le S\), the decoder-parameter gradient discrepancy is at most \(Sd_0\), assuming parameter differentiation is justified. Constants \(c_m\) may depend on decoder parameters because their derivatives cancel between the two channel objectives. This explicitly separates the receiver's dependence on value fidelity from the transmitter's dependence on input-derivative fidelity within this loss class.

These are exact dual norms for the respective function/gradient probe classes: at each input the supremum over \(\|w\|\le1\) recovers the norm in (1), with a supremum over input directions as well for \(d_1\). There is no need to fit the maximizing function. This is the useful step beyond evaluating a particular trained decoder.

Equation (2) supplies uniform objective approximation over encoders and decoders whose induced losses all belong to the stated radius-\(R\) class. The previous note then gives approximate-global-optimum transfer \(2Rd_0+\eta\), or, for a fixed decoder and an appropriate gradient-descent step,

\[
J_P(\phi^+)\le J_P(\phi)-\frac\gamma2\|\nabla_\phi J_P\|^2
+\frac\gamma2 B^2R^2d_1^2,
\]

under its true-objective smoothness and step-size conditions. Neither bound determines the final BER of arbitrary neural-network training.

### Full kernels versus finite probes

With an output RKHS, take \(\Phi(y)=k(y,\cdot)\). Then \(d_0\) is a supremum of conditional MMD values, and \(d_1\) measures derivatives of the conditional mean embedding. Kernel Gram matrices compute these norms without training a critic or downstream receiver. MMD itself and conditional mean embeddings are established tools. [Gretton et al., *A Kernel Two-Sample Test*](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), [Park and Muandet, *A Measure-Theoretic Approach to Kernel Conditional Mean Embeddings*](https://arxiv.org/abs/2002.03689).

The important unresolved issue is the loss class: a neural decoder's cross-entropy is **not automatically in a chosen RKHS with a useful bounded norm**. In particular, every function in a bounded-kernel RKHS is bounded, whereas cross-entropy can grow without bound. Tail restrictions, approximation on a bounded output domain, or a different function class would need to be explicit. A universal kernel's approximation property does not give a useful uniform norm bound for free.

If \(\ell_m=c_m+\langle w_m,\Phi\rangle+r_m\), with \(\|w_m\|\le R\) and \(\|r_m\|_\infty\le a\), the value bound becomes \(Rd_0+2a\). The gradient bound becomes \(Rd_1+b\) only if one **also** proves

\[
\sup_{x,m}\|\nabla_x E_{P_x}r_m-\nabla_x E_{Q_x}r_m\|\le b.
\]

Small uniform residual values alone do not imply that derivative bound. Resolving this approximation term for a useful decoder class is a substantive remaining theory problem.

## 4. It detects the previous counterexample without a decoder

In the fixed-power Gaussian construction of the previous note, use just the fixed bounded features

\[
\Phi(y)=(\sin y_1,\cos y_1).
\]

For first-coordinate mean \(a\) and Gaussian variance \(\sigma^2\), their expected vector is \(e^{-\sigma^2/2}(\sin a,\cos a)\). At either current codeword, true and surrogate means coincide, so \(\Delta(x)=0\). But their derivatives with respect to the second input coordinate differ by a vector of norm

\[
\|D_x\Delta(x)\|_{\rm op}=A e^{-\sigma^2/2},\qquad A=2/\sqrt3,
\]

independent of the oscillation index. Thus a fixed, untrained probe detects the persistent gradient obstruction even where all pointwise distribution metrics vanish. This is an exact arithmetic consequence of the Gaussian characteristic function and the mean-map derivative; it does not show that two trigonometric probes suffice for real channel families or certify logistic loss via (2).

Finite differences at one fixed perturbation size can miss arbitrarily rapid oscillations. Detecting this example as its frequency grows requires correspondingly small steps or access to exact derivatives. The estimator therefore needs explicit resolution/smoothness assumptions; a multiscale plot is a useful diagnostic, not a universal cure.

## 5. A finite probe bank cannot be universally sufficient

**Proposition — exact blind spot, including derivatives.** For any fixed \(r\) real-valued output probes and an output domain containing at least \(r+2\) distinct points, there are smooth-in-input channel families for which both scores vanish identically, while a decoder optimal for one channel has SER one on the other at a binary codebook.

**Proof.** Choose distinct outputs \(y_1,\ldots,y_{r+2}\). The \(r+2\) vectors \((1,\Phi(y_i))\in\mathbb R^{r+1}\) are linearly dependent. Thus nonzero coefficients \(a_i\) satisfy \(\sum_i a_i=0\) and \(\sum_i a_i\Phi(y_i)=0\). Their positive and negative parts have the same nonzero total mass. Normalize them to probability laws \(P_0,P_1\), which have disjoint supports and identical feature expectations.

Choose a smooth \(w:\mathbb R\to[0,1]\) with \(w(-1)=0,w(1)=1\), and set

\[
P_x=(1-w(x))P_0+w(x)P_1,\quad
Q_x=w(x)P_0+(1-w(x))P_1.
\]

Their feature mean maps are the same constant, so every input derivative also agrees. At equiprobable codewords \(-1,+1\), the true channel's disjoint supports permit error zero. The surrogate assigns the opposite labels to those supports, so its optimal decoder has true error one. \(\square\)

This obstruction is to a finite fixed bank serving **all** downstream tasks. A full characteristic kernel avoids this particular population feature blind spot, but finite data, input coverage, loss-class complexity, and derivative estimation remain. Do not label a finite random-feature approximation a universal certificate.

## 6. A sample-only estimator and an honest uncertainty bound

Choose anchors \(x_1,\ldots,x_s\), a finite feature bank of size \(r\), and step \(h>0\) before looking at evaluation draws. For each model and anchor, obtain \(N\) independent output samples at the base input and at \(x_a\pm h e_j\), \(j=1,\ldots,k\). All perturbations must lie in the admissible neighborhood.

1. Estimate each feature mean by its sample average and form \(\widehat\Delta(x)\).
2. Estimate derivative columns as \([\widehat\Delta(x+h e_j)-\widehat\Delta(x-h e_j)]/(2h)\).
3. Report the maximum mean-vector norm and maximum derivative-matrix operator norm across anchors, with average/quantile summaries as supplementary diagnostics.

This requires \(2s(1+2k)N\) total true-plus-surrogate output draws. It fits no downstream model. Perturbed queries are an access requirement: a passive dataset with one observation at each input requires an additional conditional estimation method and its assumptions.

**Finite-sample bound.** Suppose each probe lies in \([-1,1]\), and each coordinate of \(\Delta\) has third derivative along each coordinate direction bounded by \(M_3\) on the relevant perturbation segments. Put

\[
L=2sr(1+2k),\qquad
u=\sqrt{\frac{2\log(2L/\alpha)}{N}},\quad0<\alpha<1.
\]

Hoeffding's inequality and a union bound imply that, with probability at least \(1-\alpha\), simultaneously at all anchors,

\[
\|\widehat\Delta-\Delta\|_2\le2\sqrt r\,u,
\]

\[
\|\widehat{D\Delta}-D\Delta\|_{\rm op}
\le\sqrt{rk}\left(\frac{2u}{h}+\frac{M_3 h^2}{6}\right).
\tag{3}
\]

**Proof.** There are at most \(L\) scalar sample means. A bounded coordinate's error exceeds \(u\) with probability at most \(2e^{-Nu^2/2}\). On their simultaneous good event, a true-minus-surrogate mean has coordinate error at most \(2u\); its centered difference has error at most \(2u/h\). Taylor's theorem gives coordinate truncation error at most \(M_3h^2/6\). Vector Euclidean and matrix Frobenius bounds give (3), since operator norm is at most Frobenius norm. Cross-coordinate independence is unnecessary; independence within each sample mean is sufficient.

This bound is deliberately explicit, not claimed numerically tight. Balancing the two derivative terms suggests \(h\) of order \(N^{-1/6}\) and error of order \(N^{-1/3}\), ignoring dimension and logarithmic factors, when \(M_3\) is fixed. Exact pathwise derivative access may be cheaper, but changes the information available and needs its own variance assumptions.

A finite anchor maximum is not the population supremum. If the anchors form a verified \(\rho\)-net of \(\mathcal X\), and \(\Delta,D\Delta\) have known Lipschitz constants \(L_0,L_1\) in their respective norms, add \(L_0\rho,L_1\rho\) to the two upper bounds. Without those coverage and regularity constants, call the result a **screening diagnostic**, not a certified uniform error bound. Covering a large-dimensional input domain may be prohibitively costly.

For a full output kernel, replace finite vectors by empirical RKHS means. Centered derivative estimates are signed sums of kernel features; their inner products form a \(k\times k\) Gram matrix whose largest eigenvalue is the squared operator norm. That requires no optimization of a decoder, but (3) is the finite-coordinate bound, not a proved full-RKHS concentration theorem. A corresponding Hilbert-space bound should be established before reporting kernel confidence certificates.

## 7. What would make this a publishable and useful metric?

The generic use of conditional MMD is not new. A recent [conditional MMD framework by Moskvichev, Chau and Sejdinovic](https://arxiv.org/abs/2605.02260) explicitly compares several embedding/operator formulations. [Talwai, Shameli and Simchi-Levi](https://proceedings.mlr.press/v151/talwai22a.html) study Sobolev-norm learning rates for conditional mean embeddings; their interpolation/operator norms must not be equated with physical input derivatives without checking the connection. These are mandatory related work, not evidence that the proposed communications guarantee has already been proved.

The useful research package would contain:

- A loss class broad enough to include or quantitatively approximate a meaningful communication receiver family, with nonvacuous constants and explicit tail control.
- A score estimated from channel samples before any candidate-specific downstream training, including its sampling cost and uncertainty.
- A theorem connecting that score to loss/gradient fidelity, with coverage and smoothness assumptions that can be enforced or tested in the experiment.
- Held-out evidence that it predicts downstream transfer or training usefulness more reliably than global SWD, conditional SWD, ordinary conditional MMD, and simple moment/covariance checks.
- A demonstration that using the score to screen/select surrogates actually saves total compute without regularly discarding the useful ones.

The full score in (1) concerns **expected** gradients. Actual SGD also depends on estimator variance. Two reparameterizations of the same conditional laws can give the same \(d_0,d_1\) and different sample-gradient variance. Therefore a claim about performance per training time should additionally report sampling/backpropagation cost and gradient-estimator stability, or restrict the theorem to population-gradient updates. A single scalar fidelity metric need not encode all three.

## 8. Practical pilot and changes to the handover

### What the completed local experiment changes

The seed-7 AWGN/SSPA pilot establishes that the expected-gradient comparison
is executable with existing checkpoints. It supplies initial validation
targets for a new score, not that score itself. Its SSPA confirmation gives:

| Frozen analytic-trained SSPA codec | Conditional SWD | Encoder-gradient cosine | Encoder-gradient norm ratio |
| --- | ---:| ---:| ---:|
| Independent analytic estimate | 0.00656 | 0.998 | 1.000 |
| Condition-wise Sinkhorn | 0.02790 | 0.852 | 0.531 |
| DDIM-10 | 0.04865 | 0.956 | 0.048 |
| DDIM-100 | 0.01181 | 0.995 | 0.872 |

Sinkhorn's better SWD than DDIM-10 does not imply a better encoder-gradient
direction. DDIM-10's better direction does not imply a faithful gradient
magnitude. Both effects belong in the validation outcomes. At the
Sinkhorn-trained codec, its encoder cosine falls to 0.625 and its norm ratio
to 0.128, despite similar conditional SWD (0.02875). A task-independent score
can aim to flag fidelity risk across tasks; it cannot assign every decoder
the same realized error or claim a universal monotone ranking of final BER.

These observations support testing separate value and derivative components.
They do not establish that the proposed embedding derivative captures this
gap, that its upper bounds are tight, or that a kernel score is preferable to
a simpler channel-specific estimate. In particular, the current AWGN and SSPA
analytic laws are conditionally Gaussian with fixed additive covariance.
Include conditional mean/covariance errors **and their input derivatives** as
cheap comparators, alongside ordinary moments. Such moments do not identify an
arbitrary non-Gaussian learned law, so retain distribution-sensitive checks.

The existing one-step results use equal-norm parameter changes. Compare those
directional outcomes and common-learning-rate outcomes separately when testing
predictive value. Do not interpret a norm ratio as a predicted slowdown of Adam.
All already inspected seed-7 codec/method results are development data. Hold
out unused seeds or model families after freezing the score specification;
fresh Monte Carlo draws alone do not create a new held-out model test.

### Next implementation gate

Do not replace the main checkpoint selector yet. The initial full-kernel
feasibility part has run; the broader validation steps below remain the plan:

1. Use repeated conditional samples on AWGN and SSPA, a fixed admissible power domain, and a frozen multiscale output kernel/probe bank. Start with a full-kernel calculation on small clouds; use random features only as an explicitly measured approximation if cost requires it.
2. Compute mean-embedding error and input-derivative error at fixed anchors before inspecting any candidate-specific AE result. Compare coordinate derivatives on the small-dimensional pilot with a cheaper direction panel; the latter only certifies the probed directions unless an additional covering argument is supplied. Freeze normalization, bandwidths, step sizes, and aggregation on development data.
3. Verify the analytic oscillatory and label-reversal examples, an exact-surrogate null case, and moment-matched distributions with different decision behavior. Include the finite-bank blind spot as a stated theoretical limitation, not a benchmark the method is secretly tuned to pass.
4. Use a restricted existing/new downstream evaluation set as **ground truth for validating the metric**, not as input to its computation. The previous note's trained-decoder and one-step diagnostics remain valuable validation tools.
5. Freeze the score on development channels/methods/seeds, then evaluate its ranking and selection utility on held-out generator families or channel families, not merely held-out Monte Carlo draws. Measure rank agreement, false reassurance when the score is small, retained best downstream performance after screening, uncertainty, and total saved cost.

Use a nested comparison: SWD and conditional SWD; ordinary conditional
moments/MMD; moments plus input derivatives; embedding values plus input
derivatives. Give each the same declared channel-query budget, or show its
quality-versus-query-cost curve when budgets cannot be matched. Retain the
full `(d0,d1)` panel first; do not fit a scalar weight to the already observed
decoder rankings and call that a pre-optimization guarantee.

Proceed to a larger campaign only if the estimator resolves departures from
the exact-channel floor, is stable across the declared perturbation scales,
and adds held-out screening value beyond these cheaper controls at an
acceptable query cost. Otherwise refine the probes/estimation or retain the
gradient study as a task-specific contribution. A useful theorem alone does
not establish a useful screening instrument.

Some downstream optimization is needed once to establish that a proposed predictor works. The operational goal is then to avoid retraining a downstream model for every later surrogate checkpoint or new candidate. The unresolved proof obligation is not just finding another distribution discrepancy: it is linking an **estimable and affordable** discrepancy to a useful task class with meaningful constants.

## Numerical derivative reference and an unbounded feature component

The follow-up [resolution protocol](metric_resolution_protocol_20261009.md)
fixes an autograd reference and a moment augmentation. For a sampled output
`y_i(x)` with Jacobian `J_i`, differentiating the finite mean embedding gives
`mean_i D phi(y_i) J_i`. For an RBF with length scale `l`,

```text
D_y D_z k(y,z) = k(y,z) [I/l^2 - (y-z)(y-z)^T/l^4].
```

Contracting this matrix with sample Jacobians and signed P-Q empirical
weights yields the derivative Gram matrix without an input difference step.
This is a reference for the derivative of a **finite sampled embedding**,
not an exact population derivative. Samples from P and Q are independent.
Autograd requires differentiable simulator access; it cannot be applied
directly to a passive dataset. The finite-difference comparison shares the
same random draws at each perturbation within each simulator.

A bounded kernel cannot make a value discrepancy grow with output variance
without limit. Keep its distribution-sensitive component and add the fixed
features

```text
Phi_aug(y) = (Phi_RBF(y), y/sqrt(d), vec(yy^T)/d),
k_aug(y,z) = k_RBF(y,z) + y^T z/d + (y^T z)^2/d^2.
```

Here `d` is the real output dimension; unit per-coordinate input power is
the existing channel convention. No coefficients are fitted to gradient or
SER results. The second component estimates raw second moments, not centered
covariances. Means and covariances remain useful separate checks. Report all
components rather than relying only on a combined norm.

The direct sum enlarges the supported loss class to include linear and
quadratic output terms as well as RBF-RKHS terms. For any representation
`ell_x(y)=<a(x),Phi_aug(y)>` with bounded coefficient and coefficient-derivative
norms, the earlier gradient-error argument still gives
`||D_x(E_P ell_x-E_Q ell_x)|| <= ||D_x a|| d0 + ||a|| d1`.
This is not a claim that arbitrary decoder cross-entropy has such a
representation or controlled approximation remainder. Existence of second
moments is needed for the new mean embedding. Finite estimator variance needs
fourth moments and, for derivative estimates, appropriate integrability of
feature Jacobians. The new score is not intrinsically robust to heavy tails.

Recovering a variance failure and resolving finite-difference truncation are
necessary checks. Neither alone establishes an advantage over conditional
SWD or validates a pre-optimization selector.

### Exact-law and same-input controls completed

The [follow-up controls](metric_controls_results_20261009.md) now provide
explicit checks of the assessment's variance decomposition and same-input
mechanism. Orthogonal, input-dependent Gaussian noise rotations preserve the
full conditional law while increasing empirical derivative norms. At N=512,
the ordinary RBF derivative norm increases from 0.114 to 0.928 without any
population mismatch. Independent-split cross traces remain signed and target
zero in expectation. This proves neither unbiasedness of their eigenvalue
norms nor finite-sample confidence coverage.

The matched-value control keeps the laws at the evaluated inputs fixed while
varying local mean Jacobians. Exact kernel and polynomial task-class bounds
hold on the tested probes, including unused directions/probes. This supports
additional derivative information, but ordinary mean/Jacobian errors suffice
for these Gaussian mean perturbations. A kernel-specific selection benefit
on learned channels remains unestablished. Candidate-specific sampling noise
must accompany any later empirical derivative-fidelity claim.

### Beyond mean and covariance: an exact construction

The [post-control assessment, Section 3](metric_controls_assessment_20261009.md#3-an-exact-result-beyond-mean-and-covariance)
proves a stronger obstruction using five smooth Gaussian mixture components.
The two channel families have identical first three moments and their input
derivatives everywhere, and identical complete laws at the evaluated input.
Their RBF embedding derivatives and a fixed smooth loss's expected derivatives
nevertheless differ. A sequence even has uniformly vanishing total variation
and every fixed finite-order Wasserstein distance while the local gradient
discrepancy persists. Uniform curvature control is absent.

This establishes higher-order information beyond mean/covariance diagnostics,
not kernel necessity: a fourth-moment derivative detects this construction.
The proof also exposes an access limitation: changing categorical mixture
weights requires their derivative contribution, which naive pathwise
autodifferentiation through a sampled component misses. The bounded cosine
witness has no asserted Gaussian-RKHS norm; a separate norm-one kernel-section
witness is supplied. The accompanying bounded learned-model gate has since
completed without establishing added ordering value; see the latest decision
at the top of this note.
