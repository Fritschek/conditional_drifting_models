# Gradient fidelity for learned channel simulators

Research note, 8 October 2026. These are elementary derivations and proposed
experiments, not a novelty claim or a verified explanation of the saved results.
No manuscript text or training implementation is changed by this note.

## Prior-work boundary

- [Generative Modeling via Drifting, v1](https://arxiv.org/html/2602.04770v1)
  already includes conditional generation and per-condition batching.
- [W-Flow, v1](https://arxiv.org/html/2605.11755v1) includes conditional
  Sinkhorn velocities. Its generator-training gradient-flow analysis is a
  different question from the fidelity of gradients passed through a frozen
  simulator to a downstream encoder. I did not identify a theorem on the latter
  in these two versions. This is not an exhaustive novelty clearance.
- [Residual-Aided End-to-End Learning of Communication System Without Known Channel](https://oa.ee.tsinghua.edu.cn/~dailinglong/publications/paper/Residual-aided%20end-to-end%20learning%20of%20communication%20system%20without%20known%20channel.pdf)
  (IEEE TCCN, 2022) explicitly studies transmitter gradients through GAN channel
  surrogates and motivates residual connections by gradient vanishing.
- [Sobolev Training for Neural Networks](https://arxiv.org/abs/1706.04859)
  (2017) already trains function approximators on values and derivatives,
  including projected derivatives. Derivative matching alone is not new.

A possible contribution is a channel-law-level account of value, decision, and
gradient accuracy, validated against competitive generators, followed by a
principled improvement. Each part needs a broader literature check before a
claim of originality.

## 1. Define the downstream object before comparing derivatives

Let M be a message with a fixed distribution, x=e_phi(M) a differentiable
encoder, p_x the physical output law, and q_x a frozen surrogate law. Hold the
decoder fixed and absorb it into a differentiable loss ell_m(y). Define

\[
 F_{p,m}(x)=\mathbb E_{Y\sim p_x}\ell_m(Y),\qquad
 F_{q,m}(x)=\mathbb E_{Y\sim q_x}\ell_m(Y).
\]

The relevant input gradients are b_{p,m}=grad_x F_{p,m} and b_{q,m}.
For a reparameterized surrogate G(x,Z), with input-independent latent law,

\[
 b_{q,m}(x)=\mathbb E_Z
 [J_xG(x,Z)^\top\nabla_y\ell_m(G(x,Z))].
\]

This identity requires differentiability and an integrable local bound that
justifies exchanging differentiation and expectation. Nondifferentiable
channels may instead require finite differences or likelihood-ratio estimates.
Training cross-entropy, rather than the discontinuous hard SER indicator, is
the initial gradient target.

Compare expectations of vector-Jacobian products. Samplewise Jacobians from
two unrelated latent parameterizations need not agree even when both represent
the exact same conditional law. Equality p_x=q_x on a neighborhood implies
equal differentiable expected losses and gradients there. Equality at isolated
codewords alone does not.

For an encoder, the chain rule gives

\[
 g_q-g_p=\mathbb E_M[J_\phi e_\phi(M)^\top
 (b_{q,M}-b_{p,M})(e_\phi(M))].
\]

Consequently, under square integrability,

\[
 \|g_q-g_p\|\leq
 (\mathbb E_M\|J_\phi e_\phi(M)\|_{\rm op}^2)^{1/2}
 (\mathbb E_M\|b_{q,M}-b_{p,M}\|^2)^{1/2}.
\]

Proof: triangle inequality, the operator-norm bound, and Cauchy-Schwarz.
This is a standard chain-rule estimate, not a claimed new theorem.

## 2. Small conditional Wasserstein error does not control gradients

Consider a scalar AWGN channel and a perturbed surrogate, with sigma>0 and a>0:

\[
 Y=x+\sigma Z,\qquad
 \widehat Y=x-a\sin(x/a^2)+\sigma Z,\qquad Z\sim\mathcal N(0,1).
\]

They have equal variances and means separated by at most a. For every input,

\[
 W_2(p_x,q_x)=a|\sin(x/a^2)|\leq a.
\]

In one dimension population sliced W2 is W2. Thus even uniform conditional
SWD can approach zero. Coupling with the same X and Z also bounds the global
output W2 by a for any input law with a finite second moment.

For ell(y)=(y-1)^2/2,

\[
 F_p(x)=\tfrac12[(x-1)^2+\sigma^2],\quad
 F_q(x)=\tfrac12[(x-a\sin(x/a^2)-1)^2+\sigma^2].
\]

At x=0 the two conditional laws are identical, but

\[
 F_p'(0)=-1,\qquad F_q'(0)=a^{-1}-1.
\]

For 0<a<1 their gradients have opposite signs, and the mismatch is 1/a.
This establishes absence of an unconditional distribution-distance-only
gradient guarantee. It does not establish that trained drifting networks
exhibit this oscillation. The family has increasingly large input derivatives;
uniform regularity assumptions can rule out this construction.

## 3. A sufficient value-to-gradient bridge needs regularity

Suppose ell_m is L_l-Lipschitz in y and W1(p_x,q_x)<=d on a neighborhood.
The coupling inequality gives |F_q(x)-F_p(x)|<=delta=L_l*d.

Let r=F_q-F_p have an H-Lipschitz gradient on a ball around x, with |r|<=delta
throughout. For any unit vector u and admissible t>0,

\[
 |u^\top\nabla r(x)|\leq 2\delta/t+Ht/2.
\]

Proof: use the first-order Taylor remainder bound for r(x+tu)-r(x), then
bound the two function values by delta. If the ball contains radius
t=2 sqrt(delta/H), with H,delta>0, this yields

\[
 \|\nabla r(x)\|\leq 2\sqrt{H\delta}.
\]

For other radii use the unoptimized bound. Crucial assumptions are uniform
local accuracy and uniform input smoothness. Finite anchor SWD estimates do
not establish either. This is a standard interpolation-style estimate.
Cross-entropy composed with a fixed decoder needs an appropriate Lipschitz
bound; changing decoder weights changes the constants. ReLU kinks and the
current batch normalization need explicit treatment before applying smooth
claims to the implemented pipeline. No generic dimension-free replacement of
W1 by empirical SWD is being asserted.

## 4. Gradient error gives a local optimization guarantee

Let R_p(phi)=E_M F_{p,M}(e_phi(M)), with the decoder and surrogate held fixed.
Assume R_p has L_R-Lipschitz gradient along the update segment. Put
g=grad R_p, ghat=grad R_q. If

\[
 \|\widehat g-g\|\leq\eta\|g\|,\qquad 0\leq\eta<1,
\]

then for phi_plus=phi-alpha*ghat,

\[
 R_p(\phi_+)-R_p(\phi)
 \leq-\alpha\left[(1-\eta)-\tfrac12 L_R\alpha(1+\eta)^2\right]\|g\|^2.
\]

Proof: the smoothness inequality, g^T ghat >= (1-eta)||g||^2, and
||ghat|| <= (1+eta)||g||. A sufficiently small positive step gives descent
when g is nonzero. This is an elementary inexact-gradient result. It is local,
does not cover arbitrary Adam steps or decoder co-adaptation, and does not
guarantee improved final BER. Stochastic estimates require separate variance
terms or high-confidence error control.

## 5. Relating distribution error to fixed-decoder SER

For one message/codeword and a fixed decoder decision region D, choose a
coupling (Y,Yhat). Correctness can change only if Y is close to the decision
boundary or the coupling displacement is large. For r>0,

\[
 |P(Y\notin D)-P(\widehat Y\notin D)|
 \leq P(\operatorname{dist}(Y,\partial D)\leq r)
       +W_1(p_x,q_x)/r.
\]

Proof: if Y is farther than r from the boundary and ||Y-Yhat||<r, both
points have the same membership. Apply Markov's inequality and take the
infimum over couplings (or use near-optimal couplings). Handle ties through a
fixed decision convention and include boundary mass in the first term.
For a decoder with no boundary the decision is constant and the gap is zero.

Average this bound over messages for SER. Bitwise decision regions provide
analogous BER bounds. Nonconvex or curved boundaries do not invalidate it.
The boundary probability explains why the same transportation error can have
different communication significance for different decoders and SNRs.

This is a fixed-decoder evaluation bound, not a bound on the difference between
two separately trained systems. Encoder optimization introduces the gradient
question above, decoder learning introduces additional effects, and global
output SWD also loses the input/output association.

## 6. Experiments before modifying training

1. Start from identical frozen encoder/decoder checkpoints across surrogates.
   Evaluate current learned codewords and small neighborhoods around them.
2. Compute expected cross-entropy input gradients with independent Monte Carlo
   replicates. Report cosine alignment, absolute/relative vector error, norm
   ratio, and uncertainty. Handle near-zero reference gradients separately.
3. Verify the analytic reference gradient with finite differences of expected
   loss. Use common physical randomness across +/- perturbations where possible.
   The current analytic implant derives noise from batch power and detaches its
   scale; specify fixed-noise/power-constrained derivatives before trusting it.
   The symbolic encoder also uses batch-dependent standardization. Start with
   a fixed normalized codebook; test the full training graph separately.
4. Compare conditional SWD, conditional moments, decoder-margin distributions,
   frozen-decoder SER gaps, and gradient discrepancy on the same checkpoints.
   Distinguish fixed-decoder evaluation from subsequent retraining.
5. Take small encoder steps with each estimated gradient and measure change in
   independent analytic-channel expected loss. This directly checks utility.
6. Sweep SSPA checkpoints along controlled trajectories. Test whether gradient
   accuracy degrades before or together with conditional-law accuracy.
7. Only after identifying a mechanism, test derivative-aware training or a
   local-neighborhood objective, under matched compute. Probe expected test
   functions as well as one decoder loss to check decoder transfer. Any
   reference-gradient training uses additional simulator information and must
   be disclosed and offered to relevant comparator methods.

Do not promise a scalar adjusted SWD that universally ranks trained BER.
A useful outcome may be a complementary metric set with precise scope. The
scientific contribution would be a verified channel-specific mechanism and
effective remedy, with the elementary mathematics above supporting it.
