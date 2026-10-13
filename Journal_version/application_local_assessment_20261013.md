# Independent assessment of the local application pilots

Assessed 13 October 2026 against implementation commit `185fb96`.

**Decision: proceed with the cluster environment preflight if desired, but do
not freeze or launch the full scientific campaign yet.** The physical model,
receiver interface and exported counts are coherent. The local experiments
expose substantive model deficiencies which replication alone will not repair.
In particular, the generic Gaussian baselines have a provable representation
bottleneck, and the better-looking Sinkhorn metrics conceal a severe energy-tail
deficit. The bounded work below should precede confirmation seeds.

This assessment adds no training runs, cluster submissions, package changes or
changes to frozen numerical sources/protocols. It separates the current audit
from the execution-host tests reported by the implementation author.

## 1. What was independently verified

The [portable audit](evidence/application_local_independent_audit_20261013.json),
reproducible with [this stdlib-only script](figures_src/audit_application_local_20261013.py),
checks:

- 12 exported file hashes, the separately hashed post-hoc codebook inspection,
  and 47 distinct current source/protocol hashes.
- All 97 physical curve points: 27 SSPA, 35 clean OFDM, 35 interference OFDM.
- All 108 SSPA paired discordance records and their marginal error counts.
- OFDM symbol/bit/block rates, activity decompositions, squared block-error
  sums, marginal block-based BER standard errors and applicable zero-error bounds.
- Generator capacity/oracle-output accounting, including counted teachers;
  checkpoint hash references agree between the exported records.

The large checkpoints and pinned upstream source checkouts are absent from
this review checkout. Thus checkpoint **bytes** and upstream checkout contents
were not reverified here. Torch is also unavailable in the inspected local
runtime; the reported CPU/CUDA test results were not rerun. The current audit
does not replace the execution-host checkpoint audit or cluster preflight.

Code review and an independent mathematical audit found consistent complex
covariance orientation, proper complex noise scaling, fixed pilot placement,
hidden block activity, natural QPSK bit labels and sufficient-CP frequency-domain
modeling. The unitary FFT test covers integer-delay convolution; it does not
independently validate a time-domain fractional-delay implementation, which is
not what this simulator claims to implement.

One documentation gap: the evidence README advertises a `receivers/` directory
for preliminary CPU receiver comparisons, but it is absent from the committed
export. The final clean/interference suites do contain the useful physical-channel
receiver references. Export the preliminary records if the MLP-to-convolution
development comparison is to become paper evidence.

## 2. The OFDM task is useful; the learned channels are not yet adequate

At the nominal training/test point, Es/N0 = 10 dB:

| Receiver training source | Clean BER | Hidden-interference BER |
|---|---:|---:|
| Physical channel | 0.06349 | 0.08478 |
| Learned full Gaussian | 0.15461 | 0.15034 |
| Learned two-Gaussian mixture | 0.13591 | 0.22239 |
| Sinkhorn, one shot | 0.10843 | 0.12825 |
| Flow, 32 evaluations | 0.10465 | 0.12232 |
| Distilled consistency, one shot | 0.11264 | 0.13231 |
| Conventional pilot LMMSE, known statistics | 0.06414 | 0.08623 |

These are one generator seed and one receiver initialization per condition.
The physical receiver is close to the conventional reference at this operating
point. That makes the surrogate-transfer question meaningful, although it does
not prove receiver optimality or convergence. Each surrogate's gap from the
physical-trained reference is large compared with its marginal Monte Carlo
standard error (roughly 0.0006–0.0010). Training-seed uncertainty remains unknown.

Sinkhorn adds 4.49 percentage points of clean BER, about 71% relative to the
physical-trained receiver, and 4.35 points under interference, about 51%.
Flow is numerically a little better in both conditions. Its receiver training
cost is higher: approximately 52.2/50.5 seconds versus Sinkhorn's 14.2/13.7
seconds in these particular local runs. This is a possible accuracy/cost
tradeoff, not an established family ranking or hardware-independent speedup.

The interference breakdown is especially useful. Using the actual shared
activity fraction in the exported evaluation, decompose each BER gap as

\[
\Delta=(1-\widehat\pi)(e_{Q,0}-e_{P,0})
       +\widehat\pi(e_{Q,1}-e_{P,1}).
\]

For Sinkhorn, 0.042856 of its total 0.043472 excess BER comes from inactive
blocks: **98.58% of the gap**. Flow has essentially the same decomposition,
98.59% inactive. Sinkhorn's inactive/active BER is 0.11121/0.27978, versus
0.06353/0.27369 for physical training. Thus rare interference events do not
explain most of the present utility loss. This is a descriptive decomposition
of this paired test, not a causal intervention on the generator.

The SNR sweeps evaluate robustness of generators/receivers trained at nominal
10 dB. They are not comparisons of separately optimized models at every SNR.
The conventional receiver knows the test SNR and true second-order statistics;
retain that disclosed information advantage.

## 3. The Gaussian comparison has a structural bottleneck

The [Gaussian output head](../conditional_drifting/baselines/application_generators.py)
emits 128 means and 8,256 lower-triangular entries, totaling 8,384 outputs.
Matching the present roughly 600k parameter budget makes each hidden layer
only **69 wide**. Each component of the roughly 600k mixture has width **34**;
the two first-layer maps together have at most 68 output coordinates.

There are **112 free real data-input coordinates** after fixing the eight
complex pilots. Half the training law has a density on this entire data-input
space. This creates an elementary obstruction, stronger than an unverified
claim that the optimizer simply needs more steps.

**Claim, with its exact assumption.** Suppose each data tone k has a pilot p
with nonzero fading covariance R[k,p], and pilot values equal one. Neither
current bottleneck architecture can represent the true conditional law on
any open data-input region.

**Proof.** At a fixed input x the true complex covariance is

\[
C_b(x)=D_x R D_x^*+\sigma_b^2 I,
\qquad C_b(x)_{kp}=x_kR_{kp},\quad k\ne p.
\]

The cross-covariances therefore determine every complex data coordinate x_k.
This remains true for the unconditional interference mixture, since adding
the average diagonal noise does not change a pilot/data cross-covariance.
Consequently distinct data inputs have distinct true laws.

Restrict the Gaussian network's first affine map to the 112 data coordinates.
Its 69-by-112 linear part has a nonzero kernel vector v. For sufficiently
small nonzero t, x and x+tv remain in any chosen open region but give exactly
the same network output. Their true covariances differ, a contradiction.
For the mixture, stack both first-layer maps: the resulting 68-by-112 map
again has a nonzero kernel. Both component distributions and the constant
mixture weights are unchanged along it. The same contradiction applies. QED.

The nonzero-cross-covariance assumption is checked numerically for this
profile with a large margin: the minimum over data tones of the largest
pilot cross-covariance magnitude is 0.94135. The proof is conditional on the
stated algebraic assumption; this numerical profile check is not a formal
interval-arithmetic certificate. It can also be checked using the elementary
bound |R[k,p]| >= 1 - (2*pi*Delta_f)^2 Var(tau)/2, taking a nearby pilot.

**Limits of the claim.** This proves failure of exact representation on the
continuous input component. It does not lower-bound approximation error, prove
the observed BER is inevitable, or assert impossibility on the finite QPSK
input set. It also does not invalidate the general Gaussian/mixture families.
Increasing width removes this particular obstruction but alone guarantees
neither exact representation nor successful training.

**Required comparator.** Add a separately labeled structural baseline

\[
\widehat Y=D_xLZ+\widehat\sigma_B W,
\quad Z,W\sim\mathcal{CN}(0,I),\quad
\widehat R=LL^*,
\]

with a single input-independent learned complex covariance factor L, drawing
Z, W and B independently of one another and of the transmitted input. Use one
learned noise level for clean fading, and two ordered learned levels with a
learned input-independent mixture probability for interference. A free square
complex factor permits semidefinite R, including the physical low-rank case;
require positive observation-noise variance for likelihood evaluation. Fit
from allowed (x,y) observations without hidden taps, activity labels or the
true PDP. Count its smaller parameter count explicitly instead of padding it
to 600k. Toeplitz stationarity is a possible further assumption, but is not
necessary for this first control.

A clean-data initializer can estimate off-diagonal R[k,l] by regression of
y_k conjugate(y_l) on x_k conjugate(x_l), and estimate the common fading
diagonal/noise intercept from |y_k|^2 versus |x_k|^2 on the variable-amplitude
data tones. The mixed input law supplies the required amplitude variation;
pilot-only or fixed-modulus observations do not separate that diagonal from
noise by this regression. Use finite-sample PSD projection and constrained
likelihood refinement if needed. For interference, second moments identify
average noise only; identifying two levels/probabilities requires distributional
information, for example the unlabeled mixture likelihood. Do not pretend that
a covariance fit alone identifies the hidden mixture.

There are two cheap structural checks requiring no receiver training. First,
the pilot-output marginal must be independent of the transmitted data labels,
since every pilot input is fixed. Second, for unit-modulus QPSK on every tone,
the de-rotated block D_x*Y has an input-independent law: H plus proper isotropic
noise, or the same two-noise-state mixture. Compare these distributions across
held-out anchors. This is a distributional invariance, not a requirement that
a generator return identical samples or zero sample Jacobians when its latent
is held fixed. A violation diagnoses conditional structure error; agreement
with the invariance alone is not a downstream-risk guarantee.

These physical laws deliberately lie in the structural model classes. If the
small fitted models dominate, report that finding and treat this application
as a controlled fidelity/transfer test. It would not support a claim that a
generic neural generator is needed whenever a channel has many coordinates.

## 4. The strongest overlooked diagnostic is the missing energy tail

For each frozen QPSK anchor, let E(Y)=sum_k |Y_k|^2/64. The implementation pools
the analytic samples to choose its 90th/99th-percentile thresholds, then
evaluates fresh analytic and generated clouds. The clean 90th-percentile
threshold is about 2.121; the following are counts above the relevant
condition's threshold, out of 4,096 samples:

| Source | Clean above true 90th percentile | Interference above true 90th percentile |
|---|---:|---:|
| Independent physical samples | 412 | 455 |
| Sinkhorn | **1** | **1** |
| Flow-32 | **1** | **2** |
| Full Gaussian | 1 | 1 |
| Full mixture | 46 | 61 |
| Distilled consistency | 109 | 143 |

Sinkhorn also has zero samples above the analytic 99th-percentile threshold
in both conditions. Its observed 90th-percentile exceedance is 0.0244%, versus
10.06%/11.11% for independent physical samples. This is substantial missing
block-energy variation, even in the clean model. The reported scalar moments
do not determine whether it is mostly a mean-energy deficit, a radial-shape
error, or both. Save energy means/variances and a full quantile curve next.

This matters because Sinkhorn's clean SWD/floor is 1.524 and pilot/data
covariance-error/floor is 1.367, better than Flow's 2.154 and 3.865, while its
receiver BER is worse. Under interference the cross-covariance ratios are
1.288 versus 3.974, again with the opposite numerical BER order.
Finite-sample floors are noisy references, not acceptance tolerances. Sixteen
anchors with 256 samples each do not establish distributional equivalence
from a ratio modestly above one. The tail diagnostic independently exposes
large remaining mismatch.

Covariance alone can miss this in principle. For example, H=G a with
G~CN(0,1), and H'=exp(i Theta)a with uniform independent Theta, have the same
zero mean, proper covariance aa*, and pseudo-covariance zero, but different
radial laws. Adding the same independent proper Gaussian observation noise
preserves covariance equality and not distributional equality. This is an
exact illustrative counterexample to moment sufficiency, not a fitted
explanation or a proof of a particular receiver-performance gap.

**Connection to the gradient question.** The present OFDM runs keep the
constellation fixed. Receiver training therefore never uses the generator's
derivative with respect to x. These utility gaps cannot be attributed to a
bad transmitter-input Jacobian. The synthetic law can still bias the expected
receiver-parameter training gradient. Distribution fidelity, input-gradient
fidelity and downstream optimization should remain distinct questions.

## 5. SSPA now points strongly to decoder mismatch

At physical input Eb/N0 = 11.0103 dB, the primary evaluation has 12 errors for
physical training, 6 Gaussian, 13 Sinkhorn, 3,983 WGAN, 70 Flow-32, 310
Flow-128, 5 DDIM-20, 3 DDIM-50 and 37 distilled consistency, each out of
1,048,576 symbols. Counts of three versus six versus thirteen do not establish
a reliable across-training-seed ranking. The more accurate ordinary-input
Flow-128 SWD does not translate into better learned-system SER here.

The separate post-hoc CPU test applies both the saved neural decoder and the
exact physical ML decoder to the **same received samples and frozen codebook**:

| Codebook trained through | Saved neural errors | Physical ML errors |
|---|---:|---:|
| Physical channel | 16 | 1 |
| Gaussian | 8 | 1 |
| Sinkhorn | 22 | 0 |
| WGAN | 3,957 | 4 |
| Flow-32 | 69 | 1 |
| Flow-128 | 289 | 0 |
| DDIM-20 / DDIM-50 | 8 / 7 | 0 / 0 |
| Distilled consistency | 33 | 0 |

Equal priors and the common isotropic Gaussian noise make nearest physical
output mean exact ML for the implemented SSPA law. Even WGAN's frozen
codebook supports good physical decoding. This rules out poor physical
codeword separation as the principal explanation of its observed neural
decoder error in this test. It does not identify whether surrogate bias,
receiver optimization, or their interaction caused the mismatch, nor show
that all codebooks have equal population error.

The physical-trained neural decoder itself is imperfect. Treat it as a
finite-training reference, not a capacity-achieving lower bound. Zero observed
ML errors are not zero true risk; an iid common-probability model would give
a one-sided 95% upper bound about 2.86e-6 at this sample size. With these
fixed equal per-message allocations, the same bound remains conservative for
the uniformly averaged error under independent noise, by concavity of log(1-p).
Keep this post-hoc evaluation separate from the original primary results.

## 6. A precise, limited bridge toward a training-free utility metric

This channel allows an exact posterior calculation for a restricted receiver.
That is useful theory, but it does not yet solve evaluation for arbitrary
implicit generators or arbitrary optimized neural receivers.

Let P be the pilot-tone index set, k a data tone, Q uniform QPSK, and use only
O_k=(Y_P,Y_k). Write sigma_b^2=N0+b P_J, pi_0=1-pi, pi_1=pi, and

\[
A_b=R_{PP}+\sigma_b^2 I,\quad
\mu_b=R_{kP}A_b^{-1}Y_P,\quad
v_b=R_{kk}-R_{kP}A_b^{-1}R_{Pk}.
\]

Gaussian conditioning and independence of data-tone noise from pilot noise,
conditional on B, give

\[
w_b(Y_P)=\frac{\pi_b\,\phi_{\mathbb C}(Y_P;0,A_b)}
                   {\sum_c\pi_c\,\phi_{\mathbb C}(Y_P;0,A_c)},
\quad
p(q\mid O_k)\propto
\sum_b w_b(Y_P)\,\phi_{\mathbb C}
       (Y_k;q\mu_b,v_b+\sigma_b^2).
\]

Zero-weight states are omitted. Positive thermal noise makes all observation
densities well-defined. Unit-modulus QPSK is why v_b needs no |q|^2 factor;
an unequal-energy learned constellation requires that factor. In the clean
case there is one component and nearest q*mu is exact symbol-MAP for O_k.
The existing positive equalizer shrinkage gives the same QPSK hard decisions.
It is not full-block MAP: a neural receiver can also use the other data tones.
For the mixture, bit-MAP uses each posterior bit marginal and need not equal
the bits of a symbol-MAP decision.

For any candidate posterior q(.|O) on the **same observation**, with support
where p is positive, the excess expected logarithmic loss is exactly

\[
\mathbb E_P[-\log q(Q|O)]-H_P(Q|O)
=\mathbb E_{P_O}D_{\rm KL}(p(.|O)\Vert q(.|O))=:D.
\]

This follows by conditioning on O and subtracting the two finite-alphabet
log-loss sums. With natural logarithms and finite D, the symbol-MAP risk
satisfies

\[
0\le R_P(f_q)-R_P(f_p)\le\sqrt{2D}.
\]

Indeed the pointwise regret is at most twice posterior total variation;
Pinsker bounds this by sqrt(2 KL), and Jensen gives the displayed expectation
bound. The binary Pinsker inequality can be verified by grouping the positive
p-q coordinates and differentiating Bernoulli KL: its second derivative in
p is 1/[p(1-p)]>=4, so KL>=2(p-q)^2; log-sum gives the grouping inequality.
Thus no unverified specialized theorem is needed. Applying the same argument
to bit marginals bounds excess risk of **bit-MAP** decisions; marginal KL is
bounded by symbol KL by log-sum. The bound does not describe bits obtained
from an arbitrary symbol-MAP rule or a finitely trained neural receiver.

For fitted structural Gaussian/mixture baselines, this posterior and score can
be computed without training a receiver. For Sinkhorn/Flow, sampling does
**not** directly provide the required marginal densities or candidate posterior.
Fitting a Gaussian to their samples scores that projection, not their true
law. A controlled density/posterior estimator or a separately justified
sample-based discrepancy remains the obstruction to extending this guarantee.

A cheap immediate diagnostic is to run the frozen physical pilot decoder on
surrogate samples and compare its loss/error with physical samples. This
requires no downstream optimization and tests compatibility with a known
receiver. It is not a guarantee for a receiver trained on those surrogate
samples; an artificially easy synthetic channel can make that diagnostic
optimistic. Do not promote it to a universal selector without validation.

## 7. Concrete next handover

### A. Preserve these completed results; repair measurement exports

1. Keep `185fb96`, the current frozen manifests/protocols, final EMA weights,
   optimizer/RNG states and all failed-method outcomes. New versions go into
   child directories/manifests. Do not rerun the completed coverage studies.
2. Save per-block bit-error counts and activity masks, keyed by seed/SNR/block,
   or equivalent cross-method product sums. Current common random streams and
   marginal SEs are correct but do **not** retain enough information to recover
   a paired BER-difference SE. For per-block differences d_i in normalized
   bit loss, save sum(d_i), sum(d_i^2), and N; use
   SE=sqrt((sum(d_i^2)-sum(d_i)^2/N)/(N*(N-1))). Aggregate generator/receiver
   seed uncertainty separately from this conditional test Monte Carlo error.
3. Export generator learning traces/validation checkpoints, sampled energy
   means/variances/quantiles and per-anchor metrics. Final receiver training
   losses in the JSON are not generator-convergence evidence. For later
   learned-constellation runs, evaluate metrics at the actual learned symbols;
   the present `block_metrics` function always builds standard QPSK anchors.
4. Export the missing preliminary receiver records if used in the paper. The
   existing final-suite physical references already support the current gate.

### B. Fit the structural controls before scaling generic baselines

Implement the learned shared-covariance Gaussian and two-noise-state mixture
from Section 3 as explicitly structural baselines. Use only training (x,y),
with separate validation/test streams; retain all fitting/tuning costs. Check
recovered covariance/noise/tails against truth only as an evaluation, not as
training input. Start clean, then interference. Keep the same frequency
receiver, physical law, mixed input law and nominal SNR. The clean structural
model should approach physical-trained receiver performance across repeated
receiver seeds; otherwise inspect fitting/implementation before a large run.

### C. One bounded convergence study, then a decision

Use development seed 9401 and both conditions for a declared 10k/20k/40k
generator-update trajectory at batch 256 for Sinkhorn and Flow, alongside the
structural controls. Preserve the existing 10k endpoint and create an explicit
continuation protocol/runner; the current exact-config resume correctly rejects
a changed update budget. Do not edit frozen checkpoint identities to bypass it.
Retain K64 and the existing sampler initially so this tests budget rather
than simultaneous changes. Record oracle outputs, unique anchors and wall time:
Sinkhorn currently has only four input anchors per update and 40k anchors total,
versus 2.56m for the non-grouped models, despite equal oracle-output counts.
Equal query budgets are not equal input-diversity budgets.

At each endpoint use the same fixed validation anchor/tail panel and fresh,
paired receiver initialization/label streams. Predeclare the primary 10 dB
endpoint and a common selection rule using validation data; all five physical
SNR points remain available as diagnostics. Prefer a prespecified practical
BER tolerance to selecting whichever family wins one pilot. Check learning
curves before extending the full comparator panel. CD with a changed half
teacher/half student allocation is not a simple continuation of the old
trajectory; declare and count that separate training if included later.

For Sinkhorn, first ask whether the 90th/99th energy tails and inactive-block
transfer improve together. Saved-model analysis is enough to characterize the
current failure; do not attribute it to one mechanism without an intervention.
If four times the current budget does not repair it, inspect conditional
architecture/anchor diversity instead of launching ten replicas of the same
inadequate setup. Retain the dense Gaussian results as diagnostic controls;
merely training their present bottlenecks longer cannot remove Section 3's
representation obstruction.

For SSPA, the smaller remaining gate is a bounded WGAN adequacy check under
the common broad input law (including the already known raw-versus-EMA issue
and counted critic/generator updates). Exact physical decoding is now complete;
do not schedule it again as missing work. Receiver optimization/transfer can
be separated further with a fixed-codebook decoder study if needed, but that
is secondary to making OFDM scientifically informative.

### D. Cluster preflight versus scientific campaign

The existing `hpc/slurm_application_preflight.sh` is appropriate as an
environment/test/smoke/resume job. Create `logs/` **before** submitting, because
Slurm opens the declared log paths before the script can create directories.
Transfer/verify the pinned ignored upstream checkouts and use the cluster's
existing Torch stack. Local results used Torch 2.11/CUDA 12.8; the cluster
bootstrap defaults to Torch 2.1.2/CUDA 12.1.1. Compatibility remains untested
until that preflight succeeds. Module loads currently tolerate failure, so
inspect the actual reported runtime rather than assuming the requested module
was loaded. Do not install or upgrade packages automatically.

Only after B/C choose the final model/selection rules and create the campaign
runner. The provisional ten generator seeds by three receiver seeds remains
a later confirmation design, with at most 32 array tasks and generator-seed
aggregation. A passing 20-minute preflight cannot validate baseline adequacy,
convergence, full-budget memory/runtime or scientific conclusions. No such
campaign was submitted by this assessment.

## 8. Implications for the revised paper

- Keep the claim about learned channel simulation p(y|x), not receiver-side
  channel estimation. The new application evaluates synthetic-channel utility.
- Present the conventional receiver's statistical prior, SNR information,
  physical calibration, shared hidden activity and block-based uncertainty.
- Report the current useful finding as a mismatch between selected fidelity
  metrics and downstream receiver transfer, including the energy-tail failure.
  Neither these one-seed rankings nor this structured application establish
  Sinkhorn superiority or a universally predictive new metric.
- Separate the exact bottleneck/posterior statements above from numerical
  findings and untested mechanism hypotheses. Receiver-only failure does not
  require input-gradient failure. Reserve transmitter-gradient claims for the
  separate learned-constellation experiment after the receiver gate is met.
- Resolve the baseline defects before using this experiment to answer the
  reviewers' fairness concern. More realistic dimension alone is insufficient.

This round strengthens the experimental question and identifies concrete
obstructions. It does not yet justify freezing a TCOM resubmission or reversing
the provisional venue strategy. Venue choice should follow the repaired
application evidence, not determine which negative controls are retained.
