# Assessment of the new metric experiments

9 October 2026. Review of commit `c180522` and the gradient, feature-metric, and derivative-resolution reports. This is an assessment and next-experiment specification, not another experimental run.

**Decision:** the experiments strengthen the case that conditional distribution accuracy and transmitter-gradient fidelity are different. They do not yet establish that the proposed decoder-free kernel derivative score adds useful prediction beyond conditional SWD and simpler moment diagnostics. Continue a small controlled study; do not launch another large checkpoint-ranking campaign or replace model selection yet.

## 1. What was verified here

- Read the three new reports, their protocols, gradient runner, metric implementation, and comparison scripts. The reports carefully separate development evidence, normalized-step interventions, and final downstream performance.
- Checked the signed RBF mixed-derivative formula, empirical sample weights, moment normalization, and operator-norm construction. No material algebraic implementation error was found. An independent NumPy calculation of the mixed-kernel derivative block with unequal sample counts agreed with finite differences to `2.79e-4`, `2.79e-6`, and `3.80e-8` at steps `1e-2`, `1e-3`, and `1e-4`. This checks algebra, not the execution of the Torch/CUDA experiments.
- Record counts are consistent with the declared grids: `2430`, `1458`, `1350`, `558`, and `62` all reproduce arithmetically. Anchor/MC rows and codec cases remain nested observations, not independent trained models.
- **The twelve cited October result directories are absent in this checkout.** Numerical results below are taken from the committed reports; they were not independently regenerated here. The old result archives present locally do not substitute for the October JSON/tensor records.
- The available Python environments here lack Torch, so the reported 18 feature/gradient tests could not be rerun. No dependencies were installed. This does not contradict their reported execution on the Linux GPU host.
- The reporting-only repair below was tested locally using synthetic metadata fixtures: all nine new contract tests passed. These do not require Torch and do not validate the missing numerical records.

Sources: [gradient pilot](gradient_fidelity_local_pilot_20261008.md), [decoder-free pilot](channel_feature_metric_pilot_20261009.md), [resolution and additional seeds](metric_resolution_results_20261009.md).

## 2. The most informative empirical findings

The high-sample seed-7 SSPA comparison is the cleanest reported distribution-versus-direction example. At the same analytic-trained codec, selected Sinkhorn has lower conditional SWD than DDIM-10 (`0.02790` versus `0.04865`), but lower encoder-gradient cosine (`0.852` versus `0.956`). The independent analytic cosine is `0.998`, so this particular reported reversal is well separated from that reference control. Equal-length steps agree with the directional ordering.

Seed 9 supplies a different example: at the Sinkhorn-trained codec, WGAN has worse codeword SWD but better direction and total gradient error than selected Sinkhorn. The reported negative cosine and harmful step for the full-budget seed-9 model show a locally harmful direction at that particular operating point. They do not explain the historical instability mechanism or establish a universal full-budget failure direction.

The decoder-free metric has **not** demonstrated its intended added value. Its generic input-panel ordering misses some task-gradient orderings; it is not computed at those task codewords, and its upper-bound interpretation is not a claim of exact task ranking. Moment augmentation exposes variance failure, but SWD already exposes that failure. Retaining this negative finding is scientifically useful.

## 3. Fix the target before judging a metric

For a nonzero true gradient \(g\) and nonzero surrogate gradient \(\widehat g\), write

\[
r=\frac{\|\widehat g\|}{\|g\|},\qquad
c=\frac{\langle g,\widehat g\rangle}{\|g\|\|\widehat g\|},\qquad
e=\frac{\|\widehat g-g\|}{\|g\|}.
\]

The identity

\[
e^2=1+r^2-2rc
\tag{1}
\]

shows why direction, magnitude, and total error rank models differently. For a differentiable true objective, the first-order predicted improvement is proportional to:

- \(c\) for an equal-length parameter step \(-s\widehat g/\|\widehat g\|\);
- \(rc\) for a common plain-SGD learning rate \(-\eta\widehat g\).

The first statement follows from \(-s\langle g,\widehat g\rangle/\|\widehat g\|=-s\|g\|c\), and the second from \(-\eta\langle g,\widehat g\rangle=-\eta\|g\|^2rc\). A sufficiently small step and a suitable remainder bound are needed to infer actual loss changes. Neither formula models Adam.

The following is arithmetic from the **rounded reported seed-7 analytic-codec values**, not a new measured common-learning-rate intervention:

| Method | Direction \(c\) | Norm ratio \(r\) | Relative error from (1) | First-order progress ratio \(rc\), relative to true-gradient SGD |
| --- | ---: | ---: | ---: | ---: |
| Selected Sinkhorn | 0.852 | 0.531 | 0.614 | 0.452 |
| DDIM-10 | 0.956 | 0.048 | 0.954 | 0.0459 |
| DDIM-100 | 0.995 | 0.872 | 0.158 | 0.868 |

Thus the equal-length DDIM-10 advantage does not show better common-learning-rate progress: the reported gradients predict nearly ten times greater first-order progress for Sinkhorn in the latter comparison. It is also not a contradiction if a total-error metric prefers Sinkhorn over DDIM-10.

**Primary fidelity target for the next pilot:** absolute error of expected input-loss gradients at identical inputs and fixed tasks. Report relative error only when the reference gradient is resolved. Keep direction, magnitude, equal-length progress, and common-learning-rate progress as separate outcomes. For each claimed predictive benefit, specify which one is being predicted. Upper-bound validity alone does not imply correct ordering of every individual task.

## 4. The analytic sampling floor is not a universal error bar

This needs a stronger control before refining the score. Let the random feature-derivative operators be

\[
A_P=D_x\Phi(g_P(x,Z_P)),\quad A_Q=D_x\Phi(g_Q(x,Z_Q)),
\quad D=EA_P-EA_Q.
\]

Assume valid differentiation under expectation, finite Hilbert–Schmidt second moments, independent samples within each channel, and independent P/Q streams. For \(\widehat D=\overline A_P-\overline A_Q\),

\[
E[\widehat D^*\widehat D]
=D^*D+\frac{C_P}{N_P}+\frac{C_Q}{N_Q},
\quad C_P=E[(A_P-EA_P)^*(A_P-EA_P)],
\tag{2}
\]

with the analogous definition for \(C_Q\). Expand the product: cross terms vanish by zero means and independence; only within-sample covariance terms remain. Taking traces gives the exact signal-plus-variance decomposition of squared Hilbert–Schmidt discrepancy.

The analytic-versus-analytic control contains two analytic variance terms. It does not reveal \(C_Q\), which can be very different. The largest-eigenvalue and square-root operations used for the operator norm add nonlinear effects. Scores above the analytic floor therefore cannot, by that comparison alone, be interpreted as population discrepancy of the same magnitude. The current reports appropriately avoid scalar floor subtraction; preserve that choice.

One possible diagnostic is independent split batches:

\[
\widehat G_{\rm cross}
=\tfrac12(\widehat D_1^*\widehat D_2+\widehat D_2^*\widehat D_1),
\qquad E\widehat G_{\rm cross}=D^*D.
\]

Its trace is unbiased for squared Hilbert–Schmidt discrepancy. It may be negative or indefinite at finite sample size; clipping and taking square roots do not preserve unbiasedness. Treat it as a proposed uncertainty/bias diagnostic, not an already validated replacement score. Estimating operator norm with useful confidence requires additional analysis. Report candidate-specific repetitions/sample-size dependence rather than relying solely on the analytic null.

## 5. Next experiment A: an exact-law null with tunable derivative variance

This is inexpensive and requires no downstream optimization or trained checkpoints. It tests a necessary property of a purported **channel-law fidelity** score.

For an AWGN or additive-noise SSPA conditional mean \(f(x)\), let

\[
g_P(x,Z)=f(x)+\sigma Z,\qquad
g_{Q,\omega}(x,Z)=f(x)+\sigma R(\omega x_j)Z,
\quad Z\sim\mathcal N(0,I_d),\quad d\ge2,
\tag{3}
\]

where \(R(t)\) rotates a fixed two-dimensional output plane and leaves the other coordinates unchanged. Here \(\sigma\) is the physical per-real-component noise standard deviation, including the SSPA implementation's scaling if that channel is used. The coordinate \(j\) and rotation plane are fixed before evaluation.

**Exact claim.** For every input and every frequency, both conditional laws are \(\mathcal N(f(x),\sigma^2I_d)\). Hence their population conditional distribution and feature-derivative discrepancies are zero, wherever the latter are defined. Orthogonal invariance of the Gaussian proves this. Expected downstream-loss gradients also agree under valid differentiation.

Nevertheless, the pathwise Jacobian contains the extra term \(\sigma\omega R'(\omega x_j)Z e_j^T\). For the raw linear feature \(\Phi(y)=y\), averaging \(N\) independent draws gives

\[
E\|\widehat{D\Delta}\|_{\rm F}^2=\frac{2\sigma^2\omega^2}{N},
\tag{4}
\]

because the unrotated mean derivative is deterministic, and the derivative of a planar rotation has squared Frobenius norm two. Normalized moment features change the stated scale. Nonlinear kernel features have their own variance terms, but the population discrepancy remains exactly zero.

**Protocol proposal:** freeze frequencies \(\omega\in\{0,1,4,16\}\), sample counts \(N\in\{128,512,2048\}\), and at least eight independent repetitions for this synthetic control. Start with one declared nonzero anchor in dimension two; confirm the conclusion on the nine existing anchors only if useful. Use the same declared kernel scales and raw/moment components. Record pathwise scores, per-model variability, and optionally the split-batch Gram diagnostic. A finite-difference comparison must resolve the rotation scale; do not reuse a step that aliases rapid rotations and call agreement a proof.

The population score should remain zero; the finite-sample norm need not. The test is whether the uncertainty/bias treatment avoids claiming channel mismatch as \(\omega\) grows. Failure would be an estimator limitation, not a refutation of the population theory. If the ultimate objective also includes **SGD efficiency**, reparameterization-dependent gradient variance may be a useful separate quantity; name it as optimization noise rather than channel-law discrepancy.

## 6. Next experiment B: same inputs, matched value error, varied derivatives

After resolving experiment A, test added derivative information at identical inputs. Use a declared synthetic input panel, a fixed loss family, and channels whose value and derivative errors can be varied separately. This avoids conflating generic radial anchors with learned codewords.

For example, start from \(f(x)+\sigma Z\), and add smooth local mean perturbations near separated anchors \(x_a\). Choose bump functions equal to one near their anchor, with disjoint supports, and perturbations

\[
b_a(x)\left[\eta u+\lambda u\,v^T(x-x_a)\right].
\]

At each anchor, the mean displacement is exactly \(\eta u\) and the additional Jacobian exactly \(\lambda uv^T\). Hold \(\eta\), covariance, and directions \(u,v\) fixed while varying \(\lambda\). All population pointwise distribution distances at the anchors are then unchanged across \(\lambda\), while input derivatives change. A fixed panel of smooth bounded probes and simple logistic/quadratic receiver losses can be evaluated without optimizing any receiver. State each loss's integrability and kernel-class membership/approximation status explicitly.

Use absolute expected-gradient discrepancy as the primary target and include both signs of \(\lambda\). A norm of derivative mismatch does not encode its sign relative to an arbitrary task gradient, so do not demand universal direction ordering from it. Compare conditional SWD/MMD, moments and their derivatives, and kernel derivative scores under equal sample access or honest query-cost curves. Validate at additional held-out probe losses or perturbation directions without changing score weights.

Only after these controls should the metric be tested on new trained surrogates and unseen downstream tasks. A mechanistic synthetic demonstration by itself does not establish predictive superiority in real learned models.

## 7. Reporting repairs and provenance before expansion

1. **Codec identity protection repaired in this review.** The comparison script originally checked generator hashes and noise but could merge different codec checkpoints under the same `(seed, channel, codec label)`. It now checks codec SHA/configuration, noise/rate/EbN0, and declared/recorded step fractions across matching tasks, while allowing different Monte Carlo budgets. The joined CSV retains `codec_sha256`. Nine synthetic-fixture tests pass in `tests/test_metric_report_contract.py`. This is a reporting safeguard, not evidence that existing reported tables were misjoined. Legacy files without explicit normalization metadata cannot certify that unstored semantic contract.
2. **Keep numerical resolution wording precise.** The plotted statistic is relative error between two operator **norms**, not the norm of the difference between the operators. It validates the reported scalar score's resolution; it does not by itself establish derivative-operator orientation agreement. Samplewise finite-difference Jacobian checks provide additional evidence but are a different calculation.
3. **Measure complete screening cost.** Analytic reference generation occurs outside current per-candidate timers. Shared reference draws are reused, not newly generated for every model. Report oracle queries, one-time reference work, incremental candidate work, and total cost for a declared number of screened candidates. Do not multiply repeated reference metadata into a false total.
4. **Preserve a compact evidence bundle.** Copy the October `results.json`, manifests, comparison CSVs, and source snapshots alongside the report or into a documented archive. Saved gradient tensors enable further checks. Keep training checkpoints separate if large. A Git checkout containing only the reports is insufficient to regenerate their numerical claims.
5. **Check task-grid coverage explicitly.** Metric-suite grids are checked by the reporter; the number of joined task rows alone does not verify the intended gradient suite. Declare expected seed/channel/codec/method cells, record missing cells, and distinguish deliberately omitted methods from incomplete runs.

The journal can already use the carefully scoped observation that distribution scores and useful transmitter directions can disagree. A central claim of a better pre-optimization metric should wait for the estimator controls, same-input added-information test, and held-out selection evidence above. No new broad training campaign is needed to answer the immediate questions.
