# Manuscript revision guide and mathematical audit

Companion to [the handover](resubmission_handover.md) and [cluster protocol](resubmission_cluster_plan.md). Source locations below refer to commit `726db308c46fe1df67319934ed35ef8dd8d79945`; line numbers will move after edits. The submitted source and ZIP were preserved during this planning task.

**Write in a separate revision source/tree.** Keep historical results and submitted artifacts identifiable. Correct attribution, definitions, and unsupported claims immediately; write new performance conclusions only after the corresponding experiments. Mathematical statements below are either proved here, explicitly conditional on stated assumptions, or identified as open empirical questions.

## 1. Reframe the contribution

Keep the present descriptive title, or use **Conditional Sinkhorn Drifting for Fast Learned Channel Simulation**. A title change is optional. The central change is the contribution statement.

Delete the sentence at `drifting_vs_diffusion_summary.tex:259` claiming that W-Flow is unconditional. The originating methods already include conditioning: Drifting groups interactions by class, and W-Flow equation (36) gives a conditional Sinkhorn velocity. Attribute both. [Original Drifting](https://arxiv.org/html/2602.04770v1#S3.SS5), [W-Flow conditional formula](https://arxiv.org/html/2605.11755v1#A2.SS3).

Suggested replacement introduction paragraph:

> Conditional drifting and Sinkhorn-based one-step generation provide the methodological starting point of this work. We study their use as stochastic communication-channel surrogates, where the condition is a continuously varying transmitted vector and repeated simulator queries provide output samples at a fixed input. The questions are whether the resulting conditional approximation remains stable under practical particle budgets, whether its input gradients support communication-system optimization, and how its accuracy and computational cost compare with other fast conditional generators.

This deliberately states research questions. The final contributions must state the answers actually supported by the revised experiments. Continuous-input adaptation and diagnostics can be useful contributions, but do not call conditional batching or the barycentric formula newly invented.

| Component | Attribution / revision position |
| --- | --- |
| One-shot generator and detached particle targets | Inherited drifting/W-Flow machinery |
| Per-condition batching and conditional Sinkhorn velocity | Already present in prior conditional generation |
| Conditional objective integrated over transmitted inputs | Channel formulation and interpretation; its zero-set proof is a simple lifting argument |
| Joint-versus-fixed-input comparison | Empirical question about geometry and finite computation, requiring equal particle exposure |
| Adaptive-epsilon implementation and stability behavior | Implementation choices and empirical findings; a new correction is a contribution only if justified and tested |
| Channel-law, learned-codeword and expected-loss-gradient diagnostics | Communications-specific assessment; distinguish new analysis from standard metrics |
| Downstream utility with modern fast competitors and a joint structured block | Main practical evidence to establish |

Add a related-work paragraph on modern one/few-step conditional generation and conditional optimal transport. One relevant additional lead is [Generative Conditional Distributions by Neural (Entropic) Optimal Transport](https://arxiv.org/abs/2406.02317); inspect the method and assumptions before drawing detailed comparisons. Do not turn a title-level literature lead into an asserted theorem or identical algorithm.

Avoid “one-shot methods must use fiber transport”: other conditional learning objectives can identify the correct conditional law. Say “we impose fixed-input transport in this estimator.” Avoid “makes drifting valid for channel simulation”: the conference and original conditional methods already provide conditional generators.

## 2. Exact joint equality: correction and a rigorous illustration

**Claim.** Let X and Y be standard Borel spaces, let μ be a probability measure on X, and let p and q be probability kernels from X to Y. Define

\[
P(dx,dy)=\mu(dx)p_x(dy),\qquad
Q(dx,dy)=\mu(dx)q_x(dy).
\]

Then

\[
P=Q\quad\Longleftrightarrow\quad p_x=q_x\quad\text{for }\mu\text{-almost every }x.
\]

**Structural reason.** Conditional distributions are identifiable up to μ-null sets when their joint law and conditioning marginal are specified.

**Proof.** The reverse implication follows by integration. For the forward implication, joint equality gives equal integrals of `p_x(B)` and `q_x(B)` over every measurable A in X. Thus they agree μ-almost everywhere for each B in a countable determining class of Y. Intersect the countably many full-measure sets and extend equality to all measurable B by the determining-class theorem. The countability is what provides one common null set rather than a different exceptional set for each B. ∎

**Counterexample to marginal matching, and a scaling caution for approximate joint matching.** Let a>0, X be uniform on {-a,a}, and define Y=X/a under P and Y=−X/a under Q. Both output marginals are uniform on {-1,1}; output-only discrepancy is zero. At every input the conditional squared W2 discrepancy is 4. Nevertheless, coupling each P atom to the Q atom with the same Y gives

\[
W_2^2(P,Q)\leq 4a^2
\]

under ordinary unweighted squared Euclidean joint cost. Small joint distance can therefore reflect condition scaling, although zero joint distance still identifies the correct law.

Replace source lines 269 and 337–338 with:

> Exact equality of joint laws with a common input marginal implies equality of the conditional laws almost everywhere. Our distinction concerns the transport geometry and its finite-sample optimization: a joint cost can couple outputs associated with different inputs, and its behavior depends on the scale assigned to the condition. The proposed estimator restricts couplings to fixed inputs. Matching only the output marginal does not identify a channel law.

**Audit.** The example changes the input scale a across problems; it does not prove that a fixed valid joint divergence has incorrect zero set. An observed advantage of the restricted estimator remains an empirical claim.

## 3. Replace Proposition 1 with a zero-set proposition

At source lines 312–345, separate three statements: the global minimizer, an energy identity along a sufficiently regular flow, and actual neural optimization. They have different assumptions and conclusions.

### 3.1 Precise minimal proposition

Let X be a standard Borel space, μ a probability measure, Y a space admitting the probability kernels under discussion, and x↦p_x,q_x measurable probability kernels. Fix ε>0. Assume

\[
s(x):=S_\varepsilon(q_x,p_x)
\]

is measurable, finite μ-almost everywhere and integrable, and that for μ-almost every x the applicable fiberwise Sinkhorn divergence satisfies

\[
S_\varepsilon(q_x,p_x)\geq0,
\qquad
S_\varepsilon(q_x,p_x)=0\iff q_x=p_x.
\]

Then

\[
\mathcal F(q):=\int S_\varepsilon(q_x,p_x)\,\mu(dx)\geq0,
\qquad
\mathcal F(q)=0\iff q_x=p_x\text{ for }\mu\text{-a.e. }x.
\]

**Proof kernel.** A nonnegative measurable integrable function has integral zero if and only if it is zero almost everywhere. Apply that fact to s, then use the assumed fiberwise zero characterization. ∎

This is a conditional lifting of the fiberwise result, not a new convergence theorem. One checked sufficient regime for the fiberwise property is a compact Euclidean output domain, the half-squared Euclidean cost, and fixed ε>0: the cost is Lipschitz on that domain and the Gaussian Gibbs kernel meets the positive/universal-kernel assumptions. [Feydy et al., Theorem 1](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf).

**Domain limitation:** the actual AWGN outputs and Gaussian-latent generators are unbounded. The compact-domain theorem cannot silently be cited as a theorem for those laws. Either verify an appropriate noncompact extension with all moment/tail assumptions and citations, or explicitly restrict the verified sufficient regime and leave the general experimental interpretation conditional on the fiberwise properties. Do not quietly truncate the experimental distribution to make the theorem apply.

### 3.2 Conditional dissipation identity

For Euclidean outputs, let

\[
\phi_{t,x}(y)=\frac{\delta S_\varepsilon(\cdot,p_x)}{\delta q}(q_{t,x})(y),
\qquad v_t(x,y)=-\nabla_y\phi_{t,x}(y).
\]

Assume a solution of the fiberwise continuity equation

\[
\partial_t q_{t,x}+\nabla_y\!\cdot(q_{t,x}v_t(x,\cdot))=0
\]

exists in the appropriate weak sense. Assume its energy is absolutely continuous in time, the first-variation chain rule and integration by parts are valid (with appropriate boundary/decay conditions), and the integrability needed to integrate over μ and time holds. In particular, the squared velocity must be integrable over the relevant finite time interval. Then, for almost every time,

\[
\frac{d}{dt}\mathcal F(q_t)
=\int\!\int \nabla_y\phi_{t,x}(y)\cdot v_t(x,y)\,q_{t,x}(dy)\,\mu(dx)
=-\int\!\int\|v_t(x,y)\|^2\,q_{t,x}(dy)\,\mu(dx).
\]

This is a conditional chain-rule calculation **under the stated regularity assumptions**. It is not a proof of well-posedness or long-time convergence. Use measure-valued notation; do not assume every conditional law has a density without saying so. W-Flow's particle-limit analysis is finite-horizon and under additional regularity/asymptotic assumptions, not a convergence guarantee for this finite neural implementation. [W-Flow Appendix A](https://arxiv.org/html/2605.11755v1#A1).

### 3.3 Stationary counterexample: why the stronger conclusion fails

Test the claim on one fiber before considering any network. Let

\[
q=\delta_0,\qquad p=\tfrac12(\delta_{-1}+\delta_1),
\qquad c(u,v)=\tfrac12|u-v|^2,\qquad\varepsilon>0.
\]

The unique cross coupling is `δ_0 ⊗ p`, with barycenter 0 at the sole source point. The self coupling also has barycenter 0. Thus

\[
v(0)=T_{q,p}^\varepsilon(0)-T_{q,q}^\varepsilon(0)=0.
\]

The constant curve q_t=δ_0 solves the continuity equation, yet q≠p. Under the entropic OT convention

\[
\operatorname{OT}_\varepsilon(q,p)
=\inf_{\pi\in\Pi(q,p)}\left\{\int c\,d\pi+
\varepsilon\operatorname{KL}(\pi\|q\otimes p)\right\},
\]

the values are

\[
\operatorname{OT}_\varepsilon(q,p)=\tfrac12,\quad
\operatorname{OT}_\varepsilon(q,q)=0,\quad
\operatorname{OT}_\varepsilon(p,p)=1-\varepsilon\log\cosh(1/\varepsilon).
\]

For the last identity, the symmetric 2-by-2 plan allocates total off-diagonal mass `1/(1+exp(2/ε))`; substitution into its cost plus KL gives the expression. Therefore

\[
S_\varepsilon(q,p)=\tfrac\varepsilon2\log\cosh(1/\varepsilon)>0.
\]

**Adversarial check.** Supports are compact; masses sum to one; ε is strictly positive; the cross and self barycenters are unambiguous. Velocity zero q-almost everywhere suffices for stationarity. Requiring a chosen extension of the velocity to vanish everywhere in ambient space would be a different condition and would not establish absence of stationary measures. The counterexample refutes a universal stationary-point assertion; it does not claim that the current training necessarily collapses in this way.

**Replace the conclusion by:** the target conditional family is the unique global zero/minimizer modulo μ-null sets (equivalently, as a joint law with marginal μ) over the stated class, and is a stationary state of the exact matching field. No conclusion about other stationary states or convergence follows from the zero-set proof and dissipation alone. Delete “best representable conditional surrogate” at source line 345.

### 3.4 Entropy must also be conditional

The hard-preserving coupling is

\[
\pi(dx,dy,d\bar y)=\mu(dx)\pi_x(dy,d\bar y),\quad
\pi_x\in\Pi(q_x,p_x).
\]

Its entropy penalty is

\[
\varepsilon\int\operatorname{KL}(\pi_x\|q_x\otimes p_x)\,\mu(dx),
\]

relative to the reference `μ(dx) q_x(dy) p_x(dy')`. For nonatomic μ, imposing x=x′ inside the product of the two **full joint** laws yields a singular coupling and infinite ordinary joint KL. Thus “equivalently” at source line 284 needs the conditional reference measure specified. The fiberwise objective is not obtained by silently retaining ordinary joint-product entropy after forbidding condition motion.

## 4. Explain precisely how the implemented update relates to the objective

Let X~μ and Z~ν independently, with both laws fixed and independent of θ, and let q_{θ,x} be the pushforward of ν by g_θ(x,·). At a fixed current parameter θ₀, let g_θ be differentiable in θ and write

\[
L(\theta;\theta_0)=\mathbb E\left\|
g_\theta(X,Z)-\operatorname{stopgrad}\big[
g_{\theta_0}(X,Z)+\eta v_{\theta_0}(X,g_{\theta_0}(X,Z))\big]
\right\|^2.
\]

If the parameterized first-variation chain rule and differentiation under the expectation are justified, and v is the exact fixed-common-epsilon negative first-variation gradient, then

\[
\nabla_\theta\mathcal F(q_\theta)
=-\mathbb E[J_\theta g_\theta(X,Z)^\top v_\theta(X,g_\theta(X,Z))],
\]

and direct differentiation of the frozen-target loss gives

\[
\nabla_\theta L(\theta_0;\theta_0)
=-2\eta\mathbb E[J_{\theta_0}g_{\theta_0}^\top v_{\theta_0}]
=2\eta\nabla_\theta\mathcal F(q_{\theta_0}).
\]

Coordinate-averaged MSE adds the factor 1/d. This is an **instantaneous gradient identity**, not equality of a finite optimization trajectory to a population flow. It shows why detachment alone does not rule out a legitimate objective gradient. Parameter stationarity can also occur when a nonzero velocity is orthogonal, in expectation, to all generator parameter directions.

Suggested replacement for source lines 340–345:

> With exact population velocities and a fixed common regularization parameter, the detached-regression gradient at the current generator is proportional to the conditional objective gradient, provided the relevant chain rule holds. Our implementation replaces this ideal velocity by finite-sample transport estimates and approximate numerical solves. Its independent reference samples, regularization policy, clipping and finite optimizer updates require separate analysis. We therefore use the population functional as a design motivation and evaluate the implemented training dynamics empirically; we do not claim convergence to the target or to a globally best generator in the model class.

Complete the approximation ledger in the algorithm section:

| Layer | Actual issue | What the paper may claim |
| --- | --- | --- |
| Population | Fixed ε, valid first variation and chain rule | Conditional objective / conditional identity only under stated hypotheses |
| Finite clouds | Small per-anchor K, independent generated reference | Stochastic empirical estimator; no automatic unbiasedness for population gradient |
| Epsilon | Defaults independently resolve attraction and self scales | A heuristic field unless tied to a separately defined objective; fixed/shared option must be named |
| Solver | Ten default iterations, exponential kernel floor, final row normalization | Approximate coupling; residuals must be measured |
| Particle update | Drift norm clipping | Modified velocity, especially during instability |
| Network | Tangent-space restriction, stochastic gradients, gradient clipping, Adam, finite steps | Practical training scheme; no inherited global optimum guarantee |

Some symmetric finite-batch formulations may be gradients of **their own expected empirical objective**. Do not overcorrect by asserting that finite batches or independent references can never correspond to any objective.

Code anchors: `losses.py:217` epsilon resolution; `:297` batched projections; `:875`/`:912` separate cross/self solves; `:927` drift clipping; `:1209` detached target; `training.py:415` repeated-condition branch and `:533` optimizer handling. Verify locations after edits.

## 5. Why the expected-loss gradient experiment is necessary

The [9 October theory note](theory_swd_downstream_gradient_fidelity.md) strengthens the elementary example below. Its Proposition 5 preserves transmit power, uses a fixed logistic decoder, matches the complete laws exactly at the current codewords, and reverses the encoder gradient despite uniform conditional Wasserstein convergence. Propositions 7–8 supply a positive curvature condition and a true-descent bound. Use that precise separation as the candidate main-paper result, credit prior Sobolev/gradient-aware and residual channel-surrogate work, and retain the simpler example below as intuition. Conditional population SWD can still imply fixed-task consistency under the note's assumptions; do not assert otherwise.

The following is a rigorous counterexample to a general implication, not an explanation already established for the empirical results.

For scalar AWGN Y=x+Z, Z~N(0,σ²), consider

\[
g_n(x,Z)=x+Z+\frac{\sin(nx)}n.
\]

The surrogate law is a translation of the true conditional law, so

\[
\sup_x W_2(q_{n,x},p_x)\leq\frac1n\longrightarrow0.
\]

Yet

\[
\partial_x\mathbb E[g_n(x,Z)]=1+\cos(nx),
\]

which equals 2 at x=0 for every n, while the true derivative is 1. Thus even uniform conditional Wasserstein convergence does not control input derivatives of expected observables. All displayed expectations exist, and differentiation is elementary here. A first-moment probe suffices to disprove the universal implication; the communications experiment must use the actual fixed decoder loss.

This supports the P3 diagnostic and a possible substantive finding: a fast surrogate can be useful for sample generation yet problematic for transmitter optimization. Whether that occurs in this study remains open until measured. If a gradient-aware correction is developed after the pilot, label it a new method, supply the full objective/assumptions, and ablate it against the corrected conditional W-Flow reference.

## 6. Concrete manuscript edits by location

| Current source / asset | Revision |
| --- | --- |
| Abstract, lines 29–35 | State repeatable conditional access early; scope to channel simulation; attribute existing mechanism; replace ranked claims with the specific controlled endpoint supported by new results |
| Introduction, lines 70–94 | Remove categorical unconditional-prior narrative; state communications question, conference extension, and 3–4 contributions tied to evidence |
| Background, lines 95–257 | Shorten generic drifting tutorial and kernel variants; keep the intuition needed to understand cross/self transport; move legacy kernel detail to supplement |
| Theory, lines 258–345 | Apply Sections 2–4 above; separate proposition, conditional identity, practical estimator; remove unsupported equilibrium/global optimum conclusion |
| Algorithm table, lines 371 onward | Define B, K_g,K_p,K_r, cost/masses, epsilon policy, solver/tolerance or iterations, clipping, optimizer, validation/checkpoint rule; label one-shot as one neural forward pass at inference |
| Local-data extension, around lines 398–421 | Mark untested; replace false equivalence between fixed-anchor target smoothing and smoothing both laws; move to limitations unless P6 is completed |
| Channel models, lines 423–507 | Define real/complex variance and Eb/N0/Es/N0 conventions; document compact TDL as a toy stress test, not full 3GPP validation; add the new structured configuration |
| Protocol, lines 508–561 | Replace “matched” legacy claims with precise comparison tracks; show actual samples/updates and selection; declare μ and normalization; distinguish generator seeds from AE seeds |
| Metrics, lines 563–588 | Conditional fidelity primary; define Gaussian W2, analytic floors, exact error counts and intervals; global SWD supporting only |
| Timing discussion, lines 602–618 and `timing_table_cuda.tex` | Replace extrapolated totals with full measurements; distinguish latency, throughput and AE forward/backward time; show true sample budget |
| `wflow_conditional_metric_table.tex` | Same anchor panel for all relevant families, with intervals/floors; no Sinkhorn-only evidence for across-family claims |
| `wflow_sspa_budget_table.tex` | Replace mixed endpoint rows with actual continuous trajectory plot plus selected/last checkpoint summary and failures |
| `wflow_coding_table.tex` / `wflow_all_ser_curve_figures.tex` | Add joint SSPA M=64 and fast baselines as applicable; common test record; no artificial upper-bound floor |
| `equal_wallclock_symbolic_table.tex` | Actual time–SER curves with modern fast methods and uncertainty; distinguish AE budget from generator budget |
| `wflow_curve_figures.tex` | Explain n=2 pairwise TurboAE mapping, fixed-SNR protocol, tests and limitations; do not present as joint high-dimensional generator evidence |
| Discussion/conclusion, lines 689–723 | State supported operating regime and failure regimes; distinguish numerical rank from supported difference; no convergence/measurement applicability beyond evidence |
| `references.bib` and contribution note | Cite accepted conference with verified status; update originating and fast-baseline sources; audit reference relevance and exact bibliographic fields |

### Reporting corrections already supported by stored audit evidence

- Timing: stored AWGN DDIM-100 values imply `0.0305906 h` for one million generations and `0.1284247 h` for ten million. The latter matches the old total; the manuscript points to the wrong sample budget. Explain the original discrepancy accurately; do not present new full-run timing until measured.
- SSPA: old table uses condition-wise/analytic SER `9.3333e-5 / 1.3000e-5`; old curve at 8 dB uses `8.1333e-5 / 1.5667e-5`. The earlier audit traced these to separate Monte Carlo evaluations of the same trained configurations. Restore raw records to verify, then generate new table/prose/curve from one canonical nominal-point evaluation.
- The old curve caption says floor-clipped points are upper bounds. That is not a valid statistical argument. Retain raw counts and use a defined zero-error interval if needed.
- The old Table IX boldface indicates numerical minimum, not significant superiority. In particular, the SSPA learned rows have substantial uncertainty. Report paired differences/intervals for predeclared comparisons and label numerical rankings accurately.
- Source/code discrepancy for compact TDL noise must be resolved explicitly: per-real-coordinate noise standard deviation σ gives complex variance `2σ²`, not `σ²`. Retain historical convention when reproducing historical results; use a new configuration ID for corrected noise.

Define the reported moment-based metric, with covariance matrices positive semidefinite and numerically stabilized as disclosed:

\[
W_{2,G}^2((m_1,C_1),(m_2,C_2))
=\|m_1-m_2\|^2+
\operatorname{tr}\left(C_1+C_2-2(C_1^{1/2}C_2C_1^{1/2})^{1/2}\right).
\]

Existing code averages `W_{2,G}` over anchors after the square root. It is a Gaussian moment comparison, not full W2 between arbitrary non-Gaussian laws. Describe it as such.

## 7. Paper structure and figures

Aim for a readable roughly 12–13-page core if retaining the TCOM option, subject to its actual submission category. TMLCN scope does not require expanding the paper with every developmental variant.

1. **Introduction and related work:** use-case, prior conditional generation, what remains unknown, contribution and conference extension.
2. **Channel learning problem and population motivation:** fixed input, sampling access, zero-set statement, conditional identity and its limits.
3. **Practical training:** estimator, epsilon/solver choices, checkpoint selection, sample and compute costs.
4. **Evaluation protocol:** comparison tracks, channel/input definitions, splits, seeds, error counts, baselines.
5. **Results:** geometry/stability, modern fast comparison, downstream gradients/time, larger structured channel.
6. **Limitations and conclusion:** data access, stability/coverage, gradients, hardware dependence and actual scope.

Proposed main figures:

| Figure | Question answered | Evidence source |
| --- | --- | --- |
| 1: compact training-versus-inference schematic, plus a small fixed-input illustration if space permits | What does Sinkhorn do during training and why is inference one call? | Algorithm, representative seed/anchor chosen by a declared rule |
| 2: SSPA conditional validation/solver behavior versus updates | Is the proposed selection/stability rule reproducible, and what changes after long training? | P1 continuous trajectories, selected/last markers, multiple seeds |
| 3: fidelity and actual downstream time–accuracy curves | Is there a useful operating regime against trained fast competitors? | P2/P3 matched protocols; actual NFE and hardware |
| 4: structured-block fidelity and link performance | Does the result persist beyond toy-dimensional or pairwise generation? | P4 full-block data, correlation and BER/block-error uncertainty |

Main tables: attribution/algorithm compact comparison; complete experiment protocol and capacity/cost; conditional metrics with floors; selected downstream endpoints/paired differences. Use supplement for detailed hyperparameter screens, all channels/curves, full timing repeats, and the additional TurboAE check. Keep key limitations and the competitive comparison in the main paper.

The submitted PDF places important time and TurboAE evidence on the last pages among references, while spending several opening pages on background. Rebalance floats and shorten the tutorial so the central controlled comparison is visible before the conclusion. Do not rely on layout alone to fix the substantive gaps.

## 8. Ready-to-use scope and extension language

**Sampling scope:**

> We assume access to repeated stochastic outputs for a fixed transmitted input under a specified channel-state distribution. This access is natural in a simulator and can be approximated by controlled repeated acquisition when the relevant state and stationarity assumptions are justified. The present results do not establish performance on passive datasets containing one observation per input. Local conditional estimation in that regime additionally requires assumptions on smoothness and neighborhood coverage.

**One-shot definition:**

> One-shot denotes one neural generator evaluation per output block, with a latent draw and fixed conditioning input. Sinkhorn iterations occur during training. We report full call timing, including the stated sampling and conditioning overhead, separately from batch-amortized throughput.

**Analytic reference:**

> Analytic-channel training is a reference using the true simulator and the same downstream optimization protocol. Its observed error rate is not a proven lower bound on achievable communication performance.

**Conference extension scaffold:**

> A preliminary version studied conditional kernel drifting, direct/residual output parameterizations, and an AWGN symbolic communication example [verified conference citation]. This article studies conditional Sinkhorn training, its numerical and sampling behavior, and its use in communication-system optimization. The journal extension adds [list only completed new analyses and experiments, distinguishing additions already present in the rejected journal from those added in this revision].

**Abstract scaffold — not submission-ready until results exist:**

> Learned stochastic channel simulators are repeatedly evaluated within communication-system design loops, making both sampling cost and optimization fidelity important. We study conditional Sinkhorn drifting for channel simulation with continuously varying transmitted inputs and repeated conditional simulator access. We distinguish the population objective from the finite-particle neural update and investigate [the validated stability/selection finding]. Controlled comparisons with [implemented fast baselines] evaluate conditional distributions, expected-loss input gradients, and communication performance under matched [specified resources]. On [validated channels], the method [insert quantitative result with exact comparator, budget, and uncertainty]. [State the principal failure regime or limitation.] These results identify [the supported use-case], with one generator evaluation per channel output block.

Do not fill bracketed results with the hoped-for outcome or reuse incomparable legacy numbers.

## 9. Response-document rules

Use the 22-row matrix in the handover to build a point-by-point response. For each comment: acknowledge the precise issue, state the actual change, cite the revised section/figure, and identify the supporting run/protocol. Reply to overlapping concerns individually while pointing to the same experiment; do not rerun duplicate studies solely because two reviewers asked.

For the theoretical objection, say the earlier conclusion was too strong and has been removed. For timing, explain the sample-budget mismatch and replacement measurement. For SSPA, report whether degradation was reproduced and what the controlled evidence supports. For practical data access, explicitly choose the narrowed scope or completed sparse-data experiment. Do not imply either scope restriction proves passive-data applicability.

Before final submission, adversarially check every result sentence: exactly which methods, channel law, capacity, budget, selected checkpoint, training groups, evaluation draws, and uncertainty support it? Check every theorem's quantifiers, exceptional null sets, support/tail assumptions, regularization convention, and relationship to the implemented algorithm. Unresolved limitations belong in the paper, not only in this handover.
