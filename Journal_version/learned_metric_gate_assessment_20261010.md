# Independent assessment of the learned-model metric gate

10 October 2026; review of commit `90a4449` and the [9 October results](learned_metric_gate_results_20261009.md). No new simulation, training, checkpoint selection, or manuscript-result replacement was performed.

**Decision:** stop expanding this fixed kernel-selector candidate. Preserve the derivative-fidelity theory and the negative empirical finding. Resume the journal's measurement-contract and controlled SSPA stability work (P0/P1), with modern-baseline implementation in parallel. The broader goal of assessment before downstream optimization remains open; this experiment does not validate a new selector.

## 1. Independent verification

The transferred archive's SHA-256 matches `84a437a2007eacf7265b8b41da8215a671c398f18730ed735e96526e6e689835`. Seventeen result files were restored without replacing current sources. Independent NumPy calculations, without importing the runner or reporter, reproduce:

- Coverage: 192 metric records and 1,536 task records; 144 metric records belong to learned models. All refinement decisions and final comparator counts match.
- Every numeric summary field in the metric CSV exactly; task CSV fields to maximum absolute difference `2.22e-16`.
- Signed task cross errors directly from stored gradient vectors; moment cross scores directly from stored split vectors; self/cross/noise matrix identities. The largest checked arithmetic residual is `4.27e-14`. All stored derivative-noise matrices are positive semidefinite to numerical precision.
- All recorded source hashes match the current checked sources. The execution records also retain the protocol hash; this supports provenance but is not an external timestamp proving when the protocol was frozen.

An independent source and reference audit found no material implementation flaw. A closed-form SSPA Jacobian agrees with saved references to `1.11e-15`; finite differences of all 24 Gaussian-integrated probe expectations agree with their gradients to `1.31e-9`. The eight-dimensional probe lift, real-component noise scaling, unbiased covariance derivative, independent score/task splits, and paired model contrasts are consistent with the protocol.

Evidence: [machine-readable audit](evidence/learned_metric_gate_independent_audit_20261010.json) and [NumPy audit script](figures_src/audit_learned_metric_gate_records.py). Run the latter with NumPy available after restoring the archive. It checks saved sufficient statistics, not the omitted individual output/Jacobian tensors. The execution host reports 48 passing focused tests; Torch tests and GPU sampling were not rerun here. Float32 simulator draws are subsequently analyzed in float64; mathematical unbiasedness is not an exact floating-point claim.

## 2. A stronger negative finding than the aggregate count

The frozen final rule yields zero added-information contrasts. Its baseline is the **union** of six cheaper comparators: the successful cheaper comparator may change with the contrast. That result does not identify one deployable replacement score. Likewise, 72 contrasts reuse three trained models, three inputs and eight probes; they are not 72 independent model replications. The three-MC-SE rule is descriptive, without calibrated simultaneous coverage.

There is nevertheless a direct reversal that does not depend on this union rule. Sinkhorn has the lowest kernel derivative cross trace at every input, while DDIM-10 has the lowest absolute expected-gradient error for 21 of the 24 fixed input/probe cases. **DDIM-10 wins all six kernel-section cases**, whose losses belong to the kernel's own RKHS. Thus excluding the quadratic, softplus and cosine probes does not remove the reversal.

At input 0, for the first kernel-section loss:

| Quantity; smaller means less discrepancy | Selected Sinkhorn | DDIM-10 |
| --- | ---: | ---: |
| Conditional SWD | 0.03256 | 0.04906 |
| RBF derivative cross trace | 0.29545 | 1.08812 |
| Absolute expected-gradient error for that kernel loss | 0.009133 | 0.004018 |
| L2 norm of coordinate Monte Carlo SEs for its gradient estimate | 0.000100 | 0.000070 |

The last row is not a confidence interval for the scalar error norm. The independently sampled target cross-error contrast also passes the declared resolution rule. Across the 18 pairwise kernel-section contrasts, the RBF derivative ordering agrees in 7, opposes in 9, and is unresolved in 2. These are a descriptive breakdown of the saved development panel, not a new confirmatory test.

The final gate uses signed estimates of **squared Hilbert--Schmidt discrepancy**, not the original operator-norm \(d_1\). A post-hoc check of the already saved ordinary operator norms and the largest eigenvalues of averaged cross matrices still ranks Sinkhorn lowest at all three inputs. This makes a trace-versus-operator switch an unpromising explanation of the principal reversal here. Those nonlinear estimates are not unbiased operator norms or confidence bounds; no new selection rule was fitted.

The mismatch also need not violate an upper bound. For the 18 learned-model/kernel-section cells, squared pooled task errors divided by mean RBF cross trace range from `4.51e-7` to `3.07e-4`. These estimated ratios are consistent with a very loose envelope for these particular probes. They are not certified population ratios, a bound-coverage result, or evidence about every RKHS loss.

## 3. Why a valid fidelity bound can fail as a task ranker

This is an elementary structural distinction, not a novelty claim.

Fix an input and let \(D_A,D_B:\mathbb R^k\to\mathcal H\) be bounded derivatives of true-minus-candidate feature-embedding mismatch for two candidate channels, where \(\mathcal H\) is a real Hilbert space. For a fixed feature-linear loss \(L_w(y)=c+\langle w,\Phi(y)\rangle\), under the differentiability assumptions in the [metric theory note](theory_preoptimization_channel_metric.md), its input-gradient error is \(D_A^*w\) or \(D_B^*w\).

The exact statement that A is at least as faithful for **every** unit-ball loss is

\[
\forall w\in\mathcal H,\ \|w\|\le1:
\quad \|D_A^*w\|\le\|D_B^*w\|.
\tag{1}
\]

**Proposition.** Statement (1) holds if and only if

\[
D_A D_A^*\preceq D_B D_B^*
\quad\text{as operators on }\mathcal H.
\tag{2}
\]

**Proof.** Squaring (1) gives
\(\langle w,(D_BD_B^*-D_AD_A^*)w\rangle\ge0\) on the unit ball. Homogeneity extends it to every \(w\), which is exactly positive semidefiniteness. The same identity proves the converse. Both operators are bounded positive finite-rank operators because the input dimension is finite. No compactness, maximizer existence, or exchange of limits is needed. Zero operators are included. \(\square\)

A smaller operator norm, or a smaller Hilbert--Schmidt norm, does not imply (2). Test the simplest case, \(k=1,\mathcal H=\mathbb R^2\):

\[
D_Au=(u,0),\qquad D_Bu=(0,2u).
\]

Then \(\|D_A\|_{\rm op}=1<2=\|D_B\|_{\rm op}\), also their Hilbert--Schmidt norms, but \(w=(1,0)\) gives errors 1 and 0. For \(w=(0,1)\), the preference reverses: errors 0 and 2. Their loss-error operators \(\operatorname{diag}(1,0)\) and \(\operatorname{diag}(0,4)\) are incomparable in positive-semidefinite order. This example is realizable with feature map \(\Phi(y)=y\), scalar input, and true/candidate Gaussian channel means \(0,(-x,0),(0,-2x)\) with common fixed covariance. All losses used in this example are Gaussian-integrable. It is not an assertion that those linear losses have finite Gaussian-RKHS norm.

Thus neither perfect estimation nor an operator-norm substitution supplies a universal strict ranking of individual-task gradient fidelity. A worst-case guarantee remains meaningful: \(\sup_{\|w\|\le1}\|D^*w\|=\|D\|_{\rm op}\). It simply answers a different question. The sampled input-side Gram \(D^*D\) also lives on a different space from the loss-side operator \(DD^*\) in (2); its scalar norm alone discards the needed loss orientation.

**Implication for the original ambition.** Assessment without optimizing a downstream model is still possible for a declared task family. For example, fix a strongly measurable random loss coefficient \(w\) in advance, with \(E\|w\|^2<\infty\), and let \(C=E[w\otimes w]\) be its uncentered second-moment operator. Then

\[
E_w\|D^*w\|^2=\operatorname{tr}(D^* C D).
\tag{3}
\]

Expand the squared norm over a finite orthonormal basis of \(\mathbb R^k\) and take expectations to prove (3). This is a population identity for expected **squared** local-gradient error under that specified task distribution. It is not an automatic estimator or a prediction of final BER. Choosing \(C\) after seeing these task outcomes would tune to this panel. No such redesign or new experiment is proposed here; a future metric project would first need an independently justified loss family or task distribution and unused evaluation data.

## 4. Consequence for the paper and next work

The defensible empirical statement is:

> On the fixed SSPA development panel, lower conditional SWD did not ensure smaller expected-loss input-gradient error. A fixed RBF derivative discrepancy also failed to rank the tested losses reliably, including its own kernel-section losses, despite independent sampling and variance diagnostics.

This is a useful limitation/mechanism result. It is not evidence of a universal Sinkhorn disadvantage, a new covariance selector, or worse final BER. The covariance diagnostic's 66/72 agreement is conditional on this panel; the true channel's constant additive covariance makes it a particularly natural control. Existing checkpoints differ in their training procedures and budgets.

For the resubmission:

1. **Close this metric gate.** Retain the complete negative result, preferably as a bounded supplementary study. Do not run a larger kernel sweep, retune its bandwidths on these answers, or relabel seeds 7--9 as unused confirmation.
2. **Complete P0 numerical contracts, then P1 transport and stability controls.** The next main scientific question is whether controlled SSPA trajectories explain the existing long-budget degradation. Use the already specified matched particle exposure, common-epsilon reference, solver residuals and common checkpoint-selection protocol. Track conditional value errors, mean/covariance derivatives and fixed-task gradient errors separately. Temporal association alone will not prove causation.
3. **Prepare P2 baselines in parallel.** Include the learned Gaussian reference and competitive fast generators under the repaired exposure/compute rules. Keep the primary selection rule fixed; neither the kernel nor covariance result authorizes replacing it.
4. **Keep P3 focused on actual communications utility and mechanisms.** Distinguish distribution fidelity, expected-gradient bias, stochastic-gradient variance, normalized-step effects, and actual training outcomes. A new general-purpose selector requires a new hypothesis; it is no longer a prerequisite for finishing the channel-simulation paper.

The venue decision remains conditional on the original stability, fairness, scale and utility requirements. These negative metric results neither settle that decision nor close the reviewers' main objections.
