# Assessment after the exact-law and matched-value controls

9 October 2026. Independent review of commit `dc13495`, the [completed controls](metric_controls_results_20261009.md), their implementation, and the transferred raw evidence. This updates the [earlier assessment](metric_experiment_assessment_20261009.md); its proposed controls A and B are complete.

**Decision:** retain derivative fidelity as a research direction, but do not present the current empirical norm as a validated channel-selection metric. The controls establish a population mechanism and expose a substantial estimation confound. They do not establish kernel-specific usefulness on learned models. Resolve that narrower question with one bounded, decoder-free development check; keep the journal's stability, fairness, and modern-baseline work moving independently.

## 1. Verification and what the mixed results mean

The committed archive's SHA-256 matches `8eea91e2a70329e74f5930df174ca070270eef23cd05ce704eaceed54bef4c9a`. Its 102 result files were restored in this checkout without overwriting current sources or research notes. Rerunning the report from the raw records reproduces **all three summary CSVs byte for byte** and `checks.json` exactly. Coverage is 192 rotation records, 480 matched-value records, and 3,840 loss records. The paired value-score spread is zero, both reported population-bound checks have zero violations, and the two logistic quadrature calculations differ by at most `4.97e-16`.

This independently verifies saved records and aggregation, not a fresh simulation or training run. The execution host reports 37 passing tests; this checkout lacks Torch, so those tests were not rerun here. Training checkpoints are excluded from the evidence archive. The earlier warning that these October numerical records were absent is now superseded for this checkout.

| Question | What is established | What remains open |
| --- | --- | --- |
| Can equal conditional laws have different empirical derivative norms? | Yes. At N=512, the rotation null changes the ordinary RBF norm from 0.11401 to 0.92764 while the population discrepancy stays exactly zero. | Reliable uncertainty for each learned candidate; a useful operator-norm confidence bound. |
| Do pointwise distribution values determine local loss gradients? | No. At the same anchors, all value scores remain unchanged across slope interventions, while exact expected gradients change. | Whether this information helps screen realistic learned surrogates. |
| Does the kernel improve on simple derivative diagnostics? | Not demonstrated. Mean-Jacobian error already detects the Gaussian intervention. | Added information beyond value scores, mean/covariance derivatives, and simple higher moments. |
| Does smaller discrepancy imply better progress for every task? | No. Equal discrepancy norms occur with helpful and harmful signed perturbations. | Task-class fidelity bounds can still be useful; endpoint prediction requires additional assumptions. |

For example, the matched-value quadratic probe has the same derivative discrepancy, `2.46577`, at slopes -4 and +4. Their first-order common-SGD progress ratios are -1.56 and +3.56. This is an exact population calculation, not sampling noise. A discrepancy norm measures distance from the correct gradient; it does not encode the error's sign relative to every task gradient.

The report's interpretation is therefore appropriately cautious. The new results neither refute the population bound nor validate a replacement for checkpoint selection.

## 2. Separate population fidelity from simulation noise

Let \(D\) be the difference of expected feature-derivative operators at a fixed input. Suppose two independent, unbiased estimates \(\widehat D_1,\widehat D_2\) have finite Hilbert--Schmidt second moments. Define

\[
S=\tfrac12(\widehat D_1^*\widehat D_1+\widehat D_2^*\widehat D_2),\qquad
C=\tfrac12(\widehat D_1^*\widehat D_2+\widehat D_2^*\widehat D_1).
\]

Expanding products gives the exact identity

\[
S-C=\tfrac12(\widehat D_1-\widehat D_2)^*(\widehat D_1-\widehat D_2)\succeq0,
\qquad E C=D^*D.
\tag{1}
\]

Thus `trace(C)` targets squared Hilbert--Schmidt population discrepancy; `trace(S-C)` measures an estimator-variance contribution. For identically distributed splits the latter's expectation is \(E\|\widehat D_1-D\|_{\rm HS}^2\). Different split variances give their average. These statements require independence between splits, not merely different labels. Correlation between P and Q within a split changes its variance and must be declared.

This provides a useful **pair of diagnostics**, not one automatically corrected norm. Individual cross traces/eigenvalues can be negative; clipping, taking square roots, or taking the largest eigenvalue does not preserve unbiasedness. Hilbert--Schmidt and operator norms are also different targets. Neither the identity nor eight or sixteen repetitions supplies a calibrated confidence theorem.

An exact-law generator with greater pathwise gradient variance may train a downstream model less efficiently. That is a property of its stochastic gradient estimator and optimizer, distinct from population channel-law mismatch. The rotation control demonstrates the variance mechanism; it did not test downstream training efficiency.

## 3. An exact result beyond mean and covariance

The completed Gaussian control cannot distinguish kernel derivatives from mean derivatives. The following construction closes that particular logical gap **analytically**. It is not another simulation result, a physical-channel proposal, or a claim of publication novelty.

### Claim and assumptions

Let \(x\in\mathbb R\), \(\sigma>0\), \(\rho=1/60\), and \(\omega\ne0\). Set

\[
z=(-2,-1,0,1,2),\qquad a=(1,-4,6,-4,1),\qquad
q_i(x)=\tfrac15+\rho\sin(\omega x)a_i.
\]

Consider the smooth scalar conditional laws

\[
P_x=\sum_{i=1}^5\tfrac15\,\mathcal N(x+z_i,\sigma^2),\qquad
Q_x=\sum_{i=1}^5q_i(x)\,\mathcal N(x+z_i,\sigma^2).
\tag{2}
\]

The following statements hold:

1. These are probability laws for every real input. Their first three raw moments, and all input derivatives of those moments, agree everywhere.
2. \(P_0=Q_0\): every population pointwise distribution discrepancy vanishes at zero.
3. For every finite Gaussian RBF bandwidth \(\ell>0\), the derivative of their kernel mean-embedding difference at zero is nonzero.
4. A fixed bounded smooth loss has a nonzero expected-gradient discrepancy there.

### Proof

The structural idea is a fourth finite difference: the signed coefficients annihilate low-degree polynomials. Directly,

\[
\sum_i a_i z_i^j=0\quad(0\le j\le3),\qquad \sum_i a_i z_i^4=24.
\tag{3}
\]

The weights sum to one and lie in \([0.1,0.3]\). For each \(j\le3\), \(E(x+z_i+\sigma G)^j\), with \(G\sim\mathcal N(0,1)\), is a polynomial in \(z_i\) of degree at most \(j\). Equation (3) therefore annihilates the difference of the moments for every \(x\). The common first three raw moments are

\[
x,\qquad x^2+2+\sigma^2,\qquad x^3+3x(2+\sigma^2).
\]

In particular, the common variance is \(2+\sigma^2\). These identical smooth functions have identical derivatives. At zero, \(\sin(\omega x)=0\), so the entire mixture laws agree.

For \(k_\ell(y,y')=\exp[-(y-y')^2/(2\ell^2)]\), let \(\Phi\) be its canonical RKHS feature map and \(\mu_i(x)=E\Phi(x+z_i+\sigma G)\). These Bochner means and their derivatives exist: Gaussian RBF features have norm one and a uniformly bounded first feature derivative. Finite sums and dominated differentiation are therefore valid. Since

\[
\mu_Q(x)-\mu_P(x)=\rho\sin(\omega x)\sum_i a_i\mu_i(x),
\]

we obtain

\[
\left.\partial_x(\mu_Q-\mu_P)\right|_0
=\rho\omega\sum_i a_i\mu_i(0).
\tag{4}
\]

The sign is opposite to the earlier note's convention \(\Delta=\mu_P-\mu_Q\); the norm is unchanged. In this scalar-input example operator and Hilbert--Schmidt norms coincide. Independent Gaussian integration gives

\[
d_1(0)^2=(\rho\omega)^2a^\top K_\sigma a,\qquad
(K_\sigma)_{ij}=\frac{\ell}{\sqrt{\ell^2+2\sigma^2}}
\exp\!\left[-\frac{(z_i-z_j)^2}{2(\ell^2+2\sigma^2)}\right].
\tag{5}
\]

For completeness, strict positivity follows from the Gaussian Fourier representation:

\[
a^\top K_\sigma a
=\frac{\ell}{\sqrt{2\pi}}\int_{\mathbb R}
e^{-(\ell^2/2+\sigma^2)t^2}
\left|\sum_i a_i e^{it z_i}\right|^2dt>0,
\]

because \(\sum_i a_i e^{it z_i}=16\sin^4(t/2)\), which is nonzero away from the discrete set \(2\pi\mathbb Z\). This proves detection at each fixed finite bandwidth, not a bandwidth-uniform lower bound.

Finally, take \(L(y)=1-\cos y\), which lies in \([0,2]\) and has bounded derivatives. The Gaussian characteristic function yields

\[
E_{Q_x}L-E_{P_x}L
=-16\rho e^{-\sigma^2/2}\sin^4(1/2)\sin(\omega x)\cos x,
\]

and hence

\[
\left.\partial_x(E_{Q_x}L-E_{P_x}L)\right|_0
=-16\rho\omega e^{-\sigma^2/2}\sin^4(1/2)\ne0.
\tag{6}
\]

This is an absolute gradient-error witness. The true gradient at zero is zero by symmetry, so relative error and gradient cosine are undefined there. No Gaussian-RKHS norm is asserted for this periodic loss. A separate norm-one RKHS probe is \(L(y)=k_\ell(y,0)\): its derivative difference is \(\rho\omega\sum_i a_i E k_\ell(z_i+\sigma G,0)\), nonzero because the sum equals a positive prefactor times \(2t^4-8t+6\), where \(t=\exp[-1/(2(\ell^2+\sigma^2))]\in(0,1)\). This polynomial decreases from six to zero on that interval. \(\square\)

### Uniformly small distribution errors still do not suffice

For integer \(n\ge1\), replace \(\rho\) by \(\rho_n=1/(60n)\) and \(\omega\) by \(n\). Then, uniformly over all real inputs,

\[
\operatorname{TV}(P_x,Q_{n,x})\le8\rho_n,\qquad
W_p(P_x,Q_{n,x})\le4(8\rho_n)^{1/p}\longrightarrow0
\quad\text{for every fixed }1\le p<\infty.
\tag{7}
\]

To prove this, the total variation distance of the mixing weights is \(\frac12\rho_n|\sin(nx)|\sum_i|a_i|\le8\rho_n\). Match their common mass exactly and couple the remaining mass between centers at distance at most four. Adding the same Gaussian noise to both coupled centers gives the stated Wasserstein bound. Mixture smoothing cannot increase total variation. All moments needed for these distances are finite.

Yet (4) and (6) remain fixed and nonzero, because \(\rho_n n=1/60\). The first three moments still match for every input and every \(n\). Thus this obstruction persists even under uniform distribution convergence and exact low-order moment matching. In one output dimension, the usual sliced Wasserstein distance reduces to its unsliced counterpart. This does not contradict derivative bounds that assume uniform curvature: such uniform regularity is absent in this sequence.

Uniformly close bounded-loss values still support the usual risk-transfer and approximate-global-optimum conclusions. This example concerns local gradients; it does not refute those conclusions or prove a persistent final-performance gap for every optimizer. In particular, translating a first-order gradient discrepancy into a finite update requires controlling the remainder on the chosen step scale.

### Adversarial checks and limits

- A fourth-moment derivative detects this example too: \(E_QY^4-E_PY^4=24\rho\sin(\omega x)\). The result establishes information beyond the first three moments, **not kernel necessity or practical superiority**.
- At inputs other than zeros of \(\sin(\omega x)\), the conditional laws differ. The pointwise equality at zero must not be promoted to equality on a neighborhood. Equation (7) is convergence, not exact equality.
- This is a scalar conditional-law construction, not a demonstration under the manuscript's complete codebook, power, or physical-channel constraints.
- Differentiating a categorical draw as if its component index were constant misses the changing mixture weights. The correct derivative of a smooth probe expectation includes both \(\sum_iq_i' E L(x+z_i+\sigma G)\) and \(\sum_iq_i E L'(x+z_i+\sigma G)\). Use analytic integration, a valid score-function term, or a justified distribution-level finite difference. The current pathwise estimator is **not** a drop-in estimator for this mixture.
- For \(\ell=1,\sigma=0.25\), an arithmetic check gives \(a^\top K_\sigma a=6.9401341805\), \(d_1(0)/|\rho\omega|=2.6344134414\), and the cosine-gradient gap divided by \(\rho\omega\) is \(-0.8192811060\). These checks support the formulas; the proof is above.

## 4. The next Codex handoff: one bounded learned-model gate

Do not spend a cluster campaign reproducing the analytic example above. It already answers whether higher-order distribution derivatives can matter in principle. The unresolved question is their **measurable added usefulness in this repository**.

1. **Use existing checkpoints and no downstream optimization.** Begin with seed-7 SSPA, selected Sinkhorn, WGAN, and DDIM-10, plus the analytic channel. These checkpoints use eight real coordinates: do not pass in control B's two-dimensional anchors or probes unchanged. Take the three radius-\(\sqrt8\) inputs at zero-based indices 3, 4, 5 from `manifest.channels.SSPA.anchors` in `results/channel_metric_pathwise_n512_seed7_20261009/results.json`. Preserve these stored coordinates and serialize them in the new manifest. These are development cases. Verify checkpoint/config hashes and physical noise/input units; do not interpret the archive as containing weights. Abort with an explicit missing-input manifest if the execution host lacks the required checkpoints.
2. **Freeze dimensionally valid probes and compute every score at the same inputs.** Form unit vectors \(v_1,v_2\) by ordered Gram--Schmidt on the first two selected anchors (fail on degeneracy). Lift control B's two probe directions to \(0.8v_1+0.6v_2\) and \(-0.6v_1+0.8v_2\), keep offsets 0.2 and -0.3, and lift its kernel centers to \(0.3v_1-0.4v_2\) and \(-0.7v_1+0.6v_2\). Use the same four loss types to obtain eight fixed probes; kernel sections use the full eight-dimensional kernel. Keep the SSPA pilot's stored physical bandwidths and moment normalization, not the toy control's hard-coded noise or identity mean Jacobian. Evaluate the actual SSPA mean map and its derivative. Coordinate derivatives are ambient derivatives without re-normalizing perturbed inputs. Record SWD/MMD, mean/covariance discrepancies and derivatives, the moment derivative score, and the RBF derivative score. Include fourth raw moments along the two fixed probe directions and their derivatives, with sampling spread and moment assumptions. Do not subtract a moment norm from a kernel norm and call the result a shape metric.
3. **Separate signal from uncertainty.** Start at N=512 per independent split with eight independent repetitions. Save both self and signed cross Gram matrices and (1)'s variance term for every candidate/input. Repeat at N=2048 only for unresolved or apparently informative predeclared contrasts; record that adaptive resolution rule. Reuse reference calculations honestly and record total reference/candidate work. Use equal access or show cost differences. Increasing N makes full-kernel work quadratic; do not automatically escalate beyond 2048.
4. **Use absolute expected-gradient error as the target.** For each fixed loss, evaluate the analytic-reference and surrogate expected input gradients. Keep task-gradient evaluation streams independent of score-estimation streams; otherwise shared Monte Carlo noise can manufacture apparent prediction. Include task-gradient uncertainty. At most the initial learned panel has 72 candidate/input/repetition metric records, or 144 if all cells receive both N values, plus analytic controls and eight probe results per record. These rows remain nested within three existing trained models.
5. **Make the decision from contrasts, not pooled correlation.** Report which model/input/probe discrepancies are resolved by each comparator, and where kernel discrepancies are visible despite unresolved or small moment discrepancies. Do not claim a calibrated upper bound from noisy cross traces or turn all anchor/probe rows into independent model samples. Report failures and costs alongside successes. The Gaussian/quadratic cases calibrate the moment baseline; nonquadratic probes test its limitations. Any chosen quantitative pass threshold must be fixed before inspecting these new rows and justified by the intended screening decision.
6. **Stop or confirm.** If the kernel adds no resolved information beyond cheaper checks at this scale, keep derivative fidelity as an explanatory diagnostic and remove the better-selector claim from the paper's proposed central contribution. If it adds information, freeze the entire rule and test it on genuinely unused model seeds and probe losses before a selection claim. Previously inspected seeds 8/9 are development evidence, not fresh confirmation. No initial full encoder/decoder training is needed for this gate; predicting final training outcomes would still require a later independent downstream experiment.

This is an extension specification, not a claim that the combined runner already exists. Extend the existing feature/control runners and their reporting contracts rather than naming an unimplemented command as executable. Keep model identity, task identity, sampling independence, timing and query counts in the manifest. Preserve the complete negative panel if the gate fails.

## 5. Consequence for the journal

The supported narrative is now sharper: pointwise conditional distribution agreement does not ensure faithful transmitter gradients; low-order moments can also miss the obstruction; estimating derivative fidelity requires candidate-specific variance control. The first and third claims have the synthetic numerical controls; the second has the explicit mathematical construction above. None establishes a Sinkhorn advantage.

The original resubmission still needs matched data/compute comparisons, a resolved stability story, competitive fast baselines, and communications utility at the intended scale. These metric controls do not close those reviewer concerns. Keep the venue decision conditional on that evidence. A universal pre-optimization predictor of final BER is not a supported manuscript claim, and no novelty or acceptance claim is made for this note.
