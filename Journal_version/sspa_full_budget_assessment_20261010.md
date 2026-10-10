# Independent assessment of the full-budget SSPA results

10 October 2026. Review of commit `0a9900b`, the transport-reference audit, 4,800/30,000/390,720-update stages, their protocols, implementation and transferred evidence. This supersedes earlier instructions to launch the now-completed trajectories. No new training or solver intervention was performed in this assessment.

**Decision:** the late failure is now established in controlled continuous trajectories. Common epsilon changes its timing and severity but does not stabilize these settings. Preserve the good selected checkpoints, the failed final checkpoints, and the complete negative result. The next experiment should isolate numerical transport distortion on frozen clouds, then change only the justified solver component in a matched continuation. Do not start another epsilon/seed sweep or reopen the closed kernel-selector study.

## 1. What was independently checked

- All four new evidence archives match their documented SHA-256 checksums. Restored 555 result files without replacing current sources: 14 transport-reference, 115 short-stage, 131 30k-stage and 295 full-stage files.
- Reran the full trajectory reporter. `checks.json`, `policy_summary.json` and the summary Markdown reproduce byte for byte. All 234 validation CSV rows agree exactly after matching by policy, seed and update; filesystem traversal changes row order only.
- Reran the transport-reference reporter. Its checks and README reproduce byte for byte, including the unresolved broad-cloud reference problem and saved-plan arithmetic checks.
- All 234 checkpoint file hashes match the derivative-analysis provenance. The 54 short-stage and 72 30k-stage checkpoint files are byte-identical to the corresponding full-suite files. Recorded numerical-source hashes match the current files.
- Independently recomputed policy summaries, selected updates, descriptive transition brackets and variance amplification. The 702 full/half derivative records have unique keys.
- An independent source audit found no material implementation flaw undermining the epsilon-policy comparison. It checks the matched draw order, isolated validation/calibration RNG, legacy arithmetic, resume contracts, reference formulas and reporting scope.

The [machine-readable independent audit](evidence/sspa_full_budget_independent_audit_20261010.json) retains the derived summaries and transition records. This is verification of saved evidence and source, not independent reproduction of training. Torch tests, model evaluations and internal optimizer/RNG tensors were not rerun or inspected here; the execution host's 75 passing tests and internal-state checks remain separately reported evidence. Checkpoint byte/hash comparisons are independently verified.

## 2. The substantive result

The three policies use the same architecture, initialization, particle exposure, optimizer, learning rate, clipping and paired random streams. Each run first reaches low conditional error, then deteriorates under continued training. This resolves the old ambiguity from comparing separate short and long endpoints.

| Policy | Selected validation SWD, mean | Final SWD, mean +/- seed SD | Final variance / true variance | First saved threshold crossings, seeds 9001/9002/9003 |
| --- | ---: | ---: | ---: | --- |
| Separate adaptive | 0.03889 | 1.94482 +/- 0.178 | 457.9 | 90k / 110k / 120k |
| Fixed common | 0.03845 | 1.22944 +/- 0.0854 | 176.8 | 110k / 150k / 150k |
| Shared adaptive | 0.03852 | 1.53435 +/- 0.189 | 326.6 | 100k / 120k / 110k |

The exact per-coordinate conditional variance is 0.0528298. These ratios use mean variance over seeds, inputs and coordinates; they do not mean every input or every output direction has the same inflation. The analytic-versus-analytic SWD floor is 0.0358438 and is not subtracted from scores.

The threshold is the report's **post-hoc** first saved SWD above twice that run's 30k SWD. It is neither an exact onset nor an online stopping rule. Fixed common crosses 20k/40k/30k later than legacy in the three paired seeds; shared adaptive does not uniformly delay legacy. Three development seeds do not establish a general policy ranking or calibrated confidence interval.

The failure is predominantly severe excess variance, accompanied by mean error and numerical distortion. It is not the symmetric point-collapse stationary example from the earlier theory note, and finite observations do not prove mathematical divergence as training time tends to infinity. Good retrospectively selected validation scores also do not establish held-out test quality, final BER, or successful online early stopping.

The [late-trajectory figure](../results/sspa_epsilon_full_report_20261010/late_trajectories.png) shows why 30k was insufficient: all policies stay near the sampling floor for a substantial interval, then deteriorate sharply. A fixed 30k operating budget may be useful, but a general stopping prescription still needs independent validation and honest selection cost.

## 3. What the timing and derivative records do not identify

For every run, the last saved checkpoint before the descriptive crossing has near-correct variance, zero floor activation on the fixed diagnostic batch, and maximum converged-reference barycenter RMS error below `9e-8` on its eight checked inputs. The next saved checkpoint already has excess variance and nonzero floor activation. This brackets a joint transition; it does not establish which change happened first inside the interval or on other training inputs.

For example, fixed-common seed 9001 goes from variance 0.05289 and SWD 0.03934 at 100k to variance 0.27804 and SWD 0.08828 at 110k. Legacy seed 9001 goes from variance 0.05039 at 80k to 3.30993 at 90k. The more frequent training traces suggest growing self-coupling residuals before a large recorded regression-loss rise, but distribution fidelity was not evaluated at each of those updates. Numerical distortion could initiate, amplify, or follow distributional deterioration.

The 145 failed reference solves are unresolved numerical comparisons, not zero error and not evidence that positive-epsilon finite-dimensional OT is ill-defined. Even a successful marginal-tolerance check is an operational accuracy criterion, not a rigorous objective/gradient-error certificate. The scheduled reference panels cover eight inputs per term, not every training cloud.

The fresh-anchor derivative panel strengthens the description but not causal identification. Across policies, mean-Jacobian relative errors increase from about 0.072 to 0.097--0.180, while empirical covariance-derivative norms increase from about 0.09 to 8.91--49.08. These are different scales and estimands. Covariance-derivative norms also contain candidate-dependent Monte Carlo error; the earlier exact-law rotation control already shows why the analytic null alone cannot calibrate it. Full/half-panel norms expose variability but do not provide population derivative confidence bounds. Their growth is not proof of input-dependent covariance error of that exact magnitude, a downstream-gradient failure, or its cause.

Parameter norms increase too, but are parameterization-dependent and cannot by themselves identify a dynamical mechanism. The direct variance and SWD deterioration remain clear even with these derivative-estimation qualifications.

## 4. An exact distinction: zero expected drift is not finite-step stability

This elementary argument helps separate plausible mechanisms. It is not a claim that finite-cloud noise explains the observed runs.

Fix the current generator parameters and the complete input minibatch. Assume its conditional law equals the true channel law at every input in that batch. Condition on all generated source particles (and on their latent variables/Jacobians when considering parameter gradients). Let the positive-cloud collection \(P\) and independently generated reference-cloud collection \(R\) be conditionally independent with the same law and the same particle counts.

Let the raw source-particle drift be the cross barycentric projection minus the self/reference barycentric projection, with **unit cross/self coefficients**, computed by the same deterministic projection routine with the policy's epsilon choices. Let \(V(U,P,R)\) denote that field or its odd radially clipped version. Assume integrability. Swapping the entire collections \(P,R\) negates the drift for each of the three implemented policies:

- fixed-common epsilon is unchanged;
- the pooled shared-adaptive scale is unchanged;
- the two separate adaptive scales exchange roles.

The argument includes a scale pooled across minibatch anchors; conditioning and swapping must cover that entire batch. Identical solver settings and equal positive/reference counts are essential. Radial clipping is an odd map, so it preserves the sign reversal. Consequently,

\[
E[V\mid U,\text{inputs, parameters}]=0.
\tag{1}
\]

**Proof.** Conditional exchangeability gives \(EV(U,P,R)=EV(U,R,P)=-EV(U,P,R)\). The same argument applies after odd clipping. Multiplication by fixed source Jacobians gives zero expected unclipped detached-regression parameter gradient, with the corresponding integrability assumption. No Sinkhorn convergence or objective-gradient interpretation is needed. This is an exact-real-arithmetic statement; it is not a bitwise floating-point assertion. \(\square\)

Thus common epsilon is not necessary merely for matched-law stationarity **in expectation**, and mismatched cross/self scales away from matching do not automatically imply a nonzero mean drift exactly at matching. None of this proves local stability, convergence, or stable Adam dynamics.

To see the finite-step distinction, take positive integers \(K,d\), one source cloud \(U=(u_i)_{i=1}^K\subset\mathbb R^d\), a velocity cloud \(V=(v_i)\), and

\[
s^2(U)=\frac1{Kd}\sum_i\|u_i-\bar u\|^2.
\]

Expanding the square gives the **exact** identity for any real step \(\eta\):

\[
s^2(U+\eta V)-s^2(U)
=\frac{2\eta}{Kd}\sum_i(u_i-\bar u)\cdot(v_i-\bar v)
+\frac{\eta^2}{Kd}\sum_i\|v_i-\bar v\|^2.
\tag{2}
\]

At exact matching under (1), finite second moments make the conditional expectation of the first term zero. The second is nonnegative and, for \(\eta\ne0\), is strictly positive if the centered velocity is nonzero with positive conditional probability. For nonzero \(\eta\), equality holds exactly when the centered velocity is zero almost surely, conditional on the source batch; this includes a common random translation and \(K=1\). Therefore a literal stochastic Euler particle step can increase expected empirical variance even when the mean field is zero and transport is accurately solved.

The neural algorithm fits detached targets using shared parameters, gradient clipping and Adam state; it does **not** literally replace particles by \(U+\eta V\). Equation (2) is therefore a diagnostic and a warning about exchanging “zero expected field” with “stable finite update,” not a proof of neural variance growth. The restricted same-cloud, common-epsilon, unclipped gradient identity from the transport audit likewise does not automatically apply to this actual training update.

## 5. Concrete next experiment: numerical decomposition before retraining

This extends the [full-budget report's proposed intervention](sspa_full_budget_results_20261010.md#next-bounded-experiment). It is a specification, not an already implemented or executed study.

### A. Frozen-cloud replay

Start with seed 9001 and each policy at 30k, its last saved healthy checkpoint, its first degraded checkpoint, and 390,720. Deduplicate repeated checkpoints. Repeat the informative contrasts on seeds 9002/9003 before claiming a general mechanism. Keep the fixed-common branch as the cleanest intervention target.

For each checkpoint, save the complete inputs and source/positive/reference clouds, including the original four-particle regime and the actual policy epsilon. If reconstructing the next batch from a saved RNG state, label it as a **post-checkpoint reconstructed batch**, not the batch that caused an unsaved historical update. Use one shared saved cloud set across all solvers. Freeze the epsilon computed from the complete batch before examining subsets; recomputing a pooled median on a subset changes the question.

Compare these operations separately:

| Comparison | Intended interpretation |
| --- | --- |
| Production at 10, 30, 100 iterations | Sensitivity to practical iteration count, with the same floor/stabilizers |
| Log-domain updates on the same quantized floored kernel, at matched iteration counts | Effect of changing numerical scaling/denominator implementation; keep dtype and initialization explicit |
| Converged log-domain solve of that floored kernel | Residual practical-solve error, including iteration/stabilizer effects |
| Converged log-domain solve of the original unfloored quadratic cost, at the same epsilon | Distortion from the modified kernel/cost, including separately declared float32 quantization |

Production versus a converged floored-kernel reference alone does not isolate the denominator stabilizer. To make that attribution, add a same-kernel, same-iteration, same-dtype comparison, and isolate dtype in another comparison. Solve both cross and self terms; do not infer net-field error from one term alone. More iterations cannot recover discarded kernel information.

For every pair, retain both marginal residuals, convergence/cap status, plan and barycenter errors, raw/clipped field discrepancies, and parameter-gradient discrepancies. Report distributions and affected fractions as well as maxima. Compare only against converged references; retain unresolved cases. Warm starts or epsilon continuation may accelerate a reference, but the final solve must meet the prescribed tolerance at the **original final epsilon**.

Add the signed first-order variance term and nonnegative finite-step term in (2). This tests whether a solver changes expansion/contraction, rather than merely producing a large field-error norm. Also measure the variance change after one cloned neural optimizer update on a fixed independent evaluation panel. That update must copy model, optimizer and RNG state; it is a separate quantity from the particle identity.

**Question to answer:** does a numerical correction materially alter the expansion or parameter update on late healthy clouds, or only after the model is already degraded? A large correction on failed endpoints alone is insufficient evidence for prevention.

### B. Matched continuation only after replay

Fork the fixed-common update-80k checkpoint, initially seed 9001, with complete optimizer/RNG state. Keep the source and reference cloud counts, data, epsilon, learning rate, clipping, model and validation protocol fixed.

The current loader rejects changed numerical source. Implement this as an explicitly recorded intervention: verify the parent against its original contract, then record the changed solver backend and new source hashes in a child manifest. Do not weaken ordinary resume checks or overwrite parent states to make a changed solver appear to be an exact continuation.

- Retain an unchanged production control, which should reproduce the retained path before interpreting an intervention.
- Add the numerically corrected floored-kernel solver if finite-iteration/stabilizer error is implicated.
- Add the unfloored solver if floor distortion is implicated. If replay supports only one correction, start with just that correction and control.

Profile a bounded batch first; log-domain convergence on difficult clouds may be expensive. A branch hitting its solver cap is unresolved, not an accurate-solver success. Do not simultaneously change learning rate, particles and solver.

Run through 200k with denser predeclared validation around the known control transition. Survival to 200k establishes delay through that horizon, not full-budget stability. A claimed stabilization must subsequently survive 390,720 updates, the remaining paired development seeds and independent validation. These are mechanism-development runs, not yet new-seed confirmation of a general method advantage.

If accurate intended transport still deteriorates, retain that result and test optimizer/drift-scale or finite-cloud effects separately. Equation (1) does not eliminate those alternatives. Do not turn this into an unrestricted hyperparameter search.

## 6. Paper implications

The controlled trajectory is now a substantive result: early good fidelity does not ensure stable continued training, and common epsilon alone does not resolve the failure. It should replace speculative explanations of the old endpoint gap. The paper must not claim practical convergence or that the optimal selected budget establishes asymptotic stability.

A convincing causal intervention or a validated operating/stopping rule could make this a useful methodological contribution. The current evidence supplies neither yet. Keep modern fast baselines, matched exposure/compute, independent downstream evaluation and the larger structured channel moving in parallel. No new venue recommendation follows from this instability result alone, and the closed RBF-selector branch remains closed.
