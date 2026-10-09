# Journal resubmission handover

Prepared 8 October 2026 against commit `726db308c46fe1df67319934ed35ef8dd8d79945`.

**Start here.** This is the execution plan for revising *Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation*, rejected as TCOM-TPS-26-1722. It supersedes the recommendations in `conditional_drifting_journal_strategy.md` and `journal_execution_roadmap.md`; keep those files as historical notes. It builds on, and corrects/extends, `review_audit_20261008/README.md`.

The author has no fixed deadline or compute cap and is willing to run simulations on roughly the previous scale. The author asks us to recommend the venue. This task produced a plan; it did not launch cluster jobs, submit a paper, send correspondence, or generate new experimental results.

- [Cluster implementation and experiment protocol](resubmission_cluster_plan.md): code tasks, controls, seeds, selection, job matrix, outputs, and existing commands.
- [Manuscript revision and mathematical audit](resubmission_manuscript_plan.md): replacement claims, proofs/counterexamples, section changes, and figure plan.
- [SWD, downstream risk, and gradient fidelity](theory_swd_downstream_gradient_fidelity.md): follow-up theory, a fixed-power counterexample, sufficient conditions, and controlled P3 interventions.
- [Metric before downstream optimization](theory_preoptimization_channel_metric.md): the author's stronger target; a candidate based on conditional feature values and input derivatives, task-class guarantees, estimation bounds, and validation requirements.
- [Actual decision and all 22 reviewer comments](review_audit_20261008/README.md#full-review-reports).

## 1. Recommendation

The paper has a potentially publishable communications application, but the next submission needs a rebuilt argument and controlled evidence. More seeds of the current comparison will not answer the central objections.

**Provisional venue: IEEE TMLCN. Preserve TCOM as an option until the controlled pilots and larger-channel result are available.** TMLCN fits an ML method adapted, analyzed, and evaluated for communications. A TCOM retry is reasonable if the revised work establishes a substantive communications-specific finding beyond applying an existing conditional generator, and answers the editor's fairness, theory, and scale concerns together. This is a fit judgment, not an acceptance prediction. There is no evidence for the acceptance percentages in the old notes.

The main new reason for caution is attribution: **conditional drifting and conditional Sinkhorn/W-Flow already exist in the cited originating papers.** W-Flow explicitly writes the same conditional barycentric velocity, with class label `c` in place of channel input `x`. Continuous inputs and simulator access create useful research questions, but replacing a discrete label with a continuous vector is not, by itself, a new transport principle. [Drifting, Sections 3.5 and 4](https://arxiv.org/html/2602.04770v1#S3.SS5), [W-Flow, equation (36)](https://arxiv.org/html/2605.11755v1#A2.SS3).

Build the revision around this question:

> When does conditional Sinkhorn training provide accurate and useful one-evaluation channel simulators for communication-system optimization, compared with other fast generators under controlled sampling and compute budgets?

The strongest possible evidence would connect **conditional fidelity, fidelity of optimization gradients, and actual downstream performance per training time**, including a larger jointly generated channel block. Whether that connection holds is to be tested. Do not promise a Pareto advantage or a mechanistic explanation of instability in advance.

The 9 October [theory follow-up](theory_swd_downstream_gradient_fidelity.md) makes this direction concrete: exact conditional laws at the current unit-power codebook and arbitrarily small uniform conditional Wasserstein error can coexist with a reversed transmitter gradient. It also proves positive loss-transfer, curvature-to-gradient, and biased-descent bounds under explicit assumptions. These establish a possible mechanism, not its cause in the existing runs or a Sinkhorn-specific guarantee. Prioritize its small T1/T2 diagnostic before expanding P3; a successful mechanism study could strengthen the TCOM case, but the present venue recommendation remains conditional on evidence.

**Refined author objective:** develop a useful channel-surrogate metric that can be computed before candidate-specific downstream encoder/decoder optimization. The [metric proposal](theory_preoptimization_channel_metric.md) separates that objective from diagnostics through a trained decoder. Its candidate has explicit function-class guarantees but is not a validated predictor of final BER. Keep the small downstream runs to validate the metric; do not make their trained decoders an undeclared input to the claimed pre-optimization score, and do not switch the main selection protocol before that validation.

Use **channel simulation**, learning `p(y | x)`, throughout. Receiver-side channel estimation from pilots is a different problem and is not what the current experiments evaluate.

## 2. What was actually checked

| Evidence | Status in this assessment |
| --- | --- |
| Journal source, included result tables, source implementation, HPC wrappers | Inspected locally. The submitted PDF is 12 pages; result pages 9–12 were also rendered and visually inspected. |
| Local submission ZIP | Main TeX, timing, coding, and SSPA-budget snippets freshly verified byte-identical to the working counterparts. ZIP SHA-256: `b696487568f521faf953acbeaf684866bfbba7688222a29a608bc35f3258a4aa`. This is a local bundle, not a portal download. |
| Actual reviews | Editor rejection and 7 + 5 + 10 numbered comments preserved in the prior audit. Their source is the author's pasted decision, as documented there. |
| `tcom_review_comments.md` | Earlier pre-submission feedback, **not the rejection reports**. Its major-revision recommendation and probabilities are not decision evidence. |
| Conference | Inspected `Paper_camera_ready_checked/drifting_vs_diffusion_summary.tex`; repository README also identifies the older `(3).tex` source. Verify the exact accepted camera-ready bundle and bibliographic status before final disclosure. |
| Earlier saved numerical audit | Read `review_audit_20261008/evidence.json`; arithmetic can be checked from those stored values. Raw June records needed for independent regeneration are absent locally. |
| Audit rerun | `python3 scripts/audit_journal_review_evidence.py` failed at missing `results/timing_suite_local_cuda_20260601_1755/timing_suite_summary.json`. It did not reproduce the full audit. Restore missing archives before treating its results as freshly verified. |
| New training / cluster measurements | None performed. All new run counts and thresholds below are proposed protocol choices. |

The June timing, corrected-fiber, compact SSPA, SER-curve, baseline, and TurboAE directories must be restored from the cluster or original storage. The local `results_hpc/` contains March suites, which are not substitutes. A missing archive does not invalidate the existing result; it limits what can be independently checked from this checkout.

## 3. Findings that change the previous plan

1. **Novelty needs correction, not just stronger wording.** The prior plan says existing drifting/W-Flow are unconditional. That is false. Attribute conditional batching, Sinkhorn velocities, and detached regression. Explain the communications-specific contribution with evidence.
2. **The unique-equilibrium claim is too strong.** Unique zero of the Sinkhorn divergence and an energy-dissipation identity do not establish absence of other stationary measures. A collapsed symmetric counterexample is given in the manuscript plan. Keep a precise zero-set statement; remove unsupported global convergence and “best representable” language.
3. **Detachment is not, by itself, the obstacle to a gradient interpretation.** An exact fixed-epsilon velocity gives an instantaneous detached-regression parameter gradient proportional to the objective gradient under an appropriate chain rule. The practical gaps include finite samples, independent self-reference, adaptive unequal epsilons, approximate couplings, clipping, and finite optimizer steps.
4. **The supposedly controlled Sinkhorn comparison changes data exposure.** Current fiber training uses `B` anchors and `4B` samples of each particle type, whereas joint training uses `B` samples of each type. Equal architecture and epochs do not isolate geometry. Use identical repeated-anchor particles in the decisive comparison.
5. **SSPA long-run degradation is not explained by the saved endpoints.** The prior audit records mean SWD about `1.5344` over 100 full runs versus `0.0070714` over 30 compact runs. Actual update counts are `390720` and `4800`. This supports a serious stability question; it does not prove a particular cause or validate early stopping.
6. **Downstream evaluation has additional controls to repair.** Symbolic codewords currently depend on minibatch normalization; analytic-channel SER is used for epoch selection in `train_symbolic_autoencoder`; rates are stored without exact trial counts. Separate validation/test streams and define the power convention. TurboAE also has a decoder-noise-distribution mismatch in the earlier audit.
7. **A larger codeword is not a larger channel generator.** Existing TurboAE uses an `n=2` surrogate across independent pairs. The new structured case must generate the whole coupled block. Do not simply increase `--override-n`: its default latent dimension remains 16 and it retains the compact TDL construction.
8. **Existing scripts are building blocks, not a ready revision suite.** In particular, `hpc/check_env.sh` calls a full-budget training runner despite labeling it a small smoke test. Replace that check before using it. New scripts in the cluster plan are explicitly proposed and do not yet exist.

## 4. Work packages and order

| Package | Work and purpose | Completion evidence | Dependency |
| --- | --- | --- | --- |
| P0: provenance and numerical contracts | Restore archives, freeze source/config hashes, unify units/noise/power, integer counts and split labels, bounded smoke test | Machine-readable input manifest and audit report with missing items explicit | First |
| P1: transport correctness and stability | High-accuracy common-epsilon reference, numerical residuals, controlled SSPA trajectories, matched-particle geometry comparison | Solver checks, actual trajectory plots, selected and last checkpoints | P0 |
| P2: competitive comparison | Matched-capacity/data-access generator study; measured compute curves; low-step DDIM, flow matching, true one-step comparator, simple Gaussian reference | Conditional metrics and uncertainty for all methods; full generator costs | P0, P1 for frozen proposed method |
| P3: communication utility | Learned-codeword diagnostics, expected-loss gradient checks, equal-update and equal-time AE training, TurboAE protocol repair | Actual SER/BER/BLER versus time, gradient diagnostics, separately seeded test | P2 checkpoints |
| P4: larger structured channel | Joint 64-complex-use generation with validated multipath/boundaries/noise and a communication endpoint | Conditional temporal/cross-coordinate accuracy and receiver performance | Pilot success; P0 channel contracts |
| P5: manuscript and response | Correct theory/attribution now; integrate validated figures and claims later | Revised paper, extension statement, response matrix, source-to-number manifest | Starts immediately; ends after P1–P4 |
| P6: sparse observations, optional | Fixed one-output-per-input dataset and local conditional estimator | Held-out evaluation with neighborhood bias/coverage diagnostics | Only if retaining sparse/passive-measurement claims |

Parallelize P0 restoration with P5 theory/related-work editing and P2 baseline implementation. Run P1 before launching confirmatory Sinkhorn training. Develop P4 simulator checks in parallel, then train its generators after the pilot gate. Do not postpone correctness repairs until after a large run.

**Default main study:** AWGN for calibration, SSPA for nonlinear/stability behavior, and one larger structured block for relevance. Retain Rayleigh and compact TDL as supporting generalization cases using the same frozen protocol when feasible. Avoid adding optical fiber, MIMO, Doppler, and passive measurements simultaneously.

## 5. Pilot and submission gates

### Gate A: trustworthy computation

- Recovered evidence is traceable, or explicitly replaced by newly run data.
- Fixed-epsilon transport reference agrees with small trusted problems and reports coupling residuals.
- Matched geometry study consumes the same anchors/positive/generated/reference samples.
- Validation decisions do not inspect the final test stream.
- AWGN noise, power, independence, and analytic reference checks pass.

### Gate B: scientific value

Use 3 development seeds, expand to 5 if pilot variability warrants it, and freeze the protocol before confirmatory work. At least one reproducible, practically meaningful contribution must survive fair controls: an advantage in conditional accuracy/downstream time relative to competitive fast alternatives, or a well-supported mechanism explaining when the method succeeds or fails and how a principled modification improves it. A small numerical win without uncertainty does not meet this gate.

If the learned Gaussian baseline is sufficient on AWGN/SSPA, report that. The structured case must test whether the more flexible simulator adds value; Gaussian channel fibers cannot establish a need for non-Gaussian generative expressivity. A negative result may support a narrower efficiency or failure-mechanism contribution if that result is substantial and well evidenced. If fast flow/consistency models dominate, change the claim or method. Do not remove the winning competitor or tune against the final test set.

### Gate C: venue choice

- **Choose TCOM** if P1–P4 give a coherent substantive communications contribution, stable operating rule, fair modern baselines, and convincing structured-channel utility. Include a complete response to the original reports even though the decision was rejection.
- **Choose TMLCN** if the strongest result is a careful ML-for-communications application/analysis with reproducible empirical value, without enough new communications methodology to justify a TCOM retry. This is the current default, subject to the evidence.
- Neither venue fixes a missing scientific contribution. An unsuccessful pilot is a reason to investigate or narrow the paper, not simply change the journal name.

No acceptance guarantee or estimated acceptance percentage is justified.

## 6. Every reviewer concern has an assigned resolution

“Done” below means the evidence required to close a comment, not that it exists now.

| ID | Required response | Package / done when |
| --- | --- | --- |
| Editor 1 | Clear contribution and justified theory | P5 attribution corrected; proven/heuristic/empirical claims separated; one primary empirical question |
| Editor 2 | Fairness and modern fast comparators | P1–P3 common controls plus trained fast baselines and honest resource accounting |
| Editor 3 | Broader validation and consistent reporting | P0 canonical exports and P4 joint-block result |
| R1.1 | Explain cross/self transport, detached update, and training/inference distinction | P5 algorithm narrative and pseudocode map to actual implementation |
| R1.2 | State the sense in which the method is useful despite SER gaps | P3 time–accuracy curves with competitive one/few-step models; no universal superiority claim |
| R1.3 | One central claim and coherent empirical sequence | P5 organize results as geometry/stability → fast competitors → communication utility → structured channel |
| R1.4 | Match capacity, exposure, budgets, selection effort | P2 capacity/data-controlled track and measured compute track; report training-only parameters too |
| R1.5 | Recent one/few-step baselines | P2 conditional flow matching and a trained consistency/distillation comparator; low-step DDIM also included |
| R1.6 | Explain and measure training-cost gap | P1/P2 profiler breakdown and complete training timers; replace 2% extrapolations |
| R1.7 | Realistic scale beyond 7–8 coordinates | P4 jointly generated 128-real-coordinate structured block; TurboAE pairwise implant is insufficient |
| R2.1 | Repeated outputs at exact input | P5 restrict scope to repeatable simulator/controlled acquisition; P6 only if expanding claims |
| R2.2 | Explain practical single-observation limitation | P5 state need for stationarity/conditioning, smoothness and neighborhood coverage; no claim that local weighting resolves it automatically |
| R2.3 | Separate population guarantees from practical update | P1/P5 common-epsilon derivation and complete approximation ledger |
| R2.4 | Across-family comparison control | Same P2 evidence as R1.4 |
| R2.5 | Fast alternative paradigms | Same P2 evidence as R1.5 |
| R3.1 | Exact joint equality versus sample geometry | P5 correct statement and explicit marginal counterexample |
| R3.2 | Unique zero versus stationarity/convergence | P5 precise proposition, conditional dissipation statement, removal of unsupported conclusions |
| R3.3 | Isolate SSPA degradation, fair stopping | P1 identical long trajectories and common validation checkpoint rule for joint/fiber |
| R3.4 | Particle counts, iterations, epsilon, drift scale | P1 staged sensitivity with actual epsilon values, residuals, clipping and failure rates |
| R3.5 | Input distribution and encoder support/gradients | P0 declare power and anchor law; P3 learned-codeword fidelity and expected-loss gradients |
| R3.6 | Low-step DDIM performance and real AE time | P2/P3 newly trained AEs per sampler, matched budgets, reported actual NFEs |
| R3.7 | Timing arithmetic and latency definition | P0/P2 explicit sample count and units; batch-one latency separate from amortized throughput |
| R3.8 | TurboAE dimensions, mapping, noise independence | P3 document n=2 pair mapping and align decoder noise; measure within-pair residual covariance |
| R3.9 | Same anchor evaluation for all baselines | P2/P3 common anchors, analytic floors, GW2 definition and intervals |
| R3.10 | SER discrepancies and uncertainty | P0 one canonical nominal-point evaluation, integer errors/trials, consistent seed intervals and genuine zero-error bounds |

## 7. Submission rules and conference extension

Policy pages checked on 8 October 2026; recheck at submission.

- TCOM allows one resubmission after rejection, with disclosure and a detailed account of changes/response for a previous TCOM submission. Papers rejected by two different journals are ineligible. Its stated limits are 13 pages for submissions and 16 for revisions; do not assume a rejected-paper resubmission receives the latter allowance. The present decision is not an invitation to revise. [TCOM guidelines](https://www.comsoc.org/publications/journals/ieee-tcom/policies-and-guidelines).
- TMLCN's stated scope includes theoretical and practical ML for communications and reproducibility. Its checked guidelines require disclosure/citation of conference extensions. No specific prior-rejection limit was found on that page; confirm eligibility through the submission instructions when ready. [TMLCN guidelines](https://www.comsoc.org/publications/journals/ieee-tmlcn/policies-guidelines).
- OJ-COMS is a possible alternative, but its policy allows a manuscript rejected elsewhere only after one rejection. Do not plan an unrestricted sequence TCOM retry → another journal → OJ-COMS. [OJ-COMS guidelines](https://www.comsoc.org/publications/journals/ieee-ojcoms/policies-guidelines).
- TWC is not the default: a stronger central wireless-system question would be needed for fit. [TWC scope/policy](https://www.comsoc.org/publications/journals/ieee-twc/policies-guidelines).

The accepted conference paper already covers conditional direct/residual drifting, conditioning-aware kernels, AWGN/Rayleigh/SSPA/OptFib, timing, and an AWGN symbolic AE. The submitted journal adds Sinkhorn training, compact TDL, conditional metrics, multi-channel downstream results, equal-time AE training, and TurboAE. The revision must add the corrected methodology and evidence above, **cite the conference**, and state the extension accurately. Do not describe existing journal experiments as newly added in the response. Confirm title/venue/year/status from the accepted record; do not fabricate a DOI or page range.

## 8. Ready-to-paste instruction for the next Codex session

> Read `Journal_version/resubmission_handover.md`, `resubmission_cluster_plan.md`, and `resubmission_manuscript_plan.md` before changing anything. Treat the submitted ZIP and existing numerical reports as historical records. Start P0 and implement the P1 reference/diagnostic harness; in parallel prepare the corrected theory and attribution in a separate revision source. The current deliverable is a validated cluster run bundle and revised manuscript scaffold, not fabricated experimental conclusions. Follow the exact sampling, split, timing, and count contracts. Use bounded smoke tests; do not execute the existing misleading full-budget environment check. Produce dry-run task manifests and resource estimates from measured pilots before large submission. Keep missing archives and unresolved provenance explicit. After pilots, freeze the protocol and assemble the confirmatory run matrix. Report negative results and update the scientific claim accordingly. Do not submit a paper or contact editors without the author's instruction.

## 9. Completion criteria

- All reviewer rows have a manuscript location and evidence artifact, or a specific justified scope restriction.
- Every numerical claim can be regenerated from the canonical records; historical and revised protocols are not pooled.
- Main cross-family comparisons include competitive fast generators, common selection, and honest costs.
- All theory is stated with its domain/assumptions; no convergence claim is inferred from decreasing loss.
- Conference extension and prior rejection are disclosed as required.
- Final PDF fits the chosen venue, with figures placed near their discussion, readable uncertainty, and no unsupported captions.

This handover is a research and implementation plan. It neither establishes new performance claims nor guarantees publication.
