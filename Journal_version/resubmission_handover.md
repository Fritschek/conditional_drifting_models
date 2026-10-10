# Journal resubmission handover

Prepared 8 October 2026 against commit `726db308c46fe1df67319934ed35ef8dd8d79945`.

Updated 9 October 2026 through the gradient/metric pilots, completed synthetic
controls, and independent regeneration of their tables from the transferred
archive. Original simulations ran on the Linux host; the latest saved-record
audit ran in the macOS checkout. Those are distinct verification steps.

Updated 10 October with the independently audited negative learned-model metric
gate. That candidate's expansion is closed; P0/P1 and competitive baselines are
the immediate priorities.

**Subsequent execution:** the [small transport-reference audit](transport_reference_results_20261010.md)
has run. Its reference, solver instrumentation and restricted gradient checks are
implemented; default training outputs are unchanged. One sharp broad-cloud
reference failed the frozen cap and is explicitly unresolved. Healthy synthetic
SSPA clouds had small solver error, while stress tests exposed floor distortion.
This does not explain the historical degradation. The subsequent
[bounded SSPA trajectories](sspa_trajectory_results_20261010.md) now also ran:
three policies by three development seeds, 4,800 continuous updates each, with
resume/RNG/count contracts checked on CPU and CUDA. Common epsilon improves
the early variance deficit; final conditional SWD is nearly tied. The shared
pooled-median rule has an initial kernel-floor issue. An
[audited continuation through 30k](sspa_30k_results_20261010.md) is also complete
for all nine runs, with no degradation at saved checkpoints. The recovered
[historical loss histories](sspa_historical_training_20261010.md) place their
late transition much later, around 78k--161k by a post-hoc descriptive threshold.
The [full-budget continuation](sspa_full_budget_results_20261010.md) is now
**complete and audited for all nine trajectories** at 390,720 updates. Every
policy deteriorates in every seed. Final mean conditional SWD is 1.9448 legacy,
1.2294 fixed-common and 1.5343 shared-adaptive, versus selected values near 0.039.
Common epsilon delays or reduces degradation but does not fix it. Large late
transport errors, variance blow-up and covariance-derivative growth are observed;
causality remains unproven. Next use controlled solver replay/interventions,
not another epsilon-only sweep. The result note records the 145 unconverged
reference checks, exploratory derivative analysis and interruption overhead.
Do not repeat the reference audit, continuation interface or 30k screen
as unimplemented work. The numerical training module remains unchanged.

**Start here.** This is the execution plan for revising *Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation*, rejected as TCOM-TPS-26-1722. It supersedes the recommendations in `conditional_drifting_journal_strategy.md` and `journal_execution_roadmap.md`; keep those files as historical notes. It builds on, and corrects/extends, `review_audit_20261008/README.md`.

The author has no fixed deadline or compute cap and is willing to run simulations on roughly the previous scale. The author asks us to recommend the venue. The local checkpoint studies and nine SSPA trajectories through 390,720 updates are now complete, as recorded below. No cluster jobs, submissions, or correspondence were made.

- [Cluster implementation and experiment protocol](resubmission_cluster_plan.md): code tasks, controls, seeds, selection, job matrix, outputs, and existing commands.
- [Manuscript revision and mathematical audit](resubmission_manuscript_plan.md): replacement claims, proofs/counterexamples, section changes, and figure plan.
- [SWD, downstream risk, and gradient fidelity](theory_swd_downstream_gradient_fidelity.md): follow-up theory, a fixed-power counterexample, sufficient conditions, and controlled P3 interventions.
- [Metric before downstream optimization](theory_preoptimization_channel_metric.md): the author's stronger target; a candidate based on conditional feature values and input derivatives, task-class guarantees, estimation bounds, and validation requirements.
- [Completed local gradient-fidelity pilot](gradient_fidelity_local_pilot_20261008.md): seed-7 AWGN/SSPA results, two frozen codecs per channel, higher-sample SSPA confirmation, checks, and reproduction commands. This is task-dependent development evidence, not a validated pre-optimization score.
- [Completed decoder-free metric feasibility pilot](channel_feature_metric_pilot_20261009.md): full-kernel and moment derivatives, 128/512 samples, fixed inputs, shared versus independent noise. Estimation works, but added screening value is not established; retain sampling-floor and large-variance limitations.
- [Derivative resolution and additional-seed checks](metric_resolution_results_20261009.md): autograd reference, convergent small-step differences, fixed moment augmentation, SSPA seeds 8/9 and frozen-codec validation. Numerical checks pass, but the score still misses task-dependent gradient orderings.
- [Earlier independent assessment](metric_experiment_assessment_20261009.md): separates direction from magnitude and identifies candidate-specific estimator variance. Its A/B experiment proposals are complete and its missing-October-data warning is superseded below.
- [Completed exact-law and matched-value controls](metric_controls_results_20261009.md): the proposed A/B controls now ran locally. Rotation changes estimator variance despite identical laws; same-input perturbations isolate useful derivative information. Simple moments remain sufficient in this Gaussian control. All October raw results are present on the Linux host.
- [Assessment after the controls and next bounded gate](metric_controls_assessment_20261009.md): independently reproduces the archived tables, proves a derivative obstruction beyond the first three moments, and specifies one same-input learned-model check without downstream optimization. No kernel-specific selection advantage is established yet.
- [Completed same-input learned-model test](learned_metric_gate_results_20261009.md): the bounded SSPA test now ran at N=512/2048. All 72 final target contrasts were resolved by the frozen descriptive rule; none supplied kernel-specific added ordering information. Simple moment checks performed better on this panel. Stop expansion of this kernel-selector candidate, retain gradient fidelity as explanatory evidence, and return to P0/P1 and fair baselines. No new downstream training or manuscript changes were made.
- [Actual decision and all 22 reviewer comments](review_audit_20261008/README.md#full-review-reports).

## 1. Recommendation

The paper has a potentially publishable communications application, but the next submission needs a rebuilt argument and controlled evidence. More seeds of the current comparison will not answer the central objections.

**Provisional venue: IEEE TMLCN. Preserve TCOM as an option until the controlled pilots and larger-channel result are available.** TMLCN fits an ML method adapted, analyzed, and evaluated for communications. A TCOM retry is reasonable if the revised work establishes a substantive communications-specific finding beyond applying an existing conditional generator, and answers the editor's fairness, theory, and scale concerns together. This is a fit judgment, not an acceptance prediction. There is no evidence for the acceptance percentages in the old notes.

The main new reason for caution is attribution: **conditional drifting and conditional Sinkhorn/W-Flow already exist in the cited originating papers.** W-Flow explicitly writes the same conditional barycentric velocity, with class label `c` in place of channel input `x`. Continuous inputs and simulator access create useful research questions, but replacing a discrete label with a continuous vector is not, by itself, a new transport principle. [Drifting, Sections 3.5 and 4](https://arxiv.org/html/2602.04770v1#S3.SS5), [W-Flow, equation (36)](https://arxiv.org/html/2605.11755v1#A2.SS3).

Build the revision around this question:

> When does conditional Sinkhorn training provide accurate and useful one-evaluation channel simulators for communication-system optimization, compared with other fast generators under controlled sampling and compute budgets?

The strongest possible evidence would connect **conditional fidelity, fidelity of optimization gradients, and actual downstream performance per training time**, including a larger jointly generated channel block. Whether that connection holds is to be tested. Do not promise a Pareto advantage or a mechanistic explanation of instability in advance.

The 9 October [theory follow-up](theory_swd_downstream_gradient_fidelity.md) makes this direction concrete: exact conditional laws at the current unit-power codebook and arbitrarily small uniform conditional Wasserstein error can coexist with a reversed transmitter gradient. It also proves positive loss-transfer, curvature-to-gradient, and biased-descent bounds under explicit assumptions. These establish a possible mechanism, not its cause in the existing runs or a Sinkhorn-specific guarantee. The local pilot now supplies partial T1/T2 evidence. Extend it as specified below rather than repeating it as unstarted work. A successful mechanism study could strengthen the TCOM case, but the present venue recommendation remains conditional on evidence.

**Refined author objective:** develop a useful channel-surrogate metric that can be computed before candidate-specific downstream encoder/decoder optimization. The [metric proposal](theory_preoptimization_channel_metric.md) separates that objective from diagnostics through a trained decoder. Its candidate has explicit function-class guarantees but failed the bounded learned-model ordering gate. The broader objective remains open; do not make developing a new selector a prerequisite for finishing this paper. Keep the primary selection protocol and use derivative diagnostics to investigate mechanisms.

Use **channel simulation**, learning `p(y | x)`, throughout. Receiver-side channel estimation from pilots is a different problem and is not what the current experiments evaluate.

### Local evidence now available

**Latest execution update:** the [same-input learned-model panel](learned_metric_gate_results_20261009.md)
is complete and supersedes instructions below to run that gate. DDIM-10 has the
smallest absolute loss-gradient error for 21/24 fixed input/probe cases, while
selected Sinkhorn has the lowest conditional SWD at all three inputs. The RBF
derivative score adds no resolved ordering information beyond cheaper checks
under the frozen rule. These nested cases are not independent trained-model
replications. Do not launch a confirmatory metric sweep on this evidence. The
compact raw-statistics archive is under `Journal_version/evidence/`, outside the
ignored results tree. Larger raw sample/Jacobian tensors and model weights remain
local. This does not close the journal's stability, fairness or scale concerns.

The [10 October independent assessment](learned_metric_gate_assessment_20261010.md)
reproduces the saved summaries and confirms no material implementation flaw.
DDIM-10 wins all six kernel-section loss cases as well. It proves the exact
distinction between a worst-case norm bound and uniform individual-loss ordering,
and explains why the negative result does not refute the bound. The gate's union
of cheaper comparators does not establish one operational replacement selector.

The subsequent [decoder-free metric pilot](channel_feature_metric_pilot_20261009.md)
is also complete. Its score does not use the codecs below. It finds strong
sampling-access dependence and a useful SSPA signal, but no established benefit
over simpler distribution/moment checks. The bounded kernel understates a
large-variance failure relative to other models. The subsequent
[resolution and seed checks](metric_resolution_results_20261009.md) are also
complete. Finite differences converge to pathwise empirical derivatives,
and fixed raw-moment features expose the variance failure across seeds 7--9.
Seeds 8/9 were scored without fitting against their task outcomes. They still
do not validate added selection value: on a seed-9 frozen codec, WGAN has
worse SWD but better gradient alignment than selected Sinkhorn, and the
generic derivative score also misses this ordering. Do not rerun these checks
as unstarted work. The subsequent exact-law and same-input synthetic controls
are also complete (see below). Retain candidate-specific uncertainty as well
as analytic null controls, and separate value, direction, magnitude and actual
normalized-step outcomes. Passive-data feasibility and heavy-tail robustness
remain open.

All values below are encoder-gradient comparisons at identical frozen codecs,
using existing seed-7 generators. Each gradient averages four Monte Carlo
repeats; these repeats are not independent training seeds. SSPA uses the
higher-sample confirmation, not the initial lower-sample estimates.

| Observation | Local evidence | Consequence for the research plan |
| --- | --- | --- |
| AWGN gradients are close to the analytic reference | Condition-wise Sinkhorn cosine 0.948 / 0.943 at analytic-trained / Sinkhorn-trained codecs; independent analytic estimates 0.975 / 0.951 | Useful calibration case, not a resolved superiority ranking |
| SSPA mismatch persists after increasing samples | Sinkhorn cosine 0.852 / 0.625, norm ratio 0.531 / 0.128; DDIM-100 cosine 0.995 / 0.975 | Derivative fidelity is a concrete measurement target; it is not an established Sinkhorn advantage |
| SWD and gradient direction rank methods differently | At the analytic-trained SSPA codec, Sinkhorn conditional SWD 0.02790 vs DDIM-10 0.04865, but encoder cosine 0.852 vs 0.956 | Test whether a derivative-aware score adds information beyond distribution-only scores |
| Direction alone also misses information | DDIM-10 encoder norm ratio is only 0.048 at that codec | Report direction and magnitude separately, with sampling uncertainty |
| Sinkhorn steps remain useful | Small equal-norm encoder steps lower true-channel loss at both SSPA codecs | Do not equate imperfect fidelity with a reversed or unusable update |

The pilot uses explicit full-codebook power normalization and fixed noise. Its
SER values are not replacements for the submitted paper's minibatch-normalized
training results. Its steps normalize each gradient to a common parameter-step
norm. They test direction, not the effect of magnitude under a common SGD
learning rate or under Adam. No early/middle/late trajectory or complete
downstream retraining was run; the later exploratory seed-8/9 checkpoint
comparisons are described above.

**Immediate order:** derivative-resolution, raw-moment augmentation, and the
additional seed-8/9 checkpoint checks are complete. The report join now protects
codec identity and stored task contracts, with nine focused tests passing.
The [assessment's A/B controls](metric_controls_results_20261009.md) have now
run on the Linux host; their archived report tables also reproduce in this
macOS checkout. At N=512 the
ordinary RBF derivative norm rises from 0.114 to 0.928 for an exact-law
reparameterization; independent cross traces target zero without clipping.
The matched-value experiment shows derivative information beyond pointwise
SWD/MMD, but simple mean/Jacobian information also captures this Gaussian
example. The [latest assessment](metric_controls_assessment_20261009.md) now
proves an analytic obstruction with all first three moments matched; it also
records that a fourth-moment check detects that construction. Its bounded
same-input learned-model gate is now complete and negative for this selector.
The [independent assessment](learned_metric_gate_assessment_20261010.md) closes
that experiment queue. Resume P0 contracts, P1 stability/correctness and P2
baseline preparation. Retain absolute gradient error, direction and magnitude
as separate explanatory outcomes. Do not rerun A/B or the learned gate as
unstarted work, or call a squared-trace estimate an operator-norm confidence
bound. A replacement metric is a separate future hypothesis requiring unused
validation data, not the next automatic experiment.

## 2. What was actually checked

| Evidence | Status in this assessment |
| --- | --- |
| Journal source, included result tables, source implementation, HPC wrappers | Inspected locally. The submitted PDF is 12 pages; result pages 9–12 were also rendered and visually inspected. |
| Local submission ZIP | Main TeX, timing, coding, and SSPA-budget snippets freshly verified byte-identical to the working counterparts. ZIP SHA-256: `b696487568f521faf953acbeaf684866bfbba7688222a29a608bc35f3258a4aa`. This is a local bundle, not a portal download. |
| Actual reviews | Editor rejection and 7 + 5 + 10 numbered comments preserved in the prior audit. Their source is the author's pasted decision, as documented there. |
| `tcom_review_comments.md` | Earlier pre-submission feedback, **not the rejection reports**. Its major-revision recommendation and probabilities are not decision evidence. |
| Conference | Inspected `Paper_camera_ready_checked/drifting_vs_diffusion_summary.tex`; repository README also identifies the older `(3).tex` source. Verify the exact accepted camera-ready bundle and bibliographic status before final disclosure. |
| Earlier saved numerical audit | `review_audit_20261008/evidence.json` retained. The raw records needed by its audit are present in this local checkout. |
| Audit rerun | On 9 October, `python3 scripts/audit_journal_review_evidence.py --out /tmp/journal_review_evidence_recheck_20261009.json` completed successfully. Parsed output exactly matches the retained evidence JSON, including 100/30 full/compact SSPA runs and 30 TurboAE runs. This verifies saved records, not original training reproducibility. |
| Local gradient-fidelity study | Completed on RTX 5060 Ti using saved seed-7 checkpoints, with four frozen channel/codec comparisons and a higher-sample SSPA confirmation. See the linked pilot report and raw result directories. |
| Decoder-free feature metric | Completed 128/512-sample panels: 2,430/1,458 observations; 13 estimator/gradient tests pass. Uses existing checkpoints and fixed radial inputs. Held-out predictive/selection utility remains open. |
| Derivative resolution and moment augmentation | Completed 1,350 step-comparison records, 558 pathwise records and 62 metric/task joins, including SSPA seeds 8/9. The combined feature/gradient suite now has 18 passing tests. Still no validated selector. |
| Exact-law and same-input controls | Completed 192 rotation records, 480 matched-value records and 3,840 probe-gradient records, with exact population references. The execution host reports 37 passing tests. The independent archive audit reproduces all three summary CSVs byte for byte and checks exactly; Torch tests were not rerun on macOS. Kernel-specific selection benefit remains open. |
| Same-input learned-model gate | Completed 192 metric/1,536 task records; zero added-information contrasts under the frozen union rule. Execution host reports 48 passing tests. Independent NumPy audit reproduces numeric metric CSV fields exactly and task fields to 2.22e-16, with source/reference checks; no new Torch/GPU run here. Current selector expansion stopped. |
| New training / cluster measurements | None performed. All new run counts and thresholds below are proposed protocol choices. |

The [committed October archive](evidence/README.md) now transfers the raw metric
evidence through Git; 102 result files were restored and checked here. It does
not contain training checkpoints. Other `results/` files remain ignored, so
new systems must check availability and restore only missing required inputs.
Do not repeat bulk restoration solely because an older note reports missing
data. Unknown historical source versions and the remaining measurement-contract
repairs still belong to P0.

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
| P3: communication utility | Extend gradient/mechanism diagnostics; equal-update and equal-time AE training; TurboAE repair. Current kernel-selector gate is complete and negative | Actual SER/BER/BLER versus time, separately seeded tests, scoped gradient/variance evidence | Confirmatory comparisons need P2 checkpoints; a new selector is not a prerequisite |
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

> Start with `Journal_version/resubmission_handover.md` and `sspa_full_budget_results_20261010.md`, then the linked protocol, reports and cluster/manuscript plans. All nine SSPA trajectories completed 390,720 updates; the continuation and derivative-record audits pass. All three epsilon policies deteriorate in all seeds, so common epsilon alone is not a stability fix. Selected SWD stays near 0.039, while final policy means are 1.9448, 1.2294 and 1.5343. The late failure involves excess variance and large covariance derivatives; mean-Jacobian changes are smaller. Solver errors grow, but causality is not proved; 145 reference solves hit their cap and remain explicitly unresolved. Next perform a small fixed-cloud solver decomposition, then a paired numerical intervention from pre-transition checkpoints if justified. Do not repeat the completed epsilon sweep or revive the closed kernel-selector study. The full evidence archive includes all checkpoints and traces. Fair modern fast baselines, broad P0 interfaces and structured-channel experiments remain pending. Preserve submitted artifacts and negative outcomes; no submission or editor contact without authorization.

## 9. Completion criteria

- All reviewer rows have a manuscript location and evidence artifact, or a specific justified scope restriction.
- Every numerical claim can be regenerated from the canonical records; historical and revised protocols are not pooled.
- Main cross-family comparisons include competitive fast generators, common selection, and honest costs.
- All theory is stated with its domain/assumptions; no convergence claim is inferred from decreasing loss.
- Conference extension and prior rejection are disclosed as required.
- Final PDF fits the chosen venue, with figures placed near their discussion, readable uncertainty, and no unsupported captions.

This handover combines a research plan with explicitly labeled local development
evidence. It does not establish confirmatory performance claims or guarantee publication.
