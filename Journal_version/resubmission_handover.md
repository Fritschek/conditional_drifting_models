# Journal resubmission handover

## Independent assessment complete, 13 October

Read the [new assessment](coverage_followup_assessment_20261013.md) first.
Its portable-record audit verifies all 135 physical curve points, 270 paired
discordance records and 40 local source/protocol hashes. Coverage improves the
primary generator-block mean in all three seeds for both families, while
individual failures and one low-SNR Sinkhorn block reversal remain. Flow-8's
accuracy is limited by its sampler; cosine alone also obscures magnitude error.
The assessment proves strict underdispersion for the ideal Gaussian Euler model
and explains why an exact sample Jacobian can still give a biased loss gradient.
The SSPA software labels require an explicit +3.0103 dB conversion for the
stated input-referenced physical convention. Preserve the frozen records.
Coverage replication and sampler checks are complete; resolve the convention,
apply a common input law to competitors and define the structured-channel study.
No new training was launched by this assessment.

## Completed follow-up, 13 October

After the 12 October independent assessment, the coverage-replication and
saved-sampler simulations have been prepared in a
[separate frozen protocol](coverage_followup_protocol_20261012.md), including
commands, fresh seed assignments and a primary endpoint. The user subsequently
approved execution. Both full suites now pass their final audits: 12 generators,
45 AEs, 135 physical curve points and six saved-sampler settings. Start with the
[new results](coverage_followup_results_20261013.md) and
[compact evidence](evidence/coverage_followup_20261013/README.md). No simulation
is running. Tiny smoke profiles are execution checks only. Previous numerical
sources and frozen protocols are unchanged.

Coverage gains replicate across three fresh generator-seed means, with mixed
individual AE outcomes. Ordinary-input SWD remains an inadequate standalone
selector. Saved Flow weights benefit substantially from finer integration;
DDIM's gradient cosine decreases as steps increase in the tested early codec.
The [SSPA noise-label audit](sspa_noise_label_audit_20261012.md) identifies a
3.0103 dB convention offset that must be addressed before external comparisons.
Next decisions concern a common coverage protocol across competitors, sampler
validation, and the still-unexecuted larger structured-channel experiment.
Historical statements below that replication or sampler checks are pending
are superseded by this completed report.

## Priority update, 12 October

The [independent assessment](application_coverage_assessment_20261012.md) is now
complete. It verifies the portable application/coverage records and local source
hashes, separates the replicated grouping repair from mixed downstream rankings,
and supplies a bounded next experiment plan plus precise coverage/risk-transfer
bounds. The existing K64 and solver studies should not be repeated as missing work.
The normalization is scalar grand-mean subtraction, not vector centering; the
existing sqrt(512) radius remains valid. Generator-seed coverage replication,
saved-sampler checks and the larger structured channel are the priorities.
No new simulation was launched by the assessment.

The current experiment round has been assessed. No simulation is running.
Git includes code, reports, compact results and figures. The large component,
fresh-start, intervention and solver-replay checkpoint archives remain local
and must be transferred separately for checkpoint-level reproduction. See
[the evidence index](evidence/README.md) for availability.
The assessment addresses which application-level contribution it supports,
how far input coverage explains downstream failures, and which minimal tests
are still needed. Historical next-step lists below are superseded by the latest
completed-results reports and the current status index.

The [controlled SSPA input-coverage intervention](sspa_input_coverage_results_20261012.md)
is now complete: two fresh broader-input generators, twelve fresh AEs and three
inherited control AEs. Gaussian improves in all three paired initializations
(8.7x--76x lower 8 dB SER); Sinkhorn improves its worst case but slightly worsens
the other two. Both substantially reduce tail mean error and optimistic transfer
gaps, with no codeword clipping. Sinkhorn loses some ordinary-input accuracy.
This is one generator seed, not three independent generator replicas. Repeat
the frozen intervention with fresh generators before tuning/scaling; keep the
mixed Sinkhorn outcomes and higher training-query energy explicit.

Focus the next comparison on communication applications and fast competing
generators. Conditioning itself is established prior work. The joint-versus-
conditional geometry ablation is optional and does not block this work.
The new [fast-baseline implementation/protocol](fast_baseline_protocol_20261012.md)
uses pinned official DDIM, flow-matching, consistency and W-Flow routines,
plus learned Gaussian and WGAN references. It provides shared capacity/query
budgets, measured costs and differentiable downstream AE tests. Development
defaults still need competitive-baseline validation and a frozen tuning allowance;
implementation tests do not complete the scientific comparison.
The [initial check report](fast_baseline_checks_20261012.md) records passing
CPU/CUDA tests and the all-channel application pipeline. The subsequent
[completed AWGN baseline development](fast_baseline_development_results_20261012.md)
finds a large WGAN EMA-lag effect and validates the longer WGAN recipe on three
development seeds. Official consistency distillation with fully counted learned
EDM pretraining also passes three seeds; standalone CT remains under-dispersed.
The [SSPA application pilot](fast_sspa_application_results_20261012.md) is now
complete on development seed 9111: six matched-query generators, seven AEs,
analytic-channel SER/BER curves, common-codec gradients and own-codebook checks.
Sinkhorn has good conditional fidelity and similar SER to DDIM-20 in this seed;
WGAN has the lowest numerical SER despite poor conditional fidelity. The
Gaussian-trained encoder drives two codewords far into the training-law tails,
where the surrogate mean violates physical saturation; those words account
for 98.2% of its transfer errors. Similar error concentration appears in other
families. The coverage intervention above tests this mechanism; fresh-generator
replication and AE optimization remain ahead, rather than another conditioning
novelty study or a blanket scalar-metric claim. All final-codec gradient references
in this pilot are unresolved; the intermediate references pass the MC check.
No final across-family ranking or paper campaign is complete. Seeds 9101--9105
and 9111 are now development seeds, overriding any older confirmation assignment.

**Current queue:** [Experiment status, 11 October](experiment_status_20261011.md).
The local mechanism round and fresh-start/communication-utility extension are
complete. Start with the [fresh-start results](sspa_fresh_results_20261011.md).
All 9 generator, 42 AE and 6 gradient tasks finished, with 252 scheduled
checkpoints audited. K4 deteriorated on all three fresh seeds; K64-only and
official K64 stayed stable. Official K64 also improves expected encoder-gradient
fidelity, but healthy-checkpoint SER rankings are mixed and early-selected K4
remains competitive. Do not infer universal downstream superiority.
The broader journal campaign is not complete. The portable evidence bundle is
`Journal_version/evidence/sspa_fresh_evidence_20261011.tar.gz`.
Optical fiber with larger conditional clouds is now a requested follow-up,
not a running experiment or a prerequisite for this handoff.

## Component isolation complete, 11 October

The user asked which ingredient made the normalized K64 package work and
whether it comes from the original authors. It directly uses the authors'
W-Flow loss, with our SSPA architecture/raw coordinates and warm-start setting;
it is distinct from the original Drifting JAX release. See the
[component protocol](sspa_component_protocol_20261010.md).

The [component results](sspa_component_results_20261011.md) now isolate the
practical fix. Four deletions from the successful package reached 390720 on
seed 9001. Returning to K4 failed (independent SWD 0.13262, variance 4.88055).
Removing normalization, restoring the old epsilon, or restoring the old
optimizer individually all passed. The additional **K64-only bridge in the
unchanged old trainer also passed**, SWD 0.0206825 and variance 0.0524019.
Only grouping changed from 4096 inputs x 4 outputs to 256 x 64, retaining the
same 16384 outputs of each kind per update. The old solver, raw loss, epsilon,
Adam state, learning rate and clipping remained unchanged; no EMA was used.

Output: `results/sspa_components_20261011/` and
`results/sspa_particle_bridge_20261011/`. All five paths are complete and
audited: 85 checkpoints, parent/source hashes, exposure and optimizer checks.
The successful reference is reproduced bitwise for 2000 updates. All 22
focused CPU tests and four component CUDA tests passed. Both trajectory
figures and all raw/EMA endpoints are retained. This is one deliberately
selected development seed and a healthy warm start, not independent-seed,
fresh-start or downstream confirmation. It establishes neither a universal
failure of K4 nor K64 as the minimal required cloud size.

## Completed follow-up execution, 10 October

The user requested completion of the remaining experiments before the next
handover. The [solver replay replication](sspa_solver_replay_replication_20261010.md)
is complete on seeds 9002/9003 and audited. The
[intervention matrix and full-budget follow-up](sspa_intervention_results_20261010.md)
are also complete: eight branches by three development seeds through 200k, then
two qualifying branches by all three seeds through 390720. All 24 tasks and six
extensions finished without task errors. The source-frozen protocol, 93 focused
CPU tests, CUDA resume/control checks and final audit passed.

**Outcome:** numerical solver corrections, 100 iterations, beta1=0 and smaller
drift scale do not prevent deterioration. Learning rate 0.0001 passes at 200k
but one seed fails later. The official-style normalized K64 package remains
accurate through 390720 on all three seeds, both raw and EMA. Raw independent
SWD is 0.01963--0.01970 versus analytic floor 0.01835; variance remains close to
0.05283. This is a warm-started, multi-component operating result on development
seeds, not an isolated-K result or a downstream SER/gradient-fidelity result.

Local suite: `results/sspa_interventions_20261010/`, including `completion.json`,
`audit.json`, `endpoints.csv`, all checkpoints and three inspected trajectory
figures. Parent bytes and original numerical files are unchanged. The audit
checks all 36 historical control checkpoints, paired K4 RNG streams, exposure,
optimizer interventions and frozen extension decisions. Ten of 432 small
reference solves remain explicitly unresolved. See the result note for costs,
full per-seed endpoints and limitations.

The next decisions are fresh-initialization/held-out-seed verification of the
normalized package, component isolation if a mechanism claim is intended, and
gradient/downstream evaluation of those checkpoints. Modern fast baselines and
the larger structured channel remain separate pending work. Do not rerun the
completed replay, repeated-update screen or continuation matrix as missing work.

Portable bundle: `Journal_version/evidence/sspa_intervention_evidence_20261010.tar.gz`
(20,026,461 bytes; 290 individually verified files). SHA256:
`787670fdef33250e58cdefbbad6379273e52e58f30135a500293668dbec7f675`.
It contains the new result note, plots, records, source snapshots and selected
checkpoints. The bundle README identifies the intermediate checkpoints and
full replay tensors that remain local. Existing earlier evidence archives are
retained alongside it.

## Earlier Context

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

The [independent full-budget assessment](sspa_full_budget_assessment_20261010.md)
reproduces the reports and verifies all 234 checkpoint hashes and inherited
checkpoint bytes. Final conditional variance is about 458/177/327 times truth
for legacy/fixed/shared policies. It specifies the next frozen-cloud solver
decomposition and matched intervention, and proves why zero expected finite-cloud
drift at exact matching does not imply finite-step stability. No solver repair
has yet been tested in training.

**Latest execution:** the [frozen-cloud solver replay](sspa_solver_replay_results_20261010.md)
is complete for seed 9001: 12 checkpoints, 192 method records and 168 cloned
single-Adam-step comparisons. The six healthy reconstructed batches have tiny
solver corrections. Degraded batches have substantial numerical sensitivity;
24 reference-method updates were withheld because some anchors did not converge.
Particle variance changes can have the opposite sign to neural variance changes.
No numerical trigger or stability repair is established. The
[official-code comparison](official_drifting_code_comparison_20261010.md) found
the authors' W-Flow Torch and original Drifting JAX releases. Their W-Flow solver
and our matched log-domain primitive agree exactly in 48 CPU comparisons.
Official training also differs in particle count, normalization and optimizer.
Check these separately before another long run; do not repeat the completed
replay as unimplemented work.

**Repeated-update screen complete:** [864 single-update trials](sspa_update_screen_results_20261010.md)
now compare nine methods on the three fixed-common 100k checkpoints, with 32
fresh paired batches each. Official-log versus production variance changes differ
by at most 3.14e-10. Larger K reduces gradient noise when spending more samples,
but not at fixed total outputs. Clearing only Adam's first moment changes the
local direction substantially. Abrupt official normalization with inherited
optimizer state amplifies updates; this is not a full official-training test.
No long-run repair is established. Next isolate optimizer/step-size effects in
a separately declared continuation, retaining unchanged and log-solver controls.
Read the screen's caveats and protocol before designing it.

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
- [Full-budget assessment and next solver intervention](sspa_full_budget_assessment_20261010.md): independently verified late failure in all nine paired trajectories; separate floor distortion, numerical solve error, finite-cloud noise and actual neural updates before claiming a cause or fix.
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
that experiment queue. The restricted P0 resume/RNG/count contracts, P1a audit
and full P1b epsilon trajectories have since completed. Proceed with the
[frozen-cloud decomposition and conditional intervention](sspa_full_budget_assessment_20261010.md#5-concrete-next-experiment-numerical-decomposition-before-retraining),
plus broader P0 repairs and P2 baseline preparation. Retain absolute gradient error, direction and magnitude
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
| Full-budget SSPA trajectories | All 9 runs reach 390,720 updates and deteriorate. Independent report regeneration matches JSON/Markdown and all keyed CSV rows; all 234 checkpoint hashes and 54/72 inherited states verified. 145 reference solves remain unresolved. Execution-host tests and internal RNG audits are reported separately; no new training or Torch tests in this review. |
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
5. **SSPA late deterioration is now reproduced in controlled trajectories.** The original 100-full/30-compact endpoint comparison was ambiguous. All nine new paired trajectories first learn and then develop severe excess variance. Common epsilon is insufficient to prevent failure; solver replay and a matched correction are still needed to identify cause. The new conditional validation protocol differs from the historical SWD table, so do not compare their numerical values directly or call retrospective checkpoint selection online early stopping.
6. **Downstream evaluation has additional controls to repair.** Symbolic codewords currently depend on minibatch normalization; analytic-channel SER is used for epoch selection in `train_symbolic_autoencoder`; rates are stored without exact trial counts. Separate validation/test streams and define the power convention. TurboAE also has a decoder-noise-distribution mismatch in the earlier audit.
7. **A larger codeword is not a larger channel generator.** Existing TurboAE uses an `n=2` surrogate across independent pairs. The new structured case must generate the whole coupled block. Do not simply increase `--override-n`: its default latent dimension remains 16 and it retains the compact TDL construction.
8. **Existing scripts are building blocks, not a ready revision suite.** In particular, `hpc/check_env.sh` calls a full-budget training runner despite labeling it a small smoke test. Replace that check before using it. New scripts in the cluster plan are explicitly proposed and do not yet exist.

## 4. Work packages and order

| Package | Work and purpose | Completion evidence | Dependency |
| --- | --- | --- | --- |
| P0: provenance and numerical contracts | Restore archives, freeze source/config hashes, unify units/noise/power, integer counts and split labels, bounded smoke test | Machine-readable input manifest and audit report with missing items explicit | First |
| P1: transport correctness and stability | Reference/replay/trajectories/interventions complete; normalized K64 passes three seeds. Component isolation and K64-only old-trainer bridge complete on seed 9001. Next fresh training, unused seeds and pending geometry comparison | Audited failure and repair evidence on development warm starts; confirmatory operating rule still needs verification | Restricted P0 contracts verified; broad interfaces remain pending |
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

> Start with `Journal_version/experiment_status_20261011.md` and `sspa_component_results_20261011.md`, then the earlier three-seed intervention report and official-code comparison. The local mechanism round is complete and audited. The normalized K64 package passed three development seeds; subsequent component deletions and an old-trainer K64-only bridge all reached 390720 on seed 9001. Returning to K4 failed, while deleting normalization or restoring epsilon/optimizer individually passed. Changing only grouping to K64/B256 also rescued the old production trainer at matched output exposure. This establishes a practical grouping effect on one selected warm-start trajectory, not fresh-start convergence, a universal K threshold, downstream benefit or a complete instability theory. Decide next on cold-start/unused-seed verification and gradient/SER evaluation. Matched geometry, modern fast baselines, broad measurement repairs and the larger structured channel remain pending. Optical fiber with larger conditional clouds is requested for the backlog, not yet run. Preserve negative outcomes and submitted artifacts; do not repeat completed screens or revive the closed kernel-selector study. No submission or editor contact without authorization.

## 9. Completion criteria

- All reviewer rows have a manuscript location and evidence artifact, or a specific justified scope restriction.
- Every numerical claim can be regenerated from the canonical records; historical and revised protocols are not pooled.
- Main cross-family comparisons include competitive fast generators, common selection, and honest costs.
- All theory is stated with its domain/assumptions; no convergence claim is inferred from decreasing loss.
- Conference extension and prior rejection are disclosed as required.
- Final PDF fits the chosen venue, with figures placed near their discussion, readable uncertainty, and no unsupported captions.

This handover combines a research plan with explicitly labeled local development
evidence. It does not establish confirmatory performance claims or guarantee publication.
