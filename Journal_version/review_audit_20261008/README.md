# TCOM review audit and next submission decision

Date: 8 October 2026. Manuscript: TCOM-TPS-26-1722.

This file is a standalone handoff containing the audit, proposed plan, and
[full decision letter and reviewer reports](#full-review-reports).
Repository paths are relative to
`conditional_drifting_models/` unless indicated otherwise. Recheck venue policies
before acting on them. No next venue or experiment budget has been approved yet.

This audit compares the editor's letter and all 22 numbered reviewer comments
provided in the conversation with the local submission bundle, saved results,
and implementation. It proposes work for a revised submission. It does not
change the manuscript, rerun training, or claim that every result has been
independently reproduced.

## Evidence and scope

The main TeX and the timing, coding, SSPA-budget, and curve-figure snippets inside
`../tcom_upload.zip` are byte-identical to their current counterparts in
`Journal_version`. This is our local submission snapshot, not an independent
download from the editorial system. Its SHA-256 is in [evidence.json](evidence.json).
Earlier `tcom_review_comments.md` contains pre-submission feedback, not these
actual reviewer reports.

The saved-data checks can be repeated without installing dependencies:

```bash
python scripts/audit_journal_review_evidence.py \
  --out Journal_version/review_audit_20261008/evidence.json
```

The script checks bundle identity, timing arithmetic, both SSPA SER sources,
SSPA generator results/configurations, and all 30 TurboAE implant dimensions.
Code findings below refer to current source and, where noted, historical git
source. Many saved runs lack a source commit hash, so dates alone cannot prove
which uncommitted source was used on the HPC.

## Findings that change the assessment

### 1. The SSPA text/table discrepancy is explained by separate evaluations

Across seeds 7--36, the analytic and condition-wise Sinkhorn training histories
and configurations are identical between the compact SER suite and curve suite.
The curve runner subsequently evaluates at each SNR with a new metric seed.

| Method | Nominal-point table SER | Curve SER at 8 dB | Seeds |
| --- | ---: | ---: | ---: |
| Analytic | 1.3000001e-5 | 1.5666667e-5 | 30 |
| Condition-wise Sinkhorn | 9.3333338e-5 | 8.1333337e-5 | 30 |

These numbers reproduce the apparently conflicting manuscript values. They
are distinct Monte Carlo estimates, rather than evidence that one value was
invented. The text does not identify that distinction. Use a single canonical
evaluation record for figures, tables, and prose in the revision.

Sources: `results/journal_wflow_ser_sspa_budget_screen_20260602_073353`,
`results/journal_wflow_curves_20260602_220608`, and
`scripts/run_journal_wflow_curve_seed_channel.py:evaluate_curve`.

### 2. The timing total uses an undocumented generation budget

Saved AWGN DDIM-100 values are 0.0197201 projected training hours and
0.0391337 ms per generated sample. They produce:

- 0.0305906 hours for one million generated samples;
- 0.1284247 hours for ten million generated samples, matching Table V.

The timing presets use ten million evaluation samples on AWGN/Rayleigh/SSPA
and 100,000 on TDL. The manuscript instead links totals to Table II's different
budgets. Thus the reviewer's arithmetic objection is valid against the stated
protocol; the stored arithmetic itself is consistent with the timing preset.

Inference values are repeat elapsed time divided by `512 * 40` samples. They
measure batch-amortized generation time, not latency of an individual call.
Seven repeated measurements and their standard deviations already exist.

SSPA condition-wise timing also projects the 160-epoch, 10-million-per-epoch
configuration, whereas reported SSPA fidelity uses 120,000 per epoch. It is
not a measured cost of reaching the selected SSPA operating point. Full-run
training time and time to validation-selected quality must be recorded afresh.

Sources: `scripts/run_inference_timing_benchmark.py`,
`results/timing_suite_local_cuda_20260601_1755/timing_suite_summary.json`,
`results/timing_suite_wflow_20260601_1715/cuda/sspa/timing_summary.json`.

### 3. Long-budget SSPA degradation is systematic; its cause is unresolved

All 100 corrected full-budget SSPA runs have large final direct SWD:
mean 1.5344, median 1.5171, range 1.0581--2.0289. The 30 compact runs have
mean 0.0070714 and range 0.0052563--0.0094586. This is not a single failed seed.

Both inspected seed-7 configs use batch size 4096, 160 epochs, learning rate
0.001, four samples of each kind, drift scale 1, and norm clipping at 2.
The full and compact `dataset_size` values are 10,000,000 and 120,000.
The current training loop draws fresh Gaussian anchors every update;
`dataset_size` controls updates per epoch, not the size of a fixed stored dataset.
Therefore the different dataset-size labels do not by themselves imply a
different data distribution. They do change training exposure and the locations
of epoch boundaries. Exact implementation provenance still needs checking.

Actual `ceil(dataset_size / batch_size) * epochs` counts are 390,720 and 4,800.
The table's 390.6k and 4.69k values use unrounded estimates.

The endpoints do not establish when degradation begins, whether the same
trajectory first reached the compact quality, or which mechanism causes it.
No checkpoint selection rule with a predeclared validation threshold/patience
is established by these records. Do not describe the compact run as a validated
early-stopping procedure until that procedure is defined and tested.

### 4. The Sinkhorn comparison has additional controls to match

In `conditional_drifting/training.py`, the joint branch uses B generated and
B positive samples plus B independently generated references. The condition-wise
branch uses B anchors with 4B generated, 4B positive, and 4B reference samples.
Equal epochs, architecture, and nominal batch size therefore do not give equal
channel sampling exposure or equal numbers of generated training outputs.

The manuscript's expanded-batch complexity expression describes a possible
joint comparator, but not the actual joint branch used here. The coupling
work is approximately O(I B^2) for the implemented joint branch versus
O(I B Kg (Kp + Kr)) for the condition-wise branch, excluding feature-distance
and network costs. The large timing difference is plausible at B=5000, but
needs measured profiling and a correct accounting of the compared samples.

Use one repeated-anchor batch for both fields and change only allowed coupling
geometry in the decisive ablation. Separately report a practical equal-time
comparison. Existing SSPA Table VIII has no joint-Sinkhorn M=64 row, so the
claim of a downstream controlled comparison on every channel is too broad.

### 5. Automatic epsilon adds an unacknowledged theory/implementation gap

`_resolve_batched_sinkhorn_epsilon` uses the median positive half-squared
distance pooled over within-anchor costs, with a floor of 0.001. It is called
separately for generated-to-positive and generated-to-reference transport.
Those calls generally select different epsilons, and both values change with
the samples and generator. Ten iterations, kernel clamping, row renormalization,
independent reference samples, drift clipping, and neural regression add further
approximations.

Consequently, the implemented default field is not automatically the velocity
of the fixed-common-epsilon objective written in the proposition. A useful
small experiment compares fixed/shared epsilon with the current separate
automatic choices, measures coupling marginal errors, and checks a trusted
high-accuracy solver on small problems. Separate epsilon is a hypothesis for
instability, not an established explanation of the SSPA result.

Sources: `conditional_drifting/losses.py:_resolve_batched_sinkhorn_epsilon`,
`_batched_sinkhorn_barycentric_projection`, `compute_fiber_sinkhorn_drift`.

### 6. TurboAE exists, but needs both documentation and a protocol check

All 30 saved implant configs have n=2, 30 generator epochs, and noise standard
deviation 0.6309573 at Eb/N0=4 dB and rate 1/2. The TurboAE has 64 information
bits and 128 real transmitted coordinates in a `[batch, 64, 2]` output tensor.
The implant reshapes to `[batch * 64, 2]`, draws independent latent vectors per
row, generates two outputs, and restores the tensor. No n=7 chunking or padding
is involved. Independence between pairs follows from this sampling structure
conditional on the input; independence and correct variance within each pair
must be measured, not assumed.

The earlier conversational suggestion of a separately trained n=64 generator
was incorrect. This was a valid long-codeword downstream experiment with a
separate n=2 surrogate. The manuscript omitted that distinction.

There is an additional training mismatch. `_PerSymbolImplant` ignores SNR and
decoder-training kwargs. Analytic AWGN uses the configured decoder SNR range,
then reduces the sampled noise powers to a single scalar standard deviation.
The surrogate remains at its trained noise level. This behavior exists in both
the current code and the pre-June-1 historical implementation. The stored run
configs specify decoder offsets [-3.5, 0] dB. Thus matching the outer TurboAE
optimizer settings did not ensure matching the decoder noise distributions.
Quantify this and align them before attributing the whole BER gap to fidelity.

Also check the surrogate's temperature-estimation version: the run predates
commit `b23a404`, which replaced flattened cross-condition epsilon estimation
with within-condition estimation. The saved configs do not identify a source
commit or epsilon mode. Historical source suggests legacy behavior, but does
not prove the exact HPC code version. Treat this as unresolved provenance.

Sources: `results/turboae_long_block_hpc_20260529_165218`,
`conditional_drifting/e2e_implants.py`, `conditional_drifting/model.py`,
`scripts/train_turboae_awgn_implant.py`, `external/turbo_mingru_decoder/model_turboAE.py`.

### 7. Other setup/reporting issues

- TDL samples independent taps per codeword, constant within four complex uses.
  It tests delay coupling but no temporal evolution or Doppler process.
- TDL code adds noise with standard deviation sigma to EACH real coordinate.
  The manuscript writes complex CN(0, sigma^2). Under the usual complex-variance
  convention the code gives CN(0, 2 sigma^2). Reconcile the definition and SNR
  labeling before any rerun; do not silently alter the channel and mix results.
- Drifting and diffusion use fresh unnormalized standard-normal inputs by
  default. WGAN divides inputs by the batch standard deviation. Unify mu and
  power conventions in the new controlled harness.
- The existing 1800-second experiment controls downstream AE training time.
  It does not match the cost/capacity of training the channel generators.
  Preserve it, label that scope, and include reduced-step diffusion implants.
- Plotting code clips SSPA means below a single-run floor and labels them as
  upper bounds. A display floor is not a confidence bound, and a mean over
  seeds can legitimately be below a single-run one-error resolution. Store
  counts and trials, show zero-error upper confidence bounds when applicable,
  and separate Monte Carlo uncertainty from variability across trained seeds.
- Anchor floors and metrics already exist for many saved Sinkhorn runs. Report
  uncertainty and floors consistently, then evaluate the other families with
  the same anchors and projection draws. The code computes mean per-anchor
  Gaussian W2, taking the square root before averaging, not mean squared W2.

## Theory revision

Correct the joint-law statement: with common input marginal mu, exact equality
of joint laws implies conditional equality mu-almost everywhere. The proposed
advantage concerns coupling geometry, finite-sample estimation, and training.

Keep a carefully stated result on nonnegativity and the zero set, with explicit
measure/support, kernel/cost, epsilon, integrability, and measurability
assumptions. State a dissipation identity only under differentiability and
interchange-of-integral assumptions. A unique zero and decreasing energy do not
establish convergence, absence of other stationary measures, or a globally best
network approximation. Remove the unsupported "best representable" statement.

[W-Flow Appendix A](https://arxiv.org/html/2605.11755v1#A1) treats convergence
of particle approximations to population dynamics on finite time horizons under
regularity and asymptotic assumptions. That does not by itself establish
convergence of our finite network to the target. Check the conditions against
our unbounded Gaussian anchors and finite per-anchor clouds. Use
[Feydy et al.](https://proceedings.mlr.press/v89/feydy19a.html) for the
Sinkhorn-divergence properties actually needed, without importing stronger
equilibrium conclusions by implication.

One useful theoretical bridge is to derive the instantaneous parameter gradient
of detached regression, proportional to minus the sum of generator-Jacobian
transposes times particle velocities. Under compatible exact transport and a
shared fixed epsilon this can connect to an objective gradient. Establish those
conditions first; neither detachment alone nor the present heuristic establishes
descent or global convergence. No new theorem has been proved in this audit.

## Review-by-review disposition

| Comment | Assessment and required response |
| --- | --- |
| R1.1 algorithm rationale | Explain cross/self transport and training-only dynamics; add common-epsilon reference and map each approximation to code. |
| R1.2 SER advantage | Valid objection to universal superiority. Define a useful operating point and test modern fast competitors. |
| R1.3 central claim | Center conditional coupling geometry, then measure its practical utility under matched resources. |
| R1.4 fairness | Real gap. Match capacity, input distribution, sampling exposure, selection effort; add time-controlled generator comparisons. |
| R1.5 modern baselines | Real gap. Start with reduced-step DDIM, conditional flow matching, and one established consistency/distillation baseline. |
| R1.6 timing | Budget mismatch confirmed; small-cloud speed rationale plausible; report full runs and profiling. |
| R1.7 channel size | Long-codeword AWGN exists but does not test joint high-dimensional structured generation. Add one larger channel. |
| R2.1 repeated observations | Real scope limitation for passive datasets. Exact repeats are feasible in controlled acquisition if other conditions are held fixed. |
| R2.2 practical applicability | Either validate a sparse-observation approximation or explicitly restrict demonstrated applicability to repeatable sampling. |
| R2.3 practical gradient | Qualification already exists, but epsilon selection and projection need a precise bridge and narrower claims. |
| R2.4 fairness | Same experiment as R1.4; avoid duplicating runs to answer overlapping comments. |
| R2.5 fast generators | Same experiment as R1.5. |
| R3.1 joint equality | Correct the text. Finite transport geometry is the relevant distinction. |
| R3.2 theory | Restrict proposition; remove unsupported best-representable claim; spell out assumptions. |
| R3.3 SSPA collapse | Confirmed over 100 seeds. Need matched trajectories and common validation selection for both fields. |
| R3.4 sensitivity | Log epsilon for each term, marginal errors, clipping fraction, and validation metrics while varying key controls. |
| R3.5 mu/gradients | Specify Gaussian inputs and normalization. Evaluate learned encoder codewords and expected-loss input gradients. |
| R3.6 DDIM/time | Use existing teacher weights for lower-step samplers, but retrain downstream AEs through those samplers. |
| R3.7 timing arithmetic | Ten-million versus one-million mismatch explained above. Correct labels/budgets and measure latency separately. |
| R3.8 TurboAE mapping | Separate n=2 surrogate confirmed for all 30 seeds. Explain mapping and resolve decoder-noise/provenance issues. |
| R3.9 anchor table | Existing Sinkhorn floors can be reused; add common-protocol evaluation of other methods and GW2 definition. |
| R3.10 inconsistent SER/statistics | Separate evaluations reproduce both values. Consolidate records; report seed uncertainty and trial counts consistently. |

## Work sequence and decision points

### Stage 1: repair the evidence and establish a numerical reference

Use one manifest connecting every retained figure/table/text value to run ID,
config, checkpoint, evaluation seed, and trial count. Reconcile evaluation
budgets, noise conventions, and TurboAE provenance. Export existing anchor
floors and uncertainty. Correct mathematical overclaims in a revision branch.

Add a small, validated Sinkhorn reference with one explicit common epsilon and
coupling residual checks. Compare its velocity against the practical estimator.
This determines whether a code change is warranted before investing in new
large-scale results. Benchmark stored models on common anchors and on learned
codewords. Analytic channel access makes expected-loss gradient comparisons
possible, using enough Monte Carlo samples to report gradient-estimation noise.

### Stage 2: small controlled pilots

Start with 3--5 development seeds, explicitly excluded from confirmatory claims.
Use AWGN to check calibration and SSPA to expose the known instability.

1. Save a single SSPA trajectory at logarithmically spaced update counts,
   keeping batch size, fresh-data law, and optimizer fixed. Apply the same
   held-out anchor validation rule to joint and condition-wise methods. Continue
   far enough to reproduce degradation, with a predeclared numerical-failure
   rule. Full-SWD curves cannot be recovered from loss-only logs.
2. Vary shared/separate epsilon first, then per-anchor counts (e.g. 4, 8, 16),
   iterations (10, 30, 100), epsilon multiplier, and drift scale in staged
   comparisons. Log marginal errors, mean/covariance errors, output magnitude,
   latent sensitivity, and clipping frequency. Avoid a full Cartesian sweep.
3. Reuse diffusion teacher checkpoints for DDIM-10/20/50. Evaluate both channel
   fidelity and freshly trained downstream AEs. Existing DDIM-100 AE weights
   alone cannot establish low-step training behavior.
4. Add conditional flow matching at multiple sampling-step counts and one
   established one-step consistency/distillation implementation. Define which
   form is used and count any teacher training and distillation costs. A
   one-step Euler sample from an ordinary flow is not automatically a strong
   one-step baseline.

Measure complete channel-model training wall time plus inference at batch 1
and the actual downstream batch size, and downstream SER versus elapsed AE
training time. Use the same held-out evaluation samples and a comparable
hyperparameter-search allowance. Equal architecture alone does not equalize
capacity perfectly, but comparable parameter counts and shared backbone shapes
substantially improve attribution. Report all real channel draws, generator
draws, critic updates, and optimizer steps explicitly.

Decision: continue if conditional Sinkhorn offers a reproducible useful
conditional-fidelity or downstream-time advantage against a competitive fast
baseline, or a clearly demonstrated geometry contribution with bounded cost.
Universal wins are unnecessary. If neither holds, change the scientific claim
or method before spending a large HPC budget.

### Stage 3: one stronger wireless experiment

Prefer a 64-complex-use frequency-selective block or OFDM block with a clearly
specified multipath implementation, power/noise calibration, and CP/boundary
handling. The surrogate must generate the coupled block jointly. Include an
analytic reference and the strongest pilot competitors. A channel simulator
can be reused only after validating its dimensional, delay, and SNR conventions.
Do not merely increase the code length around the current n=2 implant.

Doppler/time-conditioned fading or MIMO is optional initially; select one
well-controlled extension. This addresses dimensionality without combining
several new difficulties in one experiment.

For measured-data claims, separately test a fixed dataset with one output per
input, no fresh positive sampling during training, and held-out evaluation
against an analytic reference. A local conditional estimator must quantify
neighborhood bias. It remains a sparse-data proof of concept rather than a
measured-channel validation. Otherwise remove the untested approximation from
the main method and retain a narrowly scoped limitation/future-work paragraph.

### Stage 4: confirmatory runs and writing

Freeze protocol after pilots, use fresh evaluation seeds, and choose final
seed/trial counts from observed variability and target precision. Thirty
training seeds may be sensible, but 100 everywhere is not required by these
reviews. Pair common evaluation draws where useful, while remembering that
the same integer training seed does not synchronize architectures' RNG streams.
Retain error counts and use longer evaluations where rare errors dominate.

Write around three questions: does condition preservation improve a controlled
one-shot model; does that help relative to modern fast generators; and does
the result persist for a larger structured channel? Keep TurboAE as additional
downstream evidence after correcting its protocol description and controls.

## Journal decision

Policies checked on 8 October 2026.

- **TCOM:** one resubmission after rejection is allowed. Disclose the previous
  paper and supply major changes plus a point-by-point response. A manuscript
  rejected by two different journals is ineligible. The guidelines distinguish
  13-page submissions and 16-page revisions; do not assume a resubmission gets
  the latter allowance. The letter is a rejection, not an invitation to revise.
  [Official TCOM policy](https://www.comsoc.org/publications/journals/ieee-tcom/policies-and-guidelines)
- **TMLCN:** closest alternative in scientific scope, explicitly covering
  ML for channel modeling and reproducible practical studies. This is a fit
  judgment, not a prediction of easier acceptance. The checked guidelines do
  not state a specific prior-rejection limit; disclose the history and check
  eligibility through the portal/editorial office before submission.
  [Scope](https://www.comsoc.org/publications/journals/ieee-tmlcn),
  [guidelines](https://www.comsoc.org/publications/journals/ieee-tmlcn/policies-guidelines)
- **OJ-COMS:** a broader option for a substantial experimental contribution.
  Its policy allows a paper rejected elsewhere only if it has been rejected
  once, with disclosure and an explanation of changes. A second failed attempt
  may therefore remove this route. It should not be assumed an unrestricted
  fallback after trying TCOM again.
  [Official OJ-COMS policy](https://www.comsoc.org/publications/journals/ieee-ojcoms/policies-guidelines)

Recommendation: preserve the TCOM option while completing Stages 1--2. If the
controlled results and larger-channel experiment answer the central objections,
a TCOM resubmission is reasonable. If the strongest contribution remains a
focused ML method and empirical study, prefer TMLCN. Discuss OJ-COMS as an
alternative before accumulating another rejection. All routes need the same
core correctness fixes. The accepted GLOBECOM paper should now be cited and
the extension explained in the manuscript and submission disclosures.

No manuscript edits, training runs, submissions, or external messages were
performed during this audit. The audit script and evidence file were checked
against the saved records. GPU execution and a new mathematical convergence
proof remain outside this completed audit.

## Full review reports

Received: 8 October 2026. Source: decision letter and reviews pasted by Rick
Fritschek in this conversation. Wording is preserved with normalized typography
and Markdown math formatting. This is not an independent editorial-system download.

### Editorial decision

08-Oct-2026

Dear Dr. Fritschek,

Based on the reviews of qualified reviewers, which you will find below or attached, and my own reading of your manuscript, I will not be able to recommend your paper entitled "Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation," paper number TCOM-TPS-26-1722 for publication in the IEEE Transactions on Communications.

Based on my reading and all three reviewer reports, the technical concerns raised are foundational rather than incremental. I therefore recommend rejection.

1. The main contribution and performance advantage remain unclear. The theoretical justification is incomplete relative to the implemented method.

2. The comparative evaluation is not sufficiently fair. Baselines differ substantially in capacity, training budget, and schedule, and modern one-step or few-step baselines are missing.

3. The experimental validation is narrow and some reported results are inconsistent in some Tables and Figures.

I would like to thank you for considering our journal as a means of publication of your work. We hope you will consider us again with regard to your future technical contributions.

Prof. Wankai Tang

Editor, IEEE Transactions on Communications

tangwk@seu.edu.cn

Bcc: Reviewers

### Reviewer 1

Comments to the Author

1. The description of Table I is insufficient. It is not clear how the proposed Sinkhorn-based update differs conceptually or algorithmically from conventional diffusion-based sampling. The algorithm is presented largely as a sequence of procedural steps, without enough explanation of the underlying design choices, the role of each transport term, or why this particular update rule is appropriate. As written, the presentation does not convey a sufficiently developed algorithmic rationale.

2. The SER results do not demonstrate a clear advantage of condition-wise Sinkhorn drifting. In Fig. 2, the proposed method is not consistently superior to the competing learned channel models and, on several channels, diffusion remains noticeably stronger. Therefore, it is difficult to determine in what sense the proposed method achieves a meaningful performance advantage beyond reduced inference latency.

3. The central claim of the paper and the empirical evidence supporting it are not sufficiently clear. It is unclear whether the main contribution is improved conditional-distribution fidelity, better downstream communication performance, lower inference latency, or a particular accuracy-latency tradeoff. The figures and tables do not provide a single, coherent empirical narrative that clearly validates the paper's primary claim.

4. The baseline comparison is not sufficiently fair. The authors explicitly acknowledge that the comparison is not a fair one in terms of model capacity. However, this makes it difficult to assess the central practical claim regarding the accuracy-latency tradeoff. The drifting generator has approximately 20k parameters, whereas the diffusion models contain approximately 60k-75k parameters, and the WGAN generators range from approximately 19k to 72k parameters depending on the channel. In addition, the training epochs, learning rates, and data budgets differ substantially across model families. Consequently, the reported performance differences cannot be attributed solely to the proposed generation mechanism.

5. More importantly, the experimental study does not include recent one-step or few-step generative baselines. Since the primary motivation is to avoid the iterative sampling cost of diffusion models, the proposed method should be compared against modern low-step alternatives, such as consistency models, diffusion distillation, conditional flow matching or rectified flow, and other one-step conditional generators. Comparisons only against DDPM, DDIM, and WGAN are insufficient to establish the competitiveness of the proposed approach.

6. The training-time comparison in Table V is difficult to interpret. For AWGN, diffusion training is reported to require 0.020 hours, whereas direct drifting requires 1.299 hours and condition-wise Sinkhorn requires only 0.048 hours. Moreover, joint Sinkhorn and condition-wise Sinkhorn use the same generator architecture, yet their reported training times differ by approximately a factor of 19, from 0.917 hours to 0.048 hours. The authors attribute this difference to the computational complexity of global versus multiple small condition-wise transport problems, but the magnitude of the gap requires more careful explanation and profiling. In addition, the reported training times are extrapolated from runs that cover only 2% of the full training procedure, rather than being measured during complete training runs.

7. The channel benchmarks are too small and simplistic to demonstrate practical relevance. Most experiments use channel dimensions of only 7 or 8. Even the TDL experiment is based on only four complex symbols and uses a compact circular-convolution approximation. These settings do not adequately represent the difficulty of realistic learned channel simulation.

### Reviewer 2

Comments to the Author

This paper proposes condition-wise Sinkhorn drifting for one-shot learned channel simulation. The reviewer has the following comments:

1. The proposed condition-wise Sinkhorn update relies on drawing multiple repeated output samples y for the exact same transmitted symbol x. While this is feasible for synthetic channel simulators, real-world measured datasets typically provide only a single observation per condition. Although a local kernel approximation is briefly formulated in Section II-D, it is not evaluated in the experiments, leaving the method's practical applicability to measured channel data unverified.

2. (Cont.) The authors should either empirically validate this setting or appropriately moderate the claims across the abstract, introduction, and conclusions. At the very least, the authors should provide an intuitive, plain-language explanation of how this single-observation limitation can be practically addressed and resolved in real-world scenarios.

3. The practical condition-wise update is explicitly not gradient descent on the conditional Sinkhorn objective. The authors are recommended to clarify which claims follow from the population analysis and which are supported only empirically.

4. The DDPM/DDIM/WGAN comparisons use different network capacities, training schedules, and budgets. A matched-compute or matched-capacity comparison is needed to support an across-family performance claim; otherwise, the conclusions should focus on the controlled Sinkhorn ablation.

5. The latency advantage of the proposed one-shot generator is primarily demonstrated against multi-step diffusion samplers. To provide a more comprehensive assessment of fast channel simulation, the authors are recommended to compare or discuss against other modern fast/one-step generative paradigms, such as Consistency Models or Flow Matching / Rectified Flows.

### Reviewer 3

Comments to the Author

This manuscript proposes condition-wise Sinkhorn drifting for one-shot learned channel simulation. By constructing transport couplings separately for each transmitted input, the method aims to improve conditional distribution matching while retaining low inference cost. The topic is relevant, and the downstream communication experiments are valuable. However, the theoretical interpretation, training stability, comparative evaluation, and consistency of the reported results require substantial revision. There are some problems that need to be addressed.

1. Following Proposition 1, the manuscript states that matching a joint sample cloud "under an arbitrary joint cost" does not enforce conditional equality. Please distinguish approximate sample-cloud matching from exact equality of joint laws, which implies conditional equality almost everywhere under a common input marginal. Clarify whether the proposed advantage concerns transport geometry, finite-sample optimization, or identifiability.

2. Equations (18)-(19) establish the objective's unique zero and a dissipation identity, but do not alone prove convergence or exclude other stationary configurations. Please specify the applicable assumptions from W-Flow [16] and Sinkhorn-divergence theory [18], justify the equilibrium interpretation, and qualify the "best representable conditional surrogate" statement. Although population and implemented dynamics are distinguished, guarantees for the latter remain unclear.

3. Table VII reports SWD increasing from \(7.07\times10^{-3}\) to \(1.53\) and SER from \(9.33\times10^{-5}\) to \(2.70\times10^{-2}\) between the 4.69k- and 390.6k-update settings. Since dataset size and seed count also differ, please isolate the cause through controlled training trajectories. Specify the validation-based stopping rule and apply comparable checkpoint selection to joint Sinkhorn in the same downstream setting.

4. Section III-B(b) uses four generated, four positive, and four reference samples per anchor with ten Sinkhorn iterations. Please evaluate sensitivity to these sample counts, iteration count, entropic regularization, and drift scale; explicitly define "automatic entropic scale selection"; and assess whether the numerical couplings adequately approximate the intended transport problems.

5. Please specify the sampling distribution and power normalization of \(\mu\) in (13)-(14), and assess whether training anchors cover the codewords encountered as the downstream encoder evolves. Evaluate conditional accuracy at learned codewords and compare surrogate versus analytic expected-loss gradients to help explain downstream performance gaps.

6. Table V reports DDIM-10/20/50 timing, but Fig. 2 and Table IX compare downstream performance only against DDIM-100. Please include reduced-step DDIM, report SER against actual training time, and consider a representative one-step consistency or distillation baseline from [12]-[13] to substantiate the speed-accuracy positioning.

7. For AWGN DDIM-100, Table V reports 0.020 training hours and 0.039 ms/sample. With the \(10^6\) evaluation samples in Table II, the total is approximately 0.031 hours, rather than 0.128 hours. Please reconcile sample budgets, units, and calculations throughout the table, and distinguish batch-amortized time per sample from individual-call latency.

8. Please explain how the \(n=7\) AWGN surrogate is used in the block-length-64, rate-\(1/2\) TurboAE experiment in Section IV-E: retraining, chunking, or another adaptation. Specify the input/output mapping, boundary handling, latent sampling, and noise calibration at 4 dB, and verify that the implanted channel preserves AWGN independence and variance.

9. Table VI includes only the two Sinkhorn variants. Please evaluate direct drifting, WGAN, and diffusion using the same anchor protocol, define Gaussian Wasserstein-2 mathematically, and report uncertainty and analytic-versus-analytic floors. Clarify whether Table VI uses the anchor/sample settings described for Fig. 1, and extend the SSPA floor reporting in Table VII to the other channels.

10. Section IV-C reports SSPA SERs of \(8.13\times10^{-5}\) and \(1.57\times10^{-5}\) for condition-wise Sinkhorn and analytic training, whereas Table VIII gives \(9.33\times10^{-5}\) and \(1.30\times10^{-5}\). Please reconcile these values. Although mixed SD/SE reporting and floor-clipped upper bounds are disclosed, provide directly comparable uncertainty measures, downstream trial counts, and the floor/upper-bound calculation, and distinguish numerical rankings from statistically supported improvements in Table IX.
