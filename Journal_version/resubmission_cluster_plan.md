# Cluster work specification

Companion to [the resubmission handover](resubmission_handover.md). The cluster
designs below remain **proposed**. The separate [local gradient pilot](gradient_fidelity_local_pilot_20261008.md)
has run on a GPU and partially implements P3/T1/T2. Its existing runner is
`scripts/run_local_gradient_fidelity.py`; newly named revision scripts below
still require implementation. Do not confuse those statuses.

## 1. Freeze the experimental contract first

### Inputs, randomness, and selection

1. Give each channel configuration an immutable ID containing dimension, power convention, noise convention, physical parameters, conditioning variables, and sampling law. Keep historical and corrected TDL noise conventions under different IDs.
2. For the initial AWGN/SSPA replication, preserve the legacy Gaussian anchor law exactly. For the controlled comparison, use one shared sampler across every family. Remove WGAN-only minibatch rescaling. A per-coordinate variance, per-complex-symbol power, and per-codeword power are different conventions: serialize the chosen one.
3. Assign independent random streams for model initialization, anchors, channel noise/state, generator latent variables, metric projections, validation, and final test. Give downstream AE initialization and training their own streams. A shared integer seed does not ensure common draws across different architectures.
4. Development seeds: `9001,9002,9003`; optionally `9004,9005` to resolve pilot uncertainty. Confirmatory generator seeds: `9101`–`9110`, with three separately seeded AEs per generator. Verify these IDs are unused before launch. Do not include development runs in final uncertainty estimates.
5. Generator selection: minimum mean conditional SWD over a fixed validation panel, earliest checkpoint on an exact tie. Use moment/covariance/gradient diagnostics to explain behavior, not as alternative post-hoc selectors. Pilot changes to this rule must be frozen before confirmation. Give methods the same number of candidate checkpoints within a comparison track.
6. Start validation with 128 anchors × 128 outputs each and 64 projection directions. Final assessment starts with 256 anchors × 512 outputs and 128 directions. Increase samples based on pilot precision/dimension, then freeze them. Retain two independent analytic clouds at each anchor to measure estimator floors. Use distinct anchor, output, and projection streams for test.
7. Diagnose Gaussian training-input radii and separately evaluate actual learned codewords. For high dimension use suitable covariance sample sizes; do not interpret an underdetermined covariance estimate as full-law validation.
8. Always retain both the validation-selected checkpoint and the final checkpoint. Stability experiments continue beyond the selected checkpoint; checkpoint selection does not mean the dynamics converged.

**AE selection:** choose either final checkpoint at fixed update/time budget (default main comparison), or the best checkpoint on a separately seeded analytic validation channel with equal evaluation allowances. If analytic validation is used, disclose and count those channel queries. Neither route may use final test errors for selection. The old runner's `eval_implant` is a validation mechanism when it selects epochs; rename the concept in outputs.

### What “matched” means

Use separate comparisons; a single equality constraint cannot answer every fairness question.

| Track | Held fixed | Allowed to vary / must report |
| --- | --- | --- |
| Geometry ablation | Same generator, initialization, optimizer, anchor cloud, generated/positive/reference counts, common epsilon, selection rule, updates | Allowed couplings and resulting drift; runtime/memory differ |
| Capacity and data access | Similar inference parameter count, identical channel/input law, same oracle-sample bank or oracle-call cap, validation/tuning allowance | Appropriate family training objectives and optimizer schedules; critic/teacher cost and reused examples reported |
| Practical compute | Same target GPU, full generator-training time cap, validation allowance and access model | Updates/data exposure may differ; report both, and all teacher cost |
| Downstream equal updates | Same AE architecture, initialization protocol, messages, power convention, optimizer, update count | Implant cost and gradient; elapsed time differs |
| Downstream equal time | Same AE and actual training-time cap | Completed updates differ; use actual time and identical final-test protocol |

For low-dimensional cross-family tests, use a primary inference-capacity target around 60k parameters, allowing a declared ±10% tolerance; record exact counts including all mean/covariance heads. Add a smaller approximately 20k tier for the most informative methods if capacity sensitivity changes conclusions. These are development targets, not claims of functional equivalence. Use exact same backbone for joint/fiber comparisons. Additionally break out head parameters and report critic/teacher parameters and EMA storage as training costs where applicable.

Tuning allowance: start with at most six configurations per method/channel/capacity tier, each evaluated on the same development seeds and capped compute. Publish search choices and costs. Specialized SSPA mechanism sweeps are an additional development expense for the proposed method; report that expense rather than pretending overall tuning was identical. Give the selected competing methods a comparable bounded tuning opportunity before freezing final claims.

## 2. P0 — restore evidence and fix the measurement interfaces

Check the following referenced archives/checkpoints (paths relative to repo),
restoring only missing inputs on the execution machine:

- `results/timing_suite_local_cuda_20260601_1755/`
- `results/timing_suite_wflow_20260601_1715/`
- `results/journal_wflow_fiber_fixed_paper_hpc_20260601_161729/`
- `results/journal_wflow_sspa_budget_screen_20260602_071106/`
- `results/journal_wflow_ser_sspa_budget_screen_20260602_073353/`
- `results/journal_wflow_curves_20260602_220608/`
- `results/turboae_long_block_hpc_20260529_165218/`
- Baseline/other suites identified by the retained figure and table export manifests.

On 9 October the local audit completed and reproduced the retained evidence
JSON exactly. These inputs are therefore available here; another checkout may
lack the git-ignored results. Run `scripts/audit_journal_review_evidence.py`
after checking its inputs. Success verifies saved records, not reproducibility
of training. Keep unknown source versions explicit; a saved model config does
not prove which source commit produced it.

Implementation tasks:

| Existing path | Required change |
| --- | --- |
| `conditional_drifting/training.py` | Exact-step limits/checkpoints, callbacks, RNG/optimizer resume, independent validation RNG, counters and timers; shared input sampler |
| `conditional_drifting/losses.py` | Optional transport diagnostic return; explicit epsilon policy; stable numerical reference, retaining named legacy behavior |
| Config dataclasses in `conditional_drifting/training.py`, `baselines/diffusion.py`, `baselines/paper_wgan.py`, and generator definitions in `conditional_drifting/model.py` | Serialize all numerical settings and expose hidden/latent dimension in revision interfaces; a central revision-config module would be new |
| `conditional_drifting/symbolic_ae.py` | Integer errors/trials; explicit validation/test; fixed normalization semantics; reusable learned-codeword evaluator |
| `conditional_drifting/e2e_implants.py` | Consistent SNR/decoder-noise handling; new baseline adapters with differentiable input path |
| `scripts/run_inference_timing_benchmark.py` | Separate full training measurement, forward throughput, batch-one latency, forward/backward cost, evaluation sample count |
| Existing aggregates and artifact exporters | Canonical result IDs; no relabeling low-step samples as DDIM-100; no floor-clipping masquerading as statistical bounds |
| `hpc/check_env.sh` | Replace final full-budget runner with bounded smoke command below |

Proposed revision entry points: `scripts/run_journal_revision_trajectory.py`, `scripts/evaluate_journal_revision_checkpoints.py`, `scripts/run_journal_revision_task.py`, and `scripts/aggregate_journal_revision.py`. These names are an implementation specification, **not existing runnable programs**. One JSON manifest row should describe one task; no auto-discovery of “latest” checkpoint for confirmatory runs.

P0 completion checks: a resumed tiny run matches an uninterrupted run within the declared deterministic tolerance; validation leaves training RNG unchanged; count totals match the actual loop; mismatched config/checkpoint hashes cause an error; aggregates refuse incomplete/duplicate tasks. Test those scientific contracts, rather than writing tests that merely duplicate the code.

## 3. P1 — numerical reference, SSPA trajectories, and coupling geometry

### P1a: small numerical reference

Define `epsilon_policy` with distinct values `fixed_common`, `shared_adaptive`, `legacy_separate_adaptive`. The existing explicit `--sinkhorn-epsilon` provides fixed common epsilon; shared adaptive behavior requires a code change. Log actual cross/self epsilon even if equal.

Use quadratic cost `0.5 * ||u-v||^2`, explicit uniform masses, and a high-accuracy float64 log-domain Sinkhorn reference. Stop that reference by both marginal residuals, not only an iteration count. Proposed small-problem tolerance: maximum relative row/column marginal error `1e-8`, with a hard iteration cap that fails visibly. Treat this as an engineering tolerance, not a theorem.

Compare the practical solver against it on unequal cloud sizes, near-duplicate particles, concentrated and broad Gaussians, small epsilon, identical clouds, and the collapsed symmetric example from the manuscript plan. Save barycentric errors and mass residuals **before** final row normalization. A row-normalized matrix need not satisfy the target marginal.

For a gradient check only, use compatible exact same-batch empirical self-transport, fixed epsilon, no clipping, and the full derivative of the debiased empirical objective. Verify centered finite differences versus the detached target parameter gradient, including MSE scaling. Then separately compare independent-reference behavior. Do not label either finite-batch estimator an unbiased population gradient without proving that claim.

Shared adaptive epsilon: a proposed pooled within-anchor cost statistic applied once to both solves; freeze the statistic definition in config. Fixed epsilon: calibrate a positive scale on training-only pilot clouds, then freeze it. Neither rule is automatically an exact gradient of a fixed objective if epsilon changes with the model.

### P1b: reproduce the SSPA issue along continuous trajectories

First run the legacy corrected fiber policy with `B=4096`, `K_g=K_p=K_r=4`, learning rate `1e-3`, drift scale `1`, drift-norm cap `2`, the original Gaussian anchor law, and the documented SSPA channel. Keep fresh-data sampling and optimizer settings fixed.

Save checkpoints at updates:

```text
0, 100, 300, 1000, 2500, 4800, 10000, 30000, 100000, 390720
```

These are points on a **single** run per seed, not ten independently restarted runs. Validate on the fixed schedule with a separate RNG. Three full legacy trajectories establish whether the historical deterioration can be reproduced with known current code. Missing historical source provenance remains a limitation even if endpoints look similar.

Compare legacy, fixed-common, and shared-adaptive epsilon first; initial cap 30k updates for screens, then extend the preselected policies to 390720 for the controlled long-run check. A numerical-failure stop (NaN/Inf, impossible coupling, unrecoverable OOM) must record failure, not silently discard the seed. Save last valid checkpoint. A poor but finite validation score is a scientific outcome, not permission to erase the run.

Do not call a compact run “early stopping” until a stopping rule was actually applied. The default selection rule here chooses a saved checkpoint; report both total cost of exploring the trajectory and cost to that selected checkpoint.

### P1c: decisive geometry comparison

Construct one repeated-anchor batch: `B` distinct inputs, `K_g` generated outputs, `K_p` true outputs, and `K_r` independently generated references for each input. Give both algorithms exactly these particles. Fiber transport constrains each coupling to its anchor. Joint transport flattens the same particles and uses a specified condition/output cost, with condition scale tuned over a declared small development grid. In this comparison preserve a common epsilon, reference rule, network, optimizer, and update count.

Start at `B=128`, `K_g=K_p=K_r=4`, giving 512 rows in the flattened joint solve. Pilot feasibility before increasing `B`. Do not compare this small-batch trajectory to the legacy B=4096 experiment as if only geometry differed.

Complexity for these deliberately matched clouds is approximately `O(I B K_g (K_p+K_r))` versus `O(I B^2 K_g (K_p+K_r))`, excluding distances/network cost. The old implemented joint branch instead uses B particles each and has approximately `O(I B^2)` coupling work. Report the actually run dimensions. A 16384-by-16384 float32 matrix already occupies 1 GiB before other matrices/intermediates; expanded joint training at legacy B=4096 is not a safe default.

Run the selected fiber and joint policies with the same validation checkpoint allowance, including SSPA with message alphabet 64 downstream. The current journal table has no joint-Sinkhorn row in that exact SSPA setup. Do not claim that existing table establishes this comparison.

### P1d: staged sensitivity

After epsilon-policy screening, vary one group at a time around the selected development configuration:

| Factor | Proposed values | Control |
| --- | --- | --- |
| Generated/positive/reference samples | Common K = 4, 8, 16 | Report fixed-B cost; add fixed-BK comparison to separate particles per condition from total sample exposure |
| Asymmetric clouds | (K_g,K_p,K_r) = (4,8,4), (4,4,8) | Tests which extra samples help, rather than changing all counts together |
| Solver iterations | 10, 30, 100 | Report residuals; more iterations are not assumed to fix statistical error |
| Common epsilon | 0.5, 1, 2 times frozen calibration scale | Same definition/units and training data law |
| Drift scale | 0.25, 0.5, 1 | Keep optimizer LR fixed initially; the two scales are not interchangeable under clipping/Adam |

No full Cartesian sweep. Use three development seeds, then confirm the selected policy and the key mechanism contrast on fresh seeds. If interactions are suggested, test the particular interaction explicitly and disclose it as development.

Required logs: raw/clipped drift norms, clipping fraction, both epsilons, coupling residuals, kernel-floor activation, generated mean/covariance/eigenvalues, output norm, latent sensitivity, conditional SWD/GW2 and analytic floors, actual updates/channel queries, elapsed training/validation seconds, and peak device memory. Do not infer a collapse mechanism from the training loss alone.

Logging cadence: update lightweight counters every step, record scalar summaries every 100 steps, and run expensive covariance/eigenvalue, latent-sensitivity, and high-accuracy-reference diagnostics only at scheduled validation checkpoints. Accumulate detached counters without forcing a device synchronization every step where possible. Measure diagnostic overhead separately; it must not silently dominate or distort the reported training time.

## 4. P2 — fast competitors and proper training costs

Minimum competitive set on AWGN and SSPA:

| Model | Role / implementation requirement |
| --- | --- |
| Analytic simulator | Calibration and downstream training reference; not a universal lower bound |
| Direct/kernel drifting | Conference continuity and one-shot control; preserve precise variant identity |
| Joint Sinkhorn | Geometry ablation, not a substitute for the existing conditional W-Flow method |
| Conditional Sinkhorn with shared fixed epsilon | Faithful conditional W-Flow-style reference in channel coordinates |
| Selected practical condition-wise variant | If identical to the previous row, merge them; if changed, state the exact novel change and ablate it |
| Conditional Gaussian | Train mean/diagonal log-variance by Gaussian NLL; draw `m(x)+s(x)*z`; account for variance floor. AWGN and additive-noise SSPA have Gaussian conditional laws, making this an essential cheap control |
| DDIM 10, 20, 50, 100 | Reuse each trained diffusion teacher; each sampler gets its own fidelity evaluation and freshly trained AE |
| Conditional flow matching | Train a conditional time-dependent velocity; report an NFE curve, e.g. 1, 2, 4, 8, 16 Euler evaluations, with solver identified |
| Trained one-step consistency/distillation | A real one-step learning algorithm from a verified reference, not simply a one-step Euler sample labeled “consistency” |
| WGAN | Retain contextual baseline; harmonize input law and count its five critic updates and training-only parameters |

Use [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) for a conditional flow baseline and [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) plus its [author implementation](https://github.com/openai/consistency_models) for one-step design/validation. Conditional *probability paths* in flow matching and conditioning on channel input are distinct concepts; the model must explicitly take channel `x` as fixed side information.

For the consistency comparator, choose and freeze one established consistency-training or consistency-distillation protocol during development. If adapting an existing VP/v-prediction diffusion teacher, explicitly derive the teacher-to-student parameterization/ODE conversion and verify it; the original EDM-style consistency implementation is not a drop-in wrapper for this repository's teacher. A separately trained compatible teacher is acceptable if its cost is counted. Validate mean/variance and sampling quality on a simple Gaussian before claiming a competitive baseline. Pin reference commit/version and disclose any adaptations.

A one-step Euler flow can be a useful point on the flow curve but does not replace the trained one-step comparator. Conversely, do not reject an established baseline after one unsuccessful untuned run; check its implementation and spend the declared development allowance.

Measure full generator training, including data generation, losses/transport, forward/backward and optimizer. Separately report validation, serialization, preprocessing, and tuning. For distilled methods show both teacher-already-available and teacher-plus-student cost; the latter is the default from-scratch comparison. Reusing one teacher for several DDIM samplers does not imply four separate training costs.

For inference measure:

- Batch-one call latency, with warmup and synchronized individual measurements; median and spread.
- Batch-amortized throughput at batch 512 and actual downstream batch size.
- Input-gradient forward/backward time through each frozen implant, because this drives AE training cost.
- Latent/input generation included or excluded consistently, actual NFE, dtype/device, repeats, warmup, peak memory, and raw elapsed measurements.

Use one unit per column. Total simulator cost for N calls is `T_train + N*t_call` only under the measured batching/workload assumptions. A break-even number is meaningful only for comparable accepted quality and positive timing differences; do not claim saved compute by comparing unequal fidelity without stating the tolerance.

## 5. P3 — learned-codeword, gradient, and downstream experiments

### Completed development pilot and remaining controls

The separate [decoder-free metric pilot](channel_feature_metric_pilot_20261009.md)
has now run using `scripts/run_channel_feature_metric_pilot.py` and
`conditional_drifting/feature_metrics.py`. It covers full-kernel MMD and
embedding derivatives, moment derivatives and SWD, with 128/512 samples,
three perturbation sizes and both shared/independent noise. Reuse these
implementations and measured sampling floors. The subsequent
[resolution study](metric_resolution_results_20261009.md) completed smaller-step
checks against pathwise empirical derivatives, fixed raw-moment augmentation,
and SSPA seed-8/9 checks against frozen-codec gradients. It resolves numerical
step bias and detects the large-variance failure, but still misses some
task-gradient orderings. The [exact-law and matched-value controls](metric_controls_results_20261009.md)
then confirmed reparameterization-dependent estimator variance and useful
population derivative information at identical inputs. The Gaussian control
also admits a simple mean/Jacobian solution. Keep the selection-value gate
open. Next separate candidate-specific estimation noise and population
error on a small same-input learned-model check; passive-data sampling,
heavy-tail robustness and useful selection beyond simple controls remain open.

The [local report](gradient_fidelity_local_pilot_20261008.md) covers seed 7 on
AWGN/SSPA, two frozen codecs per channel, analytic gradient finite differences,
codeword SWD, expected input/encoder gradients, independent Monte Carlo floors,
and equal-norm single encoder steps evaluated on the analytic channel. Six
focused tests pass. The higher-sample SSPA run uses four repeats of 8,192
surrogate draws per codeword and 65,536 analytic-reference draws per codeword.
These are existing unequal-budget generator checkpoints, not the controlled
P2 comparison. Treat them as development cases.

Reuse this runner before writing a second gradient estimator. Missing pieces
are three development generator seeds, common early/middle/late reference
codecs, Gaussian anchors and input neighborhoods, a declared reference-noise
threshold, surrogate as well as analytic finite differences at multiple step
sizes, and richer per-message/absolute-error and gradient-variance exports.
The existing JSON and tensor files retain aggregate and repeat-level data.

Add common-learning-rate SGD interventions alongside the current equal-norm
steps. The former test direction and magnitude together; the latter isolate
direction. Choose step sizes using an analytic-only development calibration,
not separately to make each surrogate improve. Keep actual Adam training as
a separate endpoint. No receiver-only, encoder-only multi-step, or joint AE
training intervention has been completed by this local pilot.

For the pre-optimization score, keep all decoder losses out of score
construction. Compare feature-value error alone against value plus derivative
error, ordinary moments against moments plus their derivatives, and SWD/MMD
references. Resolve exact-channel sampling floors and finite-difference bias
before ranking models. Freeze the design before unused seeds or model families
are evaluated. A failure to improve screening utility is a valid result; do
not replace the current selector merely because a new score has a theorem.

The [theory follow-up, Sections 10–11](theory_swd_downstream_gradient_fidelity.md#10-concrete-p3-addendum-test-mechanism-before-scaling) adds a staged mechanism study: common-checkpoint diagnostics, receiver-only versus encoder-only controls, and matched one-step gradient interventions. Run these on development seeds before expanding the confirmatory study. They supplement the experiment plan below; they do not change its validation selector or make analytic gradient access free.

The author's subsequent objective is a metric computed **before candidate-specific downstream optimization**. The [metric proposal, Section 8](theory_preoptimization_channel_metric.md#8-practical-pilot-and-changes-to-the-handover) and completed local pilot use conditional feature values and input derivatives. Keep trained decoders out of score computation; use future unused model/task cases to validate predictive value after freezing the score. Existing seed-7 codec results are development evidence, not held-out validation. The candidate's task-class bounds and present measurements do not establish a practical BER predictor, so retain the existing primary selection protocol.

Reuse `evaluate_implant_conditional_metrics` in `conditional_drifting/symbolic_ae.py`; it already evaluates learned codewords and analytic floors. Add a common saved-checkpoint adapter for all models instead of reimplementing the metrics in each runner.

Fix codeword semantics first. Current minibatch standardization makes the transmitted vector for one message depend on other messages in the batch. For the revised symbolic study, use full-codebook normalization: compute the complete M-message codebook, center/scale by its declared average energy, then index messages. Apply this identically to all implants and rerun affected results. Record that this changes the legacy AE protocol. For any retained legacy diagnostics, preserve and report the batch context.

### Expected-loss gradients

At early, middle, and final checkpoints of a predeclared analytic-trained reference AE, freeze the same decoder and evaluate

\[
\nabla_x\mathbb E[\ell(D(Y),m)]
\]

under analytic and learned channels at the **same** codeword/message. Add method-trained AE checkpoints as a separate diagnostic, not a confounded replacement for the common decoder comparison.

Estimate expected gradients from independent Monte Carlo replicates. Start at 4096 samples per selected codeword and four replicates; calibrate against analytic-versus-analytic gradient variability on development data. Report absolute error, norm error, and cosine similarity only when the reference norm exceeds a declared noise threshold. Raw samplewise Jacobians are not comparable across arbitrary latent parameterizations. Freeze implant/decoder parameters while retaining differentiation with respect to x; do not place the whole evaluation in `no_grad()`.

Check a subset against centered finite differences with controlled random numbers. Report whether the gradient through power normalization is included; use the same definition for analytic and surrogate paths. Small conditional distribution distance alone does not guarantee gradient accuracy (see the manuscript plan's counterexample).

### Downstream training

- Equal-update experiment: same AE settings, number of updates, initialization protocol, and message streams. Record time.
- Equal-time experiment: actual cumulative **training** times 60, 300, 900, and 1800 seconds as starting budgets; save checkpoints on crossing each threshold and record actual elapsed time. Pause training timers for separate validation and report total wall time too. Slow methods may overshoot by one update; retain the exact elapsed values and do not label them exact 1800 seconds.
- Evaluate all selected checkpoints on a fixed final analytic test protocol only after selection is frozen. Main headline at 1800 seconds; time curves explain convergence. Use separate test streams across SNRs or record pairing explicitly.
- Train new AEs for DDIM-10/20/50 and every other sampler. Testing a DDIM-100-trained AE with a faster channel generator does not answer the requested training question.
- Evaluate receiver metrics on the real/analytic target channel, not the learned training implant. Keep analytic-trained reference models under the same optimization budget.

### TurboAE repair

The historical experiment trains a separate n=2 AWGN surrogate and maps `[batch,64,2] → [batch*64,2] → [batch,64,2]`, with independent latent draws per pair. It has 64 information bits and 128 real transmitted coordinates, at rate 1/2. No n=7 padding/chunking is involved.

Start with a clean **fixed-SNR** comparison at Eb/N0=4 dB, with both encoder and decoder training at the same noise level for analytic and learned implants. Do not retain the old analytic decoder SNR offsets while the surrogate ignores them. A later variable-SNR comparison needs explicitly SNR-conditioned training for every learned model, or a justified channel-specific noise transformation; it is a separate protocol.

Check residual conditional mean, per-coordinate variance, within-pair and cross-pair covariance, dependence on input amplitude, and Eb/N0/rate mapping. Under the repository's real AWGN convention, rate 1/2 and 4 dB give noise standard deviation about 0.6309573. Validate rather than assuming pairwise noise independence from independent latent vectors. Recover old epsilon implementation provenance or rerun with a known version. Retain the old BER gap as historical evidence, not as proof of an isolated surrogate-fidelity effect.

## 6. P4 — one larger jointly generated wireless block

**Required for the preferred strengthened journal package.** Use the explicit five-tap configuration below as an engineering pilot, then replace it by a validated standard-profile configuration for the main journal result. The pilot alone is not full 3GPP validation.

- Single-antenna OFDM, FFT size 64, subcarrier spacing 30 kHz, cyclic prefix 16 samples, 8 fixed pilot subcarriers and 56 QPSK data subcarriers. Sampling rate is 1.92 MHz and the CP duration is about 8.33 microseconds.
- Unit-average-energy constellation and a unitary FFT/IFFT; deterministic declared codebook power normalization. Record pilot positions/values.
- A block-constant frequency-selective channel: five taps with sample delays `[0,1,3,7,12]`, powers proportional to `[0,-3,-6,-9,-12]` dB and normalized to sum one; independent circular complex Gaussian tap gains with those powers, resampled per block. This is an explicit discrete multipath benchmark. Do not label it standardized TDL-D.
- Physically implement linear convolution on the CP-extended signal, then remove CP. Compare against the expected circular-convolution equivalent in a noiseless validation check. Delays fit within CP.
- Noise convention `n ~ CN(0,N0 I)`, with each real component variance `N0/2`. Use Es/N0 for the main protocol (suggested training point 10 dB; test 0:2:20 dB). Any Eb/N0 conversion must include data/pilot/CP overhead and coding rate.
- Generator input/output are whole 64-complex-sample useful time-domain blocks, or equivalently whole frequency-domain blocks under the declared unitary transform: **128 real coordinates generated jointly**. Draw one block latent vector with at least 128 independent coordinates. Make latent/backbone dimensions configurable and record them. A smooth latent-16 map cannot exactly represent full-dimensional 128-real Gaussian noise; Wasserstein approximation can still occur, so this is an expressivity warning, not a proof of poor finite-sample metrics.
- Hold channel state hidden from the surrogate/receiver unless state is explicitly added to the conditioning for every method. State-conditioned and marginalized channel laws are different targets.

**Main channel after pilot validation:** use the TR 38.901 scalable TDL-C power/delay profile, RMS delay spread 100 ns, carrier frequency 3.5 GHz, and zero intra-symbol Doppler for the first main comparison. Pin a supported simulator/specification version and verify its power/delay tables; do not silently use the original paper's cited edition with different library defaults. Official [Sionna TDL documentation](https://nvlabs.github.io/sionna/v2.1.0/phy/api/channel/wireless/sionna.phy.channel.tr38901.TDL.html) provides the profile and scaled-delay interface; the [ETSI TR 38.901 edition](https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/17.01.00_60/tr_138901v170100p.pdf) is a primary standards reference. Treat this as a standardized profile within a specified block-static OFDM experiment, not a claim to emulate all standard scenarios.

Retain the actual fractional path delays. Evaluate the frequency response at subcarrier frequencies or use a validated fractional-delay time-domain implementation; do not round all paths to a few circular shifts. Check that the profile support fits within the CP and benchmark empirical frequency covariance against the analytic sum over path powers/delays; [Sionna's frequency-covariance definition](https://nvlabs.github.io/sionna/v2.1.0/phy/api/ofdm/sionna.phy.ofdm.tdl_freq_cov_mat.html) is a useful reference. This replaces the five-tap pilot in the main confirmatory matrix rather than doubling the whole campaign. Mobility and a second delay spread can be later robustness checks.

The 0:2:20 dB grid is **downstream receiver testing on analytic channels** after nominal-point training. The nominal-point surrogate's fidelity is claimed only at 10 dB. Multi-SNR surrogate fidelity requires an explicitly SNR-conditioned model or separately trained surrogates, with extra training cells and costs.

Start with fixed QPSK and a common neural block receiver trained on generated data, tested on the analytic block channel. Include a pilot-based conventional receiver under the stated interpolation/equalization assumptions as context. Then add one differentiable-transmitter experiment, such as learned four-point constellation coordinates with fixed average energy and the same block receiver, trained through each implant. This preserves a manageable communication endpoint while testing the intended input-gradient use. Evaluate BER and whole-block error probability; define exactly which data bits constitute a block.

Training anchors should cover both generic continuous inputs and the actual OFDM waveforms. A proposed frozen mixture uses equal weights of complex Gaussian blocks and initial QPSK-OFDM blocks, with the same declared average energy. Normalize by fixed distribution/constellation energy, not by each randomly drawn block or minibatch; per-block normalization would define a different law. Separate evaluations on learned waveforms reveal support shift; do not silently retrain only the proposed method on its final test codewords.

Use a common block-capable architecture family (e.g. 1D residual convolutional backbone) and freeze capacity after memory/runtime pilots. Initial competitors: selected conditional Sinkhorn, joint geometry control where feasible, best flow model, trained one-step comparator, and learned Gaussian covariance surrogate. Report the Gaussian baseline honestly: the five-tap Gaussian pilot is exactly conditionally Gaussian, and a Gaussian covariance model is a strong reference for the block-static NLoS case. Verify the main simulator's tap law; a finite sum-of-sinusoids implementation is not automatically exactly Gaussian. If the Gaussian baseline wins, that constrains the need for a universal generator. A further non-Gaussian impairment or expensive black-box channel is justified only by an explicit physical/use-case question, not by a need to force the proposed model to win.

Required simulator checks: noiseless identity for one tap, CP boundary equivalence, expected tap power, real/complex noise variance, unitary-transform energy, correct SNR, shared tap realization across the entire block, and independent realizations across blocks. Required model diagnostics: per-anchor SWD, full cross-coordinate covariance, temporal/frequency correlation, and downstream BER/block error. Marginals alone cannot establish that memory was learned.

## 7. P6 — optional single-observation study

R2 allows a simulator-only scope. Prefer completing P0–P4 first. If retaining a claim about passive datasets, implement a **fixed** dataset with exactly one y per x, no hidden fresh positive channel calls during training, disjoint condition splits, and held-out simulator evaluation. Compare local weighted Sinkhorn, joint training, and the strongest ordinary conditional baseline under the same dataset budget. An oracle repeated-condition run is a separately labeled access upper reference, not a fair same-data comparator.

Report bandwidth/neighbor count, effective neighborhood sample size, condition-space dimension and coverage, held-out radius strata, and neighborhood bias. Matching a fixed-anchor generator to a smoothed target and matching two smoothed conditional mixtures are different finite-bandwidth objectives. A synthetic one-observation experiment demonstrates a sampling regime; it is not measured-channel validation. Repeating a waveform in hardware yields repeated draws of the same conditional law only under a justified state/stationarity model.

## 8. Confirmatory matrix and resource planning

Do not run 100 seeds for every cell. Default after development:

| Study | Default scale | Notes |
| --- | --- | --- |
| Legacy SSPA long trajectory | 3 development seeds × legacy policy; selected epsilon policies extended after screen | Mechanism exploration, not final uncertainty |
| Numerical/sensitivity screens | 3 development seeds per staged configuration | Avoid Cartesian product |
| Frozen geometry | 2 fields × AWGN/SSPA × 10 generator seeds = 40 models | Same repeated particles; can reuse eligible matched models elsewhere |
| Competitive low-dimensional models | About 6 generator families × 2 channels × 10 seeds = 120 trained models | Direct, selected Sinkhorn, Gaussian, diffusion teacher, flow, consistency; add WGAN/joint as separately counted controls; merge duplicated Sinkhorn reference |
| Downstream main panel | 7 selected implant/sampler operating points × 2 channels × 10 generator seeds × 3 AE seeds = 420 AE runs | Select operating points on development validation; add analytic AE references and equal-update runs separately |
| Larger structured block | 4–5 learned methods × 10 generator seeds, then 3 receiver/AE seeds each | Final method panel fixed from development, not main-test winners |
| TurboAE | Analytic and selected learned methods, 10 generator groups × 3 AE seeds | Reuse independent analytic references where pairing remains valid |

Ten generator seeds × three AEs are **ten independent surrogate groups**, not thirty independent generators. For primary robustness claims, extend the selected two-method/channel contrasts to 30 independent generator seeds if pilot precision requires it; reserve 100 for a specific unresolved high-variance or failure-rate question. Choose this precision rule from development estimates before observing confirmatory outcomes.

Budget arithmetic: 420 AE runs × 1800 training seconds = **210 GPU-hours of AE training alone**, plus analytic references, equal-update runs, validation/test, and all generator/teacher training. This is a planned scale, not a runtime prediction. An exploratory panel of seven implants × two channels × three seeds costs 21 GPU-hours of AE training alone. Fewer retained operating points reduce this linearly.

Existing SLURM templates commonly reserve one GPU, four CPUs and 32 GB host RAM; some have 36-hour or 84-hour limits. Those are limits, not evidence of actual duration. Profile each new method/channel on the target GPU; extrapolate only for reservation planning and label it as an estimate. Paper timing must use complete measured runs. Record GPU model and precision; avoid pooling timing across hardware.

Cluster execution design:

1. Create an immutable task manifest containing resolved configuration, run ID and parent checkpoint IDs. Print task count and expected resource estimate in a dry run.
2. Submit arrays with a stated concurrency cap and one manifest task per logical run. Use resource classes for small MLP, joint OT, teacher/distillation, and block models based on pilots.
3. Save model/optimizer/EMA/scheduler/RNG state and counters at scheduled steps and before time limits. Resumes must verify config/source compatibility and count all segments in elapsed cost.
4. Mark `complete`, `failed`, and `incomplete` explicitly. Never aggregate only convenient successful seeds without also reporting failures.
5. Aggregate only after dependency completion and schema/hash checks. Preserve raw records, selected checkpoints and the exact code snapshot needed to reproduce them.

## 9. Statistical and artifact requirements

Use seed-level 95% intervals consistently, with variability across trained generators distinguished from finite evaluation noise. Average the three AE outcomes within each generator before a generator-group bootstrap or appropriate group-level interval; report within-group variation separately. Paired differences require explicit pairing of random inputs/AE initialization, not merely identical seed labels. Use predeclared primary comparisons; label broad exploratory rankings as exploratory.

Store exact SER symbol errors/trials, BER bit errors/bit trials, and block errors/blocks. Bit errors within one block are dependent; use block-level resampling where needed. For iid message trials with zero symbol errors, the one-sided 95% upper bound is `1 - 0.05**(1/N)`, approximately `3/N`. A plot floor `1/N` is not a confidence bound. Across heterogeneous trained seeds do not pool all trials into a binomial model without specifying that different estimand.

Choose fixed final evaluation counts from development error probabilities and desired precision. About 100 expected independent errors gives roughly 20% relative 95% Monte Carlo precision under a small-p binomial approximation; this is a planning approximation, not a seed-uncertainty bound. Do not use an unreported “stop after enough errors” rule with ordinary fixed-N confidence intervals. A valid sequential procedure is a separate option.

Minimum per-run artifacts:

```text
manifest.json                 # source/config hashes, hardware, seeds, channel, parent/teacher IDs
train_metrics.jsonl           # exact steps, counters, timing, epsilon/drift/solver diagnostics
checkpoint_step<N>.pt         # complete resume state
selected_checkpoint.json      # selection rule, split, score, selected step and hash
conditional_metrics.csv       # per anchor/metric, raw model and analytic-floor values
gradient_metrics.csv          # fixed decoder/codeword IDs, gradient and MC uncertainty
downstream_metrics.csv        # generator/AE seeds, times, SNR, integer errors/trials
timing.json                   # raw repeats, workload, NFE, units, warmup/synchronization
paper_values.csv              # unique source row for every figure/table/prose value
```

Gaussian W2 must be reported as the implemented mean of per-anchor square roots, or deliberately changed and relabeled; do not confuse it with the square root of the mean squared value. Fixed-anchor direct-output and residual SWD differ only by a common translation when evaluated identically, so they are not independent evidence.

## 10. Existing commands that can be reused

These are examples for a configured GPU environment, not a full implementation of this protocol. Run from the repository root inside an appropriate allocation. Do not run the old full-budget suite merely to check the environment.

### Bounded smoke run with existing flags

```bash
python -u scripts/run_enhanced_direct_benchmark.py \
  --device cuda --seed 9001 --channels AWGN \
  --dataset-size 256 --batch-size 64 --drifting-epochs 1 \
  --eval-size 256 --swd-projections 8 \
  --drift-field fiber_sinkhorn --conditioning-mode none \
  --sinkhorn-epsilon 1.0 --sinkhorn-iterations 30 \
  --fiber-generated-samples 4 --fiber-positive-samples 4 \
  --fiber-reference-samples 4 \
  --anchor-metrics --anchor-count 8 --anchor-samples 16 \
  --anchor-swd-projections 8 \
  --save-dir results/revision_smoke/checkpoints \
  --out results/revision_smoke/summary.json
```

Epsilon 1.0 is a smoke setting, not a selected research hyperparameter. This runner lacks the new checkpoint/diagnostic/selection contract.

### Low-step downstream pilot after restoring checkpoints

Set `WFLOW_SUITE_DIR` and `BASELINE_SUITE_DIR` to verified suite roots containing the requested seed/channel. Then:

```bash
python -u scripts/run_equal_wallclock_symbolic_implant_seed_channel.py \
  --device cuda --seed 7 --channel SSPA \
  --variants analytic,fiber_sinkhorn,wgan,diffusion_ddim10,diffusion_ddim20,diffusion_ddim50,diffusion_ddim100 \
  --wflow-suite-dir "$WFLOW_SUITE_DIR" \
  --baseline-suite-dir "$BASELINE_SUITE_DIR" \
  --suite-dir results/revision_sspa_wallclock_pilot \
  --train-seconds 1800 --batch-size 500 \
  --eval-size 100000 --eval-batch-size 1000 \
  --save-checkpoints
```

This is seven separate 1800-second training runs (3.5 GPU-hours plus overhead), with final-point evaluation. It does not yet deliver the new time curves or full split/count protocol. Use this only as labeled legacy exploration until repaired.

### Existing compact-TDL dimension check

```bash
python scripts/validate_tdl_channel.py \
  --device cuda --symbols 64 --samples 10000 \
  --noise-std 0.0 --seed 9001 \
  --out results/revision_tdl64_calibration.json
```

This checks the existing effective compact tap profile, not the proposed OFDM system.

Interface traps to fix before new exports: `evaluate_journal_baseline_direct_swd.py` can retain the label `ddim100` even when another step count is requested; `--condition-codebook-checkpoint` in the enhanced runner is used for OptFib only; `submit_journal_wflow_fiber_fixed_suite.sh` refers to a historical fiber correction, not fixed epsilon, and defaults to a large run; existing path-based checkpoint reuse can silently mix configurations. Revision tasks must reject those ambiguities.
