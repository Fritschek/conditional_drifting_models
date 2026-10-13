# Experiment status for the next handoff

Updated 13 October 2026. This is the current status index. Earlier assessment documents
retain their historical recommendations; do not interpret them as an unexecuted
queue when the work is marked complete below.

## Coverage replication and sampler panel complete, 13 October

Read the [completed-results report](coverage_followup_results_20261013.md) first.
The full coverage replication and saved-checkpoint sampler panel completed on
local CUDA. Strict reports verify 12 generators, 45 AEs, 135 curve points and
six sampler settings. No simulation remains running. All 68 focused CPU tests
and seven new CUDA tests passed during preparation.
Both CUDA smoke runs resumed with byte-identical result JSON. Smoke coverage
aggregation verified four tiny generators, 15 AEs and 45 curve points.

The [frozen child protocol](coverage_followup_protocol_20261012.md) specifies
12 fresh coverage generators, 36 surrogate AEs and nine analytic references,
using generator seeds 9201--9203 and paired AE seeds assigned before execution.
The saved-checkpoint Flow 8/32/128 and DDIM 20/50/100 panel is also complete.
New runners provide strict manifests, checkpoint resume, paired physical-noise
error counts and generator-block aggregation. Tiny smoke runs are execution
checks only. The seed-9111 sources remain unchanged. The larger structured-channel
experiment is still pending.

Broader coverage improves mean nominal-8-dB SER in all three generator blocks
for both families. Ratios of normal/broad means are 32.55 for Gaussian and 4.05
for Sinkhorn, with 8/9 and 7/9 numerical AE-pair improvements respectively.
Sinkhorn ordinary-input conditional SWD worsens despite improved mean SER.
Flow's variance deficit shrinks markedly with more steps; DDIM gradient fidelity
does not improve monotonically. These are development findings, not universal
family rankings. Compact records are in
`evidence/coverage_followup_20261013/`; no new model weights are included.

A separate [SSPA noise-label check](sspa_noise_label_audit_20261012.md) identifies
a 3.0103 dB offset if Eb/N0 is defined relative to the pre-amplifier input under
the current real-coordinate normalization. The completed replication preserves
the parent's nominal settings and physical variance. Resolve the energy-reference
convention before new paper-level or external-reference comparisons.

## Independent assessment complete, 12 October

Read [the current assessment](application_coverage_assessment_20261012.md) for
the scientific conclusions, independent portable-record audit, mathematical
coverage bounds and next bounded replication. K64 grouping has a replicated
practical stability result; Gaussian coverage gains are clear within one
generator seed; Sinkhorn coverage outcomes and healthy downstream rankings
remain mixed. No universal pre-optimization selector is established.
The audit also clarifies scalar versus coordinate-wise centering and complete
pipeline timing. No new training was run during this assessment.

## Fast competing generators: local application pilot complete

Latest: the [input-coverage intervention](sspa_input_coverage_results_20261012.md)
has completed 15 AE rows (12 new, three inherited) and two new generators.
Gaussian transfer improves across all three paired AE seeds. Sinkhorn repairs
its worst case but slightly worsens the other two, while ordinary-input accuracy
degrades. Both reduce tail errors and optimistic surrogate losses. This is still
one generator seed. The frozen broader law needs fresh-generator replication,
not post-hoc mixture tuning. All controls, budgets and parent hashes pass audit;
no experiment remains running.

Read the [SSPA application results](fast_sspa_application_results_20261012.md)
for the newest result. Official-source modern baselines have now been implemented,
AWGN development has validated WGAN and counted consistency distillation, and
a single-seed SSPA comparison has completed six learned generators and seven AEs.
All learned methods receive 40.96m oracle output observations and approximately
60k inference parameters, including teacher pretraining in the distilled row.

Sinkhorn retains good conditional fidelity and gives SER numerically close to
DDIM-20 in this seed. WGAN gives the lowest numerical SER despite inaccurate
conditional laws. Post-hoc own-codebook checks localize severe extrapolation:
the Gaussian AE sends two inputs with norms above ten into regions with almost
no training-law probability, and those words account for 98.2% of its analytic
transfer errors. Distillation also concentrates errors on an extreme codeword.
The average-power constraint is still satisfied. The broader-input test is now
complete as linked above; replication and AE optimization remain before scaling.

All 13 CPU and nine CUDA tests passed, along with the checkpoint/count/source
audit and exact per-message evaluation replay. No current experiment remains
running. The evidence archive is
`Journal_version/evidence/fast_sspa_application_evidence_20261012.tar.gz`.
These are development results, not confirmatory family rankings. In particular,
9101--9105 and 9111 must not be reused as purported untouched confirmation seeds.
The larger structured channel and optical-fiber follow-up remain pending.

## Overnight extension: complete

User approved local execution within the remaining overnight window. The
[frozen fresh-start protocol](sspa_fresh_protocol_20261011.md) specifies unused
seeds 9004--9006, identical random initialization per seed, and three arms:
old K4, old K64-only, and normalized official K64. All have equal output
exposure and a 390720-update horizon. Both validation-selected and final raw
checkpoints enter 42 paired AE trials, followed by six common-codec gradient
comparisons. Final BER/SER is always measured on analytic SSPA.

All nine generator runs, 42 AE trials and six gradient comparisons completed
without task failures by 18:15 UTC on 11 October. Read the
[fresh-start results](sspa_fresh_results_20261011.md) first. K4 failed late on
all three fresh seeds; both K64 variants stayed stable. The official recipe
also improved expected encoder-gradient fidelity, but healthy-checkpoint SER
rankings are mixed and selected K4 remains competitive. No universal downstream
superiority or new SWD-to-SER selector is established.

Output: `results/sspa_fresh_overnight_20261011/`. `status.json` and `events.jsonl`
record completion. The runner is `scripts/run_sspa_overnight.py`. CPU and
CUDA tests each passed five cases; separate end-to-end smoke runs completed
3 generator, 14 codec and 2 gradient tasks. Smoke results are not evidence.
The full run took about 3 h 18 min and finished well before the overnight
deadline. Its audit checked 252 scheduled generator/AE checkpoints, provenance,
counts, paired messages and gradient-vector hashes. No optical-fiber
experiment, HPC submission or installation is part of this extension.

## Previous local mechanism round: complete

| Work | Status and outcome |
| --- | --- |
| Gradient/metric pilots, resolution and synthetic controls | Complete. Useful distinctions between law fidelity, expected gradients and estimator variance. The learned-model RBF selector gate was negative; its expansion is closed. |
| Small transport reference and numerical gradient checks | Complete, with unresolved reference solves explicitly retained. |
| Continuous SSPA epsilon-policy trajectories | All three policies x three development seeds reached 390720. All K4 paths deteriorated. Early 4800/30000 stages are parts of these completed paths. |
| Frozen-cloud numerical replay and replication | Complete on all three development seeds; 576 solver-case rows and 504 applied cloned updates. No evidence that solver arithmetic alone repairs the instability. |
| Repeated-update screen | Complete: three checkpoints x 32 fresh batches x nine variants. Do not confuse this one-step screen with long-run training. |
| Matched solver/optimizer/scale interventions | Eight branches x three seeds through 200k, then two qualifying branches x three seeds through 390720. Normalized K64 passed all three; lowered learning rate failed late in one seed. |
| Successful-recipe component deletions | All four branches finished at 390720 on seed 9001. K4 failed; deleting normalization, restoring epsilon or restoring the optimizer individually passed. |
| K64-only bridge to old trainer | Finished at 390720 on seed 9001. Changing only grouping to 256 inputs x 64 samples rescues the old trainer at unchanged output exposure. |

The preceding mechanism handoff is ready. There are no unfinished simulations
in that round; the overnight extension above is separate. The component/bridge audit
verified 85 checkpoints and 1553600 new updates, with historical files intact.
All 22 focused CPU tests and four component CUDA tests passed.

## Main new finding

Larger per-input clouds are the decisive practical difference on the tested
SSPA warm-start trajectory. K64-only in the old trainer gives independent
conditional SWD 0.0206825 (analytic floor 0.0183507) and variance 0.0524019
(truth 0.0528298). The old K4 trajectory fails at the same full budget.

This is not proof of the detailed instability mechanism, a universal failure
of K4, or an optimal/minimal K. The package has three-seed evidence, but the
isolated K/B change has one deliberately selected seed and a healthy warm
start. The overnight extension above now supplies fresh-start and downstream
development checks, with their own separate protocol and limitations.

Read [the component report](sspa_component_results_20261011.md), then the
[three-seed intervention report](sspa_intervention_results_20261010.md).
The exact authors' code attribution is in
[the official implementation comparison](official_drifting_code_comparison_20261010.md).
The imported loss is from the authors' W-Flow release, not the original
Drifting JAX release; conditional grouping is shared prior work.

## Next decisions after assessment

These are proposed next stages, not prerequisites for handing back this round.

1. Completed above: confirm the repaired recipe from fresh initialization and on unused seeds.
   Include a matched K4 control. Use K16/K32 only if a cloud-size threshold or
   compute tradeoff is a concrete question; avoid another broad parameter sweep.
2. Completed in the overnight batch: evaluate conditional accuracy at learned codewords, expected-loss gradients,
   and downstream SER/BER for the repaired checkpoints. Low SWD and correct
   variance alone do not establish these outcomes.
3. Prioritize fair fast baselines: low-step DDIM, conditional flow matching, a genuine
   trained one-step comparator, and a conditional Gaussian reference. Existing
   historical benchmarks do not fulfill the revised matched comparisons.
   The [new implementation](fast_baseline_protocol_20261012.md) supplies the
   runners and pinned official numerical routines. The [AWGN development round](fast_baseline_development_results_20261012.md)
   now validates WGAN and counted consistency distillation on three development
   seeds each; standalone CT remains under-dispersed. The six-family SSPA pilot
   and paired input-coverage intervention are complete as described above.
   Independent-generator replication and the final seeded comparison remain
   pending. Per the 12 October decision, the
   joint-versus-condition-wise geometry study is optional. Conditioning itself
   is not the proposed novelty; communication applications drive the comparison.
4. Complete remaining measurement contracts and the larger jointly generated
   structured channel. The restricted SSPA resume/count machinery does not mean
   that all P0 interfaces or the proposed 128-real-coordinate block study exist.

Broader tasks are specified in `resubmission_cluster_plan.md` and the reviewer
response matrix in `resubmission_handover.md`. Measured-data applicability must
remain explicitly scoped unless a separate experiment validates it. The closed
RBF-selector study should not be restarted without a new hypothesis.

## Added follow-up: optical fiber

Requested by the user on 11 October. Keep this on the research backlog, **not
running** and not blocking the current handoff.

Question: does larger same-input sampling repair any of the previous optical-
fiber surrogate failures, as it did in this SSPA continuation?

Start with a bounded K4 versus K64 comparison at equal generated/positive/
reference output exposure. Preserve one documented optical-fiber channel,
input power, propagation discretization, input law and analytic reference.
Do not simultaneously change the physical approximation or add a deterministic
base model when trying to attribute an effect to K. Separate the unchanged old
recipe from the official normalized recipe if both are tested.

Evaluate conditional amplitude/phase statistics and tails as well as SWD,
then expected-loss gradients and analytic-channel downstream SER/BER if the
law-fidelity pilot is promising. SSPA success does not establish an optical-
fiber fix; its nonlinear phase evolution may expose different limitations.

## Artifacts

- `Journal_version/sspa_fresh_results_20261011.md` and
  `results/sspa_fresh_overnight_20261011/`: current fresh-start/utility handoff.
- `Journal_version/evidence/sspa_fresh_evidence_20261011.tar.gz`: verified
  bundle including all scheduled fresh-start/AE checkpoints and gradient vectors.
- `results/sspa_components_20261011/`: audited endpoints, all component
  checkpoints, traces, and both trajectory figures.
- `results/sspa_particle_bridge_20261011/`: K64-only old-trainer checkpoints
  and records.
- `Journal_version/sspa_component_results_20261011.md`: current conclusions,
  scope, attribution and reproducibility details.
- `Journal_version/evidence/`: existing portable evidence bundles for the
  earlier phases; the new component bundle accompanies this handoff.

No manuscript/public-repository changes, HPC submissions, dependency installs,
commits, pushes, or editor contact were made in this component investigation.
