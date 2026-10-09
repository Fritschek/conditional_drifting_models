# Same-input SSPA metric results

Completed locally on 9 October 2026, following the
[frozen protocol](learned_metric_gate_protocol_20261009.md) and the
[post-control assessment](metric_controls_assessment_20261009.md).

**Decision:** the bounded development test provides no evidence that the current
RBF derivative score adds useful model-ordering information beyond the simpler
checks. Do not expand this score into a large seed sweep or use it for checkpoint
selection. Retain derivative fidelity as an explanatory research direction.

## What ran

- Existing seed-7 SSPA checkpoints: selected condition-wise Sinkhorn, WGAN, and
  DDIM-10, plus an independently sampled analytic control. No model training.
- Three stored eight-real-coordinate inputs, indices 3--5 from the previous
  N=512 pathwise panel. Checkpoint hashes/configs matched that panel. These inputs
  have norm sqrt(8); derivatives are ambient, without input renormalization.
- Eight fixed output-loss probes: two directions/centers, each with a quadratic,
  softplus, cosine, and mixture-RBF-section loss. No fitted decoder.
- Two independent score splits and two independent task splits, eight repetitions.
  N=512 first, then N=2048 at all three inputs under the frozen refinement rule.
  Reference score draws were shared across models, not across independent splits.
- Exact Gaussian reference integration through the nonlinear SSPA mean/Jacobian,
  with 64-versus-128-node verification for logistic gradients. Physical real-noise
  standard deviation was 0.3250531435997416/sqrt(2). The analytic candidate's
  separate task stream supplied the noisy null control; a noisy reference task
  estimate was unnecessary.
- GPU sampling/Jacobians, CPU float64 kernel calculations. Main panel elapsed
  611.05 seconds on the local RTX 5060 Ti. Metric/probe algebra took 23.88 seconds
  at N=512 and 578.20 seconds at N=2048. These are measured execution costs,
  including the chosen unoptimized full-kernel implementation, not latency claims
  about the generators or an optimized metric implementation.

Coverage is 192 metric records, including 144 learned-model records, and 1,536
task records. The two sample levels remain separate. There were 1,105,920 sampled
output vectors, each with its full 8-by-8 Jacobian. Shared reference score draws
account for 122,880 of those vectors; each candidate's score and task streams each
account for another 122,880. A reduced preflight is excluded from these totals.

## Predeclared comparisons

At N=2048 all 72 pairwise task-error contrasts satisfy the descriptive
three-MC-SE resolution rule. This is three model pairs times three inputs times
eight losses, **not 72 independent trained-model trials**. Their target is the
signed independent-split estimate of squared expected-gradient error. The rule
does not supply calibrated confidence or a multiple-comparison guarantee.

| Comparator | Ordering agrees with target | Opposes | Unresolved |
| --- | ---: | ---: | ---: |
| Conditional SWD | 48 | 24 | 0 |
| RBF value cross score | 48 | 24 | 0 |
| Mean derivative cross score | 54 | 18 | 0 |
| Covariance derivative cross score | 66 | 6 | 0 |
| Polynomial derivative cross score | 61 | 3 | 8 |
| Projected fourth-moment derivative cross score | 64 | 8 | 0 |
| RBF derivative cross score | 32 | 32 | 8 |

There are **zero candidate added-information contrasts** under the frozen rule.
Every resolved agreeing RBF derivative ordering also has a resolved agreeing
cheaper comparator. This comparison uses the union of the cheaper comparators;
it does not establish a single universal replacement score. No analytic RBF or
task cross-error null exceeds the descriptive three-SE threshold at either N.

## A concrete value-versus-gradient mismatch

At input 0, the following values come from the final N=2048 panel. Gradient error
is for the first fixed quadratic probe, not for a trained receiver.

| Model | Conditional SWD, mean +/- MC SE | Absolute expected-gradient error |
| --- | ---: | ---: |
| Conditional Sinkhorn | 0.032558 +/- 0.00036 | 0.11253 |
| WGAN | 0.097478 +/- 0.00046 | 0.21997 |
| DDIM-10 | 0.049055 +/- 0.00034 | 0.058232 |

Gradient errors use the norm of the pooled gradient estimate. Corresponding L2
norms of coordinate MC SEs are 0.00088, 0.0018, and 0.00042. Those uncertainty
summaries are not scalar-norm confidence intervals.

Conditional Sinkhorn has the lowest SWD at all three inputs, and the lowest RBF
derivative cross score. DDIM-10 has the smallest absolute gradient error for
21/24 input/probe cases; Sinkhorn is best in the remaining three, all at input 2.
This supports the value-versus-gradient distinction without establishing either
model as universally better. It also shows that the present kernel derivative
score does not fix the ordering problem on this panel.

## Interpretation and limits

The polynomial/covariance/fourth-moment comparisons performed better here.
The analytic SSPA has input-independent additive-noise covariance, so covariance
derivatives provide a particularly simple physical check. Their association with
gradient errors in this small panel is not evidence that covariance errors caused
all downstream differences. A quadratic probe depends on projected mean and
second-moment derivatives, making the moment comparison especially relevant.

The kernel result does not refute the population RKHS gradient bound. A norm
over a function class and all input directions need not rank errors for every
particular loss in that class. The two probe directions and fixed kernel centers
are a limited development panel. Bandwidths were not adjusted after seeing it.
These are existing, differently trained checkpoints, not a matched-capacity
comparison of generation mechanisms. None of these results predicts final BER.

The variance correction alone is insufficient here: independent-split scores
and independent target streams still fail the proposed added-information test.
Do not claim a validated kernel selector, a Sinkhorn advantage, or a validated
covariance selector from these data. The analytic higher-moment counterexample
remains valid, but its practical kernel-specific advantage remains unestablished.

Next priority should return to the existing P0/P1 channel/transport contracts,
controlled SSPA stability, and fair fast-generator comparisons. Keep the simple
moment/Jacobian checks and analytic controls in those studies. Any new metric
design needs a fresh, explicit hypothesis rather than tuning this panel until
its rankings agree. Seeds 7--9 and these probes are development evidence.

## Files and reproduction

- Runner: `scripts/run_learned_metric_gate.py`.
- Reporter: `scripts/report_learned_metric_gate.py`.
- Raw per-repeat statistics, matrices, task gradient vectors, source snapshots,
  manifest, and sampled outputs/Jacobians:
  `results/learned_metric_gate_sspa_seed7_20261009/`.
- Complete tables, all contrasts, and coverage checks:
  `results/learned_metric_gate_report_20261009/`.
- Git-visible transfer archive:
  `Journal_version/evidence/learned_metric_gate_evidence_20261009.tar.gz`.
  See the [evidence README](evidence/README.md) for its checksum and scope.

The archive includes the per-repeat sufficient statistics and gradient vectors
needed to reproduce all report tables. Large raw sampled-output/Jacobian `.pt`
files and pretrained weights remain local and are excluded. Recomputing metrics
from individual samples needs those local files, or rerunning sampling.

Use a fresh output directory for either command:

```bash
python scripts/run_learned_metric_gate.py --device cuda --out-dir results/learned_metric_gate_new_run
python scripts/report_learned_metric_gate.py \
  --results results/learned_metric_gate_sspa_seed7_20261009/results.json \
  --out-dir results/learned_metric_gate_report_reproduced
```

Validation: 48 focused tests passed across the gradient, feature, control,
report-contract and new same-input suites. Coverage/refinement decisions reproduce,
negative cross scores are retained, and all split variance matrices passed their
positive-semidefiniteness checks. No dependencies were installed, no HPC jobs or
new training were launched, and submitted manuscript artifacts were not changed.
