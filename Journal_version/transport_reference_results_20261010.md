# Small Sinkhorn transport-reference audit

10 October 2026. Completed the bounded P1a check described in the
[frozen protocol](transport_reference_protocol_20261010.md). This follows the
closed [kernel-metric gate](learned_metric_gate_assessment_20261010.md).

**Outcome:** the reference harness is available and the restricted gradient
identity checks pass. Numerical failures are possible in sharp transport regimes,
but the tested SSPA-shaped clouds do not show a large solver error. This is not
an explanation of the historical SSPA training deterioration. The next experiment
should observe controlled continuous training trajectories, with numerical and
statistical quantities logged separately.

**Later status:** the restricted resume/RNG/count contracts and all nine
[full-budget trajectories](sspa_full_budget_results_20261010.md) are now complete.
The next-step text below is historical. The [independent assessment](sspa_full_budget_assessment_20261010.md)
specifies the remaining numerical-mechanism experiment.

## Scope and implementation

The reference solves uniform-mass entropic OT with cost ||u-v||^2/2 and KL relative
to the product of source and target masses. It uses float64 log-domain scaling,
checks both relative marginal residuals against 1e-8, and stops visibly at 50,000
iterations if necessary. There is no kernel floor or final row renormalization.

The production solver remains float32, with its existing kernel floor and
denominator stabilizers. Two private projection functions now accept optional
`return_diagnostics=True`. This exposes costs, actual epsilon, floored kernel,
raw coupling, masses and normalized weights. Defaults and training behavior were
preserved: 36 comparisons against the pre-edit committed implementation were
bit-identical across batched/unbatched, fixed/adaptive, and iteration settings.

The numerical panel has eight deterministic cloud cases, three epsilon values
(0.001, 0.05, 1), and three practical iteration counts (10, 30, 100): 24 reference
problems and 72 production records. It includes unequal counts, duplicates,
concentrated/broad/separated clouds, unequal cluster occupancy, identical clouds
and collapsed symmetric sources. A second panel has eight SSPA-shaped anchors,
four generated/positive/reference outputs each, four epsilon policies, and the
same three iteration counts: 12 policy records.

This is a CPU reference test, not a GPU timing benchmark. The final run took
about 1.69 seconds on this host. No generator/receiver training, cluster work,
dependencies or submitted-paper edits were involved.

## Numerical findings

**Most reference problems converged; one did not.** The broad-cloud problem at
epsilon 0.001 reached the 50,000-iteration cap with a source-marginal relative
error of 0.2 and target error about 7.5e-13. Its intended-OT accuracy columns are
left unresolved. It is preserved in the data rather than counted as ground truth.
The other 23 reference problems met both marginal tolerances. All 24 references
for the production's modified floored kernel converged.

**Row normalization can hide a marginal mismatch.** For near-duplicate clouds
at epsilon 0.001 and 10 iterations, the raw maximum relative source-marginal error
was about 0.999. After row normalization, the target-marginal relative error
remained about 0.499. A row-normalized barycentric weight matrix is not therefore
a solved balanced transport coupling.

**Kernel flooring can change the transport problem substantially.** The practical
kernel is exp(-(C-row_min(C))/epsilon), lower-bounded by 1e-8. Ignoring float32
rounding, this caps the row-shifted effective cost at epsilon*log(1e8).
Increasing iterations cannot recover information lost through that cap.

| Case | Epsilon | Iterations | Practical barycenter RMS error | Converged floored-kernel RMS error |
| --- | ---: | ---: | ---: | ---: |
| Broad clouds | 0.05 | 10 | 2.84518 | 2.82821 |
| Broad clouds | 0.05 | 100 | 2.82761 | 2.82821 |
| Unequal 4-by-7 clouds | 0.001 | 10 | 0.172979 | 0.185446 |
| Unequal 4-by-7 clouds | 0.001 | 100 | 0.183782 | 0.185446 |

Errors are against converged intended quadratic-cost transport, measured per
output coordinate. The floored-kernel reference also includes the production
kernel's float32 quantization. These stress cases are not SSPA model failures.
The practical-versus-floored-reference gap contains finite-iteration error,
denominator/normalization stabilizers and rounding; this panel does not uniquely
attribute that remaining gap to one of those effects.

## SSPA-shaped clouds

The source/reference clouds were generated from the same synthetic mean/noise
law, with independent draws. They are not checkpoint outputs. Positives use the
actual SSPA mean and real-component noise 0.3250531435997416/sqrt(2).

| Policy | Cross epsilon | Self epsilon | Drift RMS error at 10 iterations | Reference field RMS difference from fixed common |
| --- | ---: | ---: | ---: | ---: |
| Fixed common, analytic-only pilot scale | 0.353474 | 0.353474 | 2.10e-7 | 0 |
| Shared adaptive, pooled within-anchor costs | 0.746384 | 0.746384 | 9.28e-8 | 0.0456823 |
| Existing separate within-anchor medians | 0.754820 | 0.735207 | 8.00e-8 | 0.0452027 |
| Historical global/marginal scale | 9.50381 | 9.42864 | 9.06e-8 | 0.0935957 |

No kernel floor activated in this panel. The largest raw source-marginal error
at 10 iterations was 7.51e-6 for the fixed-common policy. The corresponding drift
RMS error was only 2.10e-7. The other policies were near float32 marginal precision.
Thirty iterations reduced the fixed-common residual to about 1.19e-7.

Changing epsilon changes the intended field, even with accurate transport solves.
In this particular panel the separately adaptive versus shared-adaptive reference
fields differed by only 0.0007974 RMS. The larger shifts in the table also change
the regularization scale itself. Do not attribute them solely to cross/self
epsilon mismatch or claim that one policy produces better training.

This does not justify increasing all training solves from 10 to 100 iterations.
It does justify logging marginal residuals, floor activation, generated spread,
actual cross/self epsilon and their ratio along the actual trajectory. A later
bad checkpoint may enter a regime absent from these initially plausible clouds.

## Gradient and stationary-point checks

For a small affine generator, fixed common epsilon 0.4, identical empirical
self clouds, raw coordinates and no clipping:

- Every parameter coordinate of the full debiased objective was checked against
  centered finite differences at step 1e-5. Maximum error was 6.78e-12.
- Detached coordinate-averaged MSE with drift scale eta=0.3 and output dimension
  d=2 gave (2 eta/d) times that parameter gradient. Maximum error was 2.18e-13.
- The practical 10-iteration gradient differed from that scaled reference by
  at most 7.80e-9 in this well-conditioned example.
- Replacing the self cloud with independent latent draws through the same affine
  generator changed the sampled velocity, with total Euclidean difference 0.4763.
  This is a finite-sample reference change, not a numerical error against the
  same-batch objective. No unbiased population-gradient claim follows.

For q=delta_0 and p=(delta_-1+delta_1)/2, all three epsilon settings reproduce
zero barycentric velocity and positive divergence
S_epsilon = epsilon/2 * log(cosh(1/epsilon)). Identical-law checks give zero
divergence. Thus these calculations support the manuscript's proposed correction:
unique objective zero does not by itself exclude other stationary measures.

## Validation and files

- 57 focused tests passed, including nine new transport tests.
- The reporter independently recomputes residuals and barycenter errors from
  saved plans with NumPy. The maximum difference is 1.09e-7, consistent with
  comparing float32 production reductions against float64 saved-plan arithmetic.
- The maximum batched-versus-unbatched projection difference in the stress panel
  was 9.54e-7. This is distinct from the bit-identical before/after default checks.
- Final raw data: `results/transport_reference_audit_20261010_v2/`.
- Complete tables: `results/transport_reference_report_20261010/`.
- Reference implementation: `conditional_drifting/transport_reference.py`.
- Runner/reporter: `scripts/run_transport_reference_audit.py` and
  `scripts/report_transport_reference_audit.py`.
- Compact archive: `Journal_version/evidence/transport_reference_evidence_20261010.tar.gz`.
  Checksum and restoration instructions are in the [evidence README](evidence/README.md).

The first development run, `results/transport_reference_audit_20261010/`, is
superseded: its independent-reference gradient control sampled an unmatched affine
law. The final run corrects that control. All reported numbers above refer to v2;
no runs are pooled. The other numerical panels were unchanged by that correction.

Reproduce with fresh output directories:

```bash
python scripts/run_transport_reference_audit.py --out-dir results/transport_reference_new
python scripts/report_transport_reference_audit.py \
  --results-dir results/transport_reference_audit_20261010_v2 \
  --out-dir results/transport_reference_report_new
```

Next: implement and verify exact-update/resume, validation RNG isolation and
sample-count contracts before long SSPA runs. Then compare the existing policy
and fixed/shared common-epsilon policies on continuous, identically initialized
trajectories. Track both selected and final checkpoints, retaining finite but poor
outcomes. P0 completion, shared-adaptive training support, fair baseline training
and the long-run instability mechanism remain open; this audit does not close them.
