# Three-seed SSPA solver replay

The two remaining solver replays are complete. This extends the seed-9001
analysis without changing its numerical methods or selecting new checkpoints.
See the [frozen continuation protocol](sspa_intervention_protocol_20261010.md)
for the seed-specific checkpoint pairs.

Across seeds 9001--9003, there are 36 reconstructed batches and 576 method
records. Every batch contains 4096 input conditions with four generated,
positive and reference samples each. There are 504 cloned single-update
comparisons. The other 72 records are reference solves with at least one
unresolved anchor; no neural update is reported for those records. Reference
tolerance is 1e-8 and the iteration cap is 50000. These are frozen-cloud
comparisons, not continued training or downstream SER/BER experiments.

## Replication

Relative parameter-gradient changes after replacing production scaling with
float64 log-domain scaling of the same floored kernel, at ten iterations:

| Policy | Seed | Last saved healthy | First saved degraded |
| --- | ---: | ---: | ---: |
| Separate adaptive | 9001 | 9.79e-7 | 0.156 |
| Separate adaptive | 9002 | 9.15e-7 | 0.140 |
| Separate adaptive | 9003 | 3.83e-6 | 0.292 |
| Fixed common | 9001 | 3.00e-7 | 0.801 |
| Fixed common | 9002 | 2.97e-7 | 0.111 |
| Fixed common | 9003 | 2.11e-4 | 0.529 |
| Shared adaptive | 9001 | 3.44e-7 | 0.214 |
| Shared adaptive | 9002 | 0.0236 | 0.151 |
| Shared adaptive | 9003 | 3.66e-7 | 0.129 |

The broad pattern replicates: corrections are substantially larger after
degradation. Shared-adaptive seed 9002 is an important exception to any claim
that the pre-degradation solver effect is always negligible. Its last saved
healthy checkpoint already has a 2.36% relative gradient change. "Healthy"
here is the existing saved-checkpoint fidelity classification, not a numerical
certificate or an assertion that instability has not begun.

The fixed-common seed-9002 and seed-9003 pairs strengthen the reason for a
solver-only continuation: the immediate good checkpoint has little numerical
correction, while the subsequent degraded state has a large correction.
Temporal association alone does not identify the trigger. The separate
continuation matrix tests prevention from the common 80k parent.

## Audit and artifacts

Both new audits passed completeness, parent/source hashes, saved coupling
residuals, unresolved-reference masking, particle variance identities and
single-update checks. The reporter now validates the seed explicitly and uses
the manifest seed in its scope text; the original scope string was hardcoded
to 9001. This was a reporting-label correction, with no numerical change.

- `results/sspa_solver_replay_remaining_20261010/seed9002/`
- `results/sspa_solver_replay_remaining_20261010/seed9003/`

Each has 12 cases, 192 method rows and 24 skipped unresolved-reference neural
updates. Summed case runtimes are 172.81 and 169.35 seconds, respectively.
They include reference solving and evaluation, not just neural training.
