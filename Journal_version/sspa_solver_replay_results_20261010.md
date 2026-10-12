# SSPA frozen-cloud solver replay

10 October 2026. Completed seed-9001 panel from the
[frozen protocol](sspa_solver_replay_protocol_20261010.md), followed by a
[comparison with official author code](official_drifting_code_comparison_20261010.md).

## Scope and checks

- 12 checkpoints: three epsilon policies, each at 30k, last saved healthy,
  first saved degraded and 390720 updates.
- One reconstructed next training batch per checkpoint: 4096 inputs, four
  generated/positive/reference samples per input. This is not the unsaved batch
  that initiated degradation.
- 16 methods per batch, 192 method records. All production-10 fields matched the
  original trainer bit for bit; all cloned models reproduced the source outputs
  bit for bit before their single update.
- Float64 reference cap 50000 and relative marginal tolerance 1e-8. All four
  reference variants fully converged on the six healthy batches. Each had some
  unresolved anchors on the six degraded batches. Those 24 reference methods
  deliberately have no neural-update result. Converged-subset particle and plan
  comparisons remain explicitly masked.
- 168 cloned Adam updates with original optimizer states, gradient clipping and
  paired 128-input/128-latent evaluation. No continued training, SER or BER run.
- 178.54 seconds summed case runtime on RTX 5060 Ti, Torch 2.11.0+cu128. This
  includes reference solves and replay evaluation, and is not a training-speed
  benchmark. The separate reduced-batch smoke is excluded.
- Audit verifies source and parent hashes, completeness, saved plan residuals,
  cap statuses, variance identities and evaluation changes.

## Numerical changes matter mainly after deterioration on this panel

Relative gradient change is `||g_new-g_production10|| / ||g_production10||`.
These are single-batch quantities without seed uncertainty.

| Policy and checkpoint | Production 100 iterations | Log-domain floored kernel, 10 iterations, float64 | Fully converged unfloored float64 reference |
| --- | ---: | ---: | ---: |
| Legacy, healthy 80k | 6.03e-5 | 9.79e-7 | 6.07e-5 |
| Fixed common, healthy 100k | 2.21e-7 | 3.00e-7 | 3.16e-7 |
| Shared adaptive, healthy 90k | 1.72e-5 | 3.44e-7 | 1.73e-5 |
| Legacy, degraded 90k | 0.150 | 0.156 | Unresolved anchors |
| Fixed common, degraded 110k | 0.739 | 0.801 | Unresolved anchors |
| Shared adaptive, degraded 100k | 0.207 | 0.214 | Unresolved anchors |

At all three 30k checkpoints, the unfloored reference changes the gradient by
less than 3.7e-7 relatively. No kernel entries are floored on any of the six
healthy reconstructed batches. Removing only the final barycenter denominator
barely changes gradients throughout this panel. In contrast, removing the
internal scaling denominator at a matched iteration count changes degraded
fields materially. Reinstating that denominator in log arithmetic recovers the
production gradient to about float32 precision. Moving from float32 to float64
alone does not remove the implemented denominator's effect.

These comparisons distinguish denominator effects from iteration count and
kernel flooring. They do not show that the denominator starts the failure.
In particular, numerical corrections have almost no effect on the fixed-common
100k batch immediately before its saved 110k deterioration.

## Particle and neural variance changes are distinct

At fixed-common 100k, production's particle variance change is +0.00230843,
with first-order term +0.00003191 and second-order term +0.00227652. The
converged unfloored reference gives essentially the same result. Its single
cloned neural update increases fresh-panel variance by +0.000118503, also
essentially unchanged by the reference. This is compatible with finite-cloud
stochastic effects, but one batch cannot establish their expected sign or
long-run consequence.

The distinction is particularly visible at the final legacy checkpoint:
the production particle field contracts empirical variance by 0.13062, while
the cloned Adam step increases fresh-panel conditional variance by 0.06627.
At fixed-common 110k, particles expand by 0.004279 but the network step decreases
variance by 0.002209. Shared parameters, projection through the network Jacobian,
gradient clipping and optimizer history separate these operations. Their
individual contributions have not been isolated. Particle motion alone is not
an explanation of neural variance growth.

The fresh-panel variances differ from the historical validation panel because
the inputs and latents differ. They should not replace historical values in
the earlier trajectory report.

## Interpretation and next work

The replay establishes substantial late numerical sensitivity but does not
identify a numerical trigger. A solver-only continuation can test prevention,
but should not be advertised as an expected fix based on these results.

The official implementation comparison reveals additional, concrete differences:
64 rather than four samples per condition, feature and force normalization,
and different optimization. Before another long run, use common-cloud repeated
batches to check the variance and mean of neural updates near the good model,
then isolate sample-count and scaling changes. A paired continuation can follow
under a new intervention manifest, retaining the original baseline and random
streams. Do not weaken exact-resume provenance checks or change the old trainer.

This is one development seed and one batch per checkpoint. Repeat informative
contrasts on seeds 9002/9003 before generalizing. No stability repair, downstream
improvement, or new manuscript claim is established.

## Artifacts and reproduction

Complete local suite: `results/sspa_solver_replay_20261010/`. Per case:
`clouds.pt`, `solves.pt`, `result.json`, and incremental `partial.json`.
Aggregate exports: `audit.json`, `metrics.csv`, `official_wflow_check.json`.
All unresolved anchors and negative outcomes are retained.

Portable evidence: `Journal_version/evidence/sspa_solver_replay_evidence_20261010.tar.gz`.
It includes all frozen clouds, JSON/CSV records, source snapshots, tests and
reviewed upstream loss/config files with their license. To keep the transfer
small, it excludes the roughly 0.5 GB collection of full `solves.pt` tensors,
which remains local with file hashes in `audit.json`. Thus the transferred
archive supports inspecting all reported cases but cannot rerun the full tensor
audit until those solves are regenerated or transferred. Parent checkpoints
remain in the earlier full-budget evidence archive. Reclone the pinned upstream
repository for its `.git` metadata before rerunning the official-code check.

Verification: 81 focused tests passed, including six new replay tests.

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/report_sspa_solver_replay.py \
  --suite-dir results/sspa_solver_replay_20261010

# A fresh output path is required. Original checkpoints and CUDA are needed.
/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_sspa_solver_replay.py \
  --out-dir results/sspa_solver_replay_repeat --device cuda

# Requires the pinned, unmodified author-code checkout. Does not install packages.
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/check_official_wflow_solver.py \
  --out results/sspa_solver_replay_20261010/official_wflow_check_repeat.json
```
