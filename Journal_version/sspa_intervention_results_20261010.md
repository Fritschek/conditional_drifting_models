# SSPA solver and optimizer continuations

Status: complete and audited. All 24 first-stage runs and all six prescribed
full-budget extensions finished without task errors. The normalized K64 package
remains accurate through 390720 on all three development seeds. A smaller
learning rate alone delays failure in one seed and remains accurate in two.

## Design

The [frozen protocol](sspa_intervention_protocol_20261010.md) specifies eight
branches on development seeds 9001--9003. Every branch starts from the same
seed's fixed-common 80k checkpoint. The original numerical sources and parent
checkpoints are preserved. These are warm-started interventions, not independent
training-from-scratch comparisons. Every update consumes 16384 positive,
generated and reference outputs. K4 branches use 4096 inputs; the K64 package
uses 256 inputs. The latter changes the number of independent conditions.

The first stage ends at 200k. A predeclared gate checks conditional SWD, mean
error and variance at 180k, 190k and 200k. Any passing seed extends all three
seeds of its branch to 390720. The independent endpoint panel does not select
branches. Both `log10_lr01` and `official64` pass on all three seeds; all other
branches fail on all three. Raw and EMA K64 models both pass.

## First-stage results

Conditional SWD on the common validation panel at 200k:

| Branch | Seed 9001 | Seed 9002 | Seed 9003 | Passing seeds |
| --- | ---: | ---: | ---: | ---: |
| Unchanged control | 1.104590 | 0.764211 | 0.862353 | 0/3 |
| Floored log solver, 10 iterations | 0.983718 | 0.815503 | 0.901583 | 0/3 |
| Unfloored log solver, 10 iterations | 2.550205 | 1.016281 | 2.249282 | 0/3 |
| Unfloored log solver, 100 iterations | 2.668158 | 0.834197 | 2.452042 | 0/3 |
| Log (10), Adam beta1=0 | 2.199181 | 1.091915 | 0.718621 | 0/3 |
| Log (10), learning rate 0.0001 | 0.037903 | 0.037381 | 0.037093 | 3/3 |
| Log (10), drift scale 0.1 | 2.474617 | 0.579881 | 0.940650 | 0/3 |
| Normalized K64 package, raw model | 0.037245 | 0.037174 | 0.037327 | 3/3 |

Parent SWDs are 0.038568, 0.039107 and 0.038379. Analytic conditional variance is
0.052829773 per real coordinate. The unchanged controls reproduce model,
optimizer and training-RNG states at all 36 matching historical checkpoints.
This checks that added validation and bookkeeping preserve the old trajectories.

The validation panel uses 128 inputs and 128 outputs each. The separate endpoint
panel uses 256 inputs and 512 outputs each, with a different random seed. Its
analytic-versus-analytic SWD floor is 0.018350691. Lower-learning-rate endpoint
SWDs are 0.020067, 0.019730 and 0.019576. Raw K64 endpoint SWDs are 0.019742,
0.019761 and 0.019830. The lower raw SWD numbers on this larger panel reflect
different finite-sample floors; they are not evidence of improvement relative
to the smaller validation panel. Floors use two independent analytic output
sets at each shared input.

The no-momentum branch delays failure in seed 9001 from approximately 110k to
136k, but deteriorates severely by 200k. A favorable single-update direction
does not establish a successful long-horizon intervention.

## What the numerical checks establish

Neither removal of the internal scaling denominator, removal of kernel flooring,
nor increasing the solver to 100 iterations prevents deterioration in these
three seeds. Iteration count alone does not certify convergence: all fixed-step
solvers retain their measured marginal residuals.

The small validation panel can miss rare numerical problems. For example, in
`log100/seed9001`, the 128-anchor validation residuals are near float precision
at 104k--108k. The sampled full-training-batch trace records maximum marginal
relative error 0.001214 over 100k--106k and 0.002604 over 106k--108k, before the
large validation-SWD increase at 110k. Those maxima concern rare clouds and
must not be presented as typical coupling errors. They also preclude claiming
that all numerical sensitivity appears only after deterioration. The precise
trigger remains unresolved.

The [three-seed frozen replay](sspa_solver_replay_replication_20261010.md) is
complete: 36 batches, 576 method records and 504 valid cloned neural updates.
The other 72 records have unresolved reference anchors and deliberately omit
the corresponding neural update. Solver sensitivity and continued-training
outcomes are different evidence and are reported separately.

## Why drift scale and learning rate differ under Adam

For a detached regression target `stopgrad(y_theta + alpha V)`, the mean squared
loss over N vectors of dimension d has parameter gradient

`g_theta = -2 alpha / (N d) sum_i J_theta(x_i,z_i)^T V_i`.

At fixed parameters and particles, alpha scales the raw gradient. With
consistently scaled Adam moments, its first moment scales by alpha and its
second moment by alpha squared. Their ratio is unchanged in the zero-optimizer-
epsilon limit. Thus changing alpha need not reduce the eventual parameter step,
whereas changing Adam's learning rate directly scales that step. Nonzero Adam
epsilon, clipping and inherited moments limit exact invariance. The intervention
inherits the old moments, so it also has a transient before they adapt.

`test_adam_gradient_scaling_is_not_learning_rate_scaling` verifies the exact
fresh-moment, zero-epsilon limiting case. This explains why the two changes
should be tested separately. It does not prove the cause of the observed
long-run failure or identify a globally stable optimization rule.

## Scope of the K64 result

The K64 branch uses the pinned authors' normalized quadratic loss at R=0.05,
ten log-domain iterations, independent references, no diagonal mask and no
classifier-free guidance. It has fresh AdamW with beta=(0.9,0.95), weight decay
0.01, 5000-step warmup to learning rate 0.0002, gradient cap 2 and EMA=0.999.
The raw model is primary; EMA is a separate prespecified secondary series.
This combines several changes. It does not isolate K or reproduce the image
feature-learning pipeline. The earlier equal-output single-update screen did
not establish a raw-gradient-noise advantage from increasing K alone.

## Full-budget results

Independent-panel endpoints at 390720. The analytic SWD floor is 0.018350691
and analytic variance is 0.052829773 for every row. "Late gate" checks all
three saved points at 300k, 350k and 390720 using the original numerical bounds.

| Branch/model | Seed | Conditional SWD | Conditional mean L2 | Conditional variance | Late gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Lower learning rate | 9001 | 0.442457 | 0.852632 | 6.853473 | Fail |
| Lower learning rate | 9002 | 0.019537 | 0.043925 | 0.052454 | Pass |
| Lower learning rate | 9003 | 0.019513 | 0.043938 | 0.052652 | Pass |
| Normalized K64, raw | 9001 | 0.019627 | 0.046031 | 0.052660 | Pass |
| Normalized K64, raw | 9002 | 0.019703 | 0.046350 | 0.052503 | Pass |
| Normalized K64, raw | 9003 | 0.019705 | 0.046523 | 0.052907 | Pass |
| Normalized K64, EMA | 9001 | 0.018992 | 0.043073 | 0.052633 | Pass |
| Normalized K64, EMA | 9002 | 0.018922 | 0.042675 | 0.052604 | Pass |
| Normalized K64, EMA | 9003 | 0.019017 | 0.043394 | 0.052700 | Pass |

The first completed full-budget extension, lower-learning-rate seed 9001,
fails: validation SWD rises from 0.038133 at 250k to 0.133718 at 300k and
0.462438 at 390720. Its independent endpoint SWD is 0.442457 and variance
6.853473. This branch's 200k success was a delayed failure in that seed.

The raw endpoint JSON fields `gate_raw` and `gate_ema` always record the
**200k extension-selection gate**, even in `result390720.json`. They do not
certify the final model. The auditor separately exports `late_three_bounded`
for the 300k, 350k and 390720 checkpoints. Likewise, task status `complete`
means execution finished, not that fidelity passed.

Both branches completed 390720 on all three seeds. Six failed-gate branches
finish at 200k by protocol; they are
not labeled as new 390720 tests. Historical full-budget controls remain available.

These results establish a bounded warm-start operating result for the normalized
package on this SSPA setting. Training-from-scratch schedules, held-out
seeds, learned-codeword gradients and downstream SER/BER remain separate tests.
The modern fast-baseline and larger structured-channel work packages are also
pending. No submitted manuscript or public implementation is changed here.

## Audit and cost

- All 24 first-stage tasks and six extensions are complete. No task exceptions
  or interrupted training intervals occurred in the main matrix.
- The auditor verifies parent/source hashes, all saved checkpoint hashes,
  intervention configurations, optimizer settings/step histories, sample counts,
  checkpoint/trace coverage, paired K4 RNG states and extension decisions.
- All 36 matching control checkpoints reproduce historical model, optimizer
  and training-RNG states exactly. The original numerical files are unchanged.
- There are 432 small reference checks, with ten unresolved at their 10000-step
  cap. Their barycenter-error values remain absent. None of those unresolved
  checks belongs to the normalized K64 branch. This small-panel audit is distinct
  from the 50000-step full-batch solver replay.
- The matrix performed 4,024,320 new updates after the reused 80k parents.
  Synchronized training time is 17486.48 seconds (4.86 hours), plus 107.12 seconds
  of measured validation. These figures exclude parent training, preflight,
  checkpoint I/O and reporting; they are not a complete process wall-clock total.
- RTX 5060 Ti, Torch 2.11.0+cu128, deterministic training, no compilation or
  mixed precision. The loss source is the pinned upstream commit documented
  in the official-code comparison.
- 93 focused CPU tests passed, with separate successful CUDA control/resume
  checks. The three trajectory figures were inspected after rendering.

## Next decisions

The requested solver replications and intervention experiments are finished.
The next study should start from this evidence rather than repeating them:

1. Test a frozen normalized-K64 recipe from random initialization and on held-out
   seeds. All present branches reuse a good 80k model.
2. If explaining the repair is the priority, isolate physical epsilon/normalization,
   K, optimizer reset, AdamW settings and weight decay. The successful package
   does not identify a single causal ingredient. Preserve equal-output and
   equal-input distinctions from the earlier repeated-update screen.
3. Evaluate expected-loss gradients and downstream coding at the new checkpoints.
   Near-floor conditional SWD alone does not establish gradient fidelity or SER.
4. Continue the already planned matched fast baselines and larger structured
   channel. The method's conditional construction remains prior art; these
   operating results do not create a new conditional-transport contribution.

## Artifacts

Local suite: `results/sspa_interventions_20261010/`. Each task has atomic
checkpoints, validation trajectories, training traces, invocation history and
endpoint records. Canonical exports are `audit.json` and `endpoints.csv`.
The three `trajectories_*.pdf/png` figures show conditional SWD, variance and
conditional mean error. All seeds, failed branches and the delayed failure are
visible. The lines connect saved evaluations; failure times between checkpoints
are not resolved by interpolation.

The runner is `scripts/run_sspa_interventions.py`; the auditor is
`scripts/report_sspa_interventions.py`. Portable evidence is
`Journal_version/evidence/sspa_intervention_evidence_20261010.tar.gz`.
It includes complete JSON/CSV records and plots, source snapshots, tests, 80k
parents and 80k/200k/final child checkpoints. Intermediate child checkpoints
and full replay cloud/plan tensors remain local with hashes. A full tensor audit
needs those omitted files or their regeneration. The archive records and verifies
each included file's SHA256. Pinned official loss code retains its MIT license.

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/report_sspa_interventions.py \
  --suite-dir results/sspa_interventions_20261010
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/plot_sspa_interventions.py \
  --suite-dir results/sspa_interventions_20261010
```
