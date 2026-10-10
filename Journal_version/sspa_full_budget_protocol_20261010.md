# SSPA full-budget continuation

**Completed:** all nine trajectories reached 390,720 updates. See the
[results and audit](sspa_full_budget_results_20261010.md). The pause/resume
entries below preserve the execution history, not the current suite state.

Frozen before launch on 10 October 2026, following the completed
[30k continuation](sspa_30k_results_20261010.md) and the read-only extraction of
[historical training histories](sspa_historical_training_20261010.md).

## Question

Do the three epsilon policies reproduce or avoid the late SSPA deterioration
when trained continuously to the original 390,720 optimizer updates? Earlier
endpoint comparisons and historical loss histories do not establish its cause.
No policy is selected or dropped from the nearly tied 30k results.

## Frozen experiment

- All nine trajectories: seeds 9001, 9002, 9003 crossed with legacy separately
  adaptive, fixed-common and shared-adaptive epsilon.
- Parent: `results/sspa_epsilon_trajectories_30k_20261010`.
- Child: `results/sspa_epsilon_trajectories_full_20261010`.
- Restore each update-30,000 model, Adam state, RNG states, calibration and
  counters; continue to **390,720 total updates**, not that many additional ones.
- No numerical training-module changes, model changes, optimizer changes,
  sampling changes, learning-rate decay or new epsilon statistic.
- Preserve all eight parent checkpoints. Add 50k and 70k, then every 10k from
  80k through 180k, then 200k, 250k, 300k, 350k and 390,720. This is 26 saved
  validation checkpoints per trajectory, 234 total including inherited states.
- The expanded schedule is selected before the long run using the historical
  loss-transition region. The historical post-hoc 0.05 loss threshold is not a
  new stopping rule, selection rule or definition of fidelity failure.
- Same fixed validation panel, analytic floors and projection streams. Select
  minimum mean conditional SWD with earliest tie among all 26 checkpoints,
  retaining the last checkpoint as well. No downstream AE or held-out test.
- Training summaries remain every 100 updates. Keep transport residuals, floor
  activation, raw/clipped drift, gradient clipping, validation moments, variance,
  SWD/GW2, and the same small high-accuracy reference checks.

## Execution and failures

The continuation preflight verifies all 72 inherited checkpoint hashes, parent
completeness, numerical source and configuration identity, and saved records.
The original short and 30k suites are not overwritten. The child records its
parent manifest hash and copied checkpoint hashes.

The runner's new `--keep-going` flag changes failure handling only. A numerical
failure is written to the task status and suite failure list, and other tasks
are attempted. Such a task is not declared complete or silently excluded from
aggregation. Keyboard interrupts and other non-Exception exits still propagate.
The runner exits nonzero if any task failed. Finite but poor metrics do not stop
a trajectory. Last scheduled valid checkpoints remain available for inspection.

Expected additional GPU training is approximately three hours on the RTX 5060 Ti,
based on the preceding measured throughput. Reference checks may cost more if
the generator deteriorates. Retain nonconverged reference results explicitly;
do not turn them into numerical-accuracy claims. Timings and sample counters
are cumulative; extension-only totals subtract the parent totals. As before,
each loader initialization makes 32,768 redundant calibration oracle draws
outside the trajectory counters before restoring the saved state.

## Command

```bash
python scripts/run_sspa_epsilon_trajectories.py --device cuda --keep-going \
  --continue-from results/sspa_epsilon_trajectories_30k_20261010 \
  --out-dir results/sspa_epsilon_trajectories_full_20261010 \
  --updates 390720 \
  --checkpoints 0,100,300,1000,2500,4800,10000,30000,50000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000,170000,180000,200000,250000,300000,350000,390720
```

For an interrupted suite retain the arguments and add `--resume`. No claimed
mechanism, winning policy, or successful long-run stabilization is frozen in
advance; those depend on the resulting trajectories and numerical checks.

## Additional exploratory checkpoint analysis

Added while the full run was in progress, after observing the first legacy
trajectory deteriorate between 80k and 90k updates. This is explanatory analysis,
not a preregistered test or an additional selection criterion. Training and
validation are unchanged.

For every saved checkpoint, evaluate parameter norms, conditional-mean input
Jacobians and conditional-covariance input derivatives. Use 128 fresh Gaussian
anchors and 256 fixed latent draws per anchor, CPU-generated with seed 800201,
separate from training, calibration and selection validation. Compare against
the exact Rapp mean Jacobian and constant analytic covariance. Report full-panel
and two half-panel estimates to expose Monte Carlo variability; the halves are
not confidence intervals. Keep all policies, seeds and checkpoints. These
quantities do not measure downstream encoder gradients and alone cannot prove
the cause of deterioration. Run this analysis after training to avoid GPU
contention and save checkpoint hashes for provenance.

## Pause and resume record

The process was gracefully interrupted on 10 October 2026 and its exit verified.
Seed 9001 legacy and fixed-common trajectories completed 390,720 updates.
Seed 9001 shared-adaptive stopped at update 142,197; its last validated, saved
checkpoint is update **140,000**. Optimizer and CPU/CUDA/Python/NumPy RNG states
are present, and numerical source hashes match. Resume will replay 2,197 unsaved
updates. Seeds 9002 and 9003 have not begun the full-budget extension and retain
their completed 30k parent checkpoints.

The runner records the intentional interruption as `state: failed` with
`exception: KeyboardInterrupt`; this is not a numerical failure. Preserve that
record until resume replaces it. The suite is incomplete and must not be
aggregated as nine completed trajectories. Both completed seed-9001 policies
deteriorated; the shared-adaptive trajectory also deteriorated before the pause.
No full-suite conclusion or causal mechanism is established.

Resume from the repository root using the command above with **`--resume`**
added, in the same Torch/CUDA environment. The local interpreter is
`/home/rick/.local/share/mamba/envs/ml/bin/python`. Do not alter the runner or
numerical training source before resuming, since their hashes are checked.
Completed trajectories are loaded without additional training; the partial
trajectory restores its latest checkpoint, and the remaining tasks start from
their 30k parent checkpoints. The exploratory full-suite derivative analysis
has not run. Its unit tests and small CPU execution check passed.

Resumed at the author's request on 10 October 2026 at approximately 11:28 local
time using the command above plus `--resume`. No competing trainer process was
present. Runner and numerical source hashes matched, as did Torch 2.11.0+cu128
and the RTX 5060 Ti device. The GPU driver reported 615.71.09 on resume; driver
identity is not part of the saved manifest's exact-resume check. The suite is
again in progress; the pause details above remain a historical execution record.

Resource accounting must distinguish the retained trajectory from execution
overhead. The 2,197 completed updates discarded at interruption consumed another
35,995,648 oracle outputs (plus any partially executed interrupted step) and
are not included in the restored counters/timers. Each loader initialization
also draws 32,768 calibration outputs outside restored counters: three tasks
were initialized before interruption and nine on resume. Thus the extension
has 393,216 such extra calibration draws across both invocations. The resumed
runner rewrites completed-task result metadata after loading; its peak-memory
field for the two previously completed tasks does not represent their original
training peak and must not be used for a training-memory comparison.
