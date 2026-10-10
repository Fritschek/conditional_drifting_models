# SSPA continuous extension to 30,000 updates

Frozen on 10 October 2026 before starting the extension, following the
[4,800-update comparison](sspa_trajectory_results_20261010.md).

## Question and design

Does the early policy-dependent variance deficit persist, disappear or turn into
late degradation as the same training trajectories continue? Extend **all nine**
runs, not only a favorable policy or seed. Keep model, optimizer, sampling,
epsilon calibration, clipping, fixed validation inputs and projection streams
unchanged. This remains development evidence, not a confirmatory model ranking.

Parent suite: `results/sspa_epsilon_trajectories_20261010`.
Child suite: `results/sspa_epsilon_trajectories_30k_20261010`.
Seeds 9001, 9002, 9003; legacy separate adaptive, fixed common, shared adaptive.
Continue each final state at update 4,800 through **30,000 total updates**.
Add validation/checkpoint saves at 10,000 and 30,000; keep the six inherited
checkpoints. No additional candidates before update 4,800. Select minimum
validation conditional SWD with earliest tie, retaining both selected and last.
No SER/BER training, tuning, new epsilon statistic or new architecture in this run.

## Continuation contract

The new CLI option `--continue-from` creates a separate suite. Before training,
it validates completeness of the parent, exact configurations, source hashes,
Torch/device/GPU identity, inherited schedule, and agreement between final
checkpoint contents and saved history/counters. The child manifest stores the
parent manifest SHA-256 and hashes of all 54 inherited checkpoints. Every copied
checkpoint is rechecked. Adam and all RNG states come from the parent checkpoint.
The training module is unchanged, preserving the original source checks.

New CPU/CUDA tests cover exact next updates versus uninterrupted training,
parent immutability, rejected historical-schedule changes, source/budget
mismatches and checkpoint tampering. Existing validation/RNG tests still apply.

Validation/oracle counters and measured times are cumulative, including inherited
work. Report extension-only time by subtraction, not by relabeling cumulative
totals. Initialization reconstructs the fixed calibration before restoring the
checkpoint, as in the prior resume implementation; those redundant loader-only
oracle calls are outside the saved trajectory counters and training timers.
There are 32,768 such extra analytic samples per process initialization. They do
not enter training/validation or influence the restored calibration/RNG state.

## Resource and failure policy

Prior training measured about 2.8--3.7 ms/update without competing CPU tests.
The extension is expected to take roughly 12--16 minutes of GPU training, with
runtime variation. Do not stop a finite but poorly performing seed. Preserve
numerical failures explicitly, and do not interpret an absent checkpoint as a
good outcome. Thirty thousand updates still fall well short of 390,720.

```bash
python scripts/run_sspa_epsilon_trajectories.py --device cuda \
  --continue-from results/sspa_epsilon_trajectories_20261010 \
  --out-dir results/sspa_epsilon_trajectories_30k_20261010 \
  --updates 30000 --checkpoints 0,100,300,1000,2500,4800,10000,30000
```

For resuming this child suite, retain all arguments and add `--resume`.
