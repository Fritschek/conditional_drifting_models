# Coverage replication and saved-sampler checks

Prepared 12 October 2026 after the independent application/coverage assessment.
This is a child study, not a continuation or relabeling of the seed-9111 pilot.
Full runs have not started. Tiny smoke runs test execution only.

## Fixed coverage replication

- Generator seeds: **9201, 9202, 9203**. No occurrences were found in existing
  protocol/source files or result manifests at allocation. These are additional
  development replications, not the later final-paper test seeds.
- Three paired AE initializations per generator seed: `300000 + 10*seed + i`,
  for `i=0,1,2`: 392010--392012, 392020--392022, 392030--392032.
- Two families (diagonal Gaussian and official normalized Sinkhorn K64), two
  input laws. All **12 generators** train from scratch. No inherited codecs.
- **36 surrogate AEs and nine paired analytic AEs**, 45 total.
- Preserve parent architecture (~60k parameters), optimizer, noise, final EMA,
  40,000 generator updates of 1,024 positive outputs, and 20,000 AE updates of
  500 messages. Training exposure totals **491,520,000 positive outputs**;
  evaluation queries and generated/reference outputs are additional.
- Normal law: N(0,I8). Broader law: 75% N(0,I8), 25% uniformly distributed
  direction times a radius uniform on [0,sqrt(512)]. No mixture tuning,
  clipping, new peak-power constraint, or adaptive early stopping.
- Actual codebook normalization subtracts one scalar grand mean and fixes
  average squared codeword norm to eight. It does not center each coordinate.
- Expected squared characterization-input norm is 8 versus 48.6667. Record
  observed query energy and maximum norm too. This is equal output exposure,
  not equal measurement energy, transport compute, or wall time.

### Primary endpoint and uncertainty

For each family and generator seed, average **broad-minus-normal physical-channel
SER at 8 dB** over its three paired AE initializations. Negative means improvement.
Report all three generator-block effects and their descriptive mean/SD. Do not
pool nine AEs as nine independent generator replicas. No significance gate,
winner selection, or mixture retuning is specified by this development study.

The full panel retains SER/BER at 4/6/8 dB, surrogate optimism at 8 dB, per-message
errors, maximum codeword norms, ordinary-input conditional metrics, analytic
floors, radial mean/variance at sqrt(8), 6, 10, 16, sqrt(512), and measured cost.
The historical Gaussian codebook is explicitly a post-hoc diagnostic inherited
from development, never a selection set for the new runs.

Each physical evaluation has 16,384 draws per message, 1,048,576 symbols in total.
For each paired AE block, every variant sees identical labels and standard
Gaussian noise, transformed by its own codewords and the unchanged physical
SSPA. Store all pairwise wrong/correct discordance counts. Reuse standard noise
across SNR points, so SNR points are correlated too. The saved pooled paired MC
SE is a descriptive, conservative approximation under the balanced-message
design; it does not quantify training-seed uncertainty. Per-message natural-label
bit errors and symbol errors must sum to the aggregate integer counts.

### Execution and resume

The new runner imports frozen training/evaluation components without editing
their source. Child manifests pin the parent manifest/results, historical
codebook checkpoint, upstream source, current runner/helper, protocol and runtime.
Both normal and broader generator checkpoints include optimizer/RNG state and
observed input energy. Completed checkpoints are loaded without rewriting them.
Each evaluated task records its manifest/checkpoint hashes. Resume rejects
configuration, source, runtime, parent or completed-checkpoint changes.
One process must own an output directory. Interrupting loses at most work since
the last checkpoint (1,000 updates); unfinished evaluation is recomputed.

The `smoke` profile is separately labeled, uses generator seed 19201 and tiny
budgets, and must never enter the scientific aggregation. It is not the paper
profile, even when it exercises the same numerical paths.

## Fixed-checkpoint sampler panel

Use the completed seed-9111 SSPA flow/diffusion **EMA checkpoints**, with no
generator or AE retraining:

- Explicit-Euler flow: **8, 32, 128** model evaluations.
- DDIM: **20, 50, 100** model evaluations, eta=0.
- Ordinary-input global/conditional metrics with analytic floors, mean and
  variance; common early-codec expected-loss/input and encoder gradients with
  independent analytic MC reference repeats and explicit resolution flags.
- Common-input/common-latent output differences and complete pathwise input
  Jacobians relative to the finest tested sampler. That reference is not the
  physical oracle or proof of numerical convergence.
- Batch-one and batch-500 sampling/input-backward timings, warmups excluded,
  repeated measurements retained. No diffusion teacher cost is invented here.

Explicit initial noise uses a dedicated generator. DDIM's internal eta=0 random
draws cannot change subsequent initial latents when NFE changes. Inputs, truth
clouds and initial-latent streams are shared across NFE settings. MC gradient
draws within each setting remain repeated, not identical copies.

The downstream NFE choice is **not changed by this panel**: Flow-8/DDIM-20 remain
the historical application settings. Review this bounded panel before freezing
any revised downstream setting; do not choose NFE from final-paper SER. No new
family ranking or universal gradient-to-SER predictor is claimed.

## Commands

From `/home/rick/GitHub/conditional_drifting_models`, using the existing environment
and already downloaded pinned official sources. No dependency installation or
HPC submission is part of these commands.

Preflight only, no training or evaluation:

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/run_sspa_coverage_replication.py --out-dir results/sspa_coverage_replication_20261012 --prepare-only
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/run_saved_sampler_panel.py --out-dir results/sspa_saved_sampler_panel_20261012 --prepare-only
```

Full replication, then strict aggregation:

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_sspa_coverage_replication.py --out-dir results/sspa_coverage_replication_20261012
/home/rick/.local/share/mamba/envs/ml/bin/python scripts/report_sspa_coverage_replication.py --suite-dir results/sspa_coverage_replication_20261012
```

Saved-checkpoint panel:

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_saved_sampler_panel.py --out-dir results/sspa_saved_sampler_panel_20261012
```

Resume by appending `--resume` to the same runner command. Do not run multiple
processes against the same directory. Tiny checks use `--profile smoke` and a
separate output directory, optionally `--device cpu`.

## Scope after this round

Inspect the replication without tuning away mixed Sinkhorn results. Apply any
later coverage choice fairly to competitors, including distilled consistency.
The larger jointly generated structured channel still needs a physical/receiver
protocol before implementation. Optical fiber remains a separate requested
follow-up. No checkpoint archives should be added to Git for this new round;
publish compact reports/figures and provenance only.
