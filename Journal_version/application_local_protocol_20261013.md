# Local application development before cluster handoff

This is a new, bounded development protocol. Completed October 10--13 experiments
and their numerical sources remain unchanged. No new confirmatory claims follow
from the local pilot. Freeze this file and the recorded numerical sources once
the development run starts. Corrections require a separately identified run.

## Physical calibration

SSPA keeps the earlier physical operating point, with explicit input-referenced
Eb/N0. M=64, n=8 real coordinates, average input energy eight, six information
bits, and real noise variance N0/2 imply physical Eb/N0=11.0102999566 dB at the
old software label eight. Evaluation uses physical 7.0103/9.0103/11.0103 dB.
Real noise variance is computed independently as Eb/(2 gamma). Tests compare
energy, bit counts and measured real noise, not only two conversion helpers.
No historical data are relabeled in place. This is not a new physical-8-dB run.

## Common-input SSPA comparison

Development seed 9301. All six families use 0.75 N(0,I8) plus 0.25 uniform
direction with radius uniform on [0,sqrt(512)]. The normal-input mode is an
explicit separate ablation, never an unreported family-specific exception.
There are 40,000 updates with 1024 oracle outputs per update and approximately
60k inference parameters. Count EDM teacher and consistency-distillation output
queries together, half the budget each. Distillation receives the same input
law in both stages. Different input grouping and training compute are reported.

Gaussian, Sinkhorn K64, WGAN and distilled consistency use one inference step.
Flow uses 32/128 and DDIM uses 20/50. Final EMA weights only. Each operating
point receives its own 20,000-update AE, batch 500, paired initialization
AE seed 409301. One analytic AE makes nine downstream systems in total.
Evaluation uses 16,384 symbols per message and identical physical noise across
the systems, with paired error discordances retained. This one-seed development
panel is not a replacement for the later independent-generator replication.
All original sampling/loss sources remain pinned; no package upgrade is needed.

## Structured OFDM development

One block is one OFDM symbol: 64 occupied subcarriers, including DC, in centered
frequency order -32..31, 30 kHz spacing, unitary FFT/IFFT and 16-sample CP.
No RF guard bands or coding are modeled. Pilots occupy indices 0,8,...,56 and
are fixed to 1+0j. The remaining 56 carriers contain independently labeled
unit-average-energy QPSK: 112 data bits per block. Natural binary labels map
to [(1+j),(1-j),(-1+j),(-1-j)]/sqrt(2).

Use the TDL-C delays/powers in Sionna v1.2.2's profile, from TR 38.901 Table
7.7.2-3, with RMS delay parameter 300 ns. Source:
https://github.com/NVlabs/sionna/blob/v1.2.2/src/sionna/phy/channel/tr38901/models/TDL-C.json

Normalize the listed tap powers to sum one. Draw independent proper complex
Gaussian taps once per block and evaluate their frequency response at the
actual fractional delays. Maximum delay 2.59569 microseconds fits within the
8.33333-microsecond CP. This is a block-static Gaussian-tap profile experiment,
not Sionna's sum-of-sinusoids implementation or full standardized deployment.
No time evolution, Doppler, MIMO or measured-data claim is made.

Generate all 128 real output coordinates together with a latent of at least
128 real coordinates. Pilots/data share the fading realization. Neither
generators nor receivers receive the fading realization or interference state.
Conditioning contains the complete transmitted block.

Two conditions are fixed before training:

- Clean: Y=diag(H)X+N.
- Interference: Y=diag(H)X+N+B J, B~Bernoulli(0.1) once per block,
  J~CN(0,I64) independent of X,H,N,B. This is a custom extension, not TDL-C.

Train at Es/N0=10 dB with Es=1 per occupied subcarrier and N~CN(0,0.1 I).
Each real noise component has variance 0.05. Interference power is one per
complex coordinate when active, fixed relative to Es, not rescaled with test
noise. Its active INR is 10 dB at training. Evaluate physical-channel receiver
curves at Es/N0=0,5,10,15,20 dB. These are receiver robustness tests, not
multi-SNR fidelity claims for the nominal-only surrogate. Any later Eb/N0
conversion must account explicitly for CP, pilots and coding.

All generators use the same 50/50 mixture of QPSK data and CN(0,1) data
coordinates; pilots remain fixed in both components. Use distributional energy
normalization, never random per-block normalization. This law does not claim
to cover every later optimized constellation or every direction in 128D.

Initial development seed 9401 uses 4000 updates, batch 256 and approximately
600k inference parameters per generator. Included by default: learned full
Gaussian, two-component full Gaussian mixture, condition-wise Sinkhorn, Flow-32
and counted consistency distillation. WGAN and DDIM-20 are supported additional
controls. Teacher pretraining is included in the common query budget. The
mixture learns constant weights and input-dependent component means/Cholesky
factors; it is given no activity labels or physical covariance parameters.
Constant mixing weights permit ordinary reparameterized input derivatives.

This MLP pilot validates the problem and interfaces. It does not freeze the
final larger-channel backbone or establish competitive capacity/tuning. A
Gaussian or two-component mixture is well specified for the corresponding
conditional law and is expected to be a strong reference, not a weak baseline.

A common receiver has two 256-wide SiLU layers and 56 four-class outputs.
Train 5000 updates, batch 256, Adam 0.001, identical initialization across
implants, then test on the physical simulator. Analytic-channel training is
included. A separate conventional pilot-LMMSE channel estimate and scalar
equalizer know the PDP and average noise/interference covariance, not the
instantaneous fading or activity. Disclose this statistical prior advantage.

An explicit `--learn-constellation` mode jointly optimizes four centered,
unit-average-energy constellation points and the same receiver through each
implant. It is a separate run, not mixed with fixed-QPSK results. Conventional
receiver rows remain fixed-QPSK contextual references in that mode.

Retain overall and activity-stratified BER/SER/block-error counts, trials,
block-based BER Monte Carlo SE, and exact one-sided 95% zero-block-error upper
bounds. Activity labels are used only in evaluation. Use common channel/label
random streams across receivers, without assuming bit errors within a block
are independent. Store nominal-anchor SWD, full covariance error, pilot/data
cross-covariance error, energy-tail exceedances and analytic sampling floors.
Report batch-one and batch-amortized forward/backward timing separately.

## Local checks and execution

Smoke profiles use tiny update budgets and separate output directories. They
test all samplers, exact resume, finite losses/gradients, counting and I/O only.
Simulator tests cover physical noise, frequency covariance, shared hidden
activity, pilots, FFT energy, and sufficient-CP convolution equivalence.
The larger-channel learned-constellation mode must also pass a gradient smoke.

```bash
python -m unittest discover -s tests -p test_application_local.py
python scripts/run_common_coverage_application.py --prepare-only --out-dir results/common_coverage_9301
python scripts/run_common_coverage_application.py --out-dir results/common_coverage_9301
python scripts/run_structured_ofdm_local.py --condition clean --out-dir results/ofdm_clean_9401
python scripts/run_structured_ofdm_local.py --condition interference --out-dir results/ofdm_interference_9401
```

Use `--profile smoke` with a new directory for execution checks; `--resume`
requires identical source/configuration/runtime manifests. Checkpoints retain
optimizer, EMA, RNG and update counters. Model weights remain in ignored
results directories. Record complete measured costs, not projected paper times.

## Cluster gate

Do not submit the full campaign until calibration tests, receiver references,
common-input development and both structured pilots have been inspected.
Freeze architecture, tuning allowance, seed allocation, operating points and
query/compute comparison tracks before confirmatory evaluation. Ten independent
generator seeds with three paired receiver seeds are a provisional scale, not
a launched campaign. Pack tasks into at most 32 SLURM array entries with strict
completeness validation and resume. No dependency upgrades inside HPC jobs.
