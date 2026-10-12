# Fast-baseline implementation checks, 12 October

Protocol: [fast baseline comparison](fast_baseline_protocol_20261012.md).
These checks validate the new code and identify development needs. They are not
final comparison data, and no manuscript tables have been replaced.

## Software checks

- Nine unittest cases passed on CPU and CUDA. Parameterized cases cover all six
  model families, all four channels, all default inference step counts, finite-
  difference input derivatives, actual network evaluation counts, positive
  Gaussian covariance, upstream flow interpolation and consistency boundary
  preconditioning, RNG isolation, strict manifest reuse and checkpoint loading.
- Generator/model/target/critic continuation matches uninterrupted updates
  exactly, including a WGAN checkpoint with populated generator optimizer state.
- Downstream AE continuation also matches uninterrupted training exactly.
- A full CUDA pipeline check completed 60 entries: 14 learned sampler settings
  plus one analytic reference on each of AWGN, Rayleigh, SSPA and TDL.
  It exercised 240 analytic-channel SER/BER operating points and 56 common-codec
  gradient checks. Budgets were only ten generator updates and three AE updates;
  its numerical performance is deliberately not used for scientific conclusions.
- The completed AWGN development suite was resumed successfully without new
  training updates. No dependency installation or update was performed.

Pipeline output: `results/fast_baselines_pipeline_20261012/`.

## Bounded AWGN learning pilot

Output: `results/fast_baselines_awgn_development_20261012/`.
Seed 9101, n=7, 5 dB, rate 4/7, 5000 outer updates, batch 512,
2,560,000 analytic training outputs per family, final EMA weights. Inference
networks contain 59,776--60,152 parameters. WGAN made 1000 generator updates
and 5000 critic updates; the other families made 5000 model updates. No AE
training was performed in this pilot.

RTX 5060 Ti, Torch 2.11.0+cu128, CUDA 12.8, float32, two CPU threads,
no compilation/graphs. Summed measured generator training time was 48.98 s,
excluding evaluation, initialization and checkpoint I/O. This number is a local
pilot measurement, not a projected paper-budget cost or total job wall time.

Global SWD uses 4096 outputs; conditional SWD averages 32 fixed inputs with 64
outputs each. Both use 64 projections. The shared analytic-versus-analytic
conditional SWD floor is **0.11748**. All sampler settings are shown below,
without selecting the best NFE after observing the results.

| Family | NFE | Global SWD | Conditional SWD |
|---|---:|---:|---:|
| Gaussian | 1 | 0.02156 | 0.11495 |
| DDIM | 10 | 0.07173 | 0.13269 |
| DDIM | 20 | 0.03166 | 0.12070 |
| DDIM | 50 | 0.03473 | 0.12286 |
| DDIM | 100 | 0.07686 | 0.15193 |
| Flow | 1 | 0.08330 | 0.37876 |
| Flow | 2 | 0.08592 | 0.24930 |
| Flow | 4 | 0.05532 | 0.14615 |
| Flow | 8 | 0.03895 | 0.12187 |
| Flow | 16 | 0.02934 | 0.11593 |
| Consistency | 1 | 0.84514 | 0.82619 |
| Consistency | 2 | 0.63642 | 0.68667 |
| K64 Sinkhorn | 1 | 0.03460 | 0.11846 |
| Clipped WGAN | 1 | 0.43549 | 0.43356 |

Gaussian, DDIM-20/50, multi-step flow and K64 Sinkhorn learn useful AWGN
conditional distributions at this short budget. Proximity to this finite-sample
floor does not establish equality of laws or a significant ranking. A measured
value below the floor can occur because the floor itself is sampled.

The one-step flow result illustrates why checking only global output SWD is
insufficient: its conditional score is substantially worse. This is a pilot
observation on one seed, not a general theorem about flow matching.

**Consistency and clipped WGAN have not passed a competitive-quality gate.**
Their weak default results must not be used to claim Sinkhorn superiority over
these families. Next development should test adequate schedules and a bounded
hyperparameter allowance, inspect conditional means/variances, and verify the
consistency adaptation further if it remains poor. The small consistency loss
alone does not establish useful generated distributions. Do not launch final
multi-seed comparisons or discard either family based on this pilot.

Modern fast-baseline code is now implemented and exercised. Competitive
configurations, seeded downstream conclusions, a measured-compute sweep and
the larger jointly generated wireless block are still pending.
