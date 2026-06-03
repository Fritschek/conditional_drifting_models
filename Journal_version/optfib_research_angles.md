# OptFib Surrogate Research Angles

This note tracks why the optical-fiber channel behaves differently from the
AWGN, Rayleigh, and SSPA cases, and which larger design changes we should test.

## Working Diagnosis

The current fiber channel is not just a harder additive-noise channel. It is a
stochastic nonlinear propagation:

```text
z_{k+1} = exp(j gamma |z_k|^2 DeltaL) z_k + n_k
```

Noise is injected repeatedly, and early perturbations change the later nonlinear
phase rotation. For fixed input `x`, the conditional law `p(y | x)` can therefore
be curved, anisotropic, phase-wrapped, and power-dependent.

This creates several failure modes for the present W-Flow/Sinkhorn drift:

- Raw Euclidean transport is not the natural geometry for phase-like output
  clouds. Sinkhorn barycenters can move mass through the origin instead of along
  angular directions.
- The residual `y - x` is not a clean stochastic residual. It mixes deterministic
  nonlinear phase rotation with amplifier/noise effects.
- Marginal metrics such as direct SWD can be misleading. A surrogate can match
  `p(y)` while missing `p(y | x)`, which is what the autoencoder decoder sees.
- The conditioning metric in raw `x` coordinates does not encode the local fiber
  state: power, phase, nonlinear phase slope, and noiseless trajectory.
- Coding performance depends on decoder decision boundaries, so distributional
  matching alone may not protect BER/SER.

## Test Queue

### 1. Fiber-coordinate transport features

Use target features based on amplitude and phase residuals, not only raw I/Q:

```text
rho = |y| - |x|
delta_theta = angle(y) - angle(x)
features = [rho, sin(delta_theta), cos(delta_theta)]
```

For stability, represent phase by sine/cosine instead of an unwrapped scalar.
Initial variants:

- `polar_residual`: transport cost only in fiber-adapted residual coordinates.
- `raw_plus_polar_residual`: concatenate raw I/Q with the polar residual
  features.
- Tune the polar residual scale, analogous to the `raw_plus_residual` scale.

### 2. Noiseless-base residual with a compatible metric

Subtract the deterministic split-step map and model the stochastic remainder:

```text
base(x) = OptFib(x, use_noise_std=True, noise_std=0)
e = y - base(x)
```

The naive raw version was not enough. Retest only after combining it with a
fiber-coordinate or tangent-space metric.

### 3. Stepwise surrogate

Replace the one-shot channel surrogate by a small learned split-step model:

```text
x -> block_1 -> block_2 -> ... -> block_K -> y
```

Even `K=5` or `K=10` may capture the stochastic phase accumulation better than a
single conditional generator.

### 4. Conditional diffusion or score model for OptFib

The old OptFib diffusion residual result was much stronger downstream than the
current W-Flow variants. Treat diffusion as either:

- the OptFib-specific baseline we must beat, or
- a teacher for distilling a faster one-step surrogate.

### 5. Diffusion-to-drift distillation

Train a conditional diffusion teacher, sample teacher pairs, then fit a one-step
drift model to either its samples or local score/transport directions.

### 6. Coding-aware objective

Add a decoder-boundary proxy or lightweight SER/BER validation loop. For OptFib,
conditional anchor metrics and BER/SER should rank models ahead of direct SWD.

### 7. Channel-state condition features

Replace raw `x` conditioning by features closer to the local fiber state:

```text
[Re x, Im x, |x|, |x|^2, phase(x), noiseless_base(x), local phase slope]
```

This may be needed for joint kernels, MMD, or conditional Sinkhorn.

### 8. Conditional MMD or energy distance

Avoid Sinkhorn barycentric projection entirely for OptFib and use same-condition
cloud matching with MMD/energy distance. This removes one possible source of
linear-barycenter artifacts.

### 9. More samples per condition

For OptFib, spend batch budget on richer conditional clouds rather than many
conditions. Conditional shape appears more important than broad marginal
coverage.

### 10. Explicit OptFib noise sweep

Current OptFib does not naturally follow the AWGN-style `noise_std`/Eb/N0 path.
For BER/SER curves, either sweep `Pn_dBm` or introduce a clearly named
Eb/N0-controlled OptFib variant.

## Current Priority

Start with item 1 because it is the smallest principled change: it keeps the
W-Flow machinery but changes the geometry of the matching problem. If that does
not move conditional anchor metrics and BER/SER, the next priority should be
item 3 or item 4 rather than more scalar tuning.

## First Local Result

Seed-7 OptFib GPU diagnostics support item 1:

| Variant | Direct SWD | Anchor SWD | Anchor Gaussian W2 | BER | SER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw+residual fiber Sinkhorn, scale 0.25 | 0.0418 | 0.2036 | 0.2638 | 0.2923 | 0.6429 |
| Raw+polar fiber Sinkhorn | 0.0318 | 0.1942 | 0.2593 | 0.2917 | 0.5876 |
| Raw+polar fiber Energy | 0.0373 | 0.1667 | 0.2418 | 0.1921 | 0.3845 |

The first raw+polar Sinkhorn result mainly improved SER and conditional metrics.
The larger gain came from replacing Sinkhorn barycentric projection by a
same-condition energy distance objective. This supports two parts of the
diagnosis: the geometry matters, and the barycentric projection itself is harmful
for OptFib.

Current first candidate for broader checking: `fiber_energy_rawpolar`.

## Physics Surrogate Result

Item 3 also looks promising, but in a simpler form than a learned stepwise model.
Replacing the default `Kstep=50` fiber channel by a coarse physical split-step
channel gives near-floor distribution metrics for `Kstep >= 2`:

| Surrogate | Direct SWD | Anchor SWD | Anchor Gaussian W2 |
| --- | ---: | ---: | ---: |
| K=1 | 0.0081 | 1.1108 | 1.9432 |
| K=2 | 0.0043 | 0.1699 | 0.2469 |
| K=5 | 0.0052 | 0.1719 | 0.2529 |
| K=20 | 0.0049 | 0.1643 | 0.2376 |
| K=50 vs K=50 floor | 0.0059 | 0.1765 | 0.2596 |

K=1 is another metric-mismatch example: marginal SWD is low, but the conditional
anchor metrics fail badly. K=20 trained the symbolic autoencoder stably with a
lower learning rate and evaluated against K=50 at final SER/BER about
`0.347/0.247`. This is better SER than the learned energy surrogate, but BER is
worse; it is still a strong baseline and a useful sanity check for any learned
OptFib surrogate.

## Reference-Guided Baseline Plan

Online references suggest that the OptFib baseline should not be a single
"analytic AE" number. In optical-fiber E2E learning papers, the stronger
baselines are system-level and physical:

- **True-channel AE / oracle training reference.** Train the same encoder/decoder
  directly through the implemented physical channel, with multiple random
  restarts and validation-based model selection. This is a practical reference,
  not a theoretical optimum.
- **Conventional modulation reference.** Compare against a fixed constellation
  such as QAM/PSK under the same true channel and detector. Fiber E2E papers
  commonly compare learned shaping against conventional QAM or a conventional
  reference system.
- **Information-rate reference.** For shaping claims, report achievable
  information rate, GMI, or spectral efficiency in addition to BER/SER. Li et al.
  optimize AIRs for a simplified nonlinear fiber channel; CSAE work reports GMI
  for 64-QAM and 128-QAM.
- **Same-demodulator channel-emulator fidelity.** For learned channel models,
  evaluate the emulator by feeding true-channel and emulator samples to the same
  demodulator/decoder and reporting the BER/SER mismatch. CSAE/CGAN work reports
  average BER difference, MSE, and KL divergence for this purpose.
- **Physical reduced-complexity surrogate.** In our simplified channel, coarse
  split-step OptFib with `Kstep=20` is a strong baseline. It is cheap, preserves
  conditional geometry, and should be included whenever learned surrogates claim
  speed or differentiability benefits.
- **Optical DSP baseline if we expand realism.** For more realistic fiber links,
  digital back-propagation and related DSP methods are standard baselines, with
  gains measured in BER or Q-factor.

For the journal experiments, use the following hierarchy:

1. True OptFib AE with multi-start/early stopping.
2. Fixed constellation baseline under true OptFib.
3. Coarse physical OptFib surrogate, especially `Kstep=20`.
4. Learned surrogates: W-Flow variants, WGAN, diffusion.
5. Report both BER/SER and conditional channel fidelity; do not rank OptFib
   methods by direct SWD alone.

## Concrete External References To Use

- **Li/Hager/Garcia/Wymeersch, ECOC 2018.** The `henkwymeersch/AutoencoderFiber`
  repository implements the simplified nonlinear memoryless fiber-channel AE and
  explicitly computes achievable information rate. This makes AIR/MI the right
  additional statistic for our OptFib section, alongside BER/SER.
- **KIT CEL lecture examples.** The `kit-cel/lecture-examples` repository has
  `mloc/ch8_Autoencoders/Autoencoder_Optical_Fiber_variableBatchSize.ipynb` and
  `Autoencoder_PolicyGradient_Optical_Fiber.ipynb`. The associated notebooks use
  the same dispersionless split-step form with `L=5000`, `gamma=1.27`,
  `Pn=-21.3 dBm`, and usually `Kstep=50`, with validation BER as the main
  performance check.
- **KIT CEL VAE equalizer.** The `kit-cel/vae-equalizer` repository is more
  realistic coherent-DP equalization rather than our memoryless channel, but it
  is useful for baseline language: compare against state-of-the-art equalizers
  and report system-level optical metrics, not only sample-space distances.

Immediate metric consequence: for every symbolic AE run, report

```text
cross_entropy_bits = CE_nats / log(2)
AIR_bits_per_message >= log2(M) - cross_entropy_bits
normalized_AIR = AIR_bits_per_message / log2(M)
```

This is the classifier-decoder lower-bound counterpart of the AIR viewpoint in
the Wymeersch/Hager line of work. It is not a replacement for BER/SER, but it
will tell us whether a surrogate preserves soft posterior information even when
hard-decision BER/SER is noisy.

## Corrected OptFib Autoencoder Baseline

The old seed-7 analytic OptFib baseline was not comparable to the optical AE
references because the symbolic encoder emitted unit-scale coordinates, while
the KIT/Wymeersch-style channel uses optical input power. With `Pin=4 dBm`, the
average squared complex symbol norm is

```text
P_in = 10^((4 - 30) / 10) = 2.5119e-3 W.
```

Adding explicit code-power normalization and using a larger optical-AE-style
network changes the seed-7 picture:

| Training channel | AE setup | SER | BER | AIR bits |
| --- | --- | ---: | ---: | ---: |
| Old analytic K=50 | 2 layers, 16 hidden, unit scale | 0.4050 | 0.2604 | n/a |
| Analytic K=50 | 2 layers, 50 hidden, `Pin=4 dBm` | 0.0291 | 0.0194 | 3.849 |
| Analytic K=50 | 4 layers, 50 hidden, `Pin=4 dBm` | 0.0236 | 0.0131 | 3.890 |
| Coarse physical K=20, eval K=50 | 4 layers, 50 hidden, `Pin=4 dBm` | 0.0080 | 0.0051 | 3.959 |

This puts our analytic/physical references into the same ballpark as the KIT
notebooks (`0.0111` validation SER for the direct optical AE and `0.0357` for
the policy-gradient notebook). The K=20 training channel is currently the
strongest practical reference: it preserves the K=50 conditional law at the
evaluated operating point but gives a smoother optimization path for the
autoencoder.

First learned-surrogate retest at this corrected scale:

| Surrogate | Training condition scale | Downstream SER | Downstream BER | AIR bits |
| --- | --- | ---: | ---: | ---: |
| Fiber energy raw+polar | `Pin=4 dBm` Gaussian conditions | 0.8053 | 0.3971 | -0.002 |

The learned surrogate failure is now clearer: even after matching the input
power, the model loses conditional information. Its direct SWD is moderate, but
anchor-conditioned metrics are far above the stochastic floor. The next learned
surrogate tests should therefore train on the learned/fixed constellation
support or use a physical stepwise surrogate/distillation target, not only
Gaussian conditions at the right power.

## Drift Setup Findings, 2026-05-28

The OptFib failure mode is not just a kernel scale issue. At `Pin=4 dBm`, the
nonlinear phase is highly oscillatory: two learned codewords separated by only
about `0.0057` in I/Q can have conditional residual means separated by about
`0.082`. A raw-coordinate MLP therefore smooths across nearby codewords and
compresses the conditional mean map.

Implemented/tested fixes:

| Setup | OptFib anchor y ratio | mean ratio | Gaussian-W2 ratio | residual SWD |
| --- | ---: | ---: | ---: | ---: |
| Energy residual, codebook, raw input, 4/4 samples | 5.13 | 6.96 | 5.95 | 0.0059 |
| Phase features, energy + weak supervised term, 4/4 samples | 1.83 | 1.51 | 2.20 | 0.0027 |
| Phase features, energy + weak supervised term, 8/8 samples | 1.45 | 1.12 | 1.59 | 0.0021 |
| Phase features, energy + weak supervised term, 16/16 samples | 1.31 | 1.17 | 1.32 | 0.0015 |

The best local channel-model setup so far is:

```text
condition_feature_mode = optfib_phase
drift_field = fiber_energy
target_kernel_scale = 250
latent_input_scale = 4
fiber_supervised_weight = 0.003125
fiber_generated_samples = fiber_positive_samples = fiber_reference_samples = 16
condition_codebook = strong K=20 OptFib AE codebook
condition_jitter_std = 0.001
```

However, coding is much stricter than these channel metrics. Evaluating the
strong K=20-trained AE through the learned phase surrogate still gives SER about
`0.172` for the 16/16 model, versus about `0.008` under the true K=50 channel.
The residual SWD is still roughly an order of magnitude above the analytic
sampling floor (`0.0015` vs about `0.00013`). This setup is promising for channel
fidelity, but it is not yet a watertight OptFib coding surrogate.

Conclusion for the next pass: do not send a full OptFib learned-surrogate coding
run to HPC yet. The next local tests should either push the phase-feature model
toward the analytic floor with larger per-condition sample counts/longer
training, or switch to a physics-residual model around a coarse split-step
baseline (`K=20 -> K=50` residual), which is better aligned with the successful
AE baseline.

### Physics-base refiner, 2026-05-28

The first stochastic residual refiner used a sampled `K=20` split-step output as
a base and trained a stochastic correction to `K=50`. This was not safe: it
matched neither channel nor decoder statistics, with fixed-codebook SER around
`0.452`. The failure is expected in hindsight because an independent stochastic
correction double-counts decoder-relevant noise.

A conservative variant works much better:

```text
physics_base_mode = optfib
physics_base_optfib_kstep = 20
condition_context_mode = input_base
condition_feature_mode = optfib_phase
drift_field = fiber_moment
latent_input_scale = 0
target_kernel_scale = 250
fiber_moment_mean_weight = 1
fiber_moment_cov_weight = 0
fiber_generated_samples = fiber_positive_samples = fiber_reference_samples = 16
```

This keeps the stochasticity in the physical `K=20` base and only learns a
deterministic conditional mean correction. For seed 7 around the strong
`K=20`-trained codebook:

| Surrogate / training path | Channel anchor y ratio | Gaussian-W2 ratio | K50 eval SER | K50 eval BER |
| --- | ---: | ---: | ---: | ---: |
| Phase energy surrogate, fixed K20 AE | 1.31 | 1.32 | 0.1719 | 0.0885 |
| Stochastic K20 correction, fixed K20 AE | 4.07 | 4.58 | 0.4524 | 0.2103 |
| Deterministic K20 mean refiner, fixed K20 AE | 1.04 | 1.05 | 0.0141 | 0.0092 |
| Train AE from scratch through deterministic refiner | n/a | n/a | 0.0273 | 0.0148 |
| Fine-tune K20 AE through deterministic refiner | n/a | n/a | 0.0096 final / 0.0080 best | 0.0064 final / 0.0053 best |

The important change is not another kernel scale; it is the architecture. The
surrogate must inherit the safe physical channel law and learn only a small
deterministic correction. This is now good enough for local fine-tuning and
decoder-aware validation, but it still does not beat the pure `K=20 -> K=50`
physical training baseline. For the paper, the pure physical baseline remains
the strongest OptFib reference, while the physics-base refiner is the learned
variant worth carrying into a small multi-seed check.

## Decoder-Aware Fidelity Metrics

Raw SWD can be small while BER/SER is bad because coding performance is not a
global Euclidean distribution statistic. For a fixed encoder codeword `x_m` and
decoder `D`, SER is

```text
SER = (1/M) sum_m P_{Y|x_m}[argmax_j D_j(Y) != m].
```

Therefore the relevant test functions are decoder decision regions, logits,
posterior probabilities, and margins. A good channel surrogate must preserve the
push-forward of `P(Y|x_m)` through the decoder, especially near decision
boundaries. The new diagnostics added to `run_symbolic_awgn_benchmark.py`
therefore report:

- decoder SER/BER and cross-entropy gaps between true and surrogate channels;
- per-message confusion-row total variation and worst-message indices;
- conditional SWD in decoder posterior space and log-posterior space;
- conditional SWD of decoder margins;
- near-boundary mass gaps using posterior margin.

For the best local OptFib phase-feature surrogate (`latent_input_scale=4`,
`16/16` samples per condition), raw metrics looked close:

```text
anchor_y_ratio = 1.16
anchor_gaussian_w2_ratio = 1.26
residual_swd = 0.00148
```

But the decoder-aware metrics correctly flag the coding failure:

```text
true SER through analytic channel      = 0.0068
surrogate SER through learned channel  = 0.1675
decoder confusion TV ratio             = 30.7x floor
decoder posterior SWD ratio            = 29.2x floor
decoder posterior-margin SWD ratio     = 30.6x floor
```

The failure is localized, not uniform. With the same run, per-message surrogate
SER is high for symbols `0`, `2`, `3`, `4`, `6`, and `15`, while several symbols
remain almost perfect. This matches the phase-wrap diagnosis: nearby learned
codewords can sit on different sides of a nonlinear phase transition, so a small
geometric mismatch becomes a large decoder decision-region mismatch.

Practical rule for the paper experiments: rank learned channel models by
decoder-aware metrics before spending HPC budget on BER/SER curves. Direct SWD
and anchor SWD are useful screening diagnostics, but a learned OptFib surrogate
should not be trusted for coding unless decoder confusion TV, posterior SWD, and
margin SWD are near their Monte Carlo floors.

## OptFib Partition Diagnostic

Added `scripts/run_optfib_decoder_partition_diagnostics.py` to split the OptFib
channel into interpretable pieces under a fixed strong AE/decoder. The script
compares each candidate to true K=50 OptFib with both raw conditional metrics and
decoder-aware metrics.

Seed-7 findings for the strong K20-trained AE:

| Candidate | Anchor y ratio | Decoder SER gap | Confusion TV ratio | Margin SWD ratio |
| --- | ---: | ---: | ---: | ---: |
| Deterministic nonlinear phase | 11.06 | -0.0032 | 0.76 | 1.84 |
| Phase + output AWGN | 7.67 | 0.0098 | 4.82 | 4.02 |
| Input AWGN then phase | 4.28 | 0.0728 | 18.75 | 23.50 |
| OptFib K=1 | 7.32 | 0.0085 | 4.00 | 4.13 |
| OptFib K=2 | 2.56 | -0.0017 | 0.76 | 1.07 |
| OptFib K=5 | 1.49 | -0.0002 | 0.53 | 0.68 |
| OptFib K=10 | 0.95 | -0.0020 | 1.15 | 0.89 |
| OptFib K=20 | 1.04 | -0.0029 | 0.89 | 0.86 |
| Learned phase-energy surrogate | 1.87 | 0.1555 | 65.00 | 48.66 |

This separates three facts that raw SWD blurred together:

1. The deterministic nonlinear phase can look terrible in raw sample geometry,
   but it is decoder-safe for the trained AE. It is not the main SER failure.
2. Noise before/nonlinearly inside the phase map is much more dangerous than
   output AWGN. This is the part the learned surrogate must reproduce.
3. A very coarse physical split-step channel (`K=2` or higher here) is already
   decoder-safe, which explains why the K=20 physical training baseline works so
   well.

The learned surrogate currently resembles the dangerous nonlinear-noise failure
more than the safe coarse-physics baseline in decoder space. Next local tests
should therefore focus on physics-residual learning (`K=2` or `K=20` base plus a
learned correction to K=50) instead of learning the full OptFib transition from a
generic latent MLP.

Corrected AE training through coarse physical OptFib at `Pin=4 dBm` gives:

| Training channel | Eval channel | Best SER | Best BER | Best epoch | Final AIR bits |
| --- | --- | ---: | ---: | ---: | ---: |
| K=2 | K=50 | 0.0941 | 0.0494 | 34 | 3.414 |
| K=5 | K=50 | 0.0193 | 0.0111 | 32 | 3.888 |
| K=10 | K=50 | 0.0157 | 0.0090 | 35 | 3.916 |
| K=20 | K=50 | 0.0082 | 0.0052 | 35 | 3.959 |
| K=50 | K=50 | 0.0246 | 0.0137 | 34 | 3.890 |

The fixed-codebook partition diagnostic and the training result are not the same
question. K=2 is decoder-safe around the K20-trained codebook, but training an
AE through K=2 still finds a codebook that transfers poorly to K=50. K20 remains
the strongest physical training baseline; K5/K10 are useful ablations, and K2 is
too coarse for optimization despite looking locally safe.
