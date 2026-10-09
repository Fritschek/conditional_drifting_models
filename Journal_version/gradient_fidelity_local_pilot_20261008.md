# Local gradient-fidelity pilot

Started 8 October 2026, completed 9 October 2026. Existing seed-7 checkpoints,
RTX 5060 Ti, PyTorch 2.11.0+cu128. No generator or codec was retrained. No
manuscript, checkpoint, or historical result was modified.

## Question and protocol

Does a channel surrogate provide the encoder with the same expected
cross-entropy gradient as the analytic channel? How does this relate to SWD
and the actual effect of an encoder step on the analytic channel?

- AWGN: M=16, n=7, Eb/N0=5 dB, rate=4/7.
- SSPA: M=64, n=8, Eb/N0=8 dB, rate=3/4.
- Two frozen encoder/decoder pairs per channel: one originally trained on the
  analytic channel, the other on the selected condition-wise Sinkhorn model.
  Within each comparison, every surrogate receives the identical codebook and
  decoder. These are two operating points, not independent training seeds.
- Messages are uniform. The complete codebook is centered and normalized to
  unit mean-square real coordinate power. This replaces random minibatch
  standardization for this controlled test. Noise variance stays fixed during
  differentiation and encoder perturbations.
- The existing SSPA implementation is reused, including its noise_std/sqrt(2)
  per-real-component noise convention. AWGN noise_std=0.52602214; SSPA
  noise_std=0.32505314. Checkpoint dimensions and noise calibration agree.
- Gradients average cross-entropy vector-Jacobian products over independent
  noise draws. We compare both input/codeword gradients and their propagation
  into encoder parameters, including differentiation through the power
  constraint. We do not compare samplewise Jacobians between unrelated latent
  parameterizations.
- SWD is the repository's average absolute projected quantile difference.
  Conditional SWD averages over all codewords; global SWD pools the same
  output samples. Projections and reference samples are shared across methods.
  These codeword-based SWDs are not the paper's Gaussian-input benchmark rows.
- Actual encoder steps have norm fraction * ||theta||, with fractions 1e-4,
  3e-4, and 1e-3, in the negative surrogate-gradient direction. The decoder is
  frozen, the full codebook is renormalized, and loss/SER/BER are evaluated on
  the analytic channel. Baseline and perturbed evaluations use common random
  numbers, independent of gradient estimation. Each step starts from the same
  original encoder, not from the previous step.

The step normalization isolates direction. It deliberately removes gradient
magnitude differences. This is not an Adam training experiment, and a small
gradient norm alone does not prove that Adam would train slowly.

## Sampling and checks

The first run covers analytic, kernel-target drifting, joint Sinkhorn,
condition-wise Sinkhorn, WGAN, DDIM-10, and DDIM-100. SSPA additionally includes
the degraded full-budget condition-wise checkpoint. Four Monte Carlo repeats
use 2,048 gradient samples per message, 16,384 analytic-reference samples per
message, 16,384 analytic evaluation samples per message, and 1,024 samples per
codeword for SWD, with 64 projections.

The SSPA confirmation increases these to 8,192, 65,536, 32,768, and 4,096,
respectively. It retains analytic, condition-wise Sinkhorn, WGAN, DDIM-10,
DDIM-100, and the full-budget condition-wise checkpoint. Thus each surrogate
gradient averages 2,097,152 draws; the analytic gradient reference averages
16,777,216 draws; each true-channel evaluation uses 8,388,608 draws.

The analytic gradient passed a double-precision, common-noise directional
finite-difference check at all four frozen codecs. Six focused tests passed:
balanced sampling/chunk scaling, power normalization, SSPA implementation
agreement, finite differences, gradient-alignment calculations, and the
encoder chain rule. They ran through Python's unittest harness, without
installing dependencies.

In the SSPA confirmation, independent halves of the analytic reference have
encoder-gradient cosines 0.9987 and 0.9961 for the two codecs. The independent
lower-sample analytic comparisons below provide additional sampling floors.
Monte Carlo repeats quantify sampling uncertainty only, not training-seed
uncertainty. Training capacities and budgets of these old checkpoints differ.

## Main observations

### AWGN

Condition-wise Sinkhorn encoder-gradient cosine is 0.948 at the analytic codec
and 0.943 at the Sinkhorn codec. Independent analytic estimates give 0.975 and
0.951; DDIM-100 gives 0.968 and 0.942. Small normalized Sinkhorn-gradient steps
lower true-channel cross-entropy at both codecs. These results are close to
the Monte Carlo resolution of this first run, not evidence that one of these
methods is superior.

### SSPA: analytic-trained codec, higher-sample confirmation

Norm ratio means ||surrogate encoder gradient|| / ||analytic encoder gradient||.
The analytic row uses an independent Monte Carlo estimate of the same channel.

| Channel model | Conditional SWD | Encoder cosine | Encoder norm ratio | True CE change, step 1e-4 |
|---|---:|---:|---:|---:|
| Analytic sampling floor | 0.00656 | 0.998 | 1.000 | -6.71e-6 |
| Condition-wise Sinkhorn | 0.02790 | 0.852 | 0.531 | -5.75e-6 |
| WGAN | 0.08760 | 0.227 | 3.125 | -1.51e-6 |
| DDIM-10 | 0.04865 | 0.956 | 0.048 | -6.41e-6 |
| DDIM-100 | 0.01181 | 0.995 | 0.872 | -6.70e-6 |

The paired Monte Carlo SE of the CE changes is between 5.5e-8 and 8.8e-8 for
these rows. Complete values, individual repeats, error counts, BER, all three
step sizes, and input-gradient results are in the raw reports.

One informative ranking reversal: condition-wise Sinkhorn has lower
conditional SWD than DDIM-10, but DDIM-10 has a better-aligned encoder-gradient
direction and gives a larger reduction in true loss for the same normalized
parameter step. The same SWD ranking holds globally at these codewords
(0.01024 versus 0.01400). However, DDIM-10 severely underestimates gradient
magnitude. Neither SWD nor gradient cosine alone describes all of this.

### SSPA: Sinkhorn-trained codec, higher-sample confirmation

Condition-wise Sinkhorn has conditional SWD 0.02875, encoder cosine 0.625,
and norm ratio 0.128. DDIM-100 gives 0.01108, 0.975, and 0.831. The independent
analytic encoder cosine is 0.993. Individual Sinkhorn gradient repeats have
encoder cosines 0.634, 0.689, 0.448, and 0.558: sample noise is visible, but
the mismatch persists across all four repeats.

The normalized Sinkhorn step lowers analytic CE by 5.05e-6 (MC SE 1.18e-7),
compared with 7.88e-6 (MC SE 1.21e-7) for the analytic gradient and 7.78e-6
(MC SE 1.04e-7) for DDIM-100. Sinkhorn's direction is useful, but it is not a
faithful replacement for the analytic gradient at this operating point.

WGAN is nearly orthogonal here (cosine 0.017), and its smallest step has no
resolved loss benefit. This is a single WGAN checkpoint at a fixed decoder,
not a statement about average WGAN downstream performance.

### Degraded full-budget SSPA checkpoint

The full-budget checkpoint has conditional SWD about 1.83-1.87 and an encoder
gradient norm roughly 2,271-2,549 times the analytic norm at these codecs.
Its direction depends strongly on the codec (cosines 0.704 and 0.059).
Equal-norm steps suppress this magnitude error, so their occasional loss
improvement must not be interpreted as evidence that the checkpoint is usable.
These measurements describe the saved failure; they do not identify what
caused training to fail. A controlled trajectory is still needed for that.

## What this establishes, and what remains

The pilot supports measuring expected-loss gradient fidelity alongside
conditional SWD. It exposes direction and magnitude errors at actual learned
codewords, and connects directions to observed true-channel loss changes.
This is evidence for the research direction in the theory note, not a proof
that gradient error alone explains the final BER/SER results or a new-method
claim. Distribution fidelity, decoder adaptation, input coverage, optimizer
dynamics, and sample noise remain relevant.

Next useful checks are more checkpoint seeds and saved intermediate codecs,
followed by local input-neighborhood tests. A method improvement should then
be compared on true-channel learning trajectories, with matched capacity and
budget and a modern low-step baseline. These runs did not train or tune a
gradient-correction method, test measured-data applicability, or resolve the
broader novelty question.

## Files and reproduction

- [Theory note](gradient_fidelity_theory_notes.md)
- [Experiment script](../scripts/run_local_gradient_fidelity.py)
- [Tests](../tests/test_gradient_fidelity.py)
- [First run](../results/gradient_fidelity_local_20261008/README.md)
- [Higher-sample SSPA confirmation](../results/gradient_fidelity_sspa_confirmation_20261008/README.md)

Each run directory contains JSON measurements, saved gradient tensors, and
the executed script. Checkpoint SHA-256 hashes and paths are recorded in JSON.
Result directories are locally available but ignored by git.

```bash
/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_local_gradient_fidelity.py \
  --codec-sources analytic,fiber_sinkhorn \
  --out-dir results/gradient_fidelity_repeat

/home/rick/.local/share/mamba/envs/ml/bin/python -u scripts/run_local_gradient_fidelity.py \
  --channels SSPA --codec-sources analytic,fiber_sinkhorn \
  --methods analytic,fiber_sinkhorn,wgan,ddim10,ddim100 \
  --gradient-samples 8192 --reference-samples 65536 \
  --eval-samples 32768 --metric-samples 4096 \
  --out-dir results/gradient_fidelity_sspa_repeat
```

Output directories must be new. Commands assume the repository root and the
existing local checkpoints. There are no dependency installations or HPC jobs.
