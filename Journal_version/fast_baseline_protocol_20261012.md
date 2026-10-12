# Application-focused fast channel comparison

## Research question and scope

How well do fast learned channel simulators support communication-system
optimization, at a stated generator capacity, channel-sample budget and actual
training/inference cost? Conditioning is established prior work. A controlled
joint-versus-condition-wise geometry ablation is optional, not a prerequisite
for this comparison or the central novelty claim.

Initial implementation tests and the bounded AWGN pilot are recorded in
[the check report](fast_baseline_checks_20261012.md). The subsequent
[baseline development round](fast_baseline_development_results_20261012.md)
validates longer WGAN training and learned-score consistency distillation on
AWGN. The `consistency` family below remains standalone CT; it has not silently
been replaced by the separately implemented distilled baseline.

Entry point: `scripts/run_fast_channel_comparison.py`. Historical paper
trainers, checkpoints and SSPA mechanism experiments are unchanged. These
are new capacity-controlled adapters, not reproductions of the original
image experiments or replacements for the old numerical tables.

## Implementations

| Family | Upstream numerical implementation | Channel adaptation |
|---|---|---|
| Diffusion / DDIM | [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion), `22e0df8183507e13a7813f8d38d51b072ca1e67c` | Conditional MLP, cosine 1000-step training, epsilon prediction, deterministic DDIM at 10/20/50/100 evaluations, no image clipping |
| Flow matching | [Meta flow_matching](https://github.com/facebookresearch/flow_matching), `11568d37f8d5a080e12aa7b5305d9c35ae07d136` | Exact `CondOTProbPath`, independent noise/data pairing, conditional MLP, Euler integration at 1/2/4/8/16 evaluations |
| Consistency training | [OpenAI consistency_models](https://github.com/openai/consistency_models), `e32b69ee436d518377db86fb2127a3972d0d8716` | Standalone training without a diffusion teacher, boundary preconditioning, progressive Karras noise grid, adaptive target EMA, L2 loss, conditional MLP, official one/two-step samplers |
| Condition-wise Sinkhorn | [Authors' W-Flow](https://github.com/hanjq17/W-Flow), `1ccb34baf57f9f8f4e188460ea3d19b431403293` | Exact normalized `drift_loss_ot`, quadratic cost, K64 generated/positive/reference clouds at each input, R=0.05, ten solver iterations |
| Learned Gaussian | Local conditional mean/scale network | Gaussian NLL, diagonal covariance by default, full Cholesky option, reparameterized differentiable samples |
| WGAN | Local clipped-WGAN training, following existing repository practice | Conditional generator/critic, RMSprop, five critic updates per generator update, harmonized input law |

Flow matching's *conditional probability path* is not itself conditioning on
the transmitted word. Every generator explicitly receives the fixed input x.
Its one-step Euler result is labeled flow/NFE1, never consistency or distilled
diffusion. The consistency model is separately trained and has no uncounted
teacher cost. `distillation=True` in its upstream denoiser selects boundary
preconditioning; the loss call supplies **no teacher**.

The consistency loader compiles the exact selected upstream class/function
definitions from their AST, avoiding unrelated MPI/image/LPIPS imports. L2 is
an upstream-supported loss, but is an adaptation from the published image
settings. Sampling removes inference-only `no_grad` decorators to permit
input derivatives. DDIM likewise uses the exact single-step routine with
autograd enabled. No numerical source is rewritten.

Sources are pinned and checked for modifications, with hashes recorded in the
manifest. Fetching installs no dependencies and updates no existing checkout.
Keep upstream licenses with their sources. In particular, Meta flow_matching
is CC-BY-NC 4.0, not MIT; do not merge its source into an MIT-only public bundle.

## Shared protocol

- AWGN: n=7, M=16, 5 dB, rate 4/7.
- Rayleigh: n=7, M=16, 12 dB, rate 4/7. Existing independent nonnegative
  Rayleigh amplitudes per real coordinate, no receiver CSI.
- SSPA: n=8, M=64, 8 dB, rate 6/8. Existing complex-pair Rapp channel with
  gain 5, saturation 1.5, smoothness 3. Its implemented per-real-coordinate
  noise deviation is `ebno_to_noise / sqrt(2)`; no silent recalibration.
- TDL: n=8, M=16, 10 dB, rate 4/8. Existing compact four-complex-symbol
  circular TDL-D-lite channel. This is **not** the proposed large block study.

All use iid standard-normal real training inputs and direct output coordinates.
No WGAN-only minibatch power normalization. Each inference network has two
hidden layers with width chosen near 60k total parameters; exact counts are
reported, with a 10% hard tolerance. Time-dependent networks use a 32-component
sinusoidal embedding and SiLU activations. WGAN uses ReLU. This controls size,
not functional expressivity or identical inductive bias. Gaussian heads count
toward capacity. EMA, consistency targets and WGAN critics are reported
separately as training-only parameters.

`batch_size` means oracle output observations per outer training update, not
the number of distinct conditioning inputs. Sinkhorn groups B/K distinct inputs
with K=64 repeats, and also generates B samples plus B independent reference
samples. The other models see B distinct inputs. WGAN performs one critic
update per outer update and a generator update every fifth outer update. Actual
oracle outputs, input anchors and generator/critic updates are recorded.
Equal output exposure does not give equal distinct-input coverage or identical
data-access patterns. This study assumes an oracle simulator permitting repeated
outputs at a fixed input; it is not a passive single-observation dataset test.
`training_generated_outputs` counts explicit generated samples in Sinkhorn and
WGAN, not denoiser/vector-field forward evaluations.

Default optimization is Adam at 2e-4 with gradient clipping 1; Sinkhorn uses
AdamW (0.9,0.95), weight decay .01, clipping 2 and 5000-update linear warmup
from 1e-6. WGAN uses RMSprop and critic clipping .01. Model EMA decay is .999;
inference uses EMA by default and raw weights can be chosen for the entire run.
Consistency target EMA follows the upstream adaptive .95 schedule, with the
progressive scale schedule parameterized from 2 to 150 over the requested
update budget. The default noise limits are .002 to 80, rho=7, sigma_data=.5.
Two-step inference uses indices (0,22,39) on a 40-point grid.

These are transparent development defaults, **not a frozen fair tuning result**.
Per-family JSON overrides allow learning rates, noise scales, EMA and solver
settings to be tuned. Common capacity/batch/budget fields cannot be overridden
per family. For example:

```json
{
  "wgan": {"learning_rate": 0.00005},
  "gaussian": {"gaussian_covariance": "full"},
  "consistency": {"consistency_end_scales": 80}
}
```

Give each family the same declared development allowance on separate seeds.
Freeze configurations before evaluation seeds. Do not discard a baseline after
one untuned run. A diagonal Gaussian is a restricted reference for correlated
TDL; use the full-covariance option there when making a competitive comparison.

## Cost and evaluation

Training seconds cover fresh analytic data, forward/loss/solver work, backward
and optimizer/EMA updates, with CUDA synchronization per outer update. They
exclude initialization, checkpoint I/O and evaluation. There is no extrapolation,
mixed precision, compilation or CUDA graphing. A `--train-seconds` cap can be
used for a separate measured-compute comparison; `--updates` remains a hard cap.
With a time cap, the consistency curriculum uses the elapsed-time fraction.
Time caps can overshoot by one update. They are not interchangeable with an
equal-query experiment, and time-based restarts need not be numerically identical
to uninterrupted runs because measured elapsed times vary.

Sampling timing includes latent sampling and all sampler evaluations. Report
batch-one latency separately from batch-amortized milliseconds per sample.
Default timing batches are 1 and 500, with three warmups and 20 repeats each.
Raw measurements, mean, SD, median and p95 are saved. Input-backward timing is
separate from no-grad sampling timing and uses frozen generator weights.

Final-checkpoint evaluation uses shared held-out input panels and projection
seeds across models. It reports global SWD, anchor-conditioned SWD, Gaussian
W2, mean and covariance errors. Two independent analytic output clouds at the
same anchors define each finite-sample floor. Validation preserves training RNG
state. There is no checkpoint selection on the final evaluation panel.

Optional symbolic AE training (`--ae-updates`) uses fresh, paired initializations,
NAdam .001, hidden width 16, uniform messages and full-codebook centering/unit
mean-square power normalization. Both encoder and decoder train through the
frozen differentiable surrogate; **SER/BER are evaluated on the analytic
channel**, at nominal Eb/N0 plus the requested offsets. Every NFE setting trains
its own AE. Equal AE updates do not imply equal AE training cost; actual times
are saved. Natural-label bit errors, symbol errors and exact trial counts are
retained, including zeros without artificial plot floors.

One analytic-trained AE per channel/seed supplies a common codebook/decoder for
gradient comparisons and conditional distribution checks at learned codewords.
Two independent analytic Monte Carlo gradients give a reference-noise check.
Small or poorly resolved gradients must not support population-fidelity claims.
The default gradient panel is a development check; convergence of its MC
reference still needs assessment before final inference. `--ae-eval-samples`
and `--gradient-samples` are counts **per message**.

## Running

From the repository root with the existing Torch environment:

```bash
python scripts/fetch_fast_baseline_sources.py
python scripts/run_fast_channel_comparison.py \
  --out-dir results/fast_channels_development \
  --channels AWGN,SSPA --seeds 9101 \
  --updates 20000 --batch-size 4096 --parameters 60000
```

This evaluates all default sampling step counts but skips AE training. To include
application tests, add `--ae-updates 20000 --ae-eval-samples 10000`. This runs
14 learned AEs plus one analytic AE per seed/channel and can be substantially
more expensive than generator training. Reduce the NFE grids for an initial
pilot, then retain the same declared settings across final seeds. All four
channels are enabled by `--channels AWGN,Rayleigh,SSPA,TDL`.

For a time-capped generator comparison use a **new directory** and, for example,
`--train-seconds 1800 --updates 1000000`. These are caps per family/channel/seed,
not the total job allocation. Use `--family-config path/to/settings.json` for
documented development choices. No cluster submission or package installation
is performed by this runner.

Interrupt and rerun the identical command with `--resume`. Generator and AE
weights, optimizers, random states and progress resume from the last saved
checkpoint. At most `--save-every` updates are lost. Changed source/configuration
is rejected rather than silently mixed. Results are per model/NFE JSON files
and a final `results.json`; the manifest records the complete protocol.

For independent downstream integration, `load_fast_implant` in
`conditional_drifting/baselines/fast_checkpoint.py` loads a `generator.pt` from
its intact suite directory, verifies its manifest and upstream sources, and
returns a frozen `nn.Module` whose input remains differentiable.

Tests:

```bash
python -m unittest tests.test_fast_channel_baselines -v
FAST_TEST_DEVICE=cuda python -m unittest tests.test_fast_channel_baselines -v
```

## Next scientific decisions

1. Establish competitive AWGN behavior for each family and freeze a bounded
   development allowance. A successful code test is not this quality gate.
2. Apply that allowance to SSPA; compare conditional fidelity, common-codec
   gradients, downstream SER/BER and actual total compute without assuming
   Sinkhorn wins.
3. Freeze held-out seeds and budgets, then extend to Rayleigh and structured
   channels. The larger jointly generated block remains separate work.

No new conditioning-novelty claim, passive measured-data validation, modern
baseline superiority claim, or large-dimensional channel result follows from
the implementation alone.
