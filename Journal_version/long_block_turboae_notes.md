# Long-Block TurboAE Sanity Test Notes

Date: 2026-05-29

## Purpose

Some reviewers may ask whether the channel surrogate behavior changes for
larger block lengths. The short symbolic AE experiments use small block codes;
this test uses the separate `turbo_mingru_decoder` project as a TurboAE-style
long-block baseline.

The baseline should not use the quick single-batch loop. TurboAE training is
brittle and the useful reference path is the overnight trainer configuration
from `turbo_mingru_decoder/run_mingru_overnight.py`.

## Matched Training Settings

The long-block runner in this repo now supports the overnight-style settings:

```text
model_type = cnn_turbo
epochs = 300
batch_size = 500
sample_size = 50000
eval_num_blocks = 50000
eval_every = 10
save_every = 25
learning_rate = 2e-4
weight_decay = 0.01
Eb/N0 = 4 dB
rate = 1/2
encoder/decoder phases = alternate
decoder updates per encoder phase = 5x
decoder Eb/N0 range = [Eb/N0 - 3.5 dB, Eb/N0]
grad_clip_norm = 1.0
```

This is exposed through:

```bash
python scripts/run_turboae_long_block_suite.py \
  --lengths 64,256,1000 \
  --modes analytic,checkpoint \
  --checkpoint results/turboae_long_block_20260529/channel_implants/awgn2_fiber_sinkhorn_seed7/checkpoints/enhanced_direct_awgn_seed7.pt \
  --channel AWGN \
  --device cuda \
  --seed 7 \
  --allow-tf32 \
  --out-dir results/turboae_long_block_20260529/full_seed7
```

For H100 runs, add `--amp --amp-dtype bfloat16` if the baseline remains
numerically stable.

## Current Plumbing Checks

An `n=2` AWGN fiberwise Sinkhorn implant was trained for TurboAE channel symbols:

```text
checkpoint = results/turboae_long_block_20260529/channel_implants/awgn2_fiber_sinkhorn_seed7/checkpoints/enhanced_direct_awgn_seed7.pt
direct_swd = 0.0171
anchor_y_ratio = 1.0016
gaussian_w2_ratio = 1.0010
```

Short plumbing pilots verified that the analytic and checkpoint modes run for
`L = 64`, `256`, and `1000`. These pilots used too few iterations and should not
be used as paper-quality TurboAE performance numbers.

The meaningful paper run should use the matched overnight settings above and
then compare:

```text
train analytic channel -> eval analytic channel
train surrogate channel -> eval analytic channel
```

The paper table should report BER and BLER. For `L=1000`, BLER remains close to
one unless BER is very low, so BER is the primary comparison metric until the
TurboAE baseline is fully converged.

## Full L=64 Seed-7 Result

The full overnight-style `L=64` run completed under:

```text
results/turboae_long_block_20260529/full_seed7
```

Settings:

```text
epochs = 300
batch_size = 500
sample_size = 50000
eval_num_blocks = 50000
eval_every = 10
save_every = 25
Eb/N0 = 4 dB
rate = 1/2
```

| Train channel | Eval channel | Best epoch | Eval BER | Eval BLER |
| --- | --- | ---: | ---: | ---: |
| Analytic AWGN | Analytic AWGN | 300 | 3.33e-4 | 1.50e-2 |
| Fiber-Sinkhorn AWGN surrogate | Analytic AWGN | 290 | 1.31e-3 | 6.85e-2 |

The last surrogate epoch was slightly worse than the best checkpoint:

```text
epoch 300: BER = 1.60e-3, BLER = 7.54e-2
```

Interpretation: this is sufficient as a long-block sanity check. The surrogate
does not cause a catastrophic TurboAE training failure at `L=64`, but it still
has a measurable downstream gap against the analytic channel. The paper should
use this as a robustness/scale check rather than claim exact parity.
