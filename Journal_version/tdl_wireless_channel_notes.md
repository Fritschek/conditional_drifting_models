# TDL Wireless Channel Replacement Notes

Date: 2026-05-28

## Motivation

OptFib is useful as a stress test, but it pulls the journal story away from
wireless channel simulation and toward physics-assisted optical surrogate
modeling. A tapped-delay-line channel is a better fourth channel for the journal:
it remains wireless, adds memory/inter-symbol interference, and still fits the
existing conditional channel API.

The implemented channel is named `TDL`. Its default profile is now
`TDL-D-lite`, a compact short-block approximation of 3GPP TR 38.901 TDL-D.

## Channel

For each codeword, the channel samples complex taps and applies circular
convolution over the short complex block:

```text
y[t] = sum_l h_l x[t-l mod T] + n[t].
```

The standard reference is 3GPP TR 38.901, Section 7.7.2. TDL-D is the LOS TDL
profile in Table 7.7.2-4 with `K1 = 13.3 dB`; TDL-E is also available in the
code as `TDL-E-lite` with `K1 = 22 dB`.

Our symbolic AE channel API has only a short fixed codeword, e.g. four complex
symbols for `n = 8`. It does not represent the full 3GPP fractional-delay FIR.
The compact implementation therefore rounds the normalized TDL delays to
symbol-spaced lags, wraps them through circular convolution, aggregates paths
that land on the same lag, and normalizes the total tap power.

The previous preliminary custom profile used these tap powers:

```text
0, -2, -6, -10 dB
```

The first tap had a LOS/Rician component with K-factor `10 dB`. A pure Rayleigh
TDL was tested first, but it was too noncoherent for the current symbolic AE
setup without pilots or side information. This motivated the LOS TDL-D profile.

Use this helper to inspect the exact short-block profile used in a run:

```bash
python scripts/validate_tdl_channel.py \
  --profile TDL-D-lite \
  --symbols 4 \
  --samples 100000 \
  --out results/tdl_setup_20260529/tdl_d_lite_validation.json
```

For comparison against toolchain baselines, MATLAB `nrTDLChannel` implements
the full TR 38.901 TDL profiles, including the exact path delays, average path
gains, and first-tap K-factor. Our journal text should call the implemented
channel a compact TDL-D short-block channel unless we later add a full
fractional-delay reference implementation.

Reference links:

- 3GPP TR 38.901 v17.1.0, Section 7.7.2:
  https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/17.01.00_60/tr_138901v170100p.pdf
- MATLAB `nrTDLChannel`:
  https://www.mathworks.com/help/5g/ref/nrtdlchannel-system-object.html

Default journal preset:

```text
n = 8                       # four complex symbols
Eb/N0 = 10 dB
rate = 4/8
dataset_size = 120000
batch_size = 512
eval_size = 100000
drifting_epochs = 60
```

## Local Seed-7 Sanity Results

These results were produced before switching the default from the custom
Rician profile to `TDL-D-lite`; they are still useful as feasibility checks but
should be regenerated for the journal tables.

Analytic AE baseline, `hidden_dim=50`, `hidden_layers=4`, 15 epochs:

| Channel variant | Eval SER | Eval BER |
| --- | ---: | ---: |
| Pure Rayleigh TDL, no LOS | 0.5173 | 0.2581 |
| Rician/LOS TDL, K=10 dB | 0.0060 | 0.0033 |

Short W-Flow screen on Rician/LOS TDL, 10 drift epochs, 50k train / 20k eval:

| Variant | Direct SWD | Anchor y ratio | Gaussian-W2 ratio |
| --- | ---: | ---: | ---: |
| Kernel joint | 0.0345 | 1.80 | 1.70 |
| Joint Sinkhorn | 0.0411 | 1.65 | 1.59 |
| Fiberwise Sinkhorn | 0.0176 | 1.06 | 1.20 |

Downstream AE trained through the short fiberwise Sinkhorn checkpoint and
evaluated on the analytic TDL channel:

```text
SER = 0.0087
BER = 0.0047
```

This is close enough to the analytic baseline for a first local screen. The
channel should replace OptFib in the main journal simulation plan, with OptFib
kept as an appendix/limitation experiment only if needed.

## TDL-D-lite Seed-7 Checks

After switching the default to the compact 3GPP-derived TDL-D profile, the
effective four-symbol tap powers are:

| Lag | Total power |
| ---: | ---: |
| 0 | -0.24 dB |
| 1 | -14.95 dB |
| 2 | -18.30 dB |
| 3 | -22.22 dB |

Analytic AE baseline, `hidden_dim=50`, `hidden_layers=4`, 15 epochs:

| Channel profile | Eval SER | Eval BER |
| --- | ---: | ---: |
| TDL-D-lite | 0.0087 | 0.0049 |

Short W-Flow screen on TDL-D-lite, 10 drift epochs, 50k train / 20k eval:

| Variant | Direct SWD | Anchor y ratio | Gaussian-W2 ratio | Downstream SER |
| --- | ---: | ---: | ---: | ---: |
| Kernel joint | 0.0346 | 1.79 | 1.72 | 0.0295 |
| Joint Sinkhorn | 0.0432 | 1.66 | 1.62 | 0.0334 |
| Fiberwise Sinkhorn | 0.0186 | 1.08 | 1.22 | 0.0142 |

Stronger fiberwise Sinkhorn screen on TDL-D-lite, 30 drift epochs, 120k train /
50k eval:

| Variant | Direct SWD | Anchor y ratio | Gaussian-W2 ratio | Downstream SER | Downstream BER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fiberwise Sinkhorn | 0.0127 | 1.15 | 1.33 | 0.0096 | 0.0052 |

Takeaway: the decoder-based downstream check agrees with the conditional anchor
metrics. Joint losses overfit a surrogate geometry that gives very low training
SER but poor analytic-channel SER. Fiberwise Sinkhorn remains the best TDL
candidate and is close to the analytic baseline once trained longer.
