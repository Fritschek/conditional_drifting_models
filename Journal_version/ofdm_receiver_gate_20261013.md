# OFDM receiver gate and separate frequency-structured pilot

This addendum precedes the first full surrogate OFDM pilot. It leaves the
initial [protocol](application_local_protocol_20261013.md), smoke results and
plain-MLP receiver reference records unchanged. It does not alter the running
common-coverage SSPA comparison.

## Plain receiver gate

Training the original two-hidden-layer MLP directly on the true channel for
5000 updates produced a clean-channel BER of 0.1449 at 10 dB and 0.1080 at
20 dB. The independently evaluated pilot-LMMSE receiver is substantially better.
The corresponding interference-channel MLP BER is 0.1601/0.1264. These CPU
checks use 16,384 test blocks per SNR and saved raw counts in
`results/ofdm_receiver_reference_cpu_20261013/`.

Consequently the full initial MLP-surrogate campaign was not launched. The
first response is a bounded receiver architecture check, not a change in the
physical law or an attempt to improve only the proposed generator's result.

## Frequency-structured receiver

The first candidate is a one-dimensional residual convolutional receiver with
32 hidden channels and dilation factors 1,2,4,8. Each residual block has two
kernel-three convolutions, GroupNorm and SiLU. Input features are received I/Q
and a fixed pilot-position mask. The model receives no fading, activity labels
or physical covariance matrix. Four logits are predicted at each data carrier.

`run_ofdm_receiver_development.py` tests this candidate on the physical channel
at the original 5000-update/batch-256 budget. A pilot-assisted residual receiver
is implemented as a second bounded option if the first candidate is inadequate;
its statistical covariance knowledge must be declared. It is not included in
the main candidate by default. No results from final confirmation seeds are
used to choose a receiver.

## Separate pilot after the receiver check

`run_structured_ofdm_frequency_local.py` uses the same physical settings,
generator architectures, receiver optimizer, test seeds and metrics as the
original protocol, but substitutes the frequency receiver for every learned
implant and analytic training. It uses 10,000 generator updates per family,
rather than 4000, to include training beyond the inherited 5000-update Sinkhorn
warmup. This budget adjustment is made before any full OFDM generator result,
and applies equally to all families; distillation still splits its total budget
between teacher and student. The receiver budget remains 5000 updates.

This is still a development pilot. The fixed update budget does not establish
convergence, and the matched-query panel does not have matched wall-clock cost.
Source manifests distinguish both versions; never merge their rows as repeats.
Use separate output directories for clean/interference and fixed/learned
constellation settings. Historical smoke tests only verify the original version;
the frequency version requires its own execution/resume test before a full run.

```bash
python scripts/run_ofdm_receiver_development.py --architecture conv --condition clean --out-dir results/ofdm_conv_receiver_clean_20261013
python scripts/run_ofdm_receiver_development.py --architecture conv --condition interference --out-dir results/ofdm_conv_receiver_interference_20261013
python scripts/run_structured_ofdm_frequency_local.py --profile smoke --out-dir results/ofdm_frequency_smoke_20261013
python scripts/run_structured_ofdm_frequency_local.py --condition clean --out-dir results/ofdm_frequency_clean_9401
python scripts/run_structured_ofdm_frequency_local.py --condition interference --out-dir results/ofdm_frequency_interference_9401
```
