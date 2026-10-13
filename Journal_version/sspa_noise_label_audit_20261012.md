# SSPA noise-label check

Identified while the frozen coverage replication was running. No experiment,
channel implementation or frozen protocol was changed in response.

The replication's 4/6/8 dB labels retain the parent's **nominal software
parameter**. Matching the implemented mean/noise variance is distinct from
verifying that this parameter equals an input-referenced physical Eb/N0.

## Code and algebra

- `scripts/run_local_gradient_fidelity.py:normalized_codebook` fixes the mean
  square of all real coordinates to one. With M=64 and n=8, mean input codeword
  energy is eight, and input energy per information bit is 8/6=1/R, R=0.75.
- `conditional_drifting/paper2309_presets.py:ebno_to_noise` returns
  `1/sqrt(2*R*gamma)` for `gamma=10**(ebno_db/10)`.
- `conditional_drifting/channels.py:sspa` divides that scale by sqrt(2) before
  multiplying independent standard-normal real-coordinate noise. Thus the
  actual per-real-coordinate variance is `1/(4*R*gamma)`.
- The fast application/coverage evaluations intentionally reproduce that
  variance. Their calibration tests verify agreement with the implementation.

If complex AWGN has real-coordinate variance N0/2 and Eb is referenced to the
pre-amplifier input, these conventions imply

```
Eb / N0 = (1/R) / (2 * 1/(4*R*gamma)) = 2*gamma.
```

Under that definition the physical input-referenced value is **3.0103 dB above
the software label**. For the nominal 8 dB setting, the real noise variance is
approximately 0.05283, N0 approximately 0.10566, and input Eb/N0 approximately
11.0103 dB. A post-amplifier definition additionally depends on the average
energy after the Rapp map and therefore on the codebook.

A separate CPU check with seed 192012, 131,072 constant eight-coordinate inputs
and 1,048,576 real noise observations measured variance 0.05273274, corresponding
to 11.01828 dB under that input-energy definition. This agrees with the analytic
variance 0.05282977 and 11.01030 dB within ordinary sampling variation. It checks
the implemented noise only, not a trained communication system or the SSPA's
post-amplifier signal energy.

## Consequence and next action

This does not alter the within-setting coverage intervention: all arms use the
same physical implementation and noise, and their input codebooks obey the
same normalization. Keep the original nominal labels and explicit variance in
the frozen records rather than silently relabeling or retraining mid-study.

Before a new paper-level comparison or external-reference comparison, declare
the intended energy reference and real/complex convention. Then either label
the existing operating points explicitly or define a separately versioned
noise calibration. A unit test for physical Eb/N0 should use energy per bit and
N0 independently of the channel's own conversion helper. Check the other
complex-channel presets separately; this note establishes only the SSPA case.
