# SSPA input coverage: completed intervention

See the [fixed protocol](sspa_input_coverage_protocol_20261012.md),
[full tables](evidence/sspa_input_coverage_20261012/results.md), and
[comparison figure](evidence/sspa_input_coverage_20261012/coverage_comparison.png).
Raw suite: `results/sspa_input_coverage_20261012/`.

Two new generators and twelve new AE runs completed. Together with the three
inherited control AEs, the comparison has 15 rows. Generator seed 9111 is fixed;
209111, 209112 and 209113 are paired AE initializations, not generator replicas.
All historical sources, checkpoints and results remain unchanged.

## What changed

Only generator-training inputs changed: replace 25% of iid-normal inputs with
uniform directions and radii uniform on [0,sqrt(512)]. The radius covers the
range permitted by the existing average-power constraint for 64 equiprobable
eight-real-coordinate codewords. This is a conservative radius bound, not a
claim that every point is well learned or routinely encountered.

Both methods retain their architectures, initialization, optimizer, 40k updates,
40.96m positive oracle outputs, final EMA selection and physical channel. The
AE architecture, optimizer, 20k updates, centering and average squared norm of
eight are unchanged. No output saturation rule or individual-codeword cap is
imposed. The new law increases expected training-query squared norm from eight
to 48.6667. The comparison matches oracle counts, not RF acquisition energy.
Physical noise variance remains fixed.

## Paired downstream results

SER at 8 dB, always evaluated on analytic SSPA, with 1,048,576 trials per row:

| AE seed | Gaussian, normal inputs | Gaussian, broader inputs | Sinkhorn, normal inputs | Sinkhorn, broader inputs |
|---|---:|---:|---:|---:|
| 209111 | 3.100e-3 | 4.101e-5 | 7.725e-5 | 9.918e-5 |
| 209112 | 8.307e-4 | 9.537e-5 | 3.433e-5 | 4.005e-5 |
| 209113 | 4.525e-3 | 1.116e-4 | 4.463e-4 | 9.441e-5 |

For Gaussian, analytic SER improves in all three pairs at all three evaluated
SNRs (4, 6, 8 dB). At 8 dB, the reductions are 75.6x, 8.71x and 40.6x. The
descriptive mean falls from 2.819e-3 to 8.265e-5. The original Gaussian-input
failure therefore repeats across AE initialization for this generator.

For Sinkhorn, the first two pairs have slightly worse SER at all three SNRs.
The worst control case improves from 4.463e-4 to 9.441e-5 at 8 dB, a 4.73x
reduction. The descriptive mean falls from 1.860e-4 to 7.788e-5, but that mean
must not hide the mixed paired outcomes. No statistical family ranking or
generator-seed robustness claim is justified by this panel.

The analytic-reference SERs at 8 dB are 5.72e-6, 2.67e-5 and 1.14e-5. Broader
coverage does not close the full learned-versus-analytic training gap.

## Fidelity and encoder behavior

Gaussian conditional mean RMSE at input norm 10 drops from .349 to .0263;
at norm sqrt(512), from 1.492 to .0502. On the prior problematic Gaussian-trained
codebook, it drops from .0841 to .0169. Ordinary-input anchor SWD/floor remains
close to one (1.097 to 1.061). Its new AE codebook maximum norms are 4.11, 5.31
and 3.82, compared with 10.27, 8.52 and 8.84 for the controls. This occurs without
clipping or imposing a new power constraint.

Sinkhorn mean RMSE at norm 10 drops from .458 to .0711 and at sqrt(512) from
1.872 to .135. Ordinary-input anchor SWD/floor worsens from 1.200 to 1.455.
Its high-radius conditional variance remains too small: at sqrt(512), the
variance ratio is .587 rather than one. Broad input coverage improves tail
means but does not fully repair every aspect of the conditional law at this
capacity and budget. New AE maximum norms are 4.10, 4.26 and 4.32, versus
6.92, 5.19 and 7.84 for the controls.

The large optimistic surrogate-to-analytic gaps are substantially reduced:

| Model/input law | Nominal surrogate SER, three AEs | Nominal analytic SER, three AEs |
|---|---|---|
| Gaussian / normal | 5.91e-5, 2.96e-5, 5.44e-5 | 3.10e-3, 8.31e-4, 4.53e-3 |
| Gaussian / broad | 3.43e-5, 1.03e-4, 8.68e-5 | 4.10e-5, 9.54e-5, 1.12e-4 |
| Sinkhorn / normal | 1.05e-5, 1.05e-5, 5.72e-6 | 7.72e-5, 3.43e-5, 4.46e-4 |
| Sinkhorn / broad | 9.82e-5, 2.57e-5, 1.14e-4 | 9.92e-5, 4.01e-5, 9.44e-5 |

These use independent noise draws for surrogate and analytic evaluation, so
they are calibration comparisons rather than per-sample equality tests.

## Interpretation and next decision

The controlled input-law change supports inadequate input coverage as an
important contributor to the severe transfer failures. The Gaussian result
is particularly clear across the three paired AEs. For Sinkhorn, the experiment
reveals an ordinary-input versus tail-fidelity tradeoff, alongside reduced
optimism and less extreme codebooks. It does not support a universal SER benefit.

This does not isolate every effect of radial reweighting or prove a general
gradient-fidelity theory. It also does not validate a peak-power constraint,
passive measured-data learning, or a universal SWD-based model selector.
Do not optimize the mixture repeatedly against these AE results.

The next useful check is to repeat the same fixed intervention with fresh
generator seeds before expanding it. Then apply a common input-coverage protocol
to modern competing generators, including distilled consistency. Their original
Gaussian-input rows must not be silently replaced in a cross-family table.
Sampler convergence checks, the larger structured channel, and optical fiber
remain separate pending work. This experiment does not justify reopening the
previously negative RBF selection study.

## Verification

- Three new tests passed on CPU and CUDA: unchanged-control update parity,
  broad-law RNG/optimizer resume, and mixture sampling statistics.
- Thirteen existing CPU regression tests passed.
- Tiny end-to-end execution, identical-config resume, and reporting audit passed.
- All three inherited AE error-count evaluations reproduce exactly.
- Full audit verifies 15 AE rows, 12 new complete AE checkpoints, both complete
  generator budgets, finite EMA weights, identical per-family training counts,
  correct power and integer SER/BER accounting, and unchanged parent hashes.
- Report figure inspected. Every paired result and all model weights retained.

The new generators consumed 189.09 measured training seconds and the twelve new
AEs 452.74 seconds, excluding evaluation, initialization and checkpoint I/O.
No extrapolation, dependency installation, HPC submission or manuscript change.
All experiment processes have finished.
