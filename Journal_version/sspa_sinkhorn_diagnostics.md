# SSPA Sinkhorn Diagnostics

Date: 2026-06-02

## Finding

The corrected condition-wise Sinkhorn implementation is not intrinsically bad on
SSPA. The catastrophic 100-seed rerun appears to be a long-budget training
instability under the full paper preset, not a basic failure of per-condition
Sinkhorn.

The pre-fix "wrong" model worked because it estimated the Sinkhorn entropy
temperature from cross-condition distances. For SSPA seed 7, the final old
checkpoint has within-condition pred-target median cost around `0.37`, but the
legacy cross-condition median cost is around `9.0`. Thus the old run solved
per-condition Sinkhorn systems with a much larger entropy temperature, behaving
more like a smoothed conditional mean correction than sharp sample pairing.

## Seed-7 Diagnostics

All local diagnostics below use SSPA unless noted.

| Run | Direct SWD | Anchor SWD | Anchor GW2 | Final loss | Drift norm |
|---|---:|---:|---:|---:|---:|
| Old full run, legacy temperature | 0.00346 | 0.05634 | 0.25164 | 0.02678 | 0.44609 |
| Corrected full run, within-fiber temperature | 1.53264 | 1.79974 | 12.33338 | 0.36631 | 1.61903 |
| Within, batch 512, 30 epochs, 120k samples | 0.01044 | 0.06031 | 0.25642 | 0.02752 | 0.45403 |
| Fixed epsilon 9, batch 512, 30 epochs, 120k samples | 0.00843 | 0.06781 | 0.34946 | 0.02705 | 0.44837 |
| Marginal temperature, batch 512, 30 epochs, 120k samples | 0.00836 | 0.06781 | 0.34924 | 0.02705 | 0.44834 |
| Within, batch 512, 160 epochs, 120k samples | 0.00704 | 0.05490 | 0.23002 | 0.02819 | 0.45959 |
| Within, batch 4096, 30 epochs, 120k samples | 0.04091 | 0.22431 | 0.79844 | 0.06335 | 0.67373 |
| Within, batch 4096, 160 epochs, 120k samples | 0.00674 | 0.05731 | 0.23840 | 0.02784 | 0.45680 |
| Marginal temperature, batch 4096, 160 epochs, 120k samples | 0.00865 | 0.06497 | 0.33125 | 0.02629 | 0.44259 |
| Fiber moment matching, batch 4096, 160 epochs, 120k samples | 0.01963 | 0.17857 | 0.64001 | 0.11322 | 0.32492 |

The corrected full paper-budget run uses 10M samples, batch 4096, and 160
epochs, which is about 390k optimizer updates. The stable local batch-4096
run uses the same epoch count but 120k samples, about 4.8k optimizer updates.
This suggests that the corrected SSPA failure is caused by excessive
fixed-learning-rate optimization of the sharp field.

## Downstream Check

Training a symbolic autoencoder through the stable corrected checkpoint
`within, batch 4096, 160 epochs, 120k samples` and evaluating on the analytic
SSPA channel gives, for seed 7:

| Metric | Value |
|---|---:|
| SER | 4.00e-4 |
| BER | 2.28e-4 |

This is far better than the corrected full-run failed checkpoint
(`BER` around `1.3e-2` for seed 7), but not as strong as the accidental old
legacy-temperature seed-7 checkpoint.

## Candidate Fixes

1. Use corrected within-fiber Sinkhorn but cap the SSPA effective optimizer
   budget, for example `dataset_size=120000`, `batch_size=4096`,
   `epochs=160`.
2. Use marginal-temperature condition-wise Sinkhorn when running very long SSPA
   budgets. This intentionally keeps the useful smoothing from the old run
   without reintroducing the full global cost matrix.
3. Test a lower learning rate or LR decay for the full SSPA budget if we need
   to preserve the original 10M-sample preset.

The next HPC screen should be SSPA-only over 30 seeds before a 100-seed rerun.
Recommended variants:

- `fiber_sinkhorn` with compact effective budget.
- `fiber_sinkhorn_marginal` with the same compact budget.
- `fiber_sinkhorn_marginal_lr3e4` if testing the full paper budget.
- Existing `kernel_target` as the current SSPA learned-surrogate baseline.
