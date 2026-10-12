# K64-only bridge back to the old production trainer

Exploratory follow-up to the component deletions, specified while the
fixed-epsilon and old-optimizer deletions are still running. The K4 deletion
has failed, while K64 without loss normalization has completed successfully.

Question: is changing only particle grouping sufficient to rescue the old
trainer on the same known failure seed? Leave-one-component-out tests alone
cannot establish sufficiency because several ingredients remain changed.

From the exact fixed-common seed-9001 update-80000 parent, run the untouched
production `SSPATrajectory.step` through update 390720. Change only
`batch_size=256, samples=64`, retaining 16384 outputs per positive/generated/
reference batch. Keep the probability-domain floored Sinkhorn implementation,
10 iterations, fixed physical epsilon, direct-space unnormalized regression,
radial field cap 2, parameter gradient cap 1, inherited Adam moments,
lr .001, beta(.9,.999), and no weight decay, warmup or EMA.

Use the same validation and independent panels and late three-point bounds
as the component study. Validate/save every 10000 updates through 200000,
then 250000, 300000, 350000 and 390720. Trace every 100 updates. Run the full
budget regardless of intermediate fidelity. Compare against the retained
K4 unchanged-control path (failed by 200k) and full-budget historical K4
fixed-common path. No independent seed, cold start, or downstream claim.

A successful result would show sufficiency of this regrouping on one
deliberately selected warm-start trajectory. It would not establish K64 as
minimal, optimal, or universally sufficient. A failed result would indicate
that other parts of the normalized recipe or their interactions also matter.
