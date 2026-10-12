# Official implementation comparison

10 October 2026. Read-only source inspection plus a matched solver check. No
dependencies were installed, no image-model checkpoints were downloaded, and
neither the production channel trainer nor existing trajectories were changed.

## Official sources

- [W-Flow, Han et al.](https://github.com/hanjq17/W-Flow), pinned at
  `1ccb34baf57f9f8f4e188460ea3d19b431403293` (9 June 2026).
  This is the authors' PyTorch release for the Ermon/Candes paper.
- [Generative Modeling via Drifting, Deng et al.](https://github.com/lambertae/drifting),
  pinned at `accd0cf09c33b70892d33941d2a287ca86cb92e1` (19 March 2026).
  This is the original JAX release. W-Flow's separate `drift_loss.py` is an
  unofficial Torch port of that method, not the original authors' release.
- [Sinkhorn-Drifting Generative Models, He et al.](https://mint-vu.github.io/SinkhornDrifting/)
  links to [mint-vu/sinkhorn_drift](https://github.com/mint-vu/sinkhorn_drift).
  This is another authors' implementation, distinct from W-Flow. Its landing
  page/README were checked, but its numerical code was not audited or run here.

The first two repositories were cloned under `results/official_sources_20261010/`.
Their upstream files were left unchanged. The comparison concerns those pinned
releases, not a claim about what code was available when our manuscript began.

## W-Flow versus the current SSPA experiment

| Component | Official release | Our controlled SSPA trajectories |
| --- | --- | --- |
| Basic field | Cross barycenter minus generated-reference barycenter | Same structure |
| Conditional grouping | Class-specific generated/positive clouds | Clouds at the same transmitted input |
| Self reference | Fresh generated batch in released OT configs; no diagonal mask | Independent generated cloud, no diagonal mask |
| Cost | Quadratic in released OT configs | Quadratic |
| Solver | Log-domain scaling without our kernel floor or additive scaling denominator | Probability-domain scaling, kernel floor and additive denominators |
| Iterations | 10 in released ablation and B configs | 10 |
| Particles | 64 generated and 64 positive per label, fresh reference of generated-batch size | 4 generated, 4 positive, 4 reference per input |
| Representation | Pretrained image features | Raw channel output |
| Scaling | Feature and force normalization | Raw-output field with radial cap 2 |
| Regularization | Common feature-scaled `R=0.05` in these configs | Separate adaptive, fixed common, or pooled adaptive epsilon |
| Regression | Detached target in normalized feature space | Detached target in output space |
| Optimization | AdamW, warmup, EMA, config-dependent schedule | Adam, constant 0.001, no EMA in this study |

Sources: [OT loss](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/drift_loss_ot.py),
[training loop](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/train.py),
[ablation configuration](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/configs/gen/ablation_ot_1node.yaml),
[B configuration](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/configs/gen/latent_sota_B_ot_8node.yaml),
[optimizer construction](https://github.com/hanjq17/W-Flow/blob/1ccb34baf57f9f8f4e188460ea3d19b431403293/utils/model_builder.py).

The bare loss API defaults differ from the published run configurations. In
particular, the function default is unsquared Euclidean cost, whereas these
configs explicitly select quadratic cost. `sinkhorn_stop_thr` is accepted but
unused, so 10 iterations do not certify convergence. The config's unconditional
negative samples used for classifier-free guidance are not the independent
generated self-reference count. Our comparison has no guidance term.

### Is their conditional formulation the same?

At the structural level, yes. Their training loop constructs same-class groups,
then applies cross-minus-self transport within each group (and feature location).
Our groups consist of repeated outputs at the same transmitted input. Replacing
the discrete class label by a channel input changes the application and data
requirements, not that conditional transport construction. The released OT
configs also use independent generated references, matching our two-cloud self
estimator. Continuous input coverage and channel-specific evaluation remain
substantive questions, but they do not make per-condition Sinkhorn transport a
new algorithm. Our differing normalization, sample counts and optimizer mean
the complete implementations are not identical.

The original JAX drifting loss also normalizes feature scale and force magnitude.
Its affinity is constructed from row/column softmax factors, with self masking,
rather than a converged Sinkhorn plan. See the
[original loss](https://github.com/lambertae/drifting/blob/accd0cf09c33b70892d33941d2a287ca86cb92e1/drift_loss.py).

## Direct numerical check

`scripts/check_official_wflow_solver.py` imports the unmodified official
`drift_loss_ot.py` with `DRIFT_COMPILE=0`. It checks the pinned commit and clean
checkout before importing. Only the reviewed Torch loss module is executed.

On all 12 saved seed-9001 clouds, compare the official quadratic-cost primitive
against our new unfloored log-domain primitive at 10 and 100 iterations, in
float32 and float64 on CPU. Retain each checkpoint's epsilon; do not replace it
with the official normalized-loss hyperparameters. Both cross and self problems
are checked, with 4096 inputs per cloud and no diagonal mask.

**Result: 48 case/dtype/iteration comparisons, 96 cross/self comparisons, maximum
absolute plan difference 0.0.** The saved record also contains marginal residuals
and field differences against the old GPU production field. Cross-device
rounding is included in the latter comparison, so it is not a pure one-factor
estimate of the solver change.

This verifies the matched solver primitive, not the full official training loss,
the image results, or long-run stability. Fixed-iteration agreement is not an
optimality certificate. The check is saved in
`results/sspa_solver_replay_20261010/official_wflow_check.json`.

## Consequences for the next experiment

The current channel implementation is not an exact reproduction of the released
W-Flow training recipe. Its conditional organization and cross/self structure
are shared with prior work; those alone should not be claimed as novelty.

Use the official log-domain primitive as the numerical baseline for a new,
separately identified experiment. Keep the old trainer intact for reproducibility.
Then isolate sample count, feature/field normalization and optimizer effects.
Do not change all three and attribute success to Sinkhorn arithmetic. Increasing
particles changes both variance and the minibatch estimator, so it is not merely
a numerical solver refinement.

The [completed replay](sspa_solver_replay_results_20261010.md) shows very small
solver corrections on the late healthy batches. It therefore does not support
claiming that moving to official arithmetic alone prevents the collapse.
An image-feature normalization scheme may also be unsuitable for raw channel
coordinates without adaptation. These are testable differences, not established
causes or fixes.
