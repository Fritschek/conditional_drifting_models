# Sinkhorn / W-Flow Follow-Up for Conditional Channel Drifting

## Source

Han, Li, Guo, Xu, Ermon, and Candès, "One-Step Generative Modeling via Wasserstein Gradient Flows", arXiv:2605.11755, submitted 2026-05-12.

## Why This Matters for Our Channel Model

The GLOBECOM version uses a conditional drifting objective with row-normalized Gaussian kernel attraction and a manually weighted generated-sample repulsion term. That makes the method fast and simple, but it leaves the kernel/coupling design channel-specific and heuristic.

W-Flow keeps the same one-step inference philosophy but replaces the heuristic drift field with a Wasserstein-gradient-flow velocity induced by the Sinkhorn divergence:

```text
V(x) = T_q,p(x) - T_q,q(x)
```

Here `T_q,p` is the entropic-OT barycentric projection from generated samples to true samples, and `T_q,q` is the generated self-transport projection. This is structurally close to our attraction-repulsion update, but the couplings are globally mass-constrained rather than independently row-normalized.

## Current Repository Hook

The repo now supports an experimental `drift_field="sinkhorn"` option in `DriftingConfig`. The default remains `drift_field="kernel"` to preserve the GLOBECOM experiments.

Relevant files:

- `conditional_drifting/losses.py`: Sinkhorn barycentric projection and drift field.
- `conditional_drifting/training.py`: two-batch generated self-transport during training.
- `scripts/run_enhanced_direct_benchmark.py`: `--drift-field sinkhorn`.
- `scripts/train_symbol_pair_implant.py`: `--drift-field sinkhorn`.
- `scripts/run_publication_benchmark.py`: `--drift-field sinkhorn`.

## First Experiment Grid

Run the same seed/channel budget as the current direct-vs-residual study, but add Sinkhorn variants:

1. Residual Sinkhorn, no conditioning: OT cost only on residual/output target.
2. Residual Sinkhorn, joint conditioning: OT cost on `[scaled x, scaled e]`.
3. Direct Sinkhorn, no conditioning: OT cost only on `y`.
4. Direct Sinkhorn, joint conditioning: OT cost on `[scaled x, scaled y]`.

Sweep:

- `sinkhorn_epsilon`: `auto`, `0.01`, `0.05`, `0.1`, `0.2`
- `sinkhorn_iterations`: `5`, `10`, `20`
- `condition_kernel_scale`: `0.25`, `0.5`, `1.0`
- `target_kernel_scale`: `1.0`

Primary channels:

- AWGN: sanity check and residual near-identity regime.
- SSPA: nonlinear regime where current residual drifting degrades in direct output space.
- OptFib: strongest test of whether global transport helps the proxy nonlinear channel.

## Metrics to Report

- Direct `y`-space SWD.
- Residual-space SWD.
- Conditional anchor moment metrics already used in the journal notes.
- End-to-end autoencoder SER/BER once the implant path is stable.
- Training time multiplier relative to kernel drifting.
- Inference time, expected to remain one generator pass.

## Hypothesis

Sinkhorn drifting should help most where the current row-normalized kernel field creates local matching artifacts: nonlinear channels, imbalanced conditional target clouds, and settings where residual-space quality does not transfer to direct `y`-space quality.

If this holds, the journal contribution can be framed as a principled conditional Sinkhorn/Wasserstein extension of the GLOBECOM conditional drifting model, not only as another hyperparameter sweep.

## First GPU Check, Seed 7

Setup:

- Direct-output enhanced drifting on AWGN, Rayleigh, and SSPA.
- `dataset_size=120000`, `eval_size=100000`, `epochs=60`, `batch_size=512`.
- Joint cost/feature space with `condition_kernel_scale=0.5`, `target_kernel_scale=1.0`.
- GPU: RTX 5060 Ti via the `ml` environment.

Generator-level SWD:

| Variant | AWGN SWD | Rayleigh SWD | SSPA SWD |
|---|---:|---:|---:|
| Kernel direct | 0.031749 | 0.025382 | 0.026896 |
| Sinkhorn direct, auto epsilon | 0.089810 | 0.077526 | 0.035458 |
| Sinkhorn direct, epsilon 0.05 | 0.391948 | 0.324190 | 0.458099 |
| Fiberwise conditional Sinkhorn direct | 0.009964 | 0.015427 | 0.008878 |

Interpretation:

- The naive Sinkhorn field is not a drop-in SWD win under the current raw joint cost.
- A fixed image-style `epsilon=0.05` is clearly too sharp/unstable in this channel coordinate system; drift norms saturate near the cap.
- Auto epsilon is stable, but currently too smooth to match the mature kernel baseline under SWD.
- The fiberwise conditional estimator is the first variant that matches the adjusted theory, and it improves over the kernel baseline on all three SWD values in this seed.

Symbolic AWGN coding check:

| Training channel implant | Final analytic-channel SER | Joint XY SWD | Anchor Y excess SWD | Mean excess | Cov excess | Gaussian W2 excess |
|---|---:|---:|---:|---:|---:|---:|
| Analytic channel | 0.005650 | 0.003261 | 0.001610 | 0.000000 | 0.020054 | 0.015297 |
| Kernel direct implant | 0.036310 | 0.095015 | 0.231929 | 0.234237 | 0.373594 | 0.761645 |
| Sinkhorn direct, auto epsilon implant | 0.029050 | 0.091044 | 0.206537 | 0.000000 | 0.372381 | 0.695962 |
| Fiberwise conditional Sinkhorn direct implant | 0.007600 | 0.010145 | 0.007806 | 0.024020 | 0.033270 | 0.039946 |

Interpretation:

- This confirms that plain SWD is not the decision metric: Sinkhorn has worse generator SWD than the kernel baseline, but better downstream symbolic SER in this AWGN run.
- The improvement appears tied to conditional mean behavior: mean excess is zero for the Sinkhorn implant under this anchor metric, while the kernel implant has a large mean excess.
- For the raw joint Sinkhorn variant, the remaining gap to analytic is still large, so cost scaling cannot be judged from SWD alone.
- The fiberwise conditional Sinkhorn variant nearly closes the symbolic SER gap to the analytic channel in this setup. This supports the theoretical adjustment: the right object is conditional/fiberwise Sinkhorn, not raw joint Sinkhorn over `[x, y]`.

## Theory Gap: Why W-Flow Does Not Transfer Directly

The Candes/Ermon W-Flow paper improves over the original drifting model by replacing the heuristic local attraction-repulsion field with the Sinkhorn-divergence Wasserstein-gradient-flow velocity

```text
V_q,p(u) = T_q,p(u) - T_q,q(u).
```

Their theory is for an unconditional target distribution `p` and a generated distribution `q`, where particles live in the same modeled space and are free to move under the Wasserstein flow. Under suitable assumptions, the empirical particle dynamics converge to the population continuity equation, and the Sinkhorn-divergence velocity has the target distribution as its only equilibrium.

Our conditional channel setting is different:

```text
x ~ p_X
y = channel(x)
ŷ = g_theta(x, z)
```

The condition `x` is not generated by the model and should not be transported. Only the channel output or residual can be changed. A naive joint Sinkhorn cost on `[x, y]` therefore approximates a WGF on the joint law of `(x, y)`, but the implemented update projects only the `y` component. That is not the same as the constrained conditional flow

```text
minimize  E_{x ~ p_X} S_epsilon(q_theta(. | x), p(. | x)).
```

This distinction likely explains why the raw joint Sinkhorn implementation does not reproduce the W-Flow image-generation improvement under SWD.

Important consequences:

- W-Flow's no-spurious-equilibrium claim applies to the full modeled distribution under the Sinkhorn WGF, not automatically to a conditional generator with fixed `x`.
- In image experiments, Sinkhorn is evaluated in learned feature/latent spaces; our first implementation uses raw channel coordinates or `[scaled x, scaled y]`.
- The W-Flow conditional image setup uses class-conditional batches and velocity guidance; our continuous channel conditions are not discrete classes.
- For continuous `x`, a minibatch usually contains only one true channel sample per condition, so a conditional Sinkhorn flow needs either repeated channel draws per anchor `x` or a local/blocked approximation.

## Better Conditional Sinkhorn Objectives

The next implementation should compare these alternatives:

1. **Same-condition Sinkhorn.**
   For each generated condition `x_i`, draw multiple true samples `y_{i,j}` from the analytic or measured channel and compute a small OT problem per anchor. This targets `S(q(.|x_i), p(.|x_i))` directly.
   This is implemented experimentally as `drift_field="fiber_sinkhorn"`.

2. **Blocked local Sinkhorn.**
   Group nearby conditions, then solve Sinkhorn only within each local block. This approximates the conditional objective without requiring exact repeated `x`.

3. **Residual-first Sinkhorn.**
   Compute OT in residual space `e = y - x` when the channel is near identity. For AWGN, this removes the irrelevant condition geometry and should be closer to the correct conditional law.

4. **Channel-feature Sinkhorn.**
   Replace raw `[x, y]` costs with features that matter for communication, e.g. residual, received energy, conditional mean offset, conditional covariance, and optionally decoder-relevant projections.

5. **Coding-metric selection.**
   Select Sinkhorn hyperparameters using symbolic SER/BER and anchor moment excess, not plain SWD alone.

The main journal angle may therefore be stronger than a simple benchmark claim: conditional channel simulation exposes a theoretical limitation of directly applying unconditional WGF/Sinkhorn drifting to conditional laws with fixed side information.

## Conditional WGF Formulation

A cleaner theory is possible if we change the geometry from full joint transport to **fiberwise conditional transport**.

Let the transmitted-symbol distribution be `mu(dx)`. The true and generated joint laws share the same condition marginal:

```text
p(dx, dy)       = mu(dx) p_x(dy)
q_theta(dx, dy) = mu(dx) q_theta,x(dy)
```

The condition `x` is side information, not a generated variable. Therefore the transport metric should only move mass inside each fiber `{x} x Y`. Define the conditional entropic OT cost

```text
OT_epsilon^cond(q, p)
  = integral OT_epsilon(q_x, p_x) mu(dx),
```

where each `OT_epsilon(q_x, p_x)` uses the usual quadratic cost in `y` or residual space. The debiased conditional Sinkhorn divergence is then

```text
S_epsilon^cond(q, p)
  = integral S_epsilon(q_x, p_x) mu(dx).
```

This is equivalent to restricting couplings to preserve `x`:

```text
pi(dx, dy, dy') = mu(dx) pi_x(dy, dy'),
pi_x in Couplings(q_x, p_x).
```

Under this geometry, the gradient flow no longer has an `x`-velocity. It has only a `y`-velocity:

```text
partial_t q_t(x, y) + div_y(q_t(x, y) v_t(x, y)) = 0.
```

For each fixed `x`, the Sinkhorn-divergence velocity is exactly the W-Flow velocity inside that fiber:

```text
v_t(x, y)
  = T_epsilon_{q_t(.|x), p(.|x)}(y)
    - T_epsilon_{q_t(.|x), q_t(.|x)}(y).
```

So the conditional analogue of the W-Flow statement is:

> If every conditional fiber is evolved under its own Sinkhorn-divergence WGF, then the only stationary point is `q_t(.|x) = p(.|x)` for `mu`-almost every `x`, assuming the corresponding unconditional Sinkhorn equilibrium property holds in each fiber.

This is the right object for channel simulation. The raw joint implementation currently in the repo is only an approximation to this, and not a theoretically exact conditional WGF.

## Empirical Conditional Estimator

For analytic or simulator channels, the fiberwise estimator is straightforward:

1. Sample anchors `x_i ~ mu`.
2. For each anchor, draw multiple generated samples `yhat_i,k = g_theta(x_i, z_i,k)`.
3. For the same anchor, draw multiple true samples `y_i,j ~ p(.|x_i)`.
4. Compute a small Sinkhorn projection inside each anchor fiber.
5. Use an independent generated batch at the same anchor for the self-transport term.
6. Regress `g_theta(x_i, z_i,k)` toward `yhat_i,k + eta v_i,k`.

For measured channels with only one sample per condition, exact fiberwise Sinkhorn is not identifiable. The practical approximation is local conditional Sinkhorn:

```text
p(.|x_i) approx weighted samples y_j with weights K_h(x_i, x_j),
```

then solve a weighted Sinkhorn problem for each anchor or local block. This recovers the current kernel-conditioning idea, but the target-space coupling becomes OT-based rather than row-normalized mean shift.

## Theoretical Claim We Can Aim For

A journal-ready proposition could be:

> Let `mu` be fixed and let `p_x`, `q_x` be probability kernels on output space with finite second moments. Define the conditional Sinkhorn functional `F(q)=integral S_epsilon(q_x,p_x) mu(dx)` on the space of kernels equipped with the fiberwise Wasserstein metric `W_mu^2(q,p)=integral W_2^2(q_x,p_x) mu(dx)`. Then the Wasserstein gradient flow of `F` is the family of per-condition continuity equations with velocity
>
> `v_x(y)=T_epsilon_{q_x,p_x}(y)-T_epsilon_{q_x,q_x}(y)`.
>
> If the Sinkhorn velocity vanishes only at equality in each fiber, then every stationary point satisfies `q_x=p_x` for `mu`-almost every `x`.

The proof should be short because it is mostly disintegration: apply the unconditional W-Flow derivation inside each fiber and integrate over `mu`.
