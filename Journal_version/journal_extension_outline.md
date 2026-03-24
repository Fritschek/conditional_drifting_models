# Journal Extension Outline

This outline turns the conference paper into a journal-scale extension without forcing all new text into the main manuscript immediately.

## Working Scope

The current conference manuscript already establishes:
- the drifting formulation,
- direct vs residual parameterizations,
- the benchmark against DDPM, DDIM, and WGAN,
- timing comparisons,
- the metric-space check for residual drifting and WGAN.

The journal version should add:
- downstream end-to-end communication relevance,
- broader channel families,
- a deeper parameterization study,
- stronger latency-quality comparisons against accelerated diffusion,
- a more explicit limitations / design-guideline section.

## Proposed High-Level Structure

1. Introduction
2. Problem Formulation and Drifting Background
3. Experimental Setup
4. Benchmark Results
5. End-to-End Learned Communication with Differentiable Channel Implants
6. Parameterization and Metric-Space Analysis
7. Extended Channel Studies
8. Latency-Quality Tradeoffs Beyond Standard Diffusion
9. Limitations and Design Guidelines
10. Conclusion

## Section 5: End-to-End Learned Communication with Differentiable Channel Implants

### Goal

Show that drifting is not only a generator benchmark method, but also a practical differentiable surrogate channel inside learned transmitter/receiver training.

### Core Questions

- How do direct and residual drifting behave when inserted into an end-to-end communication pipeline?
- Does the residual parameterization improve gradient flow for near-identity channels?
- Does direct-output drifting help on nonlinear channels where residual structure is weaker?
- How do drifting, diffusion, and WGAN compare under downstream communication metrics rather than only SWD?

### Proposed Experiments

- Train a simple autoencoder-based communication system with each learned channel implant:
  - drifting direct
  - drifting residual
  - DDPM
  - DDIM-100
  - WGAN
- Use at least:
  - AWGN
  - Rayleigh
  - SSPA
- Compare:
  - training stability
  - final SER / BLER
  - achievable information rate or cross-entropy surrogate
  - training wall-clock

### Proposed Figures

- Figure J1: End-to-end training schematic with learned channel implant
- Figure J2: SER vs SNR for direct vs residual drifting and diffusion baselines
- Figure J3: Training loss / validation SER curves showing optimization behavior

### Proposed Tables

- Table J1: Final downstream metrics across channel implants
- Table J2: End-to-end training time and convergence statistics

## Section 6: Parameterization and Metric-Space Analysis

### Goal

Turn the conference paper's direct-vs-residual observations into a systematic study.

### Core Questions

- When is residual parameterization beneficial?
- When does direct-output modeling become preferable?
- How sensitive are rankings to evaluation in output space vs residual space?
- Can a model trained in one space still score well in the other?

### Proposed Experiments

- Expand the current metric-space check from residual drifting and WGAN to all practical models where feasible
- Evaluate:
  - training target space
  - evaluation space
  - channel nonlinearity
- Report both:
  - direct-output SWD
  - residual-space SWD

### Proposed Figures

- Figure J4: Heatmap of best parameterization by channel and evaluation space
- Figure J5: Scatter plot of direct-space SWD vs residual-space SWD

### Proposed Tables

- Table J3: Full direct-vs-residual metric-space matrix
- Table J4: Relative degradation when evaluating outside the native target space

### Narrative Point

This section should make explicit that residual drifting is not "wrong" in direct space, but that it encodes a different modeling objective with a different inductive bias.

## Section 7: Extended Channel Studies

### Goal

Broaden the benchmark beyond the conference set and test whether the observed trends remain stable.

### Candidate Extensions

- higher-dimensional Rayleigh or fading channels
- MIMO channels
- channels with temporal memory
- stronger optical proxy or more realistic fiber channel variants
- measured or semi-measured channel datasets if available

### Proposed Experiments

- Repeat the main benchmark on at least one:
  - higher-dimensional fading channel
  - memory channel
  - MIMO-style channel
- Evaluate whether:
  - diffusion still dominates in fidelity
  - residual drifting still helps near identity
  - direct drifting becomes more attractive as nonlinearity or coupling increases

### Proposed Figures

- Figure J6: Channel-family map with best model class by regime
- Figure J7: Example generated sample clouds / projections for a new high-dimensional or memory channel

### Proposed Tables

- Table J5: Extended benchmark across added channels

## Section 8: Latency-Quality Tradeoffs Beyond Standard Diffusion

### Goal

Position drifting not only against DDPM/DDIM, but also against accelerated diffusion baselines.

### Core Questions

- Does drifting remain attractive once diffusion is accelerated?
- Is training cost still the main penalty of drifting?
- How do one-step diffusion alternatives compare to drifting in fidelity and latency?

### Proposed Baselines

- DDIM with reduced steps
- consistency models
- distribution matching distillation / one-step diffusion

### Proposed Experiments

- Compare:
  - SWD
  - samples per second or time per sample
  - projected training time
  - total deployment cost

### Proposed Figures

- Figure J8: Pareto frontier of fidelity vs inference latency
- Figure J9: Pareto frontier of fidelity vs total training-plus-inference cost

### Proposed Tables

- Table J6: Accelerated diffusion vs drifting timing summary
- Table J7: Accuracy / latency / training-cost summary across one-step and iterative generators

## Section 9: Limitations and Design Guidelines

### Goal

Make the journal version more useful as a methodological reference rather than only a benchmark report.

### Points to Cover

- Native metric-space dependence of direct and residual parameterizations
- When residual parameterization is appropriate
- When direct-output modeling is safer
- Sensitivity to minibatch field estimation
- Sensitivity to kernel bandwidth and repulsion hyperparameters
- Limits of the current optical proxy
- Limits of comparing mixed evaluation spaces in a main benchmark table

### Proposed Table

- Table J8: Practical design guidelines by channel regime

Example rows:
- near-identity additive channel -> residual drifting preferred
- multiplicative fading channel -> residual or direct, test both
- strongly nonlinear channel -> direct drifting preferred
- latency-critical deployment -> drifting or accelerated diffusion
- fidelity-first offline simulation -> diffusion preferred

## Minimal Journal Upgrade Path

If the full journal plan is too large, the minimum meaningful extension would be:

1. add one end-to-end autoencoder section,
2. add one broader nonlinear or higher-dimensional channel study,
3. add the one-step diffusion comparison section,
4. expand the limitations / design-guideline discussion.

That would already make the journal version feel substantially deeper than the conference paper.
