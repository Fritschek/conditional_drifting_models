# Journal Execution Roadmap

This roadmap turns the journal topics into an implementation plan with concrete code paths, dependencies, and deliverables.

## Overall Goal

Extend the conference paper into a stronger journal manuscript with:
- end-to-end communication experiments,
- broader channel studies,
- deeper parameterization and metric-space analysis,
- stronger latency comparisons,
- practical design guidance and limitations.

The guiding principle is to reuse the current benchmark code whenever possible and isolate new work into clearly bounded modules.

## Existing Assets

### In this repository

- `conditional_drifting/`
  - drifting, diffusion, WGAN baselines
  - benchmark channels
  - evaluation code
- `scripts/`
  - full-suite runners
  - timing runners
  - partial direct-metric reruns
- `Journal_version/`
  - current journal manuscript scaffold

### External local repos

- `/home/entropy/GitHub/TurboMinGru_End2End`
  - strongest base for topic 1 (true end-to-end communication)
- `/home/entropy/GitHub/TurboMinGru_Encoder`
  - useful model definitions, training patterns, and CNN encoder/decoder components

## Work Packages

## WP1: End-to-End Communication with Learned Channel Implants

### Goal

Train an encoder-decoder communication system through learned channel surrogates and compare:
- drifting direct
- drifting residual
- DDPM
- DDIM-100
- WGAN

### Recommended base code

- primary base: `/home/entropy/GitHub/TurboMinGru_End2End`
- secondary reference: `/home/entropy/GitHub/TurboMinGru_Encoder`

### Why this split

- `TurboMinGru_End2End` already appears to contain true joint encoder-decoder training
- `TurboMinGru_Encoder` contains CNN / GRU model variants and saved models that may be reused if the end-to-end repo is incomplete

### Implementation tasks

1. Inspect `TurboMinGru_End2End/main.py`, `train.py`, and channel injection points.
2. Identify the current AWGN channel call and wrap it behind a channel interface.
3. Start with the CNN Turbo autoencoder as the primary end-to-end architecture.
   - first target: `cnn_turbo`
   - second target: `CNN_turbo_serial` if needed
   - later robustness check: `gru_turbo`
4. Add learned channel implant wrappers:
   - direct drifting implant
   - residual drifting implant
   - DDPM implant
   - DDIM implant
   - WGAN implant
5. Standardize interfaces:
   - input: encoded symbols / codeword tensor
   - output: noisy channel observation tensor
   - all differentiable in forward pass
6. Define downstream metrics:
   - BER / SER
   - training loss
   - optional AIR / BCE proxy
7. Start with AWGN, then Rayleigh, then SSPA.

### Deliverables

- new end-to-end experiment script in this repo or a clean bridge script into `TurboMinGru_End2End`
- plot: BER / SER curves
- table: downstream performance vs channel implant
- table: end-to-end training time / convergence

### Dependencies

- none blocking other work packages

### Priority

- highest

## WP2: Parameterization and Metric-Space Study

### Goal

Turn the direct-vs-residual observations into a systematic study.

### Questions

- when does residual drifting help?
- when does direct drifting help?
- how much do rankings change between direct-output SWD and residual SWD?
- do channel nonlinearity and additive structure predict the better parameterization?

### Implementation tasks

1. Reuse the existing:
   - full benchmark suite
   - partial direct-metric rerun suite
2. Expand where needed so both metric spaces can be reported consistently for more model/channel pairs.
3. Add summary scripts for:
   - relative degradation across metric spaces
   - best parameterization per channel
4. Produce a clean results table and one visual summary:
   - heatmap or scatter plot

### Deliverables

- table: direct-space vs residual-space SWD across methods
- figure: metric-space sensitivity
- section text: inductive-bias interpretation

### Dependencies

- uses current benchmark code
- can run in parallel with WP1

### Priority

- highest

## WP3: New Channel Families

### Goal

Extend the benchmark to richer channel laws.

### Candidate A: Memory channel

Recommended first new channel family.

Why:
- easier than MIMO
- still scientifically valuable
- naturally tests whether residual modeling breaks when the channel is no longer pointwise near-identity

Suggested variants:
- finite-memory nonlinear channel
- ISI + noise channel
- simple autoregressive fading or filtered-noise channel

Implementation scope:
- add new channel functions to `conditional_drifting/channels.py`
- adapt benchmark runners to support new dimensions and evaluation

### Candidate B: MIMO channel

Recommended second extension.

Why:
- stronger journal keyword
- more ambitious and visible

Risks:
- larger architectural changes
- may require redesign of dimensions, conditioning, and evaluation plots

### Recommended order

1. memory channel first
2. MIMO second

### Deliverables

- one new memory-channel benchmark table
- optionally one MIMO benchmark table
- figure: parameterization behavior under richer coupling

### Dependencies

- mild dependence on WP2 interpretation

### Priority

- medium-high

## WP4: Accelerated Diffusion Baselines

### Goal

Strengthen the latency-quality story beyond DDPM/DDIM.

### Candidate baselines

- reduced-step DDIM (already present)
- consistency models
- one-step diffusion distillation

### Recommended scope

Keep this bounded.

The journal paper does not need a full reproduction of the one-step diffusion literature.
It only needs enough to position drifting among latency-aware alternatives.

### Implementation tasks

1. Decide whether to:
   - implement a lightweight external baseline, or
   - keep this section partially discussion-driven with limited experiments
2. Extend timing scripts to include any new accelerated diffusion baseline.
3. Add Pareto plots:
   - SWD vs inference time
   - SWD vs train-plus-infer cost

### Deliverables

- timing table extension
- Pareto plot
- discussion section positioning drifting against accelerated diffusion

### Dependencies

- none blocking WP1/WP2

### Priority

- medium

## WP5: Limitations and Design Guidelines

### Goal

Make the journal paper useful as a methodological reference.

### Content

- when residual drifting is appropriate
- when direct-output drifting is safer
- metric-space caveats
- minibatch field-estimation caveats
- limits of the current optical proxy
- limits of mixed-space ranking tables

### Deliverables

- dedicated discussion section
- practical design-guideline table

### Dependencies

- depends on WP2 and ideally WP3

### Priority

- high, but should be written after the main new experiments

## Recommended Execution Order

## Phase A: Fast high-value core

1. WP2 parameterization / metric-space deepening
2. WP1 end-to-end setup on AWGN
3. WP5 draft limitations / design-guideline notes

Outcome:
- strong journal delta already exists

## Phase B: Stronger breadth

4. WP1 extend end-to-end to Rayleigh and SSPA
5. WP3 add one memory channel benchmark

Outcome:
- stronger empirical breadth

## Phase C: Ambitious extension

6. WP3 add MIMO benchmark
7. WP4 add accelerated diffusion comparison
8. finalize WP5 based on all results

Outcome:
- full ambitious journal version

## Concrete Near-Term Coding Plan

### Step 1

Inspect `/home/entropy/GitHub/TurboMinGru_End2End` and identify:
- where the channel is applied,
- whether the encoder output tensor shape matches the current learned-channel interfaces,
- how BER / SER are already computed.

### Step 2

Create a small bridge layer in this repo:
- `conditional_drifting/e2e_channels.py`

This module should expose wrappers like:
- `DriftingChannelImplant`
- `DiffusionChannelImplant`
- `WGANChannelImplant`

### Step 3

Create a dedicated runner:
- `scripts/run_e2e_channel_implant_benchmark.py`

### Step 4

Create a first limited experiment grid:
- channels: `AWGN`, `Rayleigh`
- autoencoder: `cnn_turbo`
- implants:
  - drifting direct
  - drifting residual
  - DDIM-100
  - WGAN

### Step 4b

After the CNN-based benchmark is stable, repeat the same comparison with:
- autoencoder: `gru_turbo`

This turns the GRU model into an architecture-level robustness check rather than the starting point.

### Step 5

In parallel, add one new channel to `conditional_drifting/channels.py`:
- first memory channel

### Step 6

After those two workstreams succeed, return to the journal manuscript and replace the placeholder extension sections with real results.

## What Not To Do First

- do not start with full MIMO before the end-to-end interface works
- do not overbuild one-step diffusion baselines before the journal story around drifting is stronger
- do not write the long limitations section before the new experiments are in

## Immediate Recommendation

The next best concrete task is:

1. inspect `TurboMinGru_End2End` in detail,
2. identify the channel injection point,
3. implement a first AWGN learned-channel implant benchmark.

That gives the fastest path from journal outline to journal substance.
