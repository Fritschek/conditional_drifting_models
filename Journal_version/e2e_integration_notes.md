# End-to-End Integration Notes

These notes summarize how the current `turbo_mingru_decoder` code can be extended to use learned channel implants from `conditional_drifting_models`.

## Current Injection Point

Repository:
- `/home/entropy/GitHub/turbo_mingru_decoder`

Main path:
- `main.py`
- `train.py`

Current forward path in training:
1. `input_data = generate_data(...)`
2. `encoded_data = encoder(input_data)`
3. `noisy_data = awgn_channel(encoded_data, ebno_db, rate, device, ...)`
4. `decoded_output = decoder(noisy_data)`

The channel is therefore injected entirely through:
- `train.py::awgn_channel(...)`

This is good news, because we do not need to modify the encoder or decoder internals to start using learned channel models.

## Recommended Integration Strategy

Replace the hard-coded AWGN function with a pluggable channel interface.

### Minimal interface

Each channel implant should behave like:

```python
noisy_data = channel_implant(encoded_data, device=device, training=..., **channel_kwargs)
```

where:
- `encoded_data` is the transmitted codeword tensor,
- output shape matches `encoded_data`,
- gradients flow through the implant if the implant is differentiable.

### First integration target

Start with:
- AWGN only
- fixed CNN Turbo autoencoder
- one-shot learned implants first:
  - drifting direct
  - drifting residual
  - WGAN

Recommended initial model choice:
- primary: `cnn_turbo`
- later robustness check: `gru_turbo`

Diffusion can be added next, but one-shot models are the easiest first differentiable integration point.

## Current Bridge Implementation

The first bridge is now implemented in this repo:

- `conditional_drifting/e2e_implants.py`
- `scripts/train_symbol_pair_implant.py`
- `scripts/run_e2e_channel_implant_benchmark.py`
- `conditional_drifting/weight_registry.py`

Current supported implant families:
- analytic AWGN
- drifting direct
- drifting residual
- WGAN
- diffusion checkpoints

The bridge assumes the autoencoder emits tensors with last dimension `2`, so encoded data can be flattened into symbol pairs and restored after the implant pass. This matches the current CNN TurboAE path.

## First Practical Benchmark

### Scope

- Base repo: `turbo_mingru_decoder`
- Channel: AWGN
- Autoencoder: `cnn_turbo`
- Implants:
  - analytic AWGN
  - drifting direct
  - drifting residual
  - WGAN

### Metrics

- training loss
- BER
- training wall-clock
- checkpoint reuse cost

### Why this scope

- minimal code churn
- immediate downstream relevance
- directly tests whether residual vs direct drifting matters under end-to-end optimization

## Important Technical Questions

### 1. Does the encoder output scale match the learned channel models?

The learned channel models in `conditional_drifting_models` currently assume Gaussian-like real-valued input distributions for `x`.
The autoencoder code outputs learned codewords after its own normalization / power constraint.
We therefore need to check whether:
- the implant should be trained on encoder outputs directly,
- or whether an additional normalization layer is required.

This is likely the most important modeling decision for the end-to-end experiments.

### 2. Can diffusion be used directly as a differentiable implant?

Yes in principle, but with caveats:
- iterative sampling makes training slower,
- stochastic sampling path may complicate gradient flow and variance,
- it is probably better as a comparison baseline than as the first implant implementation.

### 3. Which target parameterization is more natural here?

- residual drifting gives a Jacobian with an explicit identity path
- direct drifting gives the cleaner conditional channel law

This is exactly why end-to-end experiments are valuable: they test not only sample quality, but also optimization behavior through the surrogate.

## Validated Workflow

The following path has already been smoke-tested locally:

1. Train a small symbol-pair implant checkpoint.
2. Register it in the local weights store.
3. Train the CNN TurboAE model through the implant by referring to the checkpoint by registry name.
4. Save the trained autoencoder back into the same weights store.

This means the next step is no longer bridge implementation, but running the first real AWGN comparison with reusable checkpoints.

## Recommended Immediate Next Step

1. Train reusable AWGN channel implants for:
   - drifting direct
   - drifting residual
   - WGAN
2. Train a CNN TurboAE baseline with analytic AWGN and save it into the weights store.
3. Fine-tune or retrain the same CNN TurboAE through each learned implant.
4. Compare BER and training cost across the resulting end-to-end systems.
