# End-to-End Weight Workflow

This note summarizes the intended artifact workflow for the journal end-to-end experiments.

## Goal

Avoid repeated retraining of:
- learned channel implants
- CNN TurboAE encoder/decoder models

The local artifact store lives under:
- `weights/`

with registry metadata in:
- `weights/index.json`

## Artifact Groups

- `weights/channel_implants/<channel>/...`
- `weights/e2e_autoencoders/<model_type>/...`

Each saved artifact should have:
- a checkpoint `.pt`
- a metadata `.json`
- one registry entry in `weights/index.json`

## Channel Implant Training

Example commands:

```bash
conda run --no-capture-output -n dl python scripts/train_symbol_pair_implant.py \
  --family drifting_direct \
  --channel AWGN \
  --device cuda \
  --weights-root weights \
  --register-name awgn_drift_direct_seed7
```

```bash
conda run --no-capture-output -n dl python scripts/train_symbol_pair_implant.py \
  --family drifting_residual \
  --channel AWGN \
  --device cuda \
  --weights-root weights \
  --register-name awgn_drift_residual_seed7
```

```bash
conda run --no-capture-output -n dl python scripts/train_symbol_pair_implant.py \
  --family paper_wgan \
  --channel AWGN \
  --device cuda \
  --weights-root weights \
  --register-name awgn_wgan_seed7
```

## End-to-End CNN Training

Analytic baseline:

```bash
conda run --no-capture-output -n dl python scripts/run_e2e_channel_implant_benchmark.py \
  --device cuda \
  --model-type cnn_turbo \
  --train-implant analytic_awgn \
  --eval-implant analytic_awgn \
  --weights-root weights \
  --save-e2e-name cnn_turbo_awgn_analytic_seed7
```

Learned implant:

```bash
conda run --no-capture-output -n dl python scripts/run_e2e_channel_implant_benchmark.py \
  --device cuda \
  --model-type cnn_turbo \
  --train-implant checkpoint \
  --train-implant-name awgn_drift_direct_seed7 \
  --eval-implant analytic_awgn \
  --weights-root weights \
  --save-e2e-name cnn_turbo_awgn_drift_direct_seed7
```

Train on analytic, evaluate on a learned implant:

```bash
conda run --no-capture-output -n dl python scripts/run_e2e_channel_implant_benchmark.py \
  --device cuda \
  --model-type cnn_turbo \
  --load-e2e-name cnn_turbo_awgn_analytic_seed7 \
  --train-implant analytic_awgn \
  --eval-implant checkpoint \
  --eval-implant-name awgn_drift_direct_seed7 \
  --weights-root weights \
  --epochs 0
```

## Reloading Saved Artifacts

Saved implant:
- `--train-implant checkpoint --train-implant-name <name> --weights-root weights`
- or `--eval-implant checkpoint --eval-implant-name <name> --weights-root weights`

Saved autoencoder:
- `--load-e2e-name <name> --weights-root weights`

This makes it possible to:
- reuse the same learned channel across multiple end-to-end runs
- resume or evaluate saved encoder/decoder models
- maintain a small, explicit model zoo for the journal experiments

## Recommended Naming Scheme

Channel implants:
- `awgn_drift_direct_seed7`
- `awgn_drift_residual_seed7`
- `awgn_wgan_seed7`

Autoencoders:
- `cnn_turbo_awgn_analytic_seed7`
- `cnn_turbo_awgn_drift_direct_seed7`
- `cnn_turbo_awgn_drift_residual_seed7`
- `cnn_turbo_awgn_wgan_seed7`

## Practical Recommendation

For the first real study:
- train one seed of each AWGN implant
- train one analytic CNN baseline
- train one CNN model per implant
- verify the workflow

Only after that should this be expanded to:
- more seeds
- GRU-based autoencoders
- additional channel families
