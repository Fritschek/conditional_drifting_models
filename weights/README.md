# Weights Store

This directory is the local artifact store for reusable checkpoints.

Recommended contents:

- `channel_implants/`
  - pretrained drifting / diffusion / WGAN channel surrogates
- `e2e_autoencoders/`
  - pretrained encoder / decoder checkpoints for end-to-end experiments
- `index.json`
  - optional registry mapping friendly names to checkpoint paths and metadata

The journal scripts can save and load through this directory so expensive training runs do not need to be repeated.
