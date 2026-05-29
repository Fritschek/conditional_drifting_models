# Vendored TurboAE Baseline

This directory is a minimal vendored subset of the local `turbo_mingru_decoder`
project used for the journal long-block sanity check.

Only the CNN TurboAE classes needed by
`scripts/run_e2e_channel_implant_benchmark.py` are included. The minGRU and
custom CUDA extension code from the original project is intentionally omitted so
the HPC run only needs this repository plus the normal PyTorch environment.

