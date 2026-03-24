# Figure Simulations and Generation Scripts

This folder contains the scripts used to generate or support the figures included in `paper/figures/`.

## Directly runnable in this standalone repo

- `make_toy_figures_matplotlib.py`
  - generates the three toy figures used in the paper draft:
    - `figures/drifting_conditional_toy_figure_mpl.pdf`
    - `figures/drifting_tiny_mlp_demo_mpl.pdf`
    - `figures/toy_diffusion_oracle_demo_mpl.pdf`
  - this copy is intended to be runnable from inside the standalone repository

## Legacy scripts copied for provenance

The `legacy/` subfolder contains scripts from the original workspace that were used to produce the benchmark-style paper figures before the standalone repository was separated.

These files are copied here for provenance and reference:

- `legacy/benchmark_awgn_rayleigh_sspa_swd.py`
  - source path in the original workspace: `examples/benchmark_awgn_rayleigh_sspa_swd.py`
  - used to generate the multi-channel benchmark figure and JSON summary
- `legacy/compare_awgn_ddpm_ddim_drifting_corrected.py`
  - source path in the original workspace: `examples/compare_awgn_ddpm_ddim_drifting_corrected.py`
  - used for the corrected AWGN comparison figure
- `legacy/benchmark_channels_swd_gan_t100_ddim20_gan_fa_ge60.json`
  - saved benchmark output used during figure generation in the original workspace

## Important note

The `legacy/` scripts still reflect the original mixed repository layout and should be treated as historical figure-generation sources, not as the clean API of this standalone repo.

For new simulations in the standalone repository, prefer:

- `scripts/run_publication_benchmark.py`
- `scripts/compare_optional_baselines.py`

If exact regeneration of the legacy benchmark figures is required, it is safest to run the legacy scripts in the original workspace they were created in.
