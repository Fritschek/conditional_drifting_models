# Local development readiness

This follows the [13 October assessment](coverage_followup_assessment_20261013.md)
and the accepted OFDM plus intermittent-interference recommendation. The
[new frozen protocol](application_local_protocol_20261013.md) defines the runs.
The historical calibration and coverage results are preserved.

## Implemented and verified

- Independent SSPA input-Eb/N0/real-noise contract. The prior physical noise is
  retained under explicit corrected labels, rather than silently doubled.
- All-family common-input trainer: Gaussian, WGAN, Sinkhorn, Flow, DDIM and
  counted EDM-to-consistency distillation. Both teacher and student use the
  common law. Flow-32/128 and DDIM-20/50 have separate downstream training.
- Joint 128-real-coordinate OFDM simulator with Gaussian TDL-C-profile taps,
  fixed pilots, fractional delays and optional hidden block interference.
- Learned full Gaussian and full two-component Gaussian-mixture baselines.
  The latter learns input-independent mixture probabilities without state labels.
- Physical BER/SER/block-error evaluation, activity-stratified counts,
  cross-covariance/tail checks, conventional pilot-LMMSE reference, and a
  separate learned-constellation mode.
- Checkpoint identity/source guards, optimizer/RNG resume, counted teacher
  pretraining, actual timing and strict completed-grid reporting.

Ten new CPU tests and ten CUDA tests pass. Full smoke pipelines pass for
all six SSPA families and all seven supported OFDM families. OFDM CUDA smoke
also exercises learned-constellation training. Completed CUDA resumes verify
the existing result/checkpoint hashes without changing outputs. The reports
audit 27 SSPA curve points and nine OFDM smoke curve points. Smoke outcomes
are not scientific performance results.

The profile was retrieved from the pinned Sionna v1.2.2 JSON. All 24 delays and
power entries match the implemented constants. Upstream file SHA-256:
`6dea193df23e9c4ea83c6ba4427f97651a12ceb1d2c511757d31c849c57829ea`.
No Sionna dependency was installed. Gaussian block taps are an explicit choice,
not a claim to reproduce Sionna's finite-sinusoid fading trajectories.

## Execution status

The full one-seed common-coverage development run completed at
`results/common_coverage_9301/`, including all 27 curve points. The clean OFDM
and interference OFDM pilots each completed with 35 curve points.
See [results and interpretation](application_local_results_20261013.md).
GPU experiments run serially to avoid concurrent experimental GPU load.

The initial OFDM MLP receiver was inadequate even with physical-channel
training. Its full surrogate campaign was not launched. The separate
[frequency-receiver version](ofdm_receiver_gate_20261013.md) improves the
reference substantially and uses 10k generator updates, beyond the inherited
5k Sinkhorn warmup. This is still a development budget, not a convergence
claim. New architecture/budget experiments must use distinct records and
protocol versions, leaving these completed runs unchanged.

## Before a cluster campaign

The scripts are testable locally; scientific cluster readiness is still gated
on the full development results. Inspect the clean/interference receiver
references, common-input baseline behavior and whole-block covariance before
freezing final models. In particular, the common-input SSPA WGAN is weak and
the full Gaussian OFDM covariance heads underperform on an exactly Gaussian
control law. Neither should be frozen as a competitive baseline without a
bounded adequacy check. Run the learned-constellation performance study only
after the receiver-only pipeline is useful; its gradient smoke is already done.

HPC environment compatibility must be checked on the existing cluster Torch
stack without automatic package upgrades. `hpc/slurm_application_preflight.sh`
provides a GPU-only, 20-minute environment/test/smoke/resume check using the
existing module setup. It performs no downloads or package changes. This is
not the final campaign submission script. New official-source checkouts are
local ignored artifacts and must be transferred or fetched at their pinned
commits before running. No HPC job was submitted and no final multi-seed
campaign has been declared ready.

## Next bounded decisions

1. Keep the calibration and input law fixed. Do not repeat the completed
   coverage replication or relabel historical records in place.
2. Validate the SSPA WGAN on development data with a small predeclared schedule
   comparison; retain all outcomes. Use the saved-codebook check to distinguish
   receiver mismatch from poor physical codeword separation. Replication of
   healthy methods belongs in the later cluster campaign.
3. For OFDM, test a learned stationary fading covariance with the known
   multiplicative input structure, rather than predicting a fresh dense
   Cholesky factor from each complete block. Estimate covariance/noise from
   training observations; do not pass hidden taps or the true PDP as learned
   parameters. A two-noise-state mixture can share this learned covariance.
   Label this explicitly as a structural baseline, separately from the
   capacity-matched generic neural comparison. The known-covariance receiver
   remains a conventional reference, not a learned baseline.
4. Predeclare a small budget trajectory for the structured generators and
   apply the same validation/checkpoint rule to every family. Then run the
   learned-constellation performance pilot, including metrics at its learned
   waveform. Current fixed-QPSK anchor metrics do not cover that last test.

Only after these gates should we freeze the multi-seed campaign, uncertainty
calculation and task packing (at most 32 array tasks). Passing a smoke test or
an exact-count audit does not establish that a comparator has converged.
