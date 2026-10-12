# Consistency and WGAN development

## Frozen first round

Purpose: give the two weak pilot baselines a bounded development allowance
before any cross-family superiority claim. This is application-oriented
baseline development, not an experiment about novelty of conditioning.

Runner: `scripts/run_fast_baseline_development.py`. Existing baseline adapters
and official numerical implementations are unchanged. All recipes retain
approximately 60k inference parameters, AWGN n=7 at 5 dB and rate 4/7,
standard-normal input coordinates, 20,000 outer updates, batch 1024, seed 9101.
Four recipes per family receive the same analytic-output exposure. WGAN makes
4000 generator updates and 20,000 critic updates per trial; consistency makes
20,000 generator updates. Actual time is recorded, not assumed equal.

Consistency ladder:
1. Existing progressive 2-to-150 scale schedule, learning rate 2e-4, sigma_data=.5.
2. Reduce the final scale schedule parameter to 20.
3. Same coarse schedule, learning rate 1e-3.
4. Same coarse schedule and rate, sigma_data=1 for order-one channel coordinates.

WGAN grid: learning rate {2e-4, 5e-5} crossed with critic clipping {.01, .1}.
Both use RMSprop and five critic updates per generator update. Model EMA .999
is common. Raw weights and consistency NFE2 are reported but not selected.

The upstream consistency launch examples use 800,000--1,000,000 updates and
150--200 final scales, with image-specific LPIPS. Our earlier 5000-update L2
MLP pilot compressed that curriculum heavily. A slower/coarser progression
is a testable explanation for its weak output distribution, not a proven cause.
See the pinned authors' `consistency_models/scripts/launch.sh` and
`cm/script_util.py` for the exact curriculum. No claim is made that L2 channel
training reproduces the image training protocol.

Evaluation every 5000 updates uses the same development panel, with 64 anchors,
256 outputs per anchor and 64 SWD projections. Global SWD uses 4096 outputs.
A separate 64-by-512 panel measures mean RMSE against the known analytic mean x
and mean conditional variance against the known analytic AWGN variance. Two
analytic sample sets define the finite-sample SWD floor.

Select each family's **final EMA, one-step** model with lowest conditional SWD.
No early checkpoint, raw-weight or NFE selection. Rerun that fixed recipe on
two previously unused development seeds, 9102 and 9103, each with a new panel.
These are development confirmations, not final paper seeds or significance tests.

Predeclared practical check: SWD/floor <=1.5, mean RMSE <=.1 and variance ratio
in [.8,1.2]. Passing is not a theorem of distributional equality or sufficient
evidence for downstream fidelity. Failed confirmations are retained. No AE
training or SER-based model selection occurs in this round.

The manifest freezes configurations, numerical source hashes, upstream revisions,
selection rule and gate before training. Periodic checkpoints include optimizer,
model, EMA/target, RNG, training counts, history and progress. Resume with the
same command plus `--resume`; changed protocols are rejected.

No final benchmark, manuscript change, package installation or dependency update
is authorized by this development protocol.

## Bounded extension declared after first development results

Inspection of the exact upstream `scripts/cm_train.py` shows that a supplied
`teacher_model_path` initializes the student even for standalone consistency
training. The published launch examples supply a score-model checkpoint. Our
randomly initialized CT is supported, but does not reproduce that initialization.
The first coarse-grid runs learn means while still underestimating variance.

Run `scripts/run_consistency_warmstart_development.py` on development seed 9101:
20k updates of conditional EDM pretraining with the exact upstream weighted L2
loss and `LogNormalSampler`, then 20k updates each of score-initialized CT and
consistency distillation. Each stage has batch 1024 and the same ~60k network.
Both student paths initialize model, EMA and target from the pretrained score
EMA and reset the optimizer. Student randomness uses seed 19101, separate from
pretraining. These are exploratory paths, not additional equal-budget grid rows.

CT uses the original 2-to-150 curriculum and no teacher in its loss. Distillation
uses the frozen learned EDM teacher, the upstream Heun targets, 40 fixed scales
and target EMA .95. Both use Adam 2e-4 and clipping 1, explicitly channel-adapted
settings rather than the image launch hyperparameters. No analytic teacher is
used. All student results include the full EDM pretraining time and sample count.
The teacher is shared for execution efficiency but its cost is not hidden in
the comparison. No cross-family ranking follows from these unequal development
allowances. No downstream training is included.

After the seed-9101 distillation path passed the predeclared AWGN check, freeze
its recipe and repeat the complete EDM/CT/CD extension on fresh development
seeds 9104 and 9105. Retain the failing warm-CT control as well. No hyperparameters
or final-checkpoint selection rules change, and teacher cost remains included.
