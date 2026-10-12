# SSPA input coverage intervention

Frozen before executing the intervention. Entry point:
`scripts/run_sspa_input_coverage.py`. Parent:
`results/fast_sspa_application_20261012/`.

## Question

Does broader generator-training input coverage reduce extrapolation and
surrogate-to-analytic transfer error under the unchanged average-power AE
constraint? This is an application-level intervention, not a conditioning
novelty claim. The previous pilot motivated this experiment; its seed is a
development seed and must remain labeled as such.

## Fixed intervention

Two families: learned diagonal Gaussian and normalized K64 Sinkhorn. Reuse the
completed iid-normal generators as controls and train one fresh broader-input
generator per family with the same initialization seed 9111, 60k parameter
target, 40k updates, batch 1024, optimizer, EMA, and exact physical SSPA law.
Each receives 40.96m positive channel observations. Sinkhorn retains 16 inputs
and 64 outputs per input per update. Final EMA only; no selection or retuning.

The new law is a mixture: 75% iid N(0,I8), 25% uniform direction on the sphere
times an independent radius uniform on [0,sqrt(64*8)]. The radius limit follows
from the unchanged average-power budget for 64 equiprobable codewords, rather
than the observed worst codeword. The law is uniform in radius, not volume.
Centering the codebook can tighten this conservative bound slightly.
The normal component remains unbounded. This does not guarantee accurate
learning everywhere in the allowed ball.

The normal direction draws use the original condition RNG; mixture choices
and radii use a separate resumable RNG. The narrow Gaussian/Sinkhorn training
fork is tested against the original updates bitwise with the intervention off.
No archived source changes, output clipping, input normalization, per-word power
cap, optimizer change, gradient penalty or additional downstream feedback.

## Evaluation

Three paired AE initializations: 209111, 209112, 209113. Each uses 20k updates,
batch 500, hidden width 16, NAdam .001 and the original full-codebook centering
and average-power normalization. Reuse the original analytic and control AEs
for 209111, requiring exact reproduction of their integer error counts. Train
the remaining AEs fresh. There are 15 AE rows, of which three are inherited.
This replicates AE initialization only; it is not three generator seeds.

Analytic-channel SER/BER at 4, 6 and 8 dB, each with 16,384 draws per message.
Record the same codec's nominal surrogate error, CE, input norms and conditional
moment error. Record ordinary iid-normal distribution metrics and radial moment
checks at norms sqrt(8), 6, 10, 16 and sqrt(512), with 64 common directions and
256 outputs per anchor. Also probe the frozen prior Gaussian-trained codebook
for all four learned generators. These reused codewords are explanatory
development inputs, not an untouched test set.

Interpret changes jointly: ordinary-input accuracy, tail accuracy, encoder
norms, transfer gap and analytic SER. Keep failures and every paired AE result.
No automatic declaration of success or further setting sweep based on results.
Do not pool these data with the old pilot or treat three paired AEs as a
population-level model-family comparison. Consistency distillation and a power
cap are not included in this bounded first intervention.

## Execution

```bash
python scripts/run_sspa_input_coverage.py --device cuda \
  --out-dir results/sspa_input_coverage_20261012
```

Resume the identical command with `--resume`. Both the parent artifacts and
new runner are hashed in the manifest. Existing environment only, no installs
and no HPC jobs. Model, optimizer, input-law RNG and AE checkpoints are retained.

## Query-energy clarification

The frozen law has expected squared input norm
`0.75*8 + 0.25*(512/3) = 48.6667`, versus eight for the control. This is an
intentional channel-characterization input-law change. Analytic noise variance
stays fixed; AE average squared codeword norm stays eight. Equal oracle counts
do not mean equal training-query energy. The experiment assumes simulator access
and does not establish feasibility under an RF measurement energy/peak constraint.
This arithmetic clarification changes no run setting.
