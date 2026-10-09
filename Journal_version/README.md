# Paper Folder

## Resubmission work

Start with [the resubmission handover](resubmission_handover.md), prepared after reviewing the actual TCOM rejection, the manuscript, implementation, earlier audit, and originating literature. It links a [cluster experiment specification](resubmission_cluster_plan.md) and [manuscript revision guide](resubmission_manuscript_plan.md). These supersede the older strategy/roadmap recommendations; the submitted paper and historical evidence remain unchanged.

The actual decision and reports are in [the prior review audit](review_audit_20261008/README.md#full-review-reports). `tcom_review_comments.md` contains earlier pre-submission feedback, not the rejection reports. Several June raw result archives are absent from this checkout; the new handover distinguishes the saved audit evidence from independently reproduced results.

The [SWD and gradient-fidelity research note](theory_swd_downstream_gradient_fidelity.md) adds a fixed-power Gaussian counterexample, positive risk/gradient bounds, literature context, and a controlled diagnostic protocol. Its mathematical constructions are separate from the paper's empirical results.

The [pre-optimization metric proposal](theory_preoptimization_channel_metric.md) develops the next target: assess a surrogate from conditional sample features and their input sensitivity without fitting a downstream encoder or decoder. It states the task-class guarantee and the remaining estimation, coverage, and validation gaps.

The [latest control assessment](metric_controls_assessment_20261009.md) reproduces the October control tables from the committed evidence archive, explains the mixed results, proves a gradient obstruction beyond the first three moments, and gives the next bounded learned-model check. A useful kernel-specific selector is still unvalidated.

That check is now [complete](learned_metric_gate_results_20261009.md): the same-input
SSPA panel provides no kernel-specific added ordering information beyond simpler
moment checks under the frozen rule. The report preserves all comparisons and
both sample budgets. Its transfer archive is in `evidence/`, not the ignored
`results/` folder. No new training or manuscript changes accompanied this test.

## Existing draft

This folder contains the current paper draft and the minimum assets needed to keep it with the standalone repository.

## Contents

- `drifting_vs_diffusion_summary.tex`
  - current journal working draft, updated against the March 27 GLOBECOM PDF/source and extended with the fiberwise Sinkhorn theory/results
- `drifting_vs_diffusion_summary.pdf`
  - compiled PDF snapshot copied from the workspace draft
- `references.bib`
  - bibliography used by the draft
- `methods_experimental_setup_section.tex`
  - standalone methods/experimental setup section
- `research_note_2026-03-16.md`
  - implementation and experiment note from the workspace
- `figures/`
  - figure assets referenced by the main draft
- `figures_src/`
  - simulation and figure-generation scripts
  - includes a repo-local toy-figure generator and legacy benchmark scripts copied for provenance

## Build

From this folder, a typical build command is:

```bash
tectonic --outdir . drifting_vs_diffusion_summary.tex
```

If you use another LaTeX toolchain, make sure `references.bib` and the files in `figures/` stay in place.

## Note

The latest GLOBECOM-style source in this repository is `Paper_camera_ready/drifting_vs_diffusion_summary (3).tex`, which matches the uploaded March 27 PDF more closely than `Paper_camera_ready/drifting_vs_diffusion_summary.tex`.
The journal draft in this folder uses that `(3)` source/PDF as the baseline and then adds journal-only diagnostics and the fiberwise Sinkhorn extension.
