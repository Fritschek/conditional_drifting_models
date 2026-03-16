# Paper Folder

This folder contains the current paper draft and the minimum assets needed to keep it with the standalone repository.

## Contents

- `drifting_vs_diffusion_summary.tex`
  - current conference-style draft
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

## Build

From this folder, a typical build command is:

```bash
tectonic --outdir . drifting_vs_diffusion_summary.tex
```

If you use another LaTeX toolchain, make sure `references.bib` and the files in `figures/` stay in place.

## Note

This folder is a snapshot copied from the active workspace draft. As the standalone repository evolves, it may make sense to tighten the draft so that the paper text refers only to the clean repository structure in `conditional_drifting/` and `conditional_drifting/baselines/`.
