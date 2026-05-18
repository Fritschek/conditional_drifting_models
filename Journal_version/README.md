# Paper Folder

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
