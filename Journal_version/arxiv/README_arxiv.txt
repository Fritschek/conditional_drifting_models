arXiv source package for:
Condition-Wise Sinkhorn Drifting for One-Shot Learned Channel Simulation

Main file:
drifting_vs_diffusion_summary.tex

The package includes the IEEEtran class/style files, resolved BibTeX output
drifting_vs_diffusion_summary.bbl, the bibliography source for reference, all
input tables, and the PDF figures referenced by the manuscript.

Local verification command from the repository root:
mkdir -p /tmp/cond_drift_arxiv_build
TEXINPUTS=Journal_version/arxiv//: pdflatex -interaction=nonstopmode \
  -output-directory=/tmp/cond_drift_arxiv_build \
  Journal_version/arxiv/drifting_vs_diffusion_summary.tex
