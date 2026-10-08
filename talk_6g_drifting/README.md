# 6G Conference Talk

The deck contains 27 main slides and 12 backup slides. The complete slide-by-slide purpose, content, transition, and speaker narrative are in `storyboard.md`.

Build from this directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error slides.tex
pdflatex -interaction=nonstopmode -halt-on-error slides.tex
```

The folder contains the complete slide source and every referenced figure in `assets/`. A standard LaTeX installation with Beamer, TikZ/PGF, AMSMath, and Booktabs is required.
