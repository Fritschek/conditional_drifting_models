# 6G Conference Talk: ITML Template Version

This is the ITML/TUD-template version of the conference deck. It contains 27 main slides and 12 backup slides. The complete slide-by-slide purpose, content, transition, and speaker narrative are in `storyboard.md`.

Build from this directory with:

```bash
TEXINPUTS=itml/moloch//: pdflatex -interaction=nonstopmode -halt-on-error slides.tex
TEXINPUTS=itml/moloch//: pdflatex -interaction=nonstopmode -halt-on-error slides.tex
```

The folder contains the complete slide source, every referenced figure in `assets/`, the ITML/TUD logos and setup files, and local copies of the Moloch theme and `pgfopts`. A standard LaTeX installation with Beamer, TikZ/PGF, AMSMath, Booktabs, Babel, and Microtype is required.
