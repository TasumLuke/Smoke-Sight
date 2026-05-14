# SmokeSight paper

The conference / journal manuscript for SmokeSight. CI compiles it via
`latexmk` on every push (see `.github/workflows/ci.yml`); the produced
PDF is uploaded as a build artifact.

## Local build

You need a TeX Live distribution with `pdflatex`, `biber`, and the
`latex-extra` / `science` / `bibtex-extra` packages. On Ubuntu:

```bash
sudo apt-get install -y \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-science texlive-bibtex-extra \
    biber latexmk
```

Then from the repo root:

```bash
cd paper
latexmk -pdf paper.tex
```

The PDF lands at `paper/paper.pdf`. To clean intermediate files:

```bash
latexmk -C
```

## Figures

The five figures in the paper are copies of the same PNGs the README
embeds (`docs/images/*.png`). Re-run `python docs/images/generate.py`
from the repo root if the underlying numerics change, then copy the
fresh PNGs over:

```bash
cp docs/images/{pipeline,tau_recovery,uncertainty_components,analytic_vs_mc,dispersion_fit}.png \
   paper/figures/
```
