#!/bin/bash
# Build the manuscript (run from anywhere). Disables microtype font expansion, which fails on
# minimal TeX Live installs lacking scalable fonts; has no effect on content.
cd "$(dirname "$0")/../psc_manuscript"
export TEXMFHOME=~/Library/texmf
J='\PassOptionsToPackage{expansion=false}{microtype}\input{ml.tex}'
pdflatex -interaction=nonstopmode -jobname=ml "$J" >/dev/null 2>&1
bibtex ml >/dev/null 2>&1
pdflatex -interaction=nonstopmode -jobname=ml "$J" >/dev/null 2>&1
pdflatex -interaction=nonstopmode -jobname=ml "$J" >/dev/null 2>&1
ls -la ml.pdf; grep -n "undefined\|^!\|Overfull" ml.log | head -20
