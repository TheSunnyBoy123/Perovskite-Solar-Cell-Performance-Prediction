#!/usr/bin/env python3
"""Compile psc_manuscript/tikz/pipeline.tex -> psc_manuscript/figures/ml_pipeline.png.
Works on TeX installs without the `standalone` class (wraps the picture in a box and ships it
out with a tight page). Requires pdflatex, lmodern, tikz, pifont and PyMuPDF."""
import os, subprocess, tempfile
from pathlib import Path
import fitz
R = Path(__file__).resolve().parents[1] / "psc_manuscript"
s = (R / "tikz" / "pipeline.tex").read_text()
s = s.replace("\\documentclass[tikz,border=6pt]{standalone}",
              "\\documentclass{article}\n\\usepackage{lmodern}\n\\usepackage{tikz}\n\\pagestyle{empty}\\setlength{\\parindent}{0pt}")
s = s.replace("\\usepackage{helvet}", "")
s = s.replace("\\begin{document}\n\\begin{tikzpicture}[x=1cm,y=1cm]",
              "\\begin{document}\n\\setbox0=\\hbox{\\kern6pt\\vbox{\\kern6pt\\hbox{\\begin{tikzpicture}[x=1cm,y=1cm]")
s = s.replace("\\end{tikzpicture}\n\\end{document}",
              "\\end{tikzpicture}}\\kern6pt}\\kern6pt}\n\\pdfhorigin=0pt \\pdfvorigin=0pt \\pdfpagewidth=\\wd0 "
              "\\pdfpageheight=\\dimexpr\\ht0+\\dp0\\relax\n\\shipout\\box0\n\\end{document}")
with tempfile.TemporaryDirectory() as d:
    Path(d, "p.tex").write_text(s)
    env = dict(os.environ, TEXMFHOME=os.path.expanduser("~/Library/texmf"))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "p.tex"], cwd=d, env=env, capture_output=True)
    page = fitz.open(Path(d, "p.pdf"))[0]
    page.get_pixmap(dpi=300).save(R / "figures" / "ml_pipeline.png")
print("wrote", R / "figures" / "ml_pipeline.png")
