"""Shared figure style and data access for the manuscript figures.

Style reproduces the original (13 Oct) manuscript figures: Arial/Helvetica 8 pt, open top/right spines, light
horizontal grid, blue/green/gold tier colours (t1/t2/t3), bold panel labels, legends without frames, 300 dpi.
All figure scripts import this module; edit colours, fonts and sizes here once.
"""
import sys
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MS = HERE.parent                       # psc_manuscript/
ROOT = MS.parent                       # ml_paper/
RES = ROOT / "results"
FIG = MS / "figures"                   # output directory (names are referenced by ml.tex)
sys.path.insert(0, str(ROOT / "src"))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42, "savefig.dpi": 300,
})
C = {"t1": "#4A78A8", "t2": "#54A24B", "t3": "#E4AA28"}
TIER_LABEL = {"t1": r"$t_1$: PDB only", "t2": r"$t_2$: + material properties",
              "t3": r"$t_3$: + LLM literature features (original block)"}
GROUP_COLOUR = {"process": "#C97B4A", "material": "#4A78A8", "composition": "#6BAF7C",
                "device": "#9B8AC4", "literature": "#E4AA28"}
GROUP_NAME = {"process": "Process", "material": "Material properties", "composition": "Composition",
              "device": "Device", "literature": "Literature (LLM)"}
ARCH = {"overall": "Overall", "pin": "p-i-n", "nip": "n-i-p"}


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, name):
    """Write figures/<name>.png (used by ml.tex) and figures/<name>.pdf."""
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", FIG / f"{name}.png")


def ablation():
    """Five DOI-grouped / random splits of the controlled ablation (results/ablation_v1 + converged reruns)."""
    from psc.results import load_ablation
    return load_ablation()
