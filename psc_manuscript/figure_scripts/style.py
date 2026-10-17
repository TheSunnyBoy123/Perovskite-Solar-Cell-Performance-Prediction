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
TIER_LABEL = {"t1": r"$\mathcal{M}_1$: PDP only", "t2": r"$\mathcal{M}_2$: + material properties",
              "t3": r"$\mathcal{M}_3$: + 36 LLM recipe flags"}
GROUP_COLOUR = {"process": "#C97B4A", "material": "#4A78A8", "composition": "#6BAF7C",
                "device": "#9B8AC4", "literature": "#E4AA28", "recipe": "#E4AA28"}
GROUP_NAME = {"process": "Process", "material": "Material properties", "composition": "Composition",
              "device": "Device", "literature": "Earlier literature block (44)", "recipe": "Recipe flags (36)"}
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


def final_ablation():
    """Same as ablation() but with the final t3 (t2 + 36 recipe flags) in place of the superseded 44-predictor block."""
    from psc.results import load_final
    return load_final()


def ablation():
    """Five DOI-grouped / random splits of the controlled ablation (results/ablation_v1 + converged reruns)."""
    from psc.results import load_ablation
    return load_ablation()
