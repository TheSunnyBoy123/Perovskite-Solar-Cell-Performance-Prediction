"""fig_pce_dist_trio: experimental vs predicted PCE histograms, first DOI-grouped split (pooled, t3):
(a) CatBoost, (b) early-stopped MLP, (c) subsample GPR.
Data: results/ablation_v1/preds and results/baselines_v1/preds (*_1.npz). Output: figures/fig_pce_dist_trio.{png,pdf}
"""
from style import *
import numpy as np
from matplotlib.patches import Patch

PRED_COL = ["#4A78A8", "#54A24B", "#C0504D"]


def main():
    panels = [RES / "ablation_v1/preds/doi_overall_t3_catboost_1.npz",
              RES / "baselines_v1/preds/doi_overall_t3_mlp_1.npz",
              RES / "baselines_v1/preds/doi_overall_t3_gpr_sub_1.npz"]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.28)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for a, f, col, t in zip(axes, panels, PRED_COL, "abc"):
        d = np.load(f); y, p = d["y"], d["pred"]
        b = np.linspace(0, np.percentile(y, 99.5), 50)
        a.hist(y, b, color="#222", alpha=.9, edgecolor="white", lw=.2)
        a.hist(p, b, color=col, alpha=.55, edgecolor="white", lw=.2)
        a.set_xlabel("PCE (%)"); a.set_ylabel("Count"); a.grid(axis="y", lw=.4, alpha=.4); despine(a)
        a.text(0.02, 0.92, f"({t})", transform=a.transAxes, fontweight="bold", fontsize=9, va="top",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=.6))
    fig.legend(handles=[Patch(fc="#222", alpha=.9, label="Experimental"), Patch(fc="#7aa0c4", alpha=.8, label="Predicted")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    save(fig, "fig_pce_dist_trio")


if __name__ == "__main__":
    main()
