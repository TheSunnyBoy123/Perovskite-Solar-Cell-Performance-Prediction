"""fig_oof_diagnostics: parity (a) and residuals (b) of matched CatBoost, pooled t2, five DOI-grouped test sets stacked.
Data: results/ablation_v1/preds/doi_overall_t2_catboost_{1..5}.npz. Output: figures/fig_oof_diagnostics.{png,pdf}
"""
from style import *
import numpy as np


def main():
    ys, ps = [], []
    for k in range(1, 6):
        d = np.load(RES / f"ablation_v1/preds/doi_overall_t2_catboost_{k}.npz"); ys.append(d["y"]); ps.append(d["pred"])
    y, p = np.concatenate(ys), np.concatenate(ps)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    lim = (0, 30)
    ax[0].scatter(y, p, s=2, alpha=.12, color=C["t2"], lw=0, rasterized=True)
    ax[0].plot(lim, lim, "k--", lw=.7); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("Experimental PCE (%)"); ax[0].set_ylabel("Predicted PCE (%)")
    ax[1].scatter(y, p - y, s=2, alpha=.12, color=C["t2"], lw=0, rasterized=True)
    ax[1].axhline(0, color="k", lw=.7); ax[1].set_xlim(lim)
    ax[1].set_xlabel("Experimental PCE (%)"); ax[1].set_ylabel("Residual, predicted - experimental (%)")
    for a, t in zip(ax, "ab"):
        a.text(-0.02, 1.03, f"({t})", transform=a.transAxes, fontweight="bold", fontsize=9)
        a.grid(lw=.4, alpha=.4); despine(a)
    fig.tight_layout(w_pad=1.5); save(fig, "fig_oof_diagnostics")


if __name__ == "__main__":
    main()
