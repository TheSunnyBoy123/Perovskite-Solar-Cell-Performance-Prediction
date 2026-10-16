"""shap: twelve highest-ranked features by mean |SHAP|, (a) pooled, (b) p-i-n, (c) n-i-p, matched CatBoost on t2 + earlier 44-predictor block (NOT t3), first DOI split.
Data: results/shap/shap_ranking_{overall,pin,nip}.csv (columns feature, mean_abs_shap, group). Output: figures/shap.{png,pdf}
"""
from style import *
import pandas as pd
from matplotlib.patches import Patch


def main():
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 3.3))
    for a, arch, t in zip(ax, ARCH, "abc"):
        r = pd.read_csv(RES / f"shap/shap_ranking_{arch}.csv").head(12)[::-1]
        a.barh(r.feature.str.replace("_", " "), r.mean_abs_shap, color=[GROUP_COLOUR[g] for g in r.group], height=.75)
        for yy, v in enumerate(r.mean_abs_shap):
            a.text(v, yy, f" {v:.2f}", va="center", fontsize=6)
        a.set_xlim(0, r.mean_abs_shap.max() * 1.25); a.tick_params(axis="y", labelsize=6)
        a.set_xlabel(r"mean $|$SHAP$|$"); a.grid(axis="x", lw=.4, alpha=.4); despine(a)
        a.set_title(f"({t}) {ARCH[arch]}", loc="left", fontweight="bold", fontsize=8.5)
    fig.legend(handles=[Patch(fc=GROUP_COLOUR[k], label=GROUP_NAME[k]) for k in GROUP_COLOUR], loc="lower center", ncol=5,
               frameon=False, fontsize=6.5, handlelength=1, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=1.2); save(fig, "shap")


if __name__ == "__main__":
    main()
