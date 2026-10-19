"""fig_shap_t3_split: top-10 mean |SHAP| features of the final M3 CatBoost models (first DOI-grouped split), material-property
rows vs process rows, one column per architecture. Data: results/shap/t3final_ranking_<arch>.csv (scripts/shap_t3final.py).
Output: figures/fig_shap_t3_split.{png,pdf}
"""
from style import *
import pandas as pd

TOPK = 10


def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    n = 0
    for r, grp in enumerate(("material", "process")):
        for c, arch in enumerate(ARCH):
            ax = axes[r][c]
            d = pd.read_csv(RES / f"shap/t3final_ranking_{arch}.csv")
            d = d[d.group == grp].sort_values("mean_abs_shap", ascending=False).head(TOPK)
            ax.barh(range(len(d)), d.mean_abs_shap, color=GROUP_COLOUR[grp], height=.7)
            ax.set_yticks(range(len(d))); ax.set_yticklabels([f.replace("_", " ") for f in d.feature], fontsize=5.5)
            ax.invert_yaxis(); despine(ax)
            ax.set_xlim(0, d.mean_abs_shap.max() * 1.25)
            for i, v in enumerate(d.mean_abs_shap):
                ax.text(v * 1.02, i, f"{v:.3f}", va="center", fontsize=5.5)
            ax.set_xlabel(r"Mean $|$SHAP$|$ (PCE pp)", fontsize=7)
            ax.set_title(f"{ARCH[arch]}: {GROUP_NAME[grp].lower()}" if grp == "material" else f"{ARCH[arch]}: process",
                         fontsize=7.5, loc="left")
            ax.text(-0.02, 1.12, f"({chr(97 + n)})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right"); n += 1
    fig.tight_layout(); save(fig, "fig_shap_t3_split")


if __name__ == "__main__":
    main()
