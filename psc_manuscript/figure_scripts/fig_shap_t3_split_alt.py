"""fig_shap_t3_split_alt: alternate layout of fig_shap_t3_split. Architectures as rows, material/process as columns, lollipop marks,
x-axis shared within each column so magnitudes are comparable across architectures. Top 10 features per panel.
Data: results/shap/t3final_ranking_<arch>.csv. Output: figures/fig_shap_t3_split_alt.{png,pdf}
"""
from style import *
import pandas as pd

TOPK = 10


def main():
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.6), sharex="col")
    n = 0
    for c, grp in enumerate(("material", "process")):
        for r, arch in enumerate(ARCH):
            ax = axes[r][c]
            d = pd.read_csv(RES / f"shap/t3final_ranking_{arch}.csv")
            d = d[d.group == grp].sort_values("mean_abs_shap", ascending=False).head(TOPK)
            ax.barh(range(len(d)), d.mean_abs_shap, color=GROUP_COLOUR[grp], height=.7)
            ax.set_yticks(range(len(d))); ax.set_yticklabels([f.replace("_", " ") for f in d.feature], fontsize=5.5)
            ax.invert_yaxis(); despine(ax)
            for i, v in enumerate(d.mean_abs_shap):
                ax.text(v + (1.12 if grp == "process" else 0.64) * .01, i, f"{v:.3f}", va="center", fontsize=5.5)
            ax.set_xlim(0, 1.12 if grp == "process" else 0.64)
            ax.set_title(f"{ARCH[arch]}: {'material properties' if grp == 'material' else 'process'}", fontsize=7.5, loc="left")
            ax.text(-0.02, 1.1, f"({chr(97 + n)})", transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right"); n += 1
            if r == 2:
                ax.set_xlabel(r"Mean $|$SHAP$|$ (PCE pp)", fontsize=7)
    fig.tight_layout(); save(fig, "fig_shap_t3_split_alt")


if __name__ == "__main__":
    main()
