"""fig_shap_groups: share of total mean |SHAP| by feature group, matched CatBoost trained on t2 + the earlier 44-predictor literature block (NOT t3), first DOI-grouped split.
Data: results/shap/shap_groups.csv (columns arch, group, sum_mean_abs_shap, share). Output: figures/fig_shap_groups.{png,pdf}
"""
from style import *
import pandas as pd


def main():
    G = pd.read_csv(RES / "shap/shap_groups.csv")
    order = ["process", "material", "composition", "device", "literature"]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    for i, arch in enumerate(ARCH):
        g = G[G.arch == arch].set_index("group"); left = 0
        for k in order:
            v = 100 * g.loc[k, "share"]
            ax.barh(i, v, left=left, color=GROUP_COLOUR[k], edgecolor="white", lw=.6, height=.6,
                    label=GROUP_NAME[k] if i == 0 else None)
            if k != "literature":
                ax.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if k in ("process", "material") else "#333")
            else:
                ax.text(101.5, i, f"{v:.1f}", ha="left", va="center", fontsize=7.5, color="#9A6B00", fontweight="bold")
            left += v
    tot = {a: G[G.arch == a].sum_mean_abs_shap.sum() for a in ARCH}
    ax.set_yticks(range(3)); ax.set_yticklabels([f"{ARCH[a]}\n$\\Sigma$ = {tot[a]:.2f}" for a in ARCH]); ax.invert_yaxis()
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel(r"Share of total mean $|$SHAP$|$ (%)"); despine(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, handlelength=1, handletextpad=.3, columnspacing=1.0,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, "fig_shap_groups")


if __name__ == "__main__":
    main()
