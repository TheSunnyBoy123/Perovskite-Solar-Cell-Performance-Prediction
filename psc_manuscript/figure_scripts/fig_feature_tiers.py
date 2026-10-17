"""fig_feature_tiers: t1, t2 and the final t3 (t2 + 36 recipe flags) under the five repeated DOI-grouped holdouts (matched CatBoost; mean +/- s.d. of 5 splits).
Data: results/ablation_v1 + results/t3final_matched_v1 (via style.final_ablation()). Output: figures/fig_feature_tiers.{png,pdf}
The earlier 44-predictor literature block is not shown; the common-split comparison is in fig_three_tier.py / fig_v4_recipe.py.
"""
from style import *


def main():
    A = final_ablation()
    x = A[(A.protocol == "doi") & (A.model == "catboost")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    D = {}
    for ax, key, ylab, lim in ((axes[0], "r2", r"$R^2$", (0.15, 0.50)),
                               (axes[1], "rmse", "RMSE (PCE percentage points)", (3.55, 4.15))):
        for gi, arch in enumerate(ARCH):
            for ti, t in enumerate(("t1", "t2", "t3")):
                g = x[(x.arch == arch) & (x.tier == t)][key]
                m, s = g.mean(), g.std()
                D[(arch, t, key)] = m
                xx = gi + (ti - 1) * 0.24
                ax.errorbar(xx, m, yerr=s, fmt="o", ms=5.5, color=C[t], ecolor=C[t], elinewidth=1.3, capsize=2.5,
                            capthick=1.2, zorder=3, label=TIER_LABEL[t] if gi == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels(list(ARCH.values())); ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(*lim); ax.set_ylabel(ylab); ax.grid(axis="y", lw=.4, alpha=.5); despine(ax)
    axes[0].text(-0.5, 0.505, "(a)", fontweight="bold", fontsize=9, va="bottom")
    axes[1].text(-0.5, 4.16, "(b)", fontweight="bold", fontsize=9, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handletextpad=.2, columnspacing=1.6, bbox_to_anchor=(0.5, -0.07))
    save(fig, "fig_feature_tiers")


if __name__ == "__main__":
    main()
