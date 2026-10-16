"""fig_split_effect: record-level random (hatched) vs DOI-grouped (solid) splits, pooled, CatBoost and ridge.
Data: results/ablation_v1 + converged record-level reruns (style.ablation()). Output: figures/fig_split_effect.{png,pdf}
"""
from style import *
from matplotlib.patches import Patch


def main():
    A = ablation()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, (model, name), panel in zip(axes, (("catboost", "CatBoost"), ("ridge", "Ridge")), "ab"):
        for ti, t in enumerate(("t1", "t2", "t3")):
            for si, (proto, hatch) in enumerate((("random", "//"), ("doi", None))):
                g = A[(A.arch == "overall") & (A.model == model) & (A.tier == t) & (A.protocol == proto)].r2
                ax.bar(ti + (si - .5) * .38, g.mean(), .36, yerr=g.std(), color=C[t], hatch=hatch,
                       edgecolor="black" if hatch else "white", lw=.5, capsize=2, error_kw=dict(lw=1.0, capthick=1.0))
        ax.set_xticks(range(3)); ax.set_xticklabels([r"$t_1$", r"$t_2$", r"$t_3$"]); ax.set_title(name, fontsize=8)
        ax.text(-0.02, 1.03, f"({panel})", transform=ax.transAxes, fontweight="bold", fontsize=9); despine(ax)
    axes[0].set_ylabel(r"$R^2$ (pooled)")
    fig.legend(handles=[Patch(fc="#999", hatch="//", ec="black", label="Random split (record-level)"),
                        Patch(fc="#999", label="DOI-grouped split")], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, "fig_split_effect")


if __name__ == "__main__":
    main()
