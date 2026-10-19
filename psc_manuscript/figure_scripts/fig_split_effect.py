"""fig_split_effect: (t1, t2, final t3) record-level random (hatched) vs DOI-grouped (solid) splits, pooled, CatBoost and ridge.
Data: results/ablation_v1 + converged record-level reruns + results/t3final_matched_v1 (style.final_ablation()). Output: figures/fig_split_effect.{png,pdf}
"""
from style import *
from matplotlib.patches import Patch


def main():
    A = final_ablation()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for ti, t in enumerate(("t1", "t2", "t3")):
        for si, (proto, hatch) in enumerate((("random", "//"), ("doi", None))):
            g = A[(A.arch == "overall") & (A.model == "catboost") & (A.tier == t) & (A.protocol == proto)].r2
            ax.bar(ti + (si - .5) * .38, g.mean(), .36, yerr=g.std(), color=C[t], hatch=hatch,
                   edgecolor="black" if hatch else "white", lw=.5, capsize=2, error_kw=dict(lw=1.0, capthick=1.0))
    ax.set_xticks(range(3)); ax.set_xticklabels([r"$\mathcal{M}_1$", r"$\mathcal{M}_2$", r"$\mathcal{M}_3$"]); despine(ax)
    ax.set_ylabel(r"$R^2$ (pooled)")
    fig.legend(handles=[Patch(fc="#999", hatch="//", ec="black", label="Random split (record-level)"),
                        Patch(fc="#999", label="DOI-grouped split")], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(); save(fig, "fig_split_effect")


if __name__ == "__main__":
    main()
