"""fig_v4_recipe: effect of the 36-flag recipe block on reserved publications (t3_rc minus t2), 95% publication-cluster
bootstrap intervals. Open symbols: same default settings (comparison A); filled: each tier separately tuned (B).
Grey bands: range of five DOI-permuted-block controls under default settings.
Data: results/v4/final/comparisons.json. Output: figures/fig_v4_recipe.{png,pdf}
Not comparable with fig_feature_tiers (different partition and, for tuned arms, different configurations).
"""
from style import *
import json


def main():
    D = json.load(open(RES / "v4/final/comparisons.json"))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=True)
    for ax, scope, title, t in zip(axes, ("covered", "full"), ("Covered records", "All records"), "ab"):
        d = D[scope]
        perm = [v["dR2"] for k, v in d.items() if k.startswith("ctl_perm")]
        ax.axhspan(min(perm), max(perm), color="#DDD", zorder=0)
        ax.axhline(0, color="k", lw=.7)
        for xx, key, filled, lab in ((0, "A_matched_default (t3 vs t2)", False, "Same default settings"),
                                     (1, "B_each_tuned (t3_tuned vs t2_tuned)", True, "Each tier tuned")):
            v = d[key]
            ax.errorbar(xx, v["dR2"], yerr=[[v["dR2"] - v["lo"]], [v["hi"] - v["dR2"]]], fmt="o", ms=5.5, color=C["t3"],
                        mfc=C["t3"] if filled else "white", mew=1.3, ecolor=C["t3"], elinewidth=1.3, capsize=2.5, capthick=1.2,
                        zorder=3, label=lab if t == "a" else None)
            ax.text(xx, v["hi"] + 0.0012, f"{v['dR2']:+.4f}", ha="center", fontsize=6.5, color="#444")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Same default\nsettings", "Each tier\ntuned"]); ax.set_xlim(-.6, 1.6)
        ax.set_title(title, fontsize=8); ax.grid(axis="y", lw=.4, alpha=.5); despine(ax)
        ax.text(-0.02, 1.08, f"({t})", transform=ax.transAxes, fontweight="bold", fontsize=9)
    axes[0].set_ylabel(r"$\Delta R^2$ ($t_3^{rc}$ minus $t_2$)"); axes[0].set_ylim(-0.012, 0.026)
    fig.tight_layout(w_pad=1.2); save(fig, "fig_v4_recipe")


if __name__ == "__main__":
    main()
