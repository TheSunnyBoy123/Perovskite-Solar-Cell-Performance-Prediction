"""fig_three_tier: common-split t1/t2/t3 comparison on the reserved v4final publications (same default CatBoost configuration,
identical DOI-disjoint partitions, identical test records per population). (a) R2 per tier with 95% publication-cluster bootstrap
intervals; (b) paired differences t2-t1 and t3-t2 with 95% paired publication-cluster bootstrap intervals.
Data: results/v4/final_t1/{metrics_3tier.csv, comparisons_3tier.json} (written by scripts/v4_t1_baseline.py). Output: figures/fig_three_tier.{png,pdf}
"""
from style import *
import json
import pandas as pd


def main():
    M = pd.read_csv(RES / "v4/final_t1/metrics_3tier.csv"); D = json.load(open(RES / "v4/final_t1/comparisons_3tier.json"))
    pops = (("covered", "LLM-covered records"), ("full", "All reserved records"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    for gi, (sc, _) in enumerate(pops):
        m = M[M.scope == sc].set_index("tier")
        for ti, t in enumerate(("t1", "t2", "t3")):
            r = m.loc[t]; xx = gi + (ti - 1) * 0.24
            ax.errorbar(xx, r.r2, yerr=[[r.r2 - r.r2_lo], [r.r2_hi - r.r2]], fmt="o", ms=5.5, color=C[t], ecolor=C[t], elinewidth=1.3,
                        capsize=2.5, capthick=1.2, zorder=3, label=TIER_LABEL[t] if gi == 0 else None)
        n = int(m.n.iloc[0]); d = int(m.dois.iloc[0])
        ax.text(gi, 0.215, f"{n:,} records\n{d:,} DOIs", ha="center", fontsize=6.5, color="#444")
    ax.set_xticks([0, 1]); ax.set_xticklabels([p[1] for p in pops]); ax.set_xlim(-0.55, 1.55); ax.set_ylim(0.2, 0.47)
    ax.set_ylabel(r"$R^2$"); ax.grid(axis="y", lw=.4, alpha=.5); despine(ax)
    ax.text(-0.02, 1.03, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax = axes[1]
    for gi, (sc, _) in enumerate(pops):
        for ti, (key, lab, col) in enumerate((("t2_minus_t1", r"$\mathcal{M}_2-\mathcal{M}_1$", C["t2"]), ("t3_minus_t2", r"$\mathcal{M}_3-\mathcal{M}_2$", C["t3"]))):
            v = D[sc][key]; xx = gi + (ti - 0.5) * 0.3
            ax.errorbar(xx, v["dR2"], yerr=[[v["dR2"] - v["lo"]], [v["hi"] - v["dR2"]]], fmt="o", ms=5.5, color=col, ecolor=col,
                        elinewidth=1.3, capsize=2.5, capthick=1.2, zorder=3, label=lab if gi == 0 else None)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels([p[1] for p in pops]); ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(r"Paired $\Delta R^2$"); ax.grid(axis="y", lw=.4, alpha=.5); despine(ax)
    ax.text(-0.02, 1.03, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    h, l = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(h + h2, l + l2, loc="lower center", ncol=5, frameon=False, handletextpad=.2, columnspacing=1.2, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig_three_tier")


if __name__ == "__main__":
    main()
