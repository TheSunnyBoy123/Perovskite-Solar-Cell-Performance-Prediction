"""fig_t3_protocols: paired change in R^2 from adding the 36 recipe flags to t2 (t3 minus t2) under every evaluation protocol
that was run for the final t3 (pooled data; record-level, DOI-grouped, lead-author-grouped, forward in time).
Each small point is one split (repeated holdout or forward window) with its 95% DOI-cluster bootstrap interval; the diamond is
the mean over splits. Filled: all test records; open: test records from publications with extraction data.
Data: results/t3final_summary/matched_pairs.csv (scripts/t3_protocol_summary.py). Output: figures/fig_t3_protocols.{png,pdf}
"""
from style import *
import numpy as np
import pandas as pd

PROTO = [("random", "Record-level"), ("doi", "DOI-grouped"), ("author", "Lead-author-grouped"), ("temporal", "Forward in time")]


def main():
    P = pd.read_csv(RES / "t3final_summary/matched_pairs.csv")
    P = P[P.arch == "overall"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (model, name), panel in zip(axes, (("catboost", "CatBoost"), ("ridge", "Ridge")), "ab"):
        for i, (proto, pn) in enumerate(PROTO):
            for scope, off, filled in (("all", -0.17, True), ("covered", 0.17, False)):
                g = P[(P.protocol == proto) & (P.model == model) & (P.scope == scope)].sort_values("split")
                if g.empty:
                    continue
                y0 = i + off
                ys = y0 + np.linspace(-0.09, 0.09, len(g)) if len(g) > 1 else [y0]
                for yy, (_, r) in zip(ys, g.iterrows()):
                    ax.plot([r.lo, r.hi], [yy, yy], color=C["t3"], lw=.6, alpha=.55, zorder=2)
                    ax.plot(r.d_r2, yy, "o", ms=2.6, color=C["t3"], mfc=C["t3"] if filled else "white", mew=.7, zorder=3)
                ax.plot(g.d_r2.mean(), y0, "D", ms=5.5, color="black", mfc="black" if filled else "white", mew=1.0, zorder=4)
        ax.axvline(0, color="k", lw=.7)
        ax.set_yticks(range(len(PROTO))); ax.set_yticklabels([n for _, n in PROTO]); ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta R^2$ ($\mathcal{M}_3-\mathcal{M}_2$)"); ax.set_title(name, fontsize=8); ax.grid(axis="x", lw=.4, alpha=.5); despine(ax)
        ax.text(-0.02, 1.03, f"({panel})", transform=ax.transAxes, fontweight="bold", fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="D", color="black", ls="", ms=5, label="mean, all test records"),
                        Line2D([], [], marker="D", color="black", mfc="white", ls="", ms=5, label="mean, extraction-covered publications"),
                        Line2D([], [], marker="o", color=C["t3"], ls="-", ms=3, lw=.6, label="single split with 95% bootstrap interval")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=7)
    fig.tight_layout(); save(fig, "fig_t3_protocols")


if __name__ == "__main__":
    main()
