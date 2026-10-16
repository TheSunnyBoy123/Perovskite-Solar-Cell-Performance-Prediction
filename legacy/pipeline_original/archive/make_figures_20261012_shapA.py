"""Figures for the Results section.

  python make_figures.py tiers      -> fig_feature_tiers.pdf/png
  python make_figures.py diag       -> fig_oof_diagnostics.pdf/png
  python make_figures.py shapgroups -> fig_shap_groups.pdf/png
  python make_figures.py split      -> fig_split_effect.pdf/png

Run from ml_paper/ (reads predictions/, results/). Outputs land in CWD;
move them to figures/ afterwards.
Colours: blue = PDB features (t1), green = + material properties (t2),
gold = + LLM literature features (t3).
"""
import csv
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "pdf.fonttype": 42, "savefig.dpi": 300,
})
C = {"t1": "#4A78A8", "t2": "#54A24B", "t3": "#E4AA28"}
TIER_LABEL = {"t1": r"$t_1$: PDB only", "t2": r"$t_2$: + material properties",
              "t3": r"$t_3$: + LLM literature features"}


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", bbox_inches="tight", dpi=300)
    print("saved", name)


# ---------------------------------------------------------------- 1. feature tiers (table values)
# (mean, sd) over five repeated DOI-grouped holdout splits.
# Source: results/results_folds_{overall,pin,nip}.csv (population SD, ddof=0).
# NOTE (audit 12 Oct): an earlier draft swapped t2/t3 RMSE (Overall) and the
# full t2/t3 rows (p-i-n, n-i-p); values below are the verified CSV means.
TIERS = {
    "Overall": {"t1": dict(r2=(.370, .011), rmse=(3.895, .051)),
                "t2": dict(r2=(.409, .012), rmse=(3.774, .055)),
                "t3": dict(r2=(.408, .012), rmse=(3.777, .049))},
    "p-i-n":   {"t1": dict(r2=(.293, .027), rmse=(3.901, .101)),
                "t2": dict(r2=(.358, .022), rmse=(3.717, .102)),
                "t3": dict(r2=(.354, .027), rmse=(3.729, .104))},
    "n-i-p":   {"t1": dict(r2=(.412, .011), rmse=(3.843, .057)),
                "t2": dict(r2=(.446, .010), rmse=(3.731, .034)),
                "t3": dict(r2=(.443, .015), rmse=(3.740, .043))},
}
_ARCH_FILE = {"Overall": "overall", "p-i-n": "pin", "n-i-p": "nip"}


def _per_run():
    out = {}
    for arch, fn in _ARCH_FILE.items():
        out[arch] = {}
        with open(os.path.join("results", f"results_folds_{fn}.csv")) as f:
            for row in csv.DictReader(f):
                out[arch].setdefault(row["tier"],
                                     {"r2": [], "rmse": []})
                out[arch][row["tier"]]["r2"].append(float(row["r2"]))
                out[arch][row["tier"]]["rmse"].append(float(row["rmse"]))
    return out


def fig_tiers(per_run=None):
    """per_run (optional): {arch: {tier: {'r2': [5 values], 'rmse': [5 values]}}} to overlay run points."""
    if per_run is None:
        per_run = _per_run()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, key, ylab, lim in ((axes[0], "r2", r"$R^2$", (0.24, 0.50)),
                               (axes[1], "rmse", "RMSE (PCE percentage points)", (3.55, 4.05))):
        for gi, arch in enumerate(TIERS):
            for ti, t in enumerate(("t1", "t2", "t3")):
                m, s = TIERS[arch][t][key]
                x = gi + (ti - 1) * 0.24
                ax.errorbar(x, m, yerr=s, fmt="o", ms=5.5, color=C[t], ecolor=C[t], elinewidth=1.3,
                            capsize=2.5, capthick=1.2, zorder=3, label=TIER_LABEL[t] if gi == 0 else None)
                if per_run:
                    pts = per_run[arch][t][key]
                    ax.plot([x] * len(pts), pts, ".", color=C[t], alpha=.35, ms=3, zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels(list(TIERS)); ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(*lim); ax.set_ylabel(ylab); ax.grid(axis="y", lw=.4, alpha=.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    d = {a: TIERS[a]["t3"]["r2"][0] - TIERS[a]["t2"]["r2"][0] for a in TIERS}
    for gi, a in enumerate(TIERS):
        axes[0].text(gi, 0.485, rf"$t_3-t_2$: {d[a]:+.3f}", ha="center", fontsize=7, color="#444")
    axes[0].text(-0.5, 0.505, "(a)", fontweight="bold", fontsize=9, va="bottom")
    axes[1].text(-0.5, 4.06, "(b)", fontweight="bold", fontsize=9, va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, handletextpad=.2, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.07))
    save(fig, "fig_feature_tiers")


# ---------------------------------------------------------------- 2. out-of-fold diagnostics
def LOAD_OOF():
    """Overall pooled Tier-3 CatBoost held-out predictions (41,451 rows; test
    sets overlap across the 5 runs by design — see predictions/oof_overall.csv)."""
    import pandas as pd
    d = pd.read_csv(os.path.join("predictions", "oof_overall.csv"))
    return d["true_pce"].values.astype(float), d["pred_pce"].values.astype(float)


def fig_diag(y, p, bins=50):
    r = p - y
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    lim = (0, np.ceil(max(y.max(), p.max())))
    ax[0].scatter(y, p, s=2, alpha=.12, color=C["t3"], lw=0, rasterized=True)
    ax[0].plot(lim, lim, "k--", lw=.7); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("Experimental PCE (%)"); ax[0].set_ylabel("Predicted PCE (%)")
    ax[1].scatter(p, r, s=2, alpha=.12, color=C["t3"], lw=0, rasterized=True)
    ax[1].axhline(0, color="k", lw=.7)
    ax[1].set_xlabel("Predicted PCE (%)"); ax[1].set_ylabel("Residual, predicted - experimental (%)")
    b = np.linspace(0, lim[1], bins)
    ax[2].hist(y, bins=b, color="#222", alpha=.9, label="Experimental")
    ax[2].hist(p, bins=b, color=C["t1"], alpha=.55, label="Predicted")
    ax[2].set_xlabel("PCE (%)"); ax[2].set_ylabel("Count"); ax[2].legend(frameon=False)
    for a, t in zip(ax, "abc"):
        a.text(-0.02, 1.03, f"({t})", transform=a.transAxes, fontweight="bold", fontsize=9)
        a.grid(lw=.4, alpha=.4)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_oof_diagnostics")


# ---------------------------------------------------------------- 3. SHAP importance by feature group
GROUP_COLOUR = {"Process": "#C97B4A", "Material properties": "#4A78A8", "Composition": "#6AAE7B",
                "Device": "#9B8BC4", "Literature (LLM)": "#E4AA28"}
# Total mean|SHAP| per group and number of features (n). Source: group table of the t3 model.
GROUP_TABLE = {
    "Overall": {"Process": (2.6040, 21), "Material properties": (2.4196, 102), "Composition": (0.9733, 47),
                "Device": (0.5950, 9), "Literature (LLM)": (0.2737, 44)},
    "p-i-n":   {"Process": (2.3043, 21), "Material properties": (1.9923, 102), "Composition": (0.7136, 47),
                "Device": (0.2940, 9), "Literature (LLM)": (0.2535, 46)},
    "n-i-p":   {"Process": (3.0133, 19), "Material properties": (2.1115, 102), "Composition": (0.8669, 47),
                "Device": (0.4943, 9), "Literature (LLM)": (0.2466, 44)},
}
INK = "#2C343E"


def fig_shap_groups(table=GROUP_TABLE):
    from matplotlib.patches import Patch
    fig, a = plt.subplots(figsize=(4.4, 2.85))
    models, groups = list(table), list(GROUP_COLOUR)
    # ---- (a) share of total, stacked
    for i, m in enumerate(models):
        tot = sum(v for v, _ in table[m].values()); left = 0
        for g in groups:
            share = 100 * table[m][g][0] / tot
            a.barh(i, share, left=left, height=.62, color=GROUP_COLOUR[g], edgecolor="white", lw=.7,
                   label=g if i == 0 else None)
            if g != "Literature (LLM)":
                a.text(left + share / 2, i, f"{share:.1f}", ha="center", va="center", fontsize=6.3,
                       color="white" if g in ("Process", "Material properties") else INK)
            else:
                a.text(103, i, f"{share:.1f}", ha="left", va="center", fontsize=7, fontweight="bold", color="#9A6B00")
            left += share
        a.text(-3, i, "", ha="right")
    a.set_yticks(range(len(models)))
    a.set_yticklabels([f"{m}\n$\\Sigma$ = {sum(v for v, _ in table[m].values()):.2f}" for m in models], fontsize=7.3)
    a.invert_yaxis(); a.set_ylim(2.55, -0.95); a.set_xlim(0, 118)
    a.set_xticks([0, 25, 50, 75, 100]); a.spines["bottom"].set_bounds(0, 100)
    a.set_xlabel(r"Share of total mean $|$SHAP$|$ (%)")
    a.tick_params(axis="y", length=0)
    a.text(-0.30, 1.04, "(a)", transform=a.transAxes, fontweight="bold", fontsize=9)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.legend([Patch(fc=GROUP_COLOUR[g], ec="white") for g in groups], groups, loc="lower center", ncol=5,
               frameon=False, handlelength=1.2, handletextpad=.4, columnspacing=1.5, bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig_shap_groups")


# ---------------------------------------------------------------- 4. random vs DOI-grouped split
# Matched experiment: same CatBoost, same tiers, same data, record-level
# ShuffleSplit(5 x 20%, seed 42). Source: results/results_random_overall.csv.
SPLIT = {"random": {"t1": (.546, .007), "t2": (.581, .006), "t3": (.589, .005)},
         "DOI-grouped": {"t1": (.370, .011), "t2": (.409, .012), "t3": (.408, .012)}}


def fig_split():
    if any(v[0] is None for v in SPLIT["random"].values()):
        raise SystemExit("fill SPLIT['random'] with your record-level random-split R2 (mean, sd) first")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for ti, t in enumerate(("t1", "t2", "t3")):
        for si, (name, hatch) in enumerate((("random", "//"), ("DOI-grouped", None))):
            m, s = SPLIT[name][t]
            ax.bar(ti + (si - .5) * .38, m, .36, yerr=s, color=C[t], hatch=hatch, edgecolor="white" if hatch is None else "black",
                   lw=.5, capsize=2, label=None)
    ax.set_xticks(range(3)); ax.set_xticklabels([r"$t_1$", r"$t_2$", r"$t_3$"]); ax.set_ylabel(r"$R^2$")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#999", hatch="//", ec="black", label="Random split (record-level)"),
                       Patch(fc="#999", label="DOI-grouped split")], frameon=False, fontsize=7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); save(fig, "fig_split_effect")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "tiers"
    if what == "tiers": fig_tiers()
    elif what == "diag": y, p = LOAD_OOF()[:2]; fig_diag(np.asarray(y), np.asarray(p))
    elif what == "shapgroups": fig_shap_groups()
    elif what == "split": fig_split()
