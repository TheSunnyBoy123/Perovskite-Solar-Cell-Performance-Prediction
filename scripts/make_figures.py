#!/usr/bin/env python3
"""Regenerate manuscript figures from results/*/runs.jsonl and preds. Run from ml_paper/."""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"; F = ROOT / "figures"; F.mkdir(exist_ok=True); M = ROOT / "psc_manuscript" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"t1": "#9aa5b1", "t2": "#2471a3", "t3": "#8e44ad"}
load = lambda n: pd.DataFrame([json.loads(l) for l in open(R / n / "runs.jsonl")])
import sys; sys.path.insert(0, str(ROOT / "src"))
from psc.results import load_ablation  # noqa: E402
A = load_ablation()
ARCHN = {"overall": "Pooled", "pin": "p-i-n", "nip": "n-i-p"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(F / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    shutil.copy(F / f"{name}.png", M / f"{name}.png"); plt.close(fig)


# fig 1: tiers
x = A[(A.protocol == "doi") & (A.model == "catboost")]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for j, (met, lab) in enumerate((("r2", "$R^2$"), ("rmse", "RMSE (PCE points)"))):
    for i, t in enumerate(("t1", "t2", "t3")):
        g = x[x.tier == t].groupby("arch")[met].agg(["mean", "std"]).loc[["overall", "pin", "nip"]]
        ax[j].bar(np.arange(3) + (i - 1) * 0.26, g["mean"], 0.26, yerr=g["std"], color=C[t], capsize=2,
                  label={"t1": "$\mathcal{M}_1$ database", "t2": "$\mathcal{M}_2$ + material", "t3": "$\mathcal{M}_3$ + LLM"}[t])
    ax[j].set_xticks(range(3)); ax[j].set_xticklabels([ARCHN[a] for a in ("overall", "pin", "nip")])
    ax[j].set_ylabel(lab)
    ax[j].set_ylim(0, 0.55 if met == "r2" else 5.0)  # zero-based axes: no exaggeration of small differences
ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
fig.tight_layout(); save(fig, "fig_feature_tiers")

# fig 2: split effect
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for k, (m, mn, off) in enumerate((("catboost", "CatBoost", -0.2), ("ridge", "Ridge", 0.2))):
    for i, t in enumerate(("t1", "t2", "t3")):
        for proto, hatch, al in (("random", "//", 0.45), ("doi", "", 1.0)):
            g = A[(A.arch == "overall") & (A.model == m) & (A.tier == t) & (A.protocol == proto)].r2
            xpos = i + off + (0.09 if proto == "doi" else -0.09)
            ax.bar(xpos, g.mean(), 0.18, yerr=g.std(), color=C[t], alpha=al, hatch=hatch, capsize=2, edgecolor="k", lw=0.4)
ax.set_xticks([i + o for i in range(3) for o in (-0.2, 0.2)])
ax.set_xticklabels([f"{mn}\n{t}" for i, t in enumerate(("$\mathcal{M}_1$", "$\mathcal{M}_2$", "$\mathcal{M}_3$")) for mn in ("CB", "Ridge")], fontsize=8)
ax.set_ylabel("$R^2$ (pooled)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="w", edgecolor="k", hatch="//", label="record-level random"),
                   Patch(facecolor="grey", edgecolor="k", label="DOI-grouped")], frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig_split_effect")

# fig 3: distributions + parity on split 1 (pooled; t3)
def p(path):
    d = np.load(path); return d["y"], d["pred"]
panels = [("CatBoost", R / "ablation_v1/preds/doi_overall_t3_catboost_1.npz"),
          ("MLP", R / "baselines_v1/preds/doi_overall_t3_mlp_1.npz"),
          ("GPR (subsample)", R / "baselines_v1/preds/doi_overall_t3_gpr_sub_1.npz")]
panels = [(n, f) for n, f in panels if f.exists()]
fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), sharey=True)
ax = np.atleast_1d(ax)
for a, (n, f) in zip(ax, panels):
    y, pr = p(f); hi = np.percentile(y, 99.5); b = np.linspace(0, hi, 50)
    a.hist(y, b, alpha=0.5, color="#555", label="experimental"); a.hist(pr, b, alpha=0.6, color="#8e44ad", label="predicted")
    a.set_xlabel("PCE (%)"); a.set_title(f"{n}: max pred. {pr.max():.1f}%", fontsize=9)
ax[0].set_ylabel("records"); ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig_pce_dist_trio")

# fig 4: parity + residual (CatBoost t3 pooled, all 5 DOI splits concatenated)
ys, ps = [], []
for k in range(1, 6):
    y, pr = p(R / f"ablation_v1/preds/doi_overall_t3_catboost_{k}.npz"); ys.append(y); ps.append(pr)
y, pr = np.concatenate(ys), np.concatenate(ps)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
h = ax[0].hexbin(y, pr, gridsize=60, bins="log", cmap="viridis", extent=(0, 30, 0, 30), mincnt=1)
ax[0].plot([0, 30], [0, 30], "k--", lw=0.8); ax[0].set_xlabel("experimental PCE (%)"); ax[0].set_ylabel("predicted PCE (%)")
ax[0].set_title("(a) parity (5 splits stacked)", loc="left", fontsize=9)
ax[1].hexbin(y, pr - y, gridsize=60, bins="log", cmap="viridis", extent=(0, 30, -20, 15), mincnt=1)
ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("experimental PCE (%)"); ax[1].set_ylabel("residual (pred. $-$ exp.)")
ax[1].set_title("(b) residuals", loc="left", fontsize=9)
fig.tight_layout(); save(fig, "fig_oof_diagnostics")
print("figures written:", sorted(p_.name for p_ in F.glob("*.png")))

# fig 5: top SHAP features per architecture (t3, split 1)
SH = ROOT / "results" / "shap"
if (SH / "shap_ranking_overall.csv").exists():
    col = {"literature": "#8e44ad", "material": "#2471a3", "composition": "#27ae60",
           "process": "#e67e22", "device": "#7f8c8d", "noise": "#c0392b"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, arch in zip(ax, ("overall", "pin", "nip")):
        r = pd.read_csv(SH / f"shap_ranking_{arch}.csv").head(12)[::-1]
        a.barh(r.feature.str.replace("_", " "), r.mean_abs_shap, color=[col[g] for g in r.group])
        a.set_title(f"({'abc'[('overall','pin','nip').index(arch)]}) {ARCHN[arch]}", loc="left"); a.tick_params(labelsize=7)
        a.set_xlabel("mean |SHAP| (PCE points)")
    from matplotlib.patches import Patch
    ax[2].legend(handles=[Patch(color=c, label=k) for k, c in col.items() if k != "noise"], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); save(fig, "shap")
