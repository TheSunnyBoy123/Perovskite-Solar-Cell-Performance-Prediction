#!/usr/bin/env python3
"""SHAP for the t3 matched CatBoost model on DOI-grouped split 1 (val-based early stopping).
Usage: python scripts/shap_analysis.py [overall pin nip]
Outputs results/shap/shap_ranking_<arch>.csv, shap_groups.csv, figures/shap_*_<arch>.png.
SHAP measures attribution of the fitted model, not held-out predictive benefit."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import CAT, TARGET, load_dataset  # noqa: E402
from psc.features import TrainFitPreprocessor, base_frame  # noqa: E402
from psc.groups import GROUPS, group  # noqa: E402
from psc.models import CATBOOST_MATCHED, ES_ROUNDS  # noqa: E402
from psc.splits import make_splits  # noqa: E402

archs = sys.argv[1:] or ["overall", "pin", "nip"]
out = ROOT / "results" / "shap"; out.mkdir(parents=True, exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
import shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

grp_rows = []
for arch in archs:
    df = load_dataset(arch=arch); y = df[TARGET].values.astype(float)
    tr, va, te = next(make_splits(df["doi_group"].values, "doi", 5, 0.2, 0.15, 42))
    X = base_frame(df, "t3"); pre = TrainFitPreprocessor().fit(X.iloc[tr])
    Xtr, Xva, Xte = (pre.transform(X.iloc[i]) for i in (tr, va, te))
    names = list(Xtr.columns); ci = [i for i, c in enumerate(names) if c in CAT]
    m = CatBoostRegressor(**CATBOOST_MATCHED, verbose=0, thread_count=-1)
    m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=ES_ROUNDS, verbose=False)
    (ROOT / "models").mkdir(exist_ok=True); m.save_model(str(ROOT / "models" / f"t3_{arch}_split1.cbm"))
    sel = np.random.RandomState(0).choice(len(te), min(1500, len(te)), replace=False)
    Xs = Xte.iloc[sel]
    sh = np.asarray(shap.TreeExplainer(m).shap_values(Xs))
    ma = np.abs(sh).mean(0)
    rk = pd.DataFrame({"feature": names, "mean_abs_shap": ma, "group": [group(n) for n in names]})
    rk = rk.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True); rk.index += 1
    rk.to_csv(out / f"shap_ranking_{arch}.csv", index_label="rank")
    g = rk.groupby("group").mean_abs_shap.sum()
    for k in g.index:
        grp_rows.append(dict(arch=arch, group=k, n_features=int((rk.group == k).sum()),
                             sum_mean_abs_shap=g[k], share=g[k] / g.sum()))
    # bar plot top 20
    top = rk.head(20)[::-1]; col = {"literature": "#8e44ad", "material": "#2471a3", "composition": "#27ae60",
                                     "process": "#e67e22", "device": "#7f8c8d", "noise": "#c0392b"}
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.feature, top.mean_abs_shap, color=[col[x] for x in top.group])
    ax.set_xlabel("mean |SHAP| (PCE percentage points)"); ax.set_title(f"{arch}: t3 CatBoost, DOI-grouped split 1")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in col.items() if k in set(top.group)], loc="lower right")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / f"shap_bar_{arch}.png", dpi=150); plt.close(fig)
    print(arch, "R2 split1 test:", round(float(m.score(Xte, y[te])), 4)); print(rk.head(8).to_string())
gdf = pd.DataFrame(grp_rows); gdf.to_csv(out / "shap_groups.csv", index=False)
# grouped share figure
piv = gdf.pivot(index="arch", columns="group", values="share").fillna(0)
fig, ax = plt.subplots(figsize=(7, 4)); piv.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
ax.set_ylabel("share of total mean |SHAP|"); ax.set_xlabel(""); ax.legend(fontsize=8, bbox_to_anchor=(1, 1))
fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig_shap_groups.png", dpi=150)
fig.savefig(ROOT / "figures" / "fig_shap_groups.pdf"); plt.close(fig)
print(gdf.round(4).to_string())
