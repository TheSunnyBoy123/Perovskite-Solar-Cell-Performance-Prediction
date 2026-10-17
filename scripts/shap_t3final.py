#!/usr/bin/env python3
"""SHAP for the final t3 (t2 + 36 recipe flags) matched CatBoost model, first DOI-grouped split of results/t3final_matched_v1.
Same split, sample (1,500 held-out records, RandomState(0)) and estimator settings as scripts/shap_analysis.py, with the documented
first-feature-use penalty on the flags. Outputs results/shap/t3final_ranking_<arch>.csv and t3final_groups.csv.
SHAP values attribute the fitted model's predictions; they do not show predictive benefit or causal effects."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import CAT, TARGET, load_dataset  # noqa: E402
from psc.features import base_frame  # noqa: E402
from psc.groups import group  # noqa: E402
from psc.models import CATBOOST_MATCHED, ES_ROUNDS  # noqa: E402
from psc.splits import make_splits  # noqa: E402
from psc import t3final  # noqa: E402
import shap
from catboost import CatBoostRegressor

out = ROOT / "results" / "shap"; rows = []
for arch in ["overall", "pin", "nip"]:
    df = load_dataset(arch=arch); y = df[TARGET].values.astype(float)
    tr, va, te = next(make_splits(df["doi_group"].values, "doi", 5, 0.2, 0.15, 42))
    Xtr, Xva, Xte, pen, _ = t3final.matrices(df, base_frame(df, "t2"), tr, va, te, "t3f")
    names = list(Xtr.columns); ci = [i for i, c in enumerate(names) if c in CAT]
    m = CatBoostRegressor(**CATBOOST_MATCHED, first_feature_use_penalties=pen, penalties_coefficient=1.0, verbose=0, thread_count=-1)
    m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=ES_ROUNDS, verbose=False)
    sel = np.random.RandomState(0).choice(len(te), min(1500, len(te)), replace=False)
    sv = np.asarray(shap.TreeExplainer(m).shap_values(Xte.iloc[sel]))
    grp = [("recipe" if n.startswith("rc_") else group(n)) for n in names]
    rk = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0), "group": grp})
    rk = rk.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True); rk.index += 1
    rk.to_csv(out / f"t3final_ranking_{arch}.csv", index_label="rank")
    g = rk.groupby("group").mean_abs_shap.sum()
    for k in g.index:
        rows.append(dict(arch=arch, group=k, n_features=int((rk.group == k).sum()), sum_mean_abs_shap=g[k], share=g[k] / g.sum()))
    print(arch, "split-1 test R2", round(float(m.score(Xte, y[te])), 4), "best_iter", m.get_best_iteration())
    print(rk.head(10).to_string()); print(rk[rk.group == "recipe"].head(6).to_string())
pd.DataFrame(rows).to_csv(out / "t3final_groups.csv", index=False)
print(pd.DataFrame(rows).round(4).to_string())
