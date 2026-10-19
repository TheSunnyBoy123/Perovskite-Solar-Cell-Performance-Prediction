#!/usr/bin/env python3
"""Run the frozen selected candidate (from selection.json) on
  (a) the full development population, DOI-grouped 5-fold seeds 42-44 (secondary; not used for selection), and
  (b) the covered-only repeated DOI-grouped holdouts: the five existing splits, restricted to covered records.

Nothing here modifies the candidate or the selection. If no candidate passed, the script exits without fitting.
Usage:  python scripts/confirm_covered.py [--smoke]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc import candidates as cd  # noqa: E402
from psc import v4  # noqa: E402
from psc.data import CAT_ALL, TARGET, load_dataset  # noqa: E402
from psc.features import TrainFitPreprocessor, base_frame  # noqa: E402
from psc.splits import make_splits  # noqa: E402

SEL = ROOT / "results" / "candidates" / "covered_dev" / "selection.json"
OUT = ROOT / "results" / "candidates" / "confirmation"
EXISTING = ROOT / "results" / "t3final_matched_v1" / "preds"


def fit_predict(Z, y, tr, va, te, cat, pen):
    p = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.7,
             min_data_in_leaf=5, random_seed=42, loss_function="RMSE")
    if pen is not None:
        p["first_feature_use_penalties"] = pen
        p["penalties_coefficient"] = 1.0
    ci = [i for i, c in enumerate(Z.columns) if c in cat]
    m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
    m.fit(Z.iloc[tr], y[tr], cat_features=ci, eval_set=(Z.iloc[va], y[va]), early_stopping_rounds=100, verbose=False)
    return m.predict(Z.iloc[te])


def metrics(y, p):
    e = p - y
    return dict(r2=float(1 - (e ** 2).mean() / y.var()), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())))


def evaluate(dfp, folds, kinds, tables, tag):
    """M2 and the selected candidate on identical test records. `dfp` carries the features and outcome."""
    y = dfp[TARGET].values.astype(float); g = dfp["doi_group"].values
    arch = dfp["Cell_architecture"].str.lower().values
    cov = (dfp["lx_covered"] == 1).values
    X2 = base_frame(dfp, "t2")
    rows = []
    for name, tr, va, te, fk in folds:
        assert not set(g[tr]) & set(g[te]) and not set(g[va]) & set(g[te]) and not set(g[tr]) & set(g[va]), "DOI overlap"
        pre = TrainFitPreprocessor().fit(X2.iloc[tr])
        Z2 = pre.transform(X2).reset_index(drop=True)
        cand = pd.concat([cd.build(k, tables, dfp, tr) for k in kinds], axis=1)
        cd.assert_prefixes(cand.columns)
        Z = pd.concat([Z2, cand], axis=1)
        arms = {"M2": (Z2, None), "CAND": (Z, cd.penalty_vector(Z.columns, cand.columns))}
        for arm, (ZZ, pen) in arms.items():
            pr = fit_predict(ZZ, y, tr, va, te, set(CAT_ALL), pen)
            for sub, m in (("all", np.ones(len(te), bool)), ("covered", cov[te]), ("uncovered", ~cov[te]),
                           ("pin", arch[te] == "pin"), ("nip", arch[te] == "nip")):
                if m.sum() == 0:
                    continue
                mm = metrics(y[te][m], pr[m])
                rows.append(dict(eval=tag, fold=fk, arm=arm, subset=sub, n=int(m.sum()), ndoi=len(set(g[te][m])),
                                 nfeat=ZZ.shape[1], **mm))
            np.save(OUT / f"pred_{tag}_{arm}_{fk}.npy", np.stack([te, pr]))
        print(tag, fk, "done", flush=True)
    return pd.DataFrame(rows)


def paired(R):
    out = []
    for (tag, sub), d in R.groupby(["eval", "subset"]):
        a = d[d.arm == "M2"].set_index("fold"); b = d[d.arm == "CAND"].set_index("fold")
        dr = b.r2 - a.r2
        out.append(dict(eval=tag, subset=sub, folds=len(dr), mean_dR2=dr.mean(), sd_dR2=dr.std(ddof=1),
                        positive=int((dr > 0).sum()), mean_dMAE=(b.mae - a.mae).mean(), mean_dRMSE=(b.rmse - a.rmse).mean(),
                        M2_r2=a.r2.mean(), CAND_r2=b.r2.mean(), n=int(a.n.mean())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sel = json.loads(SEL.read_text())
    if sel["selected"] is None:
        print("No candidate passed the pre-registered rule; nothing to confirm.")
        return
    name = sel["selected"]
    kinds = sel["passing_C1_C3"] if name == "C4" else [name]
    print(f"selected: {name} (blocks: {kinds})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tables = cd.paper_tables()

    # (a) secondary: full development population, same kfold design
    df = load_dataset()
    df["v4part"] = df["doi_group"].map(v4.v4_partition)
    dev = df[df.v4part == "v4dev"].reset_index(drop=True)
    seeds = (42,) if a.smoke else (42, 43, 44)
    folds = [(f"s{s}f{k}", tr, va, te, f"s{s}f{k}") for s in seeds
             for k, (tr, va, te) in enumerate(make_splits(dev["doi_group"].values, "kfold", 5, 0.2, 0.15, s), 1)
             if not a.smoke or k == 1]
    Rs = evaluate(dev, folds, kinds, tables, "secondary_full_dev")
    Rs.to_csv(OUT / "secondary_full_dev_folds.csv", index=False)
    paired(Rs).to_csv(OUT / "secondary_full_dev_paired.csv", index=False)

    # (b) confirmation: the existing repeated DOI holdouts (GroupShuffleSplit, seed 42), covered records only
    full = load_dataset()
    g_full = full["doi_group"].values
    existing = list(make_splits(g_full, "doi", 5, 0.2, 0.15, 42))
    cfolds = []
    for k, (tr, va, te) in enumerate(existing, 1):
        stored = np.load(EXISTING / f"doi_overall_t3f_catboost_{k}.npz")
        assert np.array_equal(np.sort(stored["test_idx"]), np.sort(te)), f"split {k} test indices differ from stored"
        assert np.array_equal(np.sort(stored["train_idx"]), np.sort(tr)), f"split {k} train indices differ from stored"
        cov = full["lx_covered"].values == 1
        tr_c, va_c, te_c = tr[cov[tr]], va[cov[va]], te[cov[te]]
        cfolds.append((f"doi{k}", tr_c, va_c, te_c, k))
        if a.smoke:
            break
    print("existing repeated-holdout splits match stored indices", flush=True)
    Rc = evaluate(full, [(f"c{k}", tr, va, te, k) for _, tr, va, te, k in cfolds], kinds, tables, "confirm_covered_doi")
    Rc.to_csv(OUT / "confirm_covered_doi_folds.csv", index=False)
    paired(Rc).to_csv(OUT / "confirm_covered_doi_paired.csv", index=False)
    print(paired(Rc).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
