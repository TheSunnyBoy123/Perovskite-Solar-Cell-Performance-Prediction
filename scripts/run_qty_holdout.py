#!/usr/bin/env python3
"""Train on all quantity-covered 'qdev' publications, score once on the reserved 'qconf' publications.
Per seed: DOI-disjoint 15% of qdev publications form the early-stopping validation set (test never used).
Outputs results/qty_holdout/runs.jsonl + preds/*.npz.   usage: run_qty_holdout.py [--smoke]"""
import json, sys, argparse
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
from psc.evaluate import metrics
from psc.features import TrainFitPreprocessor, base_frame
from psc.models import fit_predict

ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
TIERS = ["t1", "t2", "t3", "t3qq", "t3qq_cat", "t3_qq"] + [f"t2_lxs{k}_s{i}" for k in ("perm", "noise") for i in range(3)]
df = load_dataset(ROOT / "data/dataset_json_enriched.csv", "overall", subset="qty_covered")
y = df[TARGET].values.astype(float); g = df["doi_group"].values
tr_all = np.where(df["qpartition"] == "qdev")[0]; te = np.where(df["qpartition"] == "qconf")[0]
assert not set(g[tr_all]) & set(g[te])
out = ROOT / "results" / ("qty_holdout_smoke" if a.smoke else "qty_holdout"); (out / "preds").mkdir(parents=True, exist_ok=True)
frames = {t: base_frame(df, t) for t in TIERS}
for seed in range(42, 47):
    rng = np.random.default_rng(seed); ud = np.unique(g[tr_all]); va_d = set(rng.choice(ud, int(0.15 * len(ud)), replace=False))
    va = np.array([i for i in tr_all if g[i] in va_d]); tr = np.array([i for i in tr_all if g[i] not in va_d])
    for t in TIERS:
        pre = TrainFitPreprocessor().fit(frames[t].iloc[tr])
        Xtr, Xva, Xte = (pre.transform(frames[t].iloc[i]) for i in (tr, va, te))
        params = {"iterations": 300} if a.smoke else {}
        p, info = fit_predict("catboost", Xtr, y[tr], Xva, y[va], Xte, params)
        m = metrics(y[te], p)
        rec = dict(seed=seed, tier=t, n_train=len(tr), n_val=len(va), n_test=len(te), n_doi_test=len(set(g[te])),
                   n_features=Xtr.shape[1], n_qty_used=sum("lx_qty" in c and not c.endswith("_na") for c in Xtr.columns), **m)
        np.savez_compressed(out / "preds" / f"{t}_s{seed}.npz", test_idx=te, y=y[te], pred=p)
        open(out / "runs.jsonl", "a").write(json.dumps(rec) + "\n"); print(seed, t, round(m["r2"], 4), flush=True)
    if a.smoke: break
