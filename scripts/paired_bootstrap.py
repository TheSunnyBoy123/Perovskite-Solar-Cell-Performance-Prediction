#!/usr/bin/env python3
"""Cluster (DOI) bootstrap of paired tier differences within each test split.

For each split, test DOIs are resampled with replacement (B draws) and the paired
difference in R2/MAE/RMSE between two tiers' predictions on the SAME test records is
recomputed. Reports a per-split 95% percentile interval and the number of splits whose
interval excludes 0. This handles within-publication dependence in the test set but NOT
overlap between repeated-holdout test sets, so no pooled p-value is reported.
Usage: python scripts/paired_bootstrap.py results/ablation_v1 [model=catboost] [protocol=doi]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset  # noqa: E402

d = Path(sys.argv[1]); model = sys.argv[2] if len(sys.argv) > 2 else "catboost"
protocol = sys.argv[3] if len(sys.argv) > 3 else "doi"
B = 1000; rng = np.random.default_rng(0)
CONTR = [("t2", "t1"), ("t3", "t2"), ("t3", "t1")]


def stats(y, p):
    e = p - y
    return np.array([1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.abs(e).mean(), np.sqrt((e ** 2).mean())])


rows = []
for arch in ("overall", "pin", "nip"):
    g = load_dataset(arch=arch)["doi_group"].values
    for hi, lo in CONTR:
        for k in range(1, 6):
            fh = d / "preds" / f"{protocol}_{arch}_{hi}_{model}_{k}.npz"
            fl = d / "preds" / f"{protocol}_{arch}_{lo}_{model}_{k}.npz"
            if not (fh.exists() and fl.exists()):
                continue
            a, b = np.load(fh), np.load(fl)
            assert (a["test_idx"] == b["test_idx"]).all()
            y, ph, pl, gi = a["y"], a["pred"], b["pred"], g[a["test_idx"]]
            _, inv = np.unique(gi, return_inverse=True); nd = inv.max() + 1
            order = np.argsort(inv); bounds = np.searchsorted(inv[order], np.arange(nd + 1))
            idx_by = [order[bounds[i]:bounds[i + 1]] for i in range(nd)]
            obs = stats(y, ph) - stats(y, pl)
            bs = np.empty((B, 3))
            for r in range(B):
                ii = np.concatenate([idx_by[j] for j in rng.integers(0, nd, nd)])
                bs[r] = stats(y[ii], ph[ii]) - stats(y[ii], pl[ii])
            lo_, hi_ = np.percentile(bs, [2.5, 97.5], axis=0)
            rows.append(dict(arch=arch, contrast=f"{hi}-{lo}", split=k, n_test_doi=nd,
                             d_r2=obs[0], r2_lo=lo_[0], r2_hi=hi_[0],
                             d_mae=obs[1], mae_lo=lo_[1], mae_hi=hi_[1],
                             d_rmse=obs[2], rmse_lo=lo_[2], rmse_hi=hi_[2]))
out = pd.DataFrame(rows); out.to_csv(d / f"paired_bootstrap_{protocol}_{model}.csv", index=False)
sm = out.groupby(["arch", "contrast"]).apply(lambda x: pd.Series(dict(
    splits=len(x), d_r2=x.d_r2.mean(), r2_pos=(x.d_r2 > 0).sum(), r2_ci_excl0=((x.r2_lo > 0) | (x.r2_hi < 0)).sum(),
    d_rmse=x.d_rmse.mean(), rmse_neg=(x.d_rmse < 0).sum()))).reset_index()
sm.to_csv(d / f"paired_bootstrap_summary_{protocol}_{model}.csv", index=False)
print(sm.round(4).to_string())
