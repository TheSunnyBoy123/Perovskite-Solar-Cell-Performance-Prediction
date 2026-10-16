#!/usr/bin/env python3
"""Paired comparison on the reserved quantity-covered 'qconf' publications (results/qty_holdout).
Per tier: mean R2/MAE/RMSE over 5 seeds; paired dR2/dRMSE vs a reference (seed-wise, same test records);
95% CI from a publication-cluster bootstrap of the seed-averaged squared-error difference (dMSE>0 = tier better)."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
ref = sys.argv[1] if len(sys.argv) > 1 else "t2"
df = load_dataset(ROOT / "data/dataset_json_enriched.csv", "overall", subset="qty_covered", partition="qconf")
y = df[TARGET].values; g = df["doi_group"].values; ug, gi = np.unique(g, return_inverse=True); var = y.var()
d = ROOT / "results/qty_holdout/preds"
tiers = sorted({f.name.rsplit("_s", 1)[0] for f in d.glob("*.npz")})
def P(t):
    return np.mean([np.load(d / f"{t}_s{s}.npz")["pred"] for s in range(42, 47) if (d / f"{t}_s{s}.npz").exists()], axis=0)
def per_seed(t):
    return [np.load(d / f"{t}_s{s}.npz")["pred"] for s in range(42, 47)]
rng = np.random.default_rng(0); rows = []
pr = per_seed(ref)
print(f"qconf: {len(y)} records, {len(ug)} publications; ref={ref}")
for t in tiers:
    pt = per_seed(t)
    r2 = [1 - ((y - p) ** 2).mean() / var for p in pt]; mae = [np.abs(y - p).mean() for p in pt]; rmse = [np.sqrt(((y - p) ** 2).mean()) for p in pt]
    r2r = [1 - ((y - p) ** 2).mean() / var for p in pr]; rmr = [np.sqrt(((y - p) ** 2).mean()) for p in pr]
    e = np.mean([(y - a) ** 2 - (y - b) ** 2 for a, b in zip(pr, pt)], axis=0)
    s = np.bincount(gi, e, len(ug)); c = np.bincount(gi, minlength=len(ug)).astype(float)
    bs = [s[b].sum() / c[b].sum() for b in (rng.integers(0, len(ug), len(ug)) for _ in range(2000))]
    dr2 = np.array(r2) - np.array(r2r)
    row = dict(tier=t, r2=np.mean(r2), mae=np.mean(mae), rmse=np.mean(rmse), d_r2=dr2.mean(), d_rmse=np.mean(rmse) - np.mean(rmr),
               d_r2_ci=[np.percentile(bs, 2.5) / var, np.percentile(bs, 97.5) / var], seeds_pos=int((dr2 > 0).sum()))
    rows.append(row)
    print(f"{t:16s} R2={row['r2']:.4f} MAE={row['mae']:.3f} RMSE={row['rmse']:.3f} dR2={row['d_r2']:+.4f} CI[{row['d_r2_ci'][0]:+.4f},{row['d_r2_ci'][1]:+.4f}] dRMSE={row['d_rmse']:+.4f} seeds>0={row['seeds_pos']}/5")
json.dump(rows, open(ROOT / "results/qty_holdout/summary.json", "w"), indent=1, default=float)
