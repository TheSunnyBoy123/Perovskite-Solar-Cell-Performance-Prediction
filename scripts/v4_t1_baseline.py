#!/usr/bin/env python3
"""Common-split t1/t2/t3 comparison on the reserved v4final publications.

Trains ONLY the missing t1 arm (default CatBoost config, 5 seed models, same DOI-disjoint early-stopping splits as t2/t3 in
scripts/v4_final.py) and reuses the saved t2_default / t3_default predictions from results/v4/final. Writes to
results/v4/final_t1/ (the original results/v4/final files are not touched). Paired cluster bootstrap over publications
(2000 resamples, rng seed 0), identical to v4_final.py; per-tier R2 intervals use the same publication-cluster bootstrap.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
from psc import v4
from psc.v4model import PSCModel

SRC, OUT = ROOT / "results/v4/final", ROOT / "results/v4/final_t1"; OUT.mkdir(parents=True, exist_ok=True)
df = load_dataset(); df["part"] = df["doi_group"].map(v4.v4_partition)
dev, fin = df[df.part == "v4dev"].reset_index(drop=True), df[df.part == "v4final"].reset_index(drop=True)
ydev, yfin = dev[TARGET].values.astype(float), fin[TARGET].values.astype(float)
print(f"dev n={len(dev)} final n={len(fin)} dois={fin.doi_group.nunique()}", flush=True)

f = OUT / "pred_t1_default.npy"
if f.exists(): P1 = np.load(f)
else:
    m = PSCModel("t1", {}, seeds=range(5)); m.fit(dev, ydev, dev.doi_group.values)
    P1 = m.predict(fin, per_seed=True); np.save(f, P1)
    json.dump({"n_features_t1": len(m.cols_)}, open(OUT / "t1_matrix.json", "w"))
preds = {"t1": P1, "t2": np.load(SRC / "pred_t2_default.npy"), "t3": np.load(SRC / "pred_t3_default.npy")}
assert all(P.shape == (5, len(fin)) for P in preds.values())

cov = fin.lx_covered.values == 1; g = fin.doi_group.values; ug, inv = np.unique(g, return_inverse=True)
def stats(P, mask, nb=2000):
    """Point metrics of the seed-mean prediction + publication-cluster bootstrap interval of R2 itself (resample publications,
    recompute R2 with the resampled variance; rng seed 0)."""
    e = P.mean(0) - yfin; yy = yfin[mask]; ee = e[mask]; ii = inv[mask]; k = len(ug)
    n = np.bincount(ii, minlength=k).astype(float); se = np.bincount(ii, weights=ee ** 2, minlength=k)
    sy = np.bincount(ii, weights=yy, minlength=k); sy2 = np.bincount(ii, weights=yy ** 2, minlength=k)
    ok = n > 0; n, se, sy, sy2 = n[ok], se[ok], sy[ok], sy2[ok]
    idx = np.random.default_rng(0).integers(0, len(n), (nb, len(n)))
    N = n[idx].sum(1); var = sy2[idx].sum(1) / N - (sy[idx].sum(1) / N) ** 2; r2b = 1 - (se[idx].sum(1) / N) / var
    return dict(r2=1 - (ee ** 2).mean() / yy.var(), r2_lo=float(np.percentile(r2b, 2.5)), r2_hi=float(np.percentile(r2b, 97.5)),
                mae=float(np.abs(ee).mean()), rmse=float(np.sqrt((ee ** 2).mean())))
def paired(A, B, mask, nb=2000):
    d = ((A.mean(0) - yfin) ** 2 - (B.mean(0) - yfin) ** 2)
    s = np.bincount(inv[mask], weights=d[mask], minlength=len(ug)); n = np.bincount(inv[mask], minlength=len(ug)); ok = n > 0
    s, n = s[ok], n[ok]; rng = np.random.default_rng(0); idx = rng.integers(0, len(s), (nb, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1) / yfin[mask].var()
    per_seed = [float((((A[i] - yfin) ** 2 - (B[i] - yfin) ** 2)[mask]).mean() / yfin[mask].var()) for i in range(len(A))]
    return dict(dR2=float(d[mask].mean() / yfin[mask].var()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                seeds_pos=int(sum(x > 0 for x in per_seed)), n_seeds=len(per_seed), per_seed=per_seed)
rows, comp = [], {}
for scope, mask in (("covered", cov), ("full", np.ones(len(fin), bool))):
    for t, P in preds.items():
        rows.append({"scope": scope, "tier": t, "n": int(mask.sum()), "dois": int(len(set(g[mask]))), **stats(P, mask)})
    comp[scope] = {"t2_minus_t1": paired(preds["t1"], preds["t2"], mask), "t3_minus_t2": paired(preds["t2"], preds["t3"], mask),
                   "t3_minus_t1": paired(preds["t1"], preds["t3"], mask)}
pd.DataFrame(rows).to_csv(OUT / "metrics_3tier.csv", index=False); json.dump(comp, open(OUT / "comparisons_3tier.json", "w"), indent=1)
print(pd.DataFrame(rows).round(4).to_string())
for sc in comp:
    for k, v in comp[sc].items(): print(f"{sc:8s} {k:12s} dR2={v['dR2']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] seeds>0 {v['seeds_pos']}/{v['n_seeds']}")
