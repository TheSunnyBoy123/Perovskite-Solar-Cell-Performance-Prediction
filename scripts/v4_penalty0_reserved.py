#!/usr/bin/env python3
"""Robustness run: t3 with NO first-feature-use penalty on the recipe flags (default 0.5 in the main comparison).

Same reserved v4final records, same development training records, same five seeds / DOI-disjoint validation splits,
same default CatBoost configuration as results/v4/final; only `pen_lit` changes (0.5 -> 0). The reserved publications were
examined in earlier experiments, so this is a sensitivity check and not independent confirmation. Writes
results/revision/penalty0/pred_t3_pen0.npy and penalty0_comparison.json (paired publication-cluster bootstrap, 2000
resamples, rng seed 0, same estimator as scripts/v4_t1_baseline.py)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
from psc import v4
from psc.v4model import PSCModel

OUT = ROOT / "results/revision/penalty0"; OUT.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "results/v4/final"
df = load_dataset(); df["part"] = df["doi_group"].map(v4.v4_partition)
dev, fin = df[df.part == "v4dev"].reset_index(drop=True), df[df.part == "v4final"].reset_index(drop=True)
ydev, yfin = dev[TARGET].values.astype(float), fin[TARGET].values.astype(float)
f = OUT / "pred_t3_pen0.npy"
if f.exists(): P0 = np.load(f)
else:
    m = PSCModel("t3", {}, pen_lit=0.0, seeds=range(5)); m.fit(dev, ydev, dev.doi_group.values)
    P0 = m.predict(fin, per_seed=True); np.save(f, P0)
P2, P3 = np.load(SRC / "pred_t2_default.npy"), np.load(SRC / "pred_t3_default.npy")
cov = fin.lx_covered.values == 1; g = fin.doi_group.values; ug, inv = np.unique(g, return_inverse=True)
def paired(A, B, mask, nb=2000):
    d = ((A.mean(0) - yfin) ** 2 - (B.mean(0) - yfin) ** 2)
    s = np.bincount(inv[mask], weights=d[mask], minlength=len(ug)); n = np.bincount(inv[mask], minlength=len(ug)); ok = n > 0
    s, n = s[ok], n[ok]; idx = np.random.default_rng(0).integers(0, len(s), (nb, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1) / yfin[mask].var()
    return dict(dR2=float(d[mask].mean() / yfin[mask].var()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
def metr(P, mask):
    e = P.mean(0)[mask] - yfin[mask]
    return dict(r2=float(1 - (e ** 2).mean() / yfin[mask].var()), mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())))
res = {}
for sc, mask in (("covered", cov), ("full", np.ones(len(fin), bool))):
    res[sc] = {"t3_pen0": metr(P0, mask), "t3_pen0.5": metr(P3, mask), "t2": metr(P2, mask),
               "t3_pen0_minus_t2": paired(P2, P0, mask), "t3_pen0.5_minus_t2": paired(P2, P3, mask), "t3_pen0_minus_t3_pen0.5": paired(P3, P0, mask)}
json.dump(res, open(OUT / "penalty0_comparison.json", "w"), indent=1)
for sc in res:
    print(sc, {k: (v if "minus" not in k else {a: round(b, 4) for a, b in v.items()}) for k, v in res[sc].items()})
