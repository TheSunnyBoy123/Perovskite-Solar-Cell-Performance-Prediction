#!/usr/bin/env python3
"""Paired comparison of tiers vs a reference tier on disjoint grouped K-fold out-of-fold predictions.

usage: lx_compare.py <config-glob-prefix e.g. lx_dev_kfold_s> --ref t2 [--partition dev --subset lx_covered]
For each seed the 5 disjoint folds give one OOF prediction per record. Delta = MSE(ref) - MSE(tier)
(positive = tier better). 95% CI: publication-cluster bootstrap (resampling DOIs) on the per-record
squared-error difference, per seed, then seeds averaged; fold-level wins counted over all seed x fold cells.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET


def oof(res, tier, arch, n):
    p = np.full(n, np.nan)
    for f in sorted((res / "preds").glob(f"kfold_{arch}_{tier}_catboost_*.npz")):
        z = np.load(f); p[z["test_idx"]] = z["pred"]
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--ref", default="t2")
    ap.add_argument("--subset", default="lx_covered"); ap.add_argument("--partition", default=None)
    ap.add_argument("--arch", default="overall"); ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load_dataset(ROOT / "data/dataset_json_enriched.csv", a.arch, subset=a.subset or None, partition=a.partition)
    y = df[TARGET].values; g = df["doi_group"].values
    ug, gi = np.unique(g, return_inverse=True)
    dirs = sorted(Path(d) for d in glob.glob(str(ROOT / "results" / (a.prefix + "*"))) if Path(d).is_dir())
    tiers = sorted({f.name.split("_")[2] + ("_" + f.name.split("_")[3] if f.name.split("_")[3] != "catboost" else "") if False else
                    f.name[len(f"kfold_{a.arch}_"):].rsplit("_catboost_", 1)[0] for d in dirs for f in (d / "preds").glob("kfold_*.npz")})
    rng = np.random.default_rng(0); rows = []
    for t in tiers:
        if t == a.ref: continue
        per_seed = []; r2s = []; fold_wins = []; fold_d = []
        for d in dirs:
            pr, pt = oof(d, a.ref, a.arch, len(y)), oof(d, t, a.arch, len(y))
            if np.isnan(pr).any() or np.isnan(pt).any(): continue
            e = (y - pr) ** 2 - (y - pt) ** 2
            s = np.bincount(gi, e, len(ug)); c = np.bincount(gi, minlength=len(ug)).astype(float)
            boots = []
            for _ in range(a.nboot):
                b = rng.integers(0, len(ug), len(ug)); boots.append(s[b].sum() / c[b].sum())
            var = y.var()
            per_seed.append((e.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
            r2s.append((1 - ((y - pr) ** 2).mean() / var, 1 - ((y - pt) ** 2).mean() / var))
            for z in sorted((d / "preds").glob(f"kfold_{a.arch}_{t}_catboost_*.npz")):
                k = z.name.rsplit("_", 1)[1]; zr = np.load(d / "preds" / z.name.replace(f"_{t}_", f"_{a.ref}_"))
                zt = np.load(z); dm = ((zr["y"] - zr["pred"]) ** 2).mean() - ((zt["y"] - zt["pred"]) ** 2).mean()
                fold_d.append(dm); fold_wins.append(dm > 0)
        if not per_seed: continue
        ps = np.array(per_seed); r = np.array(r2s)
        rows.append(dict(tier=t, ref=a.ref, n_seeds=len(ps), r2_ref=r[:, 0].mean(), r2_tier=r[:, 1].mean(),
                         d_r2=(r[:, 1] - r[:, 0]).mean(), d_mse_mean=ps[:, 0].mean(),
                         ci_lo_mean=ps[:, 1].mean(), ci_hi_mean=ps[:, 2].mean(),
                         seeds_positive=int((r[:, 1] > r[:, 0]).sum()),
                         folds_win=f"{int(sum(fold_wins))}/{len(fold_wins)}", n_rec=len(y), n_doi=len(ug)))
    out = pd.DataFrame(rows).round(4)
    print(out.to_string(index=False))
    if a.out: out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
