#!/usr/bin/env python3
"""Seed-averaged paired analysis + robustness for the stage-2 literature block (block x) vs matched controls.
Delta(record) = mean over seeds of [mean_c SE(control_c) - SE(block)] ; DOI-cluster bootstrap on the record mean.
Also: leave-top-k-publications-out (publications with largest positive contribution removed), per-architecture subsets,
and share of the total gain from the top 5% of publications."""
import argparse, glob, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET


def oof(d, tier, n):
    p = np.full(n, np.nan)
    for f in (d / "preds").glob(f"kfold_overall_{tier}_catboost_*.npz"):
        z = np.load(f); p[z["test_idx"]] = z["pred"]
    return p


ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--partition"); ap.add_argument("--nboot", type=int, default=4000)
ap.add_argument("--block", default="x"); a = ap.parse_args()
df = load_dataset(ROOT / "data/dataset_json_enriched.csv", "overall", subset="lx_covered", partition=a.partition)
y = df[TARGET].values; var = y.var()
dirs = sorted(Path(d) for d in glob.glob(str(ROOT / "results" / (a.prefix + "*"))) if Path(d).is_dir())
E_ = []
for d in dirs:
    pt = oof(d, f"stack_{a.block}", len(y))
    cs = [oof(d, f"stack_{a.block}_{k}_s{s}", len(y)) for k in ("perm", "noise") for s in range(3)]
    if np.isnan(pt).any() or any(np.isnan(c).any() for c in cs): continue
    E_.append(np.mean([(y - c) ** 2 for c in cs], 0) - (y - pt) ** 2)
e = np.mean(E_, 0); print("seeds", len(E_), "records", len(y))
g = df["doi_group"].values; ug, gi = np.unique(g, return_inverse=True); rng = np.random.default_rng(1)


def boot(e, gi, n):
    ug_n = gi.max() + 1; s = np.bincount(gi, e, ug_n); c = np.bincount(gi, minlength=ug_n).astype(float); r = []
    for _ in range(a.nboot):
        b = rng.integers(0, ug_n, ug_n); r.append(s[b].sum() / c[b].sum())
    return np.mean(e), *np.percentile(r, [2.5, 97.5])


m, lo, hi = boot(e, gi, a.nboot)
print(f"ALL  dMSE={m:.4f} CI[{lo:.4f},{hi:.4f}] dR2={m/var:+.4f} (dR2 CI [{lo/var:+.4f},{hi/var:+.4f}])  per-seed dR2:", [round(x.mean() / var, 4) for x in E_])
s = np.bincount(gi, e); order = np.argsort(-s); tot = s.sum()
print("share of total gain from top 5% of publications:", round(s[order[:max(1, len(ug) // 20)]].sum() / tot, 2) if tot > 0 else "n/a (total<=0)")
for k in (1, 5, 10, 25):
    keep = ~np.isin(gi, order[:k]); mk, lk, hk = boot(e[keep], np.unique(gi[keep], return_inverse=True)[1], a.nboot)
    print(f"  leave top-{k:2d} pubs out: dMSE={mk:+.4f} CI[{lk:+.4f},{hk:+.4f}]")
arch = df["Cell_architecture"].str.lower().values
for ar in ("nip", "pin"):
    mk = arch == ar
    if mk.sum() > 500:
        m2, l2, h2 = boot(e[mk], np.unique(gi[mk], return_inverse=True)[1], a.nboot)
        print(f"  {ar}: n={mk.sum()} dMSE={m2:+.4f} CI[{l2:+.4f},{h2:+.4f}] dR2={m2/var:+.4f}")
