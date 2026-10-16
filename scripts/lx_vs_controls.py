#!/usr/bin/env python3
"""Stage-2 literature block vs its matched DOI-level controls (permuted-real / Gaussian), same stage 1 & stage 2.

usage: lx_vs_controls.py <results-prefix> [--partition dev|confirm] [--subset lx_covered]
Delta(record) = mean_c SE(control_c) - SE(block);  >0 means the real block beats the average control.
Per seed: pooled OOF over the 5 disjoint folds; 95% CI by DOI-cluster bootstrap; then reported per seed and
averaged. Also: fold-level win counts (block MSE < mean control MSE in that fold), and leave-one-seed-out sign.
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET


def load(d, tier, n, arch="overall"):
    p = np.full(n, np.nan); fo = np.full(n, -1)
    for f in sorted((d / "preds").glob(f"kfold_{arch}_{tier}_catboost_*.npz")):
        z = np.load(f); p[z["test_idx"]] = z["pred"]; fo[z["test_idx"]] = int(f.stem.rsplit("_", 1)[1])
    return p, fo


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--subset", default="lx_covered")
    ap.add_argument("--partition"); ap.add_argument("--arch", default="overall"); ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out"); ap.add_argument("--blocks", default="chem,x")
    a = ap.parse_args()
    df = load_dataset(ROOT / "data/dataset_json_enriched.csv", a.arch, subset=a.subset or None, partition=a.partition)
    y = df[TARGET].values; ug, gi = np.unique(df["doi_group"].values, return_inverse=True)
    cnt = np.bincount(gi).astype(float); var = y.var(); rng = np.random.default_rng(0)
    dirs = sorted(Path(d) for d in glob.glob(str(ROOT / "results" / (a.prefix + "*"))) if Path(d).is_dir())
    rows = []
    for b in a.blocks.split(","):
        for refname, pats in [("all_controls", ("perm", "noise")), ("perm", ("perm",)), ("noise", ("noise",)), ("stack_base", None)]:
            per = []; fw = []
            for d in dirs:
                pt, fo = load(d, f"stack_{b}", len(y), a.arch)
                if np.isnan(pt).any(): continue
                if pats is None: ctrl_se = (y - load(d, "stack_base", len(y), a.arch)[0]) ** 2
                else:
                    cs = [load(d, t.name.split("_catboost")[0].replace(f"kfold_{a.arch}_", ""), len(y), a.arch)[0]
                          for t in sorted((d / "preds").glob(f"kfold_{a.arch}_stack_{b}_*_catboost_1.npz")) if any(f"_{p}_s" in t.name for p in pats)]
                    cs = [c for c in cs if not np.isnan(c).any()]
                    if not cs: continue
                    ctrl_se = np.mean([(y - c) ** 2 for c in cs], axis=0)
                e = ctrl_se - (y - pt) ** 2
                s = np.bincount(gi, e, len(ug)); bs = []
                for _ in range(a.nboot):
                    ix = rng.integers(0, len(ug), len(ug)); bs.append(s[ix].sum() / cnt[ix].sum())
                per.append((e.mean(), *np.percentile(bs, [2.5, 97.5]), ))
                for k in range(1, 6): fw.append(e[fo == k].mean() > 0)
            if not per: continue
            P = np.array(per)
            rows.append(dict(block=b, ref=refname, n_seeds=len(P), d_mse=P[:, 0].mean(), d_r2=P[:, 0].mean() / var,
                             ci_lo=P[:, 1].mean() / 1, ci_hi=P[:, 2].mean() / 1, seeds_pos=int((P[:, 0] > 0).sum()),
                             folds_win=f"{int(sum(fw))}/{len(fw)}", n_rec=len(y), n_doi=len(ug)))
    out = pd.DataFrame(rows).round(4); print(out.to_string(index=False))
    if a.out: out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
