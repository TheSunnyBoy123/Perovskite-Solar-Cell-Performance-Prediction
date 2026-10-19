#!/usr/bin/env python3
"""Paired analysis of a dev run against a reference arm (default M2).

Per seed, the 5 disjoint folds give one out-of-fold prediction per dev record and arm. Reported per arm and subset:
  R2 / MAE / RMSE  mean and SD over seeds (variability = fold-assignment + model seed, NOT sampling uncertainty)
  dR2 vs ref       mean over seeds, per-seed values, #seeds > 0, #folds > 0 (fold-level paired R2 differences)
  95% CI dR2/dMAE  publication-cluster bootstrap (2000 resamples of DOIs) of the seed-averaged per-record
                   paired difference; dR2 = dMSE / var(y_subset). Resampling unit = publication.
Subsets: all, covered (publication has a joined, non-empty JSON; 'fixed' resolution), uncovered, pin, nip,
covered_pin, covered_nip.

  python analyze.py <run_name> [--ref M2] [--arms a,b] [--out file.csv] [--boot 2000]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))


def covered_mask(groups):
    import m3feat as F
    from psc.data import load_dataset
    F.set_known(set(load_dataset().doi_group))
    T = F.flags("fixed").dropna(how="all")
    return np.isin(groups, T.index)


def load_run(name):
    d = EXP / "runs" / name
    y = np.load(d / "y.npy"); rows = pd.read_csv(d / "rows.csv")
    led = [json.loads(l) for l in open(d / "ledger.jsonl")]
    led = pd.DataFrame(led).drop_duplicates(["arm", "seed", "fold"], keep="last")
    return d, y, rows, led


def oof(d, arm, seed, n, folds):
    p = np.full(n, np.nan); fold_of = np.zeros(n, int)
    for k in range(1, folds + 1):
        f = d / "preds" / f"{arm}_s{seed}_f{k}.npz"
        if not f.exists():
            return None, None
        z = np.load(f); p[z["test_idx"]] = z["pred"]; fold_of[z["test_idx"]] = k
    return p, fold_of


def r2(y, p):
    return 1 - np.mean((p - y) ** 2) / np.var(y)


def cluster_boot(diff, groups, B, seed=0):
    """diff: per-record values; returns bootstrap distribution of the mean over resampled publications."""
    s = pd.DataFrame({"g": groups, "d": diff}).groupby("g")["d"].agg(["sum", "size"])
    S, N = s["sum"].values, s["size"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(B, len(S)))
    return S[idx].sum(1) / N[idx].sum(1)


def analyze(name, ref="M2", arms=None, B=2000, folds=5):
    d, y, rows, led = load_run(name)
    g = rows.doi_group.values; arch = rows.Cell_architecture.fillna("").str.lower().values
    cov = covered_mask(g)
    subsets = {"all": np.ones(len(y), bool), "covered": cov, "uncovered": ~cov, "pin": arch == "pin", "nip": arch == "nip",
               "covered_pin": cov & (arch == "pin"), "covered_nip": cov & (arch == "nip")}
    seeds = sorted(led.seed.unique())
    arms = arms or list(dict.fromkeys(led.arm))
    P = {}
    for a in arms + [ref]:
        for s in seeds:
            p, fo = oof(d, a, s, len(y), folds)
            if p is not None:
                P[(a, s)] = p; FO = fo
    out = []
    for a in arms:
        ss = [s for s in seeds if (a, s) in P and (ref, s) in P]
        if not ss:
            continue
        for sub, m in subsets.items():
            yy = y[m]
            r = [r2(yy, P[(a, s)][m]) for s in ss]
            mae = [np.mean(np.abs(P[(a, s)][m] - yy)) for s in ss]
            rmse = [np.sqrt(np.mean((P[(a, s)][m] - yy) ** 2)) for s in ss]
            dr = [r2(yy, P[(a, s)][m]) - r2(yy, P[(ref, s)][m]) for s in ss]
            # fold-level paired differences
            fpos, nf = 0, 0
            for s in ss:
                _, fo = oof(d, a, s, len(y), folds)
                for k in range(1, folds + 1):
                    mk = m & (fo == k)
                    if mk.sum() < 30:
                        continue
                    nf += 1; fpos += int(r2(y[mk], P[(a, s)][mk]) > r2(y[mk], P[(ref, s)][mk]))
            se_ref = np.mean([(P[(ref, s)][m] - yy) ** 2 for s in ss], axis=0)
            se_a = np.mean([(P[(a, s)][m] - yy) ** 2 for s in ss], axis=0)
            ae_ref = np.mean([np.abs(P[(ref, s)][m] - yy) for s in ss], axis=0)
            ae_a = np.mean([np.abs(P[(a, s)][m] - yy) for s in ss], axis=0)
            bt = cluster_boot(se_ref - se_a, g[m], B) / np.var(yy)
            bm = cluster_boot(ae_a - ae_ref, g[m], B, seed=1)
            out.append(dict(arm=a, subset=sub, n=int(m.sum()), ndoi=int(len(set(g[m]))), n_seeds=len(ss),
                            r2=np.mean(r), r2_sd=np.std(r, ddof=1) if len(r) > 1 else np.nan,
                            mae=np.mean(mae), rmse=np.mean(rmse),
                            dR2=np.mean(dr), dR2_seed_sd=np.std(dr, ddof=1) if len(dr) > 1 else np.nan,
                            dR2_lo=np.quantile(bt, 0.025), dR2_hi=np.quantile(bt, 0.975),
                            p_boot_le0=float((bt <= 0).mean()),
                            dMAE=float(np.mean(ae_a - ae_ref)), dMAE_lo=np.quantile(bm, 0.025), dMAE_hi=np.quantile(bm, 0.975),
                            dRMSE=float(np.mean(rmse) - np.mean([np.sqrt(np.mean((P[(ref, s)][m] - yy) ** 2)) for s in ss])),
                            seeds_pos=int(sum(x > 0 for x in dr)), folds_pos=fpos, n_folds=nf,
                            per_seed_dR2=" ".join(f"{x:+.4f}" for x in dr)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--ref", default="M2")
    ap.add_argument("--arms", default=None); ap.add_argument("--out", default=None); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--subsets", default="all,covered,pin,nip")
    a = ap.parse_args()
    res = analyze(a.run, a.ref, a.arms.split(",") if a.arms else None, a.boot)
    out = Path(a.out) if a.out else EXP / "runs" / a.run / f"paired_vs_{a.ref}.csv"
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    show = res[res.subset.isin(a.subsets.split(","))]
    cols = ["arm", "subset", "n", "r2", "r2_sd", "mae", "dR2", "dR2_lo", "dR2_hi", "seeds_pos", "folds_pos", "n_folds", "dMAE"]
    print(show[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
