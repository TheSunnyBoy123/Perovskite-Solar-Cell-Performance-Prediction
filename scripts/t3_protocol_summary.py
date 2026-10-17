#!/usr/bin/env python3
"""Paired t3 (final, 't3f') versus t2 summaries for every protocol that now has the final t3 arm.

For each protocol / architecture / model / split, the t2 predictions are the stored ones (ablation_v1, strict_validation_v1,
random_convergence_cap20000, model_dev_v1, baselines) and the t3f predictions come from results/t3final_*_v1; test
indices are identical (asserted). Delta R^2 = R^2(t3f) - R^2(t2) on the same test records, with a 95% DOI-cluster
bootstrap interval within each split (1,000 resamples). Because test sets of repeated holdouts overlap, no pooled interval
is formed; the mean over splits, the count of splits with a positive difference and the count with an interval above zero
are reported. 'covered' restricts the test records to publications with extraction data (lx_covered == 1).
Output: results/t3final_summary/{matched_pairs.csv,matched_summary.csv,summary.json}
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset  # noqa: E402

R = ROOT / "results"; OUT = R / "t3final_summary"; OUT.mkdir(exist_ok=True)
NB = 1000


def runs(name):
    f = R / name / "runs.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(f)]) if f.exists() else None


def t2_path(protocol, arch, model, k):
    c = []
    if protocol == "random" and model == "catboost" and arch == "overall":
        c.append(R / "random_convergence_cap20000/preds" / f"random_overall_t2_catboost_{k}.npz")
    c += [R / d / "preds" / f"{protocol}_{arch}_t2_{model}_{k}.npz" for d in ("ablation_v1", "strict_validation_v1")]
    return next(p for p in c if p.exists())


def boot(err2_a, err2_b, y, g, nb=NB, seed=0):
    """Delta R2 = R2(b) - R2(a) on the same records; cluster bootstrap over DOI groups (squared-error sums per group)."""
    ug, inv = np.unique(g, return_inverse=True)
    sa = np.bincount(inv, weights=err2_a); sb = np.bincount(inv, weights=err2_b)
    sy = np.bincount(inv, weights=y); sy2 = np.bincount(inv, weights=y * y); n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ug), (nb, len(ug)))
    N = n[idx].sum(1); var = sy2[idx].sum(1) / N - (sy[idx].sum(1) / N) ** 2
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / N / var
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pair(y, pa, pb, g):
    ea, eb = (pa - y) ** 2, (pb - y) ** 2
    v = y.var()
    lo, hi = boot(ea, eb, y, g)
    return dict(n=len(y), n_doi=len(set(g)), r2_a=1 - ea.mean() / v, r2_b=1 - eb.mean() / v, d_r2=(ea.mean() - eb.mean()) / v,
                d_mae=float(np.abs(pb - y).mean() - np.abs(pa - y).mean()),
                d_rmse=float(np.sqrt(eb.mean()) - np.sqrt(ea.mean())), lo=lo, hi=hi)


def main():
    M = runs("t3final_matched_v1"); rows = []
    data = {}
    for arch in ("overall", "pin", "nip"):
        df = load_dataset(arch=arch); data[arch] = (df["doi_group"].values, df["lx_covered"].values == 1)
    for (protocol, arch, model, k), _ in M.groupby(["protocol", "arch", "model", "split"]):
        if protocol == "random" and arch != "overall":
            ref = R / "ablation_v1/preds" / f"random_{arch}_t2_{model}_{k}.npz"        # default-iteration t2 arm for the default-iteration t3f arm
            if not ref.exists(): continue
            a = np.load(ref)
        else:
            a = np.load(t2_path(protocol, arch, model, k))
        b = np.load(R / "t3final_matched_v1/preds" / f"{protocol}_{arch}_t3f_{model}_{k}.npz")
        assert np.array_equal(a["test_idx"], b["test_idx"]) and np.allclose(a["y"], b["y"])
        g_all, cov_all = data[arch]; te = b["test_idx"]; g = g_all[te]; cov = cov_all[te]; y = b["y"]
        for scope, m in (("all", np.ones(len(te), bool)), ("covered", cov)):
            if m.sum() < 50: continue
            rows.append(dict(protocol=protocol, arch=arch, model=model, split=k, scope=scope,
                             **pair(y[m], a["pred"][m], b["pred"][m], g[m])))
    P = pd.DataFrame(rows); P.to_csv(OUT / "matched_pairs.csv", index=False)
    S = (P.groupby(["protocol", "arch", "model", "scope"])
         .apply(lambda x: pd.Series(dict(splits=len(x), n_test=x.n.mean(), r2_t2=x.r2_a.mean(), r2_t3=x.r2_b.mean(),
                                         r2_t2_sd=x.r2_a.std(), r2_t3_sd=x.r2_b.std(), d_r2=x.d_r2.mean(), d_r2_sd=x.d_r2.std(),
                                         d_mae=x.d_mae.mean(), d_rmse=x.d_rmse.mean(), pos=int((x.d_r2 > 0).sum()),
                                         ci_above0=int((x.lo > 0).sum()), ci_below0=int((x.hi < 0).sum()))))
         .reset_index())
    S.to_csv(OUT / "matched_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(4).to_string())


if __name__ == "__main__":
    main()
