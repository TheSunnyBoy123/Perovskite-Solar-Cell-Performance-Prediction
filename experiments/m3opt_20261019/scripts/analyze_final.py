#!/usr/bin/env python3
"""Analysis of the frozen final runs (runs/F_final_R, runs/F_final_H).

R (reserved v4final diagnostic): per arm the 5-seed mean prediction; R2/MAE/RMSE per subset; paired dR2 vs M2 with a
  publication-cluster bootstrap (2000 resamples, as scripts/v4_final.py) and per-seed paired deltas. Subsets: full,
  covered_v4 (production lx_covered definition, = manuscript 'covered'), covered_fixed (corrected join, non-empty JSON),
  pin, nip. Reproduction check: M2 / M3_v4 vs results/v4/final/pred_t2_default.npy / pred_t3_default.npy.
H (manuscript repeated DOI holdouts, overall / pin / nip): per split R2, mean +- SD over the 5 splits, per-split paired
  dR2 vs M2, and a publication-cluster bootstrap of the split-averaged dR2 (publications resampled jointly across splits).
  Reproduction check: H M2 vs results/ablation_v1 stored t2 CatBoost predictions.

  python analyze_final.py [--name F_final]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from analyze import covered_mask  # noqa: E402

B = 2000


def r2(y, p):
    return 1 - np.mean((p - y) ** 2) / np.var(y)


def paired_boot(d, g, var, seed=0):
    s = pd.DataFrame({"g": g, "d": d}).groupby("g")["d"].agg(["sum", "size"])
    S, N = s["sum"].values, s["size"].values
    idx = np.random.default_rng(seed).integers(0, len(S), size=(B, len(S)))
    return S[idx].sum(1) / N[idx].sum(1) / var


def final_R(name):
    d = EXP / "runs" / f"{name}_R"
    y = np.load(d / "y_final.npy"); rows = pd.read_csv(d / "rows_final.csv")
    g = rows.doi_group.values; arch = rows.Cell_architecture.fillna("").str.lower().values
    from psc.data import load_dataset, TARGET
    from psc import v4
    full = load_dataset(); full["part"] = full.doi_group.map(v4.v4_partition)
    fin = full[full.part == "v4final"].reset_index(drop=True)
    assert np.array_equal(fin.doi_group.values, g) and np.allclose(fin[TARGET].values, y)
    subsets = {"full": np.ones(len(y), bool), "covered_v4": fin.lx_covered.values == 1, "covered_fixed": covered_mask(g),
               "pin": arch == "pin", "nip": arch == "nip"}
    P = {f.stem[5:]: np.load(f) for f in sorted(d.glob("pred_*.npy"))}
    rep = {}
    for arm, ref in (("M2", "pred_t2_default.npy"), ("M3_v4", "pred_t3_default.npy")):
        if arm in P:
            Q = np.load(ROOT / "results" / "v4" / "final" / ref)
            rep[arm] = dict(max_abs_diff=float(np.abs(P[arm].mean(0) - Q.mean(0)).max()),
                            r2_here=float(r2(y, P[arm].mean(0))), r2_stored=float(r2(y, Q.mean(0))))
    out = []
    for arm, A in P.items():
        for sub, m in subsets.items():
            yy = y[m]; pa = A.mean(0)[m]; pr = P["M2"].mean(0)[m]
            dd = (pr - yy) ** 2 - (pa - yy) ** 2
            bt = paired_boot(dd, g[m], yy.var())
            per_seed = [r2(yy, A[i][m]) - r2(yy, P["M2"][i][m]) for i in range(len(A))]
            out.append(dict(protocol="R_v4final", arm=arm, subset=sub, n=int(m.sum()), ndoi=int(len(set(g[m]))),
                            r2=r2(yy, pa), mae=float(np.abs(pa - yy).mean()), rmse=float(np.sqrt(((pa - yy) ** 2).mean())),
                            dR2=float(dd.mean() / yy.var()), dR2_lo=float(np.quantile(bt, .025)), dR2_hi=float(np.quantile(bt, .975)),
                            p_boot_le0=float((bt <= 0).mean()), dMAE=float(np.abs(pa - yy).mean() - np.abs(pr - yy).mean()),
                            seeds_pos=int(sum(x > 0 for x in per_seed)), per_seed_dR2=" ".join(f"{x:+.4f}" for x in per_seed)))
    return pd.DataFrame(out), rep


def final_H(name):
    d = EXP / "runs" / f"{name}_H"
    out, rep = [], {}
    for archname in ("overall", "pin", "nip"):
        files = sorted(d.glob(f"H_{archname}_*_*.npz"))
        if not files:
            continue
        arms = sorted({f.stem[len(f"H_{archname}_"):].rsplit("_", 1)[0] for f in files})
        Z = {(a, k): np.load(d / f"H_{archname}_{a}_{k}.npz", allow_pickle=True) for a in arms for k in range(1, 6)
             if (d / f"H_{archname}_{a}_{k}.npz").exists()}
        ks = sorted({k for (_, k) in Z})
        st = []
        for k in ks:
            ref = np.load(ROOT / "results" / "ablation_v1" / "preds" / f"doi_{archname}_t2_catboost_{k}.npz")
            st.append(float(np.abs(Z[("M2", k)]["pred"] - ref["pred"]).max()) if "pred" in ref.files else np.nan)
        rep[archname] = dict(M2_vs_ablation_v1_max_abs_diff=st,
                             M2_r2_here=[float(r2(Z[("M2", k)]["y"], Z[("M2", k)]["pred"])) for k in ks])
        alld = sorted(set(np.concatenate([Z[("M2", k)]["doi"] for k in ks])))
        pos = {x: i for i, x in enumerate(alld)}
        idx = np.random.default_rng(0).integers(0, len(alld), size=(B, len(alld)))
        W = np.zeros((B, len(alld))); np.add.at(W, (np.repeat(np.arange(B), len(alld)), idx.ravel()), 1)
        for a in arms:
            if not all((a, k) in Z for k in ks):
                continue
            r, dr, boots = [], [], []
            for k in ks:
                z, zr = Z[(a, k)], Z[("M2", k)]
                yy = z["y"]; r.append(r2(yy, z["pred"])); dr.append(r2(yy, z["pred"]) - r2(yy, zr["pred"]))
                dd = (zr["pred"] - yy) ** 2 - (z["pred"] - yy) ** 2
                gi = np.array([pos[x] for x in z["doi"]])
                S = np.bincount(gi, weights=dd, minlength=len(alld)); N = np.bincount(gi, minlength=len(alld))
                boots.append((W @ S) / np.maximum(W @ N, 1) / yy.var())
            bt = np.mean(boots, axis=0)
            out.append(dict(protocol="H_manuscript_holdouts", arm=a, subset=archname, n=int(np.mean([len(Z[(a, k)]["y"]) for k in ks])),
                            r2=float(np.mean(r)), r2_sd=float(np.std(r, ddof=1)), dR2=float(np.mean(dr)),
                            dR2_lo=float(np.quantile(bt, .025)), dR2_hi=float(np.quantile(bt, .975)), p_boot_le0=float((bt <= 0).mean()),
                            splits_pos=int(sum(x > 0 for x in dr)), per_split_dR2=" ".join(f"{x:+.4f}" for x in dr),
                            per_split_r2=" ".join(f"{x:.4f}" for x in r)))
    return pd.DataFrame(out), rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default="F_final"); a = ap.parse_args()
    pd.set_option("display.width", 250)
    res, reps = [], {}
    if (EXP / "runs" / f"{a.name}_R").exists():
        t, reps["R"] = final_R(a.name); res.append(t)
    if (EXP / "runs" / f"{a.name}_H").exists():
        t, reps["H"] = final_H(a.name); res.append(t)
    res = pd.concat(res, ignore_index=True)
    res.to_csv(EXP / "report" / "final_metrics.csv", index=False)
    (EXP / "report" / "final_reproduction.json").write_text(json.dumps(reps, indent=1))
    print(json.dumps(reps, indent=1))
    cols = [c for c in ["protocol", "arm", "subset", "n", "r2", "r2_sd", "dR2", "dR2_lo", "dR2_hi", "p_boot_le0", "seeds_pos", "splits_pos", "dMAE"] if c in res]
    print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
