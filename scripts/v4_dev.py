#!/usr/bin/env python3
"""Development harness for lx4 blocks: fixed t2 matrix + candidate LLM blocks, DOI-grouped k-fold on v4dev only."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET, CAT_ALL
from psc.features import base_frame, TrainFitPreprocessor
from psc.splits import make_splits
from psc import v4
from catboost import CatBoostRegressor

def blocks(df):
    Tt, Tn = v4.paper_table("tok"), v4.paper_table("num")
    B = {"tok": v4.attach(df, Tt), "num": v4.attach(df, Tn), "dev": v4.device_block(df)}
    B["rec"] = v4.attach(df, v4.paper_table("rec"))
    for g in v4.REC_GROUPS:
        B["rec" + g] = B["rec"][v4.rec_group_cols(g)]
    for g in v4.REC_GROUPS:
        B["recno" + g] = B["rec"][v4.rec_group_cols([x for x in v4.REC_GROUPS if x != g])]
    Trec = v4.paper_table("rec")
    for j in range(5):
        rng = np.random.default_rng(100 + j)
        Tp = pd.DataFrame(Trec.values[rng.permutation(len(Trec))], index=Trec.index, columns=Trec.columns)
        B[f"recperm{j}"] = v4.attach(df, Tp)
    B["rec2"] = v4.attach(df, v4.paper_table("rec2"))
    prev = B["rec"].mean()
    B["recM"] = B["rec"][prev[(prev >= 0.03) & (prev <= 0.97) | prev.isna()].index]
    B["recM"] = B["recM"].loc[:, B["recM"].columns.isin(prev[(prev >= 0.03)].index)]
    B["ix"] = interactions(df, B["rec"])
    B["fill"] = v4.fill_block(df, v4.attach(df, Tn))
    return B

def interactions(df, R):
    """device-conditioned recipe flags: a paper-level dopant/additive flag is only informative for devices that use that layer."""
    htl = df["HTL"].fillna("").str.lower(); etl = df["ETL"].fillna("").str.lower(); arch = df["Cell_architecture"].fillna("")
    spiro_like = htl.str.contains("spiro|ptaa|p3ht|poly|tpd|cuscn|nio|cui").astype(float)
    out = pd.DataFrame(index=df.index)
    for f in ("htl_tbp", "htl_litfsi", "htl_fk", "htl_f4tcnq", "htl_cu"):
        out["ix_" + f + "_x_organicHTL"] = R["rc_" + f] * (htl.str.contains("spiro|ptaa|p3ht|tpd|poly").astype(float))
    out["ix_etl_ticl4_x_tio2"] = R["rc_etl_ticl4"] * etl.str.contains("tio2").astype(float)
    out["ix_etl_dopant_x_oxide"] = R["rc_etl_dopant"] * etl.str.contains("tio2|sno2|zno").astype(float)
    out["ix_cl_x_nip"] = R["rc_cl_source"] * (arch == "nip").astype(float)
    out["ix_cl_x_pin"] = R["rc_cl_source"] * (arch == "pin").astype(float)
    out["ix_pea_x_pin"] = R["rc_pea_family"] * (arch == "pin").astype(float)
    out["ix_pea_x_nip"] = R["rc_pea_family"] * (arch == "nip").astype(float)
    out["ix_anti_x_pin"] = R["rc_antisolvent"] * (arch == "pin").astype(float)
    out["ix_vac_x_pin"] = R["rc_vacuum"] * (arch == "pin").astype(float)
    return out


def fit_eval(Z, y, tr, va, te, cat, params):
    p = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.7,
             min_data_in_leaf=5, random_seed=42, loss_function="RMSE"); p.update(params)
    ci = [i for i, c in enumerate(Z.columns) if c in cat]
    pen = p.pop("pen_lit", None)
    if pen:
        p["first_feature_use_penalties"] = [pen if c.startswith(("rc_", "tk_", "nm_", "dv_", "fl_", "ix_")) else 0.0 for c in Z.columns]
        p["penalties_coefficient"] = 1.0
    m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
    m.fit(Z.iloc[tr], y[tr], cat_features=ci, eval_set=(Z.iloc[va], y[va]), early_stopping_rounds=100, verbose=False)
    return m.predict(Z.iloc[te])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", required=True, help="comma list: t2, t2+tok, t2+tok+num ...")
    ap.add_argument("--seeds", default="42,43"); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--scope", default="covered", choices=["covered", "full"])
    ap.add_argument("--out", required=True); ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    df = load_dataset()
    df["v4part"] = df["doi_group"].map(v4.v4_partition)
    df = df[df.v4part == "v4dev"]
    if a.scope == "covered":
        df = df[df.lx_covered == 1]
    df = df.reset_index(drop=True)
    y = df[TARGET].values.astype(float); g = df["doi_group"].values
    X2 = base_frame(df, "t2"); B = blocks(df)
    print(f"n={len(df)} dois={len(set(g))}", flush=True)
    rows = []
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for seed in map(int, a.seeds.split(",")):
        for k, (tr, va, te) in enumerate(make_splits(g, "kfold", a.folds, 0.2, 0.15, seed), 1):
            pre = TrainFitPreprocessor().fit(X2.iloc[tr]); Z2 = pre.transform(X2)
            for tier in a.tiers.split(","):
                parts = tier.split("+"); Z = Z2
                for b in parts[1:]:
                    Z = pd.concat([Z, B[b]], axis=1)
                t0 = time.time()
                pr = fit_eval(Z, y, tr, va, te, set(CAT_ALL), json.loads(a.params))
                e = pr - y[te]
                rows.append(dict(seed=seed, fold=k, tier=tier, n=len(te), ndoi=len(set(g[te])), nfeat=Z.shape[1],
                                 mse=float((e**2).mean()), mae=float(np.abs(e).mean()), var=float(y[te].var()), sec=time.time()-t0))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
                np.save(str(out) + f".{tier.replace('+','_')}_s{seed}_f{k}.npy", np.stack([te, pr]))
    d = pd.DataFrame(rows); d["r2"] = 1 - d.mse / d["var"]
    s = d.groupby("tier").agg(r2=("r2", "mean"), mae=("mae", "mean"), rmse=("mse", lambda x: np.sqrt(x).mean()))
    ref = d[d.tier == "t2"].set_index(["seed", "fold"])
    s["dR2_vs_t2"] = [ (d[d.tier == t].set_index(["seed","fold"]).r2 - ref.r2).mean() for t in s.index]
    s["folds_pos"] = [ int(((d[d.tier == t].set_index(["seed","fold"]).r2 - ref.r2) > 0).sum()) for t in s.index]
    print(s.round(4))
main()
