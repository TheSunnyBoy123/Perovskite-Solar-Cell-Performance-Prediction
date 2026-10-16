#!/usr/bin/env python3
"""Two-stage residual design for sparse, paper-level literature features.

Stage 1: matched t2 CatBoost (train / DOI-disjoint-val early stopping) -> identical to the ablation's t2 model.
Stage 2: small regularised CatBoost on the literature block ONLY, fit to DOI-grouped inner out-of-fold stage-1
         residuals of the training publications (no in-sample residuals); early-stopped on validation residuals.
Final  : p = p1 + lambda * r_hat, lambda in {0, .25, .5, .75, 1} chosen on the DOI-disjoint validation set only
         (lambda=0 reproduces stage 1). Test data is never used for any choice.
Every stage-2 block has the same DOI-level permutation / Gaussian controls (tiers *_perm_s<k>, *_noise_s<k>).
Outputs follow run_experiment.py (preds/kfold_<arch>_<tier>_catboost_<k>.npz) so scripts/lx_compare.py applies.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from psc.data import TARGET, LX_TIERS, load_dataset
from psc.evaluate import metrics
from psc.features import TrainFitPreprocessor, base_frame, _doi_level_control, _lx_cols
from psc.models import CATBOOST_MATCHED, ES_ROUNDS
from psc.data import CAT
from psc.splits import make_splits

LAMBDAS = [0, .25, .5, .75, 1.0]
BLOCKS = {"chem": ("chem",), "x": ("proc", "chem", "ctx"), "all": ("proc", "chem", "ctx", "char", "meta")}


def cb(Xtr, ytr, Xva, yva, params, iters=None):
    ci = [i for i, c in enumerate(Xtr.columns) if c in CAT]
    p = {**CATBOOST_MATCHED, **params}
    if iters: p["iterations"] = iters
    m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
    if iters: m.fit(Xtr, ytr, cat_features=ci, verbose=False)
    else: m.fit(Xtr, ytr, cat_features=ci, eval_set=(Xva, yva), early_stopping_rounds=ES_ROUNDS, verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text())
    out = ROOT / "results" / (cfg["name"] + ("_smoke" if a.smoke else "")); (out / "preds").mkdir(parents=True, exist_ok=True)
    runs = out / "runs.jsonl"; done = {json.loads(l)["tier"] + str(json.loads(l)["split"]) for l in runs.read_text().splitlines()} if runs.exists() else set()
    (out / "meta.json").write_text(json.dumps({"config": cfg}, indent=1))
    df = load_dataset(ROOT / cfg["data"], cfg["arch"], subset=cfg.get("subset"), partition=cfg.get("partition"))
    y = df[TARGET].values.astype(float); g = df["doi_group"].values
    X1 = base_frame(df, "t2")
    s2_params = dict(depth=4, learning_rate=0.03, l2_leaf_reg=10.0, subsample=0.7, colsample_bylevel=0.5,
                     min_data_in_leaf=20, iterations=600)
    s2_params.update(cfg.get("stage2", {}))
    blocks = {}
    for b in cfg["blocks"]:
        cols = _lx_cols(df, BLOCKS[b]); raw = df[cols].copy()
        blocks[f"stack_{b}"] = raw
        for kind in ("perm", "noise"):
            for sd in cfg["control_seeds"]:
                c = _doi_level_control(df, "lxperm" if kind == "perm" else "lxnoise", sd, cols)
                blocks[f"stack_{b}_{kind}_s{sd}"] = c
    for k, (tr, va, te) in enumerate(make_splits(g, cfg["protocol"], cfg["n_splits"], cfg["test_size"], cfg["val_size"], cfg["seed"]), 1):
        if k > 2 and a.smoke: break
        t0 = time.time()
        pre = TrainFitPreprocessor().fit(X1.iloc[tr]); A, V, T = (pre.transform(X1.iloc[i]) for i in (tr, va, te))
        m1 = cb(A, y[tr], V, y[va], {}); p1v, p1t = m1.predict(V), m1.predict(T); bi = max(int(m1.get_best_iteration()), 50)
        # inner OOF residuals on training publications (grouped by DOI), fixed iterations = stage-1 best
        oof = np.zeros(len(tr))
        for itr, ite in GroupKFold(4).split(A, groups=g[tr]):
            mi = cb(A.iloc[itr], y[tr][itr], None, None, {}, iters=bi); oof[ite] = mi.predict(A.iloc[ite])
        rtr = y[tr] - oof; rva = y[va] - p1v
        for tier, raw in [("stack_base", None)] + list(blocks.items()):
            key = tier + str(k)
            if key in done: continue
            if raw is None:
                pred, lam, nz = p1t, 0.0, 0
            else:
                R = raw.iloc[tr]; keep = [c for c in raw.columns if R[c].notna().mean() >= 0.03]
                P = lambda i: raw.iloc[i][keep].fillna(-999.0).reset_index(drop=True)  # constant fill: no learned statistics
                m2 = CatBoostRegressor(**{**s2_params, "random_seed": 42}, loss_function="RMSE", verbose=0, thread_count=-1)
                if cfg.get("stage2_mode") == "paper":   # paper-level features -> fit per-publication mean residuals, capped weights
                    cap = cfg.get("weight_cap", 20)
                    def agg(idx, r):
                        d = P(idx).assign(_g=g[idx], _r=r); gg = d.groupby("_g")
                        return gg[keep].first(), gg["_r"].mean(), np.minimum(gg.size(), cap).astype(float)
                    Xg, rg, wg = agg(tr, rtr); Xv, rv_, wv = agg(va, rva)
                    m2.fit(Xg, rg.values, sample_weight=wg.values, eval_set=(Xv, rv_.values), early_stopping_rounds=ES_ROUNDS, verbose=False)
                else:
                    m2.fit(P(tr), rtr, eval_set=(P(va), rva), early_stopping_rounds=ES_ROUNDS, verbose=False)
                rv, rt = m2.predict(P(va)), m2.predict(P(te))
                lam = min(LAMBDAS, key=lambda l: float(np.mean((y[va] - p1v - l * rv) ** 2)))
                pred, nz = p1t + lam * rt, int(m2.get_best_iteration())
            mt = metrics(y[te], pred)
            rec = dict(protocol=cfg["protocol"], arch=cfg["arch"], tier=tier, model="catboost", split=k, lam=lam, s2_iters=nz,
                       n_train=len(tr), n_val=len(va), n_test=len(te), doi_overlap_test_train=len(set(g[te]) & set(g[tr])), **mt)
            np.savez_compressed(out / "preds" / f"{cfg['protocol']}_{cfg['arch']}_{tier}_catboost_{k}.npz",
                                test_idx=te, y=y[te], pred=pred, train_idx=tr, val_idx=va)
            with open(runs, "a") as f: f.write(json.dumps(rec) + "\n")
            print(f"s{k} {tier:28s} R2={mt['r2']:.4f} lam={lam} s2it={nz}", flush=True)
        print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
