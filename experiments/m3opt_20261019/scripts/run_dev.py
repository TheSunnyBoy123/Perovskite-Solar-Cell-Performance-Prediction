#!/usr/bin/env python3
"""Development harness: M1 / M2 / M3 candidate arms on identical DOI-grouped folds of the v4dev partition.

Population: load_dataset() (PCE >= 1), v4dev partition only (salt 'psc-v4', 65% of publications). v4final is never
loaded here. Folds: psc.splits.make_splits(groups, 'kfold', 5, seed) -> disjoint DOI-grouped 5-fold; inside each
training part a DOI-disjoint 15% validation set (early stopping only). Every arm of a config sees the same folds,
the same train-fitted M1/M2 preprocessing and the same CatBoost seed(s). Training uses all dev records (covered and
uncovered); metrics are computed later on pooled / covered / p-i-n / n-i-p records (analyze.py).

  python run_dev.py --config ../configs/<name>.json [--smoke]
Outputs: runs/<name>/preds/<arm>_s<seed>_f<fold>.npz (test_idx, pred), runs/<name>/ledger.jsonl, runs/<name>/meta.json
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
from psc.data import load_dataset, TARGET, CAT_ALL, DEFAULT_DATA  # noqa: E402
from psc.features import base_frame, TrainFitPreprocessor  # noqa: E402
from psc.splits import make_splits  # noqa: E402
from psc import v4  # noqa: E402
import m3feat as F  # noqa: E402

CB_DEFAULT = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.7,
                  min_data_in_leaf=5, loss_function="RMSE")
LIT_PREFIX = ("rc_", "rcg_", "kn", "tt_", "sv_", "jp_", "ds_", "ix_", "ctl_")


def dev_frame(partition="v4dev"):
    df = load_dataset()
    df["v4part"] = df["doi_group"].map(v4.v4_partition)
    df = df[df.v4part == partition].reset_index(drop=True)
    F.set_known(set(load_dataset().doi_group))          # resolution against all PDP publications (label-free)
    return df


def build_block(spec, df, y, g, tr, va, ctx):
    """One literature block (record-aligned DataFrame) for this fold."""
    kind = spec["kind"]
    perm = spec.get("perm")                               # DOI-level permutation control seed
    if kind == "flags":
        T = F.flags(spec.get("variant", "fixed"))
        if spec.get("subset"):
            T = T[[c for c in T.columns if c in set(sum((F.GROUPS[s] for s in spec["subset"]), []))]]
        if spec.get("drop"):
            T = T.drop(columns=[c for c in T.columns if c in set(sum((F.GROUPS[s] for s in spec["drop"]), []))])
        if spec.get("only"):
            T = T[spec["only"]]
        if perm is not None:
            T = F.permute_papers(T.dropna(how="all"), perm)
        R = F.attach(g, T).add_prefix("rc_")
        return F.represent(R, spec.get("repr", "nan"), tr)
    if kind == "covflag":
        T = F.flags(spec.get("variant", "fixed")).dropna(how="all")
        return pd.DataFrame({"rc_covered": pd.Series(g).isin(T.index).astype(float).values})
    if kind in ("jprop", "design"):
        T = F.jprop() if kind == "jprop" else F.design()
        if perm is not None:
            T = F.permute_papers(T, perm)
        return F.attach(g, T)
    pool = np.concatenate([tr, va])
    if kind in ("knn", "svd"):
        if "tfidf" not in ctx:
            txt = F.text(spec.get("text", "v1"))
            if perm is not None:
                txt = pd.Series(txt.values[np.random.default_rng(perm).permutation(len(txt))], index=txt.index)
            fit_p = sorted(set(g[pool]) & set(txt.index))
            ctx["tfidf"] = F.tfidf(txt, fit_p)[:2]
            ctx["fit_rows"] = [ctx["tfidf"][1][p] for p in fit_p]
        if kind == "knn":
            return F.knn_te(g, y, pool, ctx["tfidf"], k=spec.get("k", 25), seed=spec.get("seed", 0), prefix=spec.get("prefix", "kn"))
        return F.svd_block(g, ctx["tfidf"], ctx["fit_rows"], k=spec.get("k", 16))
    if kind == "tokte":
        toks = F.tokens()
        if perm is not None:
            keys = sorted(toks); vals = [toks[k] for k in keys]
            idx = np.random.default_rng(perm).permutation(len(keys))
            toks = {k: vals[i] for k, i in zip(keys, idx)}
        return F.tok_te(g, y, pool, toks, m=spec.get("m", 20.0), min_df=spec.get("min_df", 10))
    raise ValueError(kind)


def fit_predict(Z, y, tr, va, te, params, pen, model, seed):
    if model == "catboost":
        from catboost import CatBoostRegressor
        p = {**CB_DEFAULT, **params, "random_seed": seed}
        if pen:
            p["first_feature_use_penalties"] = [pen if c.startswith(LIT_PREFIX) else 0.0 for c in Z.columns]
            p["penalties_coefficient"] = 1.0
        ci = [i for i, c in enumerate(Z.columns) if c in set(CAT_ALL) or Z[c].dtype == object]
        m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
        m.fit(Z.iloc[tr], y[tr], cat_features=ci, eval_set=(Z.iloc[va], y[va]), early_stopping_rounds=100, verbose=False)
        return m.predict(Z.iloc[te]), int(m.get_best_iteration())
    if model == "lgbm":
        import lightgbm as lgb
        Zc = Z.copy()
        for c in Zc.columns:
            if c in set(CAT_ALL) or Zc[c].dtype == object:
                Zc[c] = Zc[c].astype("category")
        p = dict(n_estimators=4000, learning_rate=0.03, num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.7, reg_lambda=3.0, max_cat_to_onehot=8, cat_smooth=20, verbose=-1, random_state=seed, n_jobs=-1)
        p.update(params)
        m = lgb.LGBMRegressor(**p)
        m.fit(Zc.iloc[tr], y[tr], eval_set=[(Zc.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.predict(Zc.iloc[te]), int(m.best_iteration_ or 0)
    raise ValueError(model)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default=None, help="comma list to restrict")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    name = cfg["name"] + ("_smoke" if a.smoke else "")
    out = EXP / "runs" / name; (out / "preds").mkdir(parents=True, exist_ok=True)
    df = dev_frame(cfg.get("partition", "v4dev"))
    y = df[TARGET].values.astype(float); g = df["doi_group"].values
    meta = {"config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "data_sha256": hashlib.sha256(DEFAULT_DATA.read_bytes()).hexdigest(), "git_head": sh(["git", "rev-parse", "HEAD"]),
            "n_records": len(df), "n_dois": int(len(set(g))), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    np.save(out / "y.npy", y)
    df[["doi_group", "Cell_architecture"]].to_csv(out / "rows.csv", index=False)
    X = {t: base_frame(df, t) for t in ("t1", "t2")}
    arms = cfg["arms"] if not a.arms else [x for x in cfg["arms"] if x["name"] in a.arms.split(",")]
    ledger = out / "ledger.jsonl"
    seeds = cfg["seeds"][:1] if a.smoke else cfg["seeds"]
    for seed in seeds:
        for k, (tr, va, te) in enumerate(make_splits(g, "kfold", cfg.get("folds", 5), 0.2, 0.15, seed), 1):
            if a.smoke and k > 1:
                break
            todo = [arm for arm in arms if not (out / "preds" / f"{arm['name']}_s{seed}_f{k}.npz").exists()]
            if not todo:
                continue
            Zb = {}
            for t in ("t1", "t2"):
                pre = TrainFitPreprocessor().fit(X[t].iloc[tr]); Zb[t] = pre.transform(X[t]).reset_index(drop=True)
            ctx_cache = {}
            for arm in todo:
                t0 = time.time()
                parts = [Zb[arm.get("base", "t2")]]
                for spec in arm.get("blocks", []):
                    key = json.dumps({kk: vv for kk, vv in spec.items() if kk in ("perm", "text")}, sort_keys=True)
                    ctx = ctx_cache.setdefault(key, {})
                    parts.append(build_block(spec, df, y, g, tr, va, ctx).reset_index(drop=True))
                Z = pd.concat(parts, axis=1)
                assert Z.columns.is_unique, "duplicate columns"
                params = dict(cfg.get("params", {})); params.update(arm.get("params", {}))
                if a.smoke:
                    params["iterations"] = 200
                preds, iters = [], []
                for b in range(arm.get("bag", cfg.get("bag", 1))):
                    p, it = fit_predict(Z, y, tr, va, te, params, arm.get("pen", cfg.get("pen", 0.5)) if len(parts) > 1 else 0.0,
                                        arm.get("model", "catboost"), seed * 100 + b + arm.get("seed_offset", 0))
                    preds.append(p); iters.append(it)
                p = np.mean(preds, axis=0)
                np.savez_compressed(out / "preds" / f"{arm['name']}_s{seed}_f{k}.npz", test_idx=te, pred=p)
                e = p - y[te]
                rec = dict(arm=arm["name"], seed=seed, fold=k, n_train=len(tr), n_val=len(va), n_test=len(te),
                           ndoi_test=int(len(set(g[te]))), doi_overlap=int(len(set(g[te]) & (set(g[tr]) | set(g[va])))),
                           n_features=int(Z.shape[1]), n_lit=int(sum(c.startswith(F_PREFIX) for c in Z.columns)),
                           best_iter=iters, r2=float(1 - (e ** 2).mean() / y[te].var()), mae=float(np.abs(e).mean()),
                           rmse=float(np.sqrt((e ** 2).mean())), runtime_s=round(time.time() - t0, 1))
                with open(ledger, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{arm['name']:28s} s{seed} f{k} R2={rec['r2']:.4f} nf={rec['n_features']} lit={rec['n_lit']} it={iters} [{rec['runtime_s']}s]", flush=True)


F_PREFIX = LIT_PREFIX
if __name__ == "__main__":
    main()
