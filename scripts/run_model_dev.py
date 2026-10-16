#!/usr/bin/env python3
"""Model-development experiment (separate from the controlled ablation).

Per (arch, tier, DOI-grouped split): select among a predeclared CatBoost grid by
VALIDATION RMSE (identical grid/budget for every tier), then (i) tuned CatBoost,
(ii) 3-seed bag of the selected config, (iii) XGBoost (ES on val), (iv) mean of
bag and XGB. Test partition is used only for the final scores.
Usage: python scripts/run_model_dev.py --config configs/model_dev.json [--smoke]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import CAT, TARGET, load_dataset  # noqa: E402
from psc.evaluate import metrics  # noqa: E402
from psc.features import TrainFitPreprocessor, base_frame  # noqa: E402
from psc.splits import make_splits  # noqa: E402

GRID = [dict(depth=6, l2_leaf_reg=3), dict(depth=8, l2_leaf_reg=3), dict(depth=10, l2_leaf_reg=3),
        dict(depth=8, l2_leaf_reg=10), dict(depth=10, l2_leaf_reg=10), dict(depth=8, l2_leaf_reg=1)]
BASE = dict(learning_rate=0.05, iterations=3000, subsample=0.8, colsample_bylevel=0.7,
            min_data_in_leaf=5, loss_function="RMSE")
SEEDS = [42, 7, 123]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    out = ROOT / "results" / (cfg["name"] + ("_smoke" if a.smoke else "")); out.mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(exist_ok=True)
    runs = out / "runs.jsonl"; done = set()
    if runs.exists():
        done = {(r["arch"], r["tier"], r["split"]) for r in map(json.loads, runs.read_text().splitlines())}
    base = dict(BASE); grid = GRID[:2] if a.smoke else GRID
    if a.smoke: base["iterations"] = 200
    for arch in cfg["archs"]:
        df = load_dataset(arch=arch); y = df[TARGET].values.astype(float); g = df["doi_group"].values
        splits = list(make_splits(g, "doi", 2 if a.smoke else cfg["n_splits"], cfg["test_size"], cfg["val_size"], cfg["seed"]))
        for tier in cfg["tiers"]:
            Xraw = base_frame(df, tier)
            for k, (tr, va, te) in enumerate(splits, 1):
                if (arch, tier, k) in done: continue
                t0 = time.time()
                pre = TrainFitPreprocessor().fit(Xraw.iloc[tr])
                Xtr, Xva, Xte = (pre.transform(Xraw.iloc[i]) for i in (tr, va, te))
                ci = [i for i, c in enumerate(Xtr.columns) if c in CAT]
                best = None
                for gp in grid:
                    m = CatBoostRegressor(**base, **gp, random_seed=42, verbose=0, thread_count=-1)
                    m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=150, verbose=False)
                    rm = float(np.sqrt(np.mean((m.predict(Xva) - y[va]) ** 2)))
                    if best is None or rm < best[0]: best = (rm, gp, m)
                _, gp, m0 = best
                p_cb = m0.predict(Xte); bag = [p_cb]
                for s in SEEDS[1:]:
                    m = CatBoostRegressor(**base, **gp, random_seed=s, verbose=0, thread_count=-1)
                    m.fit(Xtr, y[tr], cat_features=ci, eval_set=(Xva, y[va]), early_stopping_rounds=150, verbose=False)
                    bag.append(m.predict(Xte))
                p_bag = np.mean(bag, axis=0)
                cats = {c: {v: i for i, v in enumerate(sorted(Xtr[c].unique()))} for c in CAT}
                def enc(X):
                    Z = X.copy()
                    for c in CAT: Z[c] = Z[c].map(cats[c]).fillna(-1)
                    return Z.values.astype(np.float32)
                gx = XGBRegressor(n_estimators=1500 if not a.smoke else 100, max_depth=8, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.7, reg_lambda=3.0, tree_method="hist",
                                  early_stopping_rounds=100, random_state=42, n_jobs=-1)
                gx.fit(enc(Xtr), y[tr], eval_set=[(enc(Xva), y[va])], verbose=False)
                p_x = gx.predict(enc(Xte)); p_ens = (p_bag + p_x) / 2
                preds = {"cb_tuned": p_cb, "cb_bag": p_bag, "xgb": p_x, "ens": p_ens}
                rec = dict(arch=arch, tier=tier, split=k, n_train=len(tr), n_val=len(va), n_test=len(te),
                           selected=gp, n_features=Xtr.shape[1], runtime_s=round(time.time() - t0, 1),
                           doi_overlap=len(set(g[te]) & (set(g[tr]) | set(g[va]))),
                           **{f"{n}_{mk}": v for n, p in preds.items() for mk, v in metrics(y[te], p).items()})
                np.savez_compressed(out / "preds" / f"{arch}_{tier}_{k}.npz", test_idx=te, y=y[te], **preds)
                with open(runs, "a") as f: f.write(json.dumps(rec) + "\n")
                print(arch, tier, k, gp, {n: round(rec[f"{n}_r2"], 4) for n in preds}, f"[{rec['runtime_s']}s]", flush=True)


if __name__ == "__main__":
    main()
