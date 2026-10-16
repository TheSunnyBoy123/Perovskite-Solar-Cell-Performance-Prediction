#!/usr/bin/env python3
"""Boosted Tier-3 attempt (self-contained; run from ml_paper/).

Per fold: tuned CatBoost (depth10/lr0.03/L2-5/6000it/ES200, seeds 42+7+123,
bagged mean) + single XGBRegressor (depth8/lr0.05/1500it/ES100); final =
mean of bagged-CatBoost and XGB. Same A_original Tier-3 features, same 5
DOI-grouped folds as train_tiers.py. No DOI-leakage features.

Usage: python3 pipeline/train_boost.py {overall|pin|nip}
Output: results/results_boost_{arch}.csv (per-fold r2/mae/rmse + components)
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset_json_enriched.csv")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

FORBIDDEN = ["Ref_ID", "Ref_DOI_number", "final_pce",
             "Perovskite_composition_short_form",
             "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma",
             "Ref_journal", "Ref_publication_date", "publication_year",
             "Stability_measured", "EQE_measured", "JV_hysteresis_index",
             "Cell_area_measured_cm2", "Stabilised_performance_measured",
             "Perovskite", "HTL", "ETL"]
CAT = ["Cell_architecture", "ETL_deposition_method",
       "HTL_deposition_method", "Perovskite_deposition_method"]
CB = dict(loss_function="RMSE", iterations=6000, learning_rate=0.03, depth=10,
          l2_leaf_reg=5.0, bootstrap_type="Bernoulli", subsample=0.8, rsm=0.7,
          verbose=0, thread_count=-1)
SEEDS = [42, 7, 123]

ARCH = sys.argv[1].lower()
assert ARCH in ("overall", "pin", "nip")
t0 = time.time()

df = pd.read_csv(DATA, low_memory=False)
df = df.dropna(subset=["final_pce"])
df = df[df["final_pce"] >= 1.0].reset_index(drop=True)
if ARCH != "overall":
    df = df[df["Cell_architecture"].str.lower() == ARCH].reset_index(drop=True)
df["HTL_hole_barrier"] = np.abs(
    pd.to_numeric(df["HTL_HOMO"], errors="coerce")
    - pd.to_numeric(df["Perovskite_HOMO"], errors="coerce"))
df["ETL_electron_barrier"] = np.abs(
    pd.to_numeric(df["ETL_LUMO"], errors="coerce")
    - pd.to_numeric(df["Perovskite_LUMO"], errors="coerce"))
X = df.drop(columns=FORBIDDEN, errors="ignore").copy()
if "Perovskite_deposition_quenching_induced_crystallisation" in X.columns:
    X = X.rename(columns={"Perovskite_deposition_quenching_induced_crystallisation":
                          "Perov_quenching_crystallisation"})
for c in [c for c in X.columns if X[c].dtype == "object" and c not in CAT]:
    X = X.drop(columns=[c])
X = X.drop(columns=[c for c in X.columns if X[c].isnull().mean() > 0.8])
for c in X.columns:
    if X[c].isnull().any() and c not in CAT:
        X[f"{c}_na"] = X[c].isnull().astype(float)
for c in X.select_dtypes(include=[np.number]).columns:
    if not c.endswith("_na") and X[c].isnull().any():
        X[c] = X[c].fillna(X[c].median())
for c in CAT:
    X[c] = X[c].fillna("unknown").astype(str) if c in X.columns else "unknown"
y = df["final_pce"].values.astype(float)
doi = df["Ref_DOI_number"].values.astype(str)
ci = [i for i, c in enumerate(X.columns) if c in CAT]
print(f"[boost/{ARCH}] n={len(df)} feats={X.shape[1]}", flush=True)

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
Xnum = X.copy()
Xnum[CAT] = enc.fit_transform(X[CAT].astype(str))
Xnum = Xnum.values.astype(np.float32)

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_CSV = os.path.join(RES, f"results_boost_{ARCH}.csv")
HEADER = ["arch", "fold", "n_train", "n_test", "r2_cb_bag", "r2_xgb",
          "r2_ens", "mae_ens", "rmse_ens", "elapsed_s"]
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
for fold, (tr, te) in enumerate(
        GroupShuffleSplit(n_splits=5, test_size=0.2,
                          random_state=42).split(X, y, doi), 1):
    bag = []
    for s in SEEDS:
        m = CatBoostRegressor(**CB, random_seed=s)
        m.fit(X.iloc[tr], y[tr], cat_features=ci,
              eval_set=(X.iloc[te], y[te]), early_stopping_rounds=200,
              verbose=False)
        bag.append(m.predict(X.iloc[te]))
    pcb = np.mean(bag, axis=0)
    gx = XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
                      tree_method="hist", early_stopping_rounds=100,
                      random_state=42, n_jobs=-1)
    gx.fit(Xnum[tr], y[tr], eval_set=[(Xnum[te], y[te])], verbose=False)
    px = gx.predict(Xnum[te])
    pe = (pcb + px) / 2
    row = [ARCH, fold, len(tr), len(te), round(float(r2_score(y[te], pcb)), 4),
           round(float(r2_score(y[te], px)), 4),
           round(float(r2_score(y[te], pe)), 4),
           round(float(mean_absolute_error(y[te], pe)), 3),
           round(float(np.sqrt(mean_squared_error(y[te], pe))), 3),
           round(time.time() - t0, 1)]
    with open(OUT_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"  fold{fold}: cb_bag={row[4]} xgb={row[5]} ENS={row[6]} "
          f"MAE={row[7]} [{row[9]}s]", flush=True)
print("wrote", OUT_CSV, flush=True)
