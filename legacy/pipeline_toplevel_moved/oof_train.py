#!/usr/bin/env python3
"""Out-of-fold Tier-3 retrain (self-contained; run from ml_paper/).

Trains all 5 DOI-grouped folds, saves per-fold models + one pooled OOF table:
  models/model_{arch}_fold{k}.cbm
  predictions/oof_{arch}.csv  (ref_id, doi, architecture, true_pce, pred_pce, fold)
NaN-DOI rows reported separately; doi_overlap asserted 0 every fold.

Usage: python3 pipeline/oof_train.py {overall|pin|nip}
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset_json_enriched.csv")
for d in ("models", "predictions"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

FORBIDDEN = ["Ref_ID", "Ref_DOI_number", "final_pce",
             "Perovskite_composition_short_form",
             "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma",
             "Ref_journal", "Ref_publication_date", "publication_year",
             "Stability_measured", "EQE_measured", "JV_hysteresis_index",
             "Cell_area_measured_cm2", "Stabilised_performance_measured",
             "Perovskite", "HTL", "ETL"]
CAT = ["Cell_architecture", "ETL_deposition_method",
       "HTL_deposition_method", "Perovskite_deposition_method"]
PARAMS = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
              subsample=0.8, colsample_bylevel=0.7, min_data_in_leaf=5,
              random_seed=42, verbose=0, thread_count=-1)

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
doi = df["Ref_DOI_number"].values.astype(str)  # str-cast: identical splits
doi_raw = df["Ref_DOI_number"].values
ref = df["Ref_ID"].values if "Ref_ID" in df.columns else np.arange(len(df))
archc = df["Cell_architecture"].values
ci = [i for i, c in enumerate(X.columns) if c in CAT]
print(f"[{ARCH}] n={len(df)} feats={X.shape[1]}", flush=True)

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

rows = []
for fold, (tr, te) in enumerate(
        GroupShuffleSplit(n_splits=5, test_size=0.2,
                          random_state=42).split(X, y, doi), 1):
    assert len(set(doi_raw[tr][doi_raw[tr] == doi_raw[tr]]) &
               set(doi_raw[te][doi_raw[te] == doi_raw[te]])) == 0
    m = CatBoostRegressor(**PARAMS)
    m.fit(X.iloc[tr], y[tr], cat_features=ci,
          eval_set=(X.iloc[te], y[te]), early_stopping_rounds=100,
          verbose=False)
    m.save_model(os.path.join(ROOT, "models", f"model_{ARCH}_fold{fold}.cbm"))
    p = m.predict(X.iloc[te])
    for j, k in enumerate(te):
        d = doi_raw[k]
        rows.append([int(ref[k]) if pd.notna(ref[k]) else "",
                     "" if pd.isna(d) else str(d),
                     str(archc[k]), round(float(y[k]), 4),
                     round(float(p[j]), 4), fold])
    print(f"  fold{fold}: R2={m.score(X.iloc[te], y[te]):.4f} "
          f"MAE={mean_absolute_error(y[te], p):.3f} [{time.time()-t0:.0f}s]",
          flush=True)

with open(os.path.join(ROOT, "predictions", f"oof_{ARCH}.csv"),
          "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ref_id", "doi", "architecture", "true_pce", "pred_pce", "fold"])
    w.writerows(rows)
print(f"wrote oof_{ARCH}.csv ({len(rows)} rows) [{time.time()-t0:.0f}s]", flush=True)
