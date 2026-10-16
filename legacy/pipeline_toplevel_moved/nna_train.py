#!/usr/bin/env python3
"""Missingness-indicator ablation (self-contained; run from ml_paper/).

Tier-3 features MINUS every *_na missingness indicator (base features kept,
same A_original protocol otherwise). Isolates the indicators' contribution.

Usage: python3 pipeline/nna_train.py {overall|pin|nip}
Output: results/results_nona_{arch}.csv (same columns as results_folds_*)
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
for c in X.select_dtypes(include=[np.number]).columns:
    if X[c].isnull().any():
        X[c] = X[c].fillna(X[c].median())
# THE ablation: no _na missingness indicators are ever created (base numeric
# columns are median-filled directly above), so X holds base features only.
for c in CAT:
    X[c] = X[c].fillna("unknown").astype(str) if c in X.columns else "unknown"
print(f"[{ARCH}/no-na-flags] n={len(df)} feats={X.shape[1]} "
      f"(base features only; no _na indicators created)", flush=True)
y = df["final_pce"].values.astype(float)
doi = df["Ref_DOI_number"].values.astype(str)
doi_raw = df["Ref_DOI_number"].values
ci = [i for i, c in enumerate(X.columns) if c in CAT]

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT_CSV = os.path.join(RES, f"results_nona_{ARCH}.csv")
HEADER = ["variant", "arch", "fold", "n_train", "n_test", "n_doi_train",
          "n_doi_test", "doi_overlap", "r2", "mae", "rmse", "n_feats",
          "elapsed_s", "n_no_doi_train", "n_no_doi_test"]
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)
for fold, (tr, te) in enumerate(
        GroupShuffleSplit(n_splits=5, test_size=0.2,
                          random_state=42).split(X, y, doi), 1):
    m = CatBoostRegressor(**PARAMS)
    m.fit(X.iloc[tr], y[tr], cat_features=ci,
          eval_set=(X.iloc[te], y[te]), early_stopping_rounds=100,
          verbose=False)
    p = m.predict(X.iloc[te])
    rtr, rte = doi_raw[tr], doi_raw[te]
    row = ["no-na-flags", ARCH, fold, len(tr), len(te),
           int(pd.Series(rtr).nunique()), int(pd.Series(rte).nunique()),
           int(len(set(rtr[rtr == rtr]) & set(rte[rte == rte]))),
           round(float(m.score(X.iloc[te], y[te])), 4),
           round(float(mean_absolute_error(y[te], p)), 3),
           round(float(np.sqrt(mean_squared_error(y[te], p))), 3),
           X.shape[1], round(time.time() - t0, 1),
           int(pd.isna(rtr).sum()), int(pd.isna(rte).sum())]
    with open(OUT_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"  fold{fold}: R2={row[8]} MAE={row[9]} [{row[12]}s]", flush=True)
print("wrote", OUT_CSV, flush=True)
