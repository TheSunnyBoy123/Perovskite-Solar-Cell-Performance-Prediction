#!/usr/bin/env python3
"""Tiered training for reviewer pipeline (self-contained; run from ml_paper/).

Tiers (drop-only, nothing added):
  t1 = PDB-only            (drop material-49 + literature-32)
  t2 = PDB + material      (drop literature-32)
  t3 = PDB + material + LLM literature (full)

Protocol: A_original CatBoost(depth8/lr0.05/1500it/ES100), 5-fold
GroupShuffleSplit by DOI (test_size 0.2, random_state 42).

Usage: python3 pipeline/train_tiers.py {overall|pin|nip} {t1|t2|t3|all}
Output: results/results_folds_{arch}.csv (one row per fold; includes record
counts, DOI counts, overlap flag, r2/mae/rmse — no JSON).
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

PROPS = ["band_gap", "LUMO", "HOMO", "electron_mobility", "hole_mobility",
         "dielectric_constant", "min_atomic_mass", "max_atomic_mass",
         "avg_atomic_mass", "min_radius", "max_radius", "avg_radius",
         "en_min", "en_max", "en_avg"]
MAT49 = ({f"{L}_{p}" for L in ("ETL", "HTL", "Perovskite") for p in PROPS}
         | {"perovskite_tolerance_factor", "perovskite_octahedral_factor",
            "perovskite_tau_descriptor", "perovskite_cation_radius_ratio"})
LIT32 = {"has_BCP", "has_LiF", "has_PEAI", "has_SAM", "has_C60", "has_PCBM",
         "has_Spiro", "has_PEDOT", "has_NiOx", "has_SnO2", "has_TiO2", "has_ZnO",
         "has_PTAA", "has_fullerene", "has_passivation_layer", "is_flexible_y",
         "is_encapsulated", "has_stability_measurement", "used_antisolvent",
         "used_glovebox", "used_vacuum_evaporation", "json_numeric_values_found",
         "spin_speed_rpm", "perov_concentration", "etl_thickness_nm",
         "htl_thickness_nm", "etl_anneal_temp_C", "htl_anneal_temp_C",
         "perov_anneal_temp_C", "perov_anneal_time_min",
         "perovskite_thickness_nm", "contact_thickness_nm"}
TIERS = {"t1": MAT49 | LIT32, "t2": LIT32, "t3": set()}

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

ARCH, TIERARG = sys.argv[1].lower(), sys.argv[2].lower()
TIERLIST = ["t1", "t2", "t3"] if TIERARG == "all" else [TIERARG]

df0 = pd.read_csv(DATA, low_memory=False)
df0 = df0.dropna(subset=["final_pce"])
df0 = df0[df0["final_pce"] >= 1.0].reset_index(drop=True)
if ARCH != "overall":
    df0 = df0[df0["Cell_architecture"].str.lower() == ARCH].reset_index(drop=True)

OUT_CSV = os.path.join(RES, f"results_folds_{ARCH}.csv")
HEADER = ["tier", "arch", "fold", "n_train", "n_test", "n_doi_train",
          "n_doi_test", "doi_overlap", "r2", "mae", "rmse", "n_feats",
          "elapsed_s"]
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(HEADER)

from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

for tier in TIERLIST:
    t0 = time.time()
    df = df0.drop(columns=[c for c in TIERS[tier] if c in df0.columns]).copy()
    if "HTL_HOMO" in df.columns:
        df["HTL_hole_barrier"] = np.abs(
            pd.to_numeric(df["HTL_HOMO"], errors="coerce")
            - pd.to_numeric(df["Perovskite_HOMO"], errors="coerce"))
    else:
        df["HTL_hole_barrier"] = np.nan
    if "ETL_LUMO" in df.columns:
        df["ETL_electron_barrier"] = np.abs(
            pd.to_numeric(df["ETL_LUMO"], errors="coerce")
            - pd.to_numeric(df["Perovskite_LUMO"], errors="coerce"))
    else:
        df["ETL_electron_barrier"] = np.nan
    X = df.drop(columns=FORBIDDEN, errors="ignore").copy()
    if "Perovskite_deposition_quenching_induced_crystallisation" in X.columns:
        X = X.rename(columns={
            "Perovskite_deposition_quenching_induced_crystallisation":
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
    doi = df["Ref_DOI_number"].values.astype(str)  # str-cast preserved: identical splits
    doi_raw = df["Ref_DOI_number"].values  # NaN-aware counting only
    ci = [i for i, c in enumerate(X.columns) if c in CAT]
    print(f"[{ARCH}/{tier}] n={len(df)} feats={X.shape[1]}", flush=True)
    for fold, (tr, te) in enumerate(
            GroupShuffleSplit(n_splits=5, test_size=0.2,
                              random_state=42).split(X, y, doi), 1):
        m = CatBoostRegressor(**PARAMS)
        m.fit(X.iloc[tr], y[tr], cat_features=ci,
              eval_set=(X.iloc[te], y[te]), early_stopping_rounds=100,
              verbose=False)
        p = m.predict(X.iloc[te])
        # NaN-DOI rows form one GroupShuffleSplit block; count real DOIs only
        rtr, rte = doi_raw[tr], doi_raw[te]
        nodoi_tr = int(pd.isna(rtr).sum())
        nodoi_te = int(pd.isna(rte).sum())
        row = [tier, ARCH, fold, len(tr), len(te),
               int(pd.Series(rtr).nunique()),
               int(pd.Series(rte).nunique()),
               int(len(set(rtr[rtr == rtr]) & set(rte[rte == rte]))),
               round(float(m.score(X.iloc[te], y[te])), 4),
               round(float(mean_absolute_error(y[te], p)), 3),
               round(float(np.sqrt(mean_squared_error(y[te], p))), 3),
               X.shape[1], round(time.time() - t0, 1),
               nodoi_tr, nodoi_te]
        with open(OUT_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  fold{fold}: R2={row[8]} MAE={row[9]} "
              f"dois={row[6]} overlap={row[7]} [{row[12]}s]", flush=True)
print("wrote", OUT_CSV, flush=True)
