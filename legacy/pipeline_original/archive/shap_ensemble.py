#!/usr/bin/env python3
"""Ensemble SHAP for final boosted t3 (self-contained; run from ml_paper/).

Per arch: fold-1 tuned CatBoost x3 seeds + fold-1 XGB (params identical to
train_boost.py), same Tier-3 features / DOI fold-1 / 1500-sample.
Ensemble SHAP = (mean CB-bag SHAP + XGB SHAP)/2 — exact SHAP of the reported
(r2_ens) predictor. XGB retrain is asserted bit-close to the recorded fold-1
r2_xgb from results_boost_{arch}.csv (determinism check).

Overwrites results/shap_ranking_boost_{arch}.csv with ensemble values and
prints the group table for make_figures.py GROUP_TABLE.
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset_json_enriched.csv")

PROPS = ["band_gap", "LUMO", "HOMO", "electron_mobility", "hole_mobility",
         "dielectric_constant", "min_atomic_mass", "max_atomic_mass",
         "avg_atomic_mass", "min_radius", "max_radius", "avg_radius",
         "en_min", "en_max", "en_avg"]
MAT = ({f"{L}_{p}" for L in ("ETL", "HTL", "Perovskite") for p in PROPS}
       | {"perovskite_tolerance_factor", "perovskite_octahedral_factor",
          "perovskite_tau_descriptor", "perovskite_cation_radius_ratio",
          "HTL_hole_barrier", "ETL_electron_barrier"})
LIT = {"has_BCP", "has_LiF", "has_PEAI", "has_SAM", "has_C60", "has_PCBM",
       "has_Spiro", "has_PEDOT", "has_NiOx", "has_SnO2", "has_TiO2", "has_ZnO",
       "has_PTAA", "has_fullerene", "has_passivation_layer", "is_flexible_y",
       "is_encapsulated", "has_stability_measurement", "used_antisolvent",
       "used_glovebox", "used_vacuum_evaporation", "json_numeric_values_found",
       "spin_speed_rpm", "perov_concentration", "etl_thickness_nm",
       "htl_thickness_nm", "etl_anneal_temp_C", "htl_anneal_temp_C",
       "perov_anneal_temp_C", "perov_anneal_time_min",
       "perovskite_thickness_nm", "contact_thickness_nm"}
COMP_PREFIX = ("a_", "b_", "halide_", "num_")
COMP_EXTRA = {"c_coeff_sum", "total_ion_count", "is_inorganic", "is_leadfree",
              "is_3D", "is_2D", "is_2D3D_mixture", "a_organic_fraction"}
DEVICE = {"Cell_architecture", "Add_lay_front", "Encapsulation",
          "is_flexible_x", "is_semitransparent", "Perovskite_band_gap_measured",
          "Perovskite_pl_max_nm"}


def base(f):
    return f[:-3] if f.endswith("_na") else f


def provenance(b):
    return "literature" if b in LIT else ("material" if b in MAT else "pdb")


def group(b):
    if b in LIT:
        return "literature"
    if b in MAT:
        return "material"
    if b.startswith(COMP_PREFIX) or b in COMP_EXTRA or \
            b.startswith("Perovskite_composition_"):
        return "composition"
    if b in DEVICE:
        return "device"
    return "process"


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
RECORDED_XGB = {"overall": 0.3987, "pin": 0.3866, "nip": 0.4198}

t0 = time.time()
from sklearn.metrics import r2_score
for ARCH in ["overall", "pin", "nip"]:
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
    names = X.columns.tolist()
    y = df["final_pce"].values.astype(float)
    doi = df["Ref_DOI_number"].values.astype(str)
    ci = [i for i, c in enumerate(names) if c in CAT]

    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=5, test_size=0.2,
                                    random_state=42).split(X, y, doi))
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xnum = X.copy()
    Xnum[CAT] = enc.fit_transform(X[CAT].astype(str))
    Xnum = Xnum.values.astype(np.float32)
    from xgboost import XGBRegressor
    gx = XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
                      tree_method="hist", early_stopping_rounds=100,
                      random_state=42, n_jobs=-1)
    gx.fit(Xnum[tr], y[tr], eval_set=[(Xnum[te], y[te])], verbose=False)
    r2x = r2_score(y[te], gx.predict(Xnum[te]))
    assert abs(r2x - RECORDED_XGB[ARCH]) < 2e-3, (ARCH, r2x)  # determinism check

    import shap
    rng = np.random.RandomState(0)
    sel = rng.choice(len(te), min(1500, len(te)), replace=False)
    Xs = X.iloc[te].reset_index(drop=True).iloc[sel]
    Xs_num = pd.DataFrame(Xnum[te][sel], columns=names)
    sh_x = np.asarray(shap.TreeExplainer(gx).shap_values(Xs_num))
    sh_sum, n = None, 0
    for s in (42, 7, 123):
        cb = CatBoostRegressor(**CB, random_seed=s)
        cb.fit(X.iloc[tr], y[tr], cat_features=ci,
               eval_set=(X.iloc[te], y[te]), early_stopping_rounds=200,
               verbose=False)
        sh_s = np.asarray(shap.TreeExplainer(cb).shap_values(Xs))
        sh_sum = sh_s if sh_sum is None else sh_sum + sh_s
        n += 1
    sh = (sh_sum / n + sh_x) / 2  # exact SHAP of reported ensemble predictor
    mean_abs = np.abs(sh).mean(0)
    order_all = np.argsort(-mean_abs)
    with open(os.path.join(ROOT, "results", f"shap_ranking_boost_{ARCH}.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "mean_abs_shap", "provenance", "group"])
        for r, j in enumerate(order_all, 1):
            b = base(names[j])
            w.writerow([r, names[j], round(float(mean_abs[j]), 6),
                        provenance(b), group(b)])
    from collections import defaultdict
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for j in order_all:
        g = group(base(names[j]))
        tot[g] += float(mean_abs[j])
        cnt[g] += 1
    print(f"[{ARCH}] XGB-R2={r2x:.4f} (recorded {RECORDED_XGB[ARCH]}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    for g in ["process", "material", "composition", "device", "literature"]:
        print(f"  {g}: total={tot[g]:.4f} n={cnt[g]}", flush=True)
print("done", flush=True)
