#!/usr/bin/env python3
"""High-res SHAP beeswarm composite (self-contained; run from ml_paper/).

Reuses trained Tier-3 fold-1 models (models/model_{arch}.cbm); recomputes
SHAP on 1500 DOI-held-out rows per arch. One 3-row composite (shared x)
at dpi 200 + leaves individual shap_beeswarm_{arch}.png untouched.

Usage: python3 pipeline/shap_composite.py
Output: figures/shap_beeswarm_composite.png
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset_json_enriched.csv")

FORBIDDEN = ["Ref_ID", "Ref_DOI_number", "final_pce",
             "Perovskite_composition_short_form",
             "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma",
             "Ref_journal", "Ref_publication_date", "publication_year",
             "Stability_measured", "EQE_measured", "JV_hysteresis_index",
             "Cell_area_measured_cm2", "Stabilised_performance_measured",
             "Perovskite", "HTL", "ETL"]
CAT = ["Cell_architecture", "ETL_deposition_method",
       "HTL_deposition_method", "Perovskite_deposition_method"]

LIT = {"has_BCP", "has_LiF", "has_PEAI", "has_SAM", "has_C60", "has_PCBM",
       "has_Spiro", "has_PEDOT", "has_NiOx", "has_SnO2", "has_TiO2", "has_ZnO",
       "has_PTAA", "has_fullerene", "has_passivation_layer", "is_flexible_y",
       "is_encapsulated", "has_stability_measurement", "used_antisolvent",
       "used_glovebox", "used_vacuum_evaporation", "json_numeric_values_found",
       "spin_speed_rpm", "perov_concentration", "etl_thickness_nm",
       "htl_thickness_nm", "etl_anneal_temp_C", "htl_anneal_temp_C",
       "perov_anneal_temp_C", "perov_anneal_time_min",
       "perovskite_thickness_nm", "contact_thickness_nm"}


def base(f):
    return f[:-3] if f.endswith("_na") else f


def build(arch):
    df = pd.read_csv(DATA, low_memory=False)
    df = df.dropna(subset=["final_pce"])
    df = df[df["final_pce"] >= 1.0].reset_index(drop=True)
    if arch != "overall":
        df = df[df["Cell_architecture"].str.lower() == arch].reset_index(drop=True)
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
    return X


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit

t0 = time.time()
fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
for ax, arch in zip(axes, ["overall", "pin", "nip"]):
    X = build(arch)
    names = X.columns.tolist()
    y = pd.read_csv(DATA, low_memory=False)
    y = y.dropna(subset=["final_pce"])
    y = y[y["final_pce"] >= 1.0].reset_index(drop=True)
    if arch != "overall":
        y = y[y["Cell_architecture"].str.lower() == arch].reset_index(drop=True)
    yv = y["final_pce"].values.astype(float)
    doi = y["Ref_DOI_number"].values.astype(str)
    tr, te = next(GroupShuffleSplit(n_splits=5, test_size=0.2,
                                    random_state=42).split(X, yv, doi))
    m = CatBoostRegressor()
    m.load_model(os.path.join(ROOT, "models", f"model_{arch}.cbm"))
    Xs = X.iloc[te].reset_index(drop=True)
    sel = np.random.RandomState(0).choice(len(Xs), min(1500, len(Xs)),
                                         replace=False)
    Xs = Xs.iloc[sel]
    sh = np.asarray(shap.TreeExplainer(m).shap_values(Xs))
    mean_abs = np.abs(sh).mean(0)
    oo = np.argsort(-mean_abs)[:15][::-1]
    Xv = np.zeros((len(Xs), len(names)))
    for j, c in enumerate(names):
        col = Xs[c]
        Xv[:, j] = (pd.factorize(col.astype(str))[0] if c in CAT
                    else pd.to_numeric(col, errors="coerce").fillna(0).values)
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for yi, j in enumerate(oo):
        v = sh[:, j]
        x = Xv[:, j]
        lo, hi = np.nanpercentile(x, [5, 95])
        c = np.full_like(v, 0.5) if hi - lo < 1e-12 else np.clip((x - lo) / (hi - lo), 0, 1)
        yy = yi + rng.normal(0, 0.12, size=len(v))
        idx = np.argsort(c)
        ax.scatter(v[idx], yy[idx], c=c[idx], cmap=cmap, s=7, alpha=0.7,
                   linewidths=0, vmin=0, vmax=1)
    ax.set_yticks(range(len(oo)))
    ax.set_yticklabels(
        [names[j] + (" †" if base(names[j]) in LIT else "") for j in oo],
        fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{arch} — top 15 by mean|SHAP| († = LLM-literature)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.grid(alpha=0.2, axis="x")
    print(f"[{arch}] shap done [{time.time()-t0:.0f}s]", flush=True)

axes[-1].set_xlabel("SHAP value (impact on PCE %)", fontsize=10)
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("coolwarm"),
                           norm=plt.Normalize(0, 1))
sm.set_array([])
fig.tight_layout(rect=[0, 0, 0.93, 0.97])
cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Low", "High"])
cbar.set_label("Feature value", fontsize=9)
fig.suptitle("SHAP summary — Tier-3 CatBoost, DOI fold-1 held-out "
             "(red = high feature value)", fontsize=12,
             fontweight="bold")
path = os.path.join(ROOT, "figures", "shap_beeswarm_composite.png")
fig.savefig(path, dpi=200)
plt.close(fig)
print("saved", path)
