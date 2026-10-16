#!/usr/bin/env python3
"""SHAP for final (t3) arch models (self-contained; run from ml_paper/).

Trains the Tier-3 fold-1 CatBoost for {overall|pin|nip} (A_original protocol),
saves model/predictions, then:
  - shap_ranking_{arch}.csv : feature, mean_abs_shap, provenance, group (ALL features)
  - shap_beeswarm_{arch}.png : top-15 beeswarm, literature predictors marked †
  - shap_bar_{arch}.png      : top-25 mean|SHAP| bars
Provenance: literature (LLM JSON) / material (catalogue) / pdb (database).
Groups: literature / material / composition / process / device.

Usage: python3 pipeline/shap_final.py {overall|pin|nip}
"""
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset_json_enriched.csv")
for d in ("models", "predictions", "figures", "results"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

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
FORBIDDEN = ["Ref_ID", "Ref_DOI_number", "final_pce",
             "Perovskite_composition_short_form",
             "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma",
             "Ref_journal", "Ref_publication_date", "publication_year",
             "Stability_measured", "EQE_measured", "JV_hysteresis_index",
             "Cell_area_measured_cm2", "Stabilised_performance_measured",
             "Perovskite", "HTL", "ETL"]
CAT = ["Cell_architecture", "ETL_deposition_method",
       "HTL_deposition_method", "Perovskite_deposition_method"]
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
names = X.columns.tolist()
# coverage assertion: every feature must map to a known group
assert all(group(base(f)) in
           ("literature", "material", "composition", "process", "device")
           for f in names), "group() must cover all features"
y = df["final_pce"].values.astype(float)
doi = df["Ref_DOI_number"].values.astype(str)

from sklearn.model_selection import GroupShuffleSplit
tr, te = next(GroupShuffleSplit(n_splits=5, test_size=0.2,
                                random_state=42).split(X, y, doi))
from catboost import CatBoostRegressor
m = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8,
                      l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.7,
                      min_data_in_leaf=5, random_seed=42, verbose=0,
                      thread_count=-1)
ci = [i for i, c in enumerate(names) if c in CAT]
m.fit(X.iloc[tr], y[tr], cat_features=ci,
      eval_set=(X.iloc[te], y[te]), early_stopping_rounds=100, verbose=False)
m.save_model(os.path.join(ROOT, "models", f"model_{ARCH}.cbm"))
pred = m.predict(X.iloc[te])
np.save(os.path.join(ROOT, "predictions", f"pred_{ARCH}.npy"), pred)
np.save(os.path.join(ROOT, "predictions", f"yte_{ARCH}.npy"), y[te])
print(f"[{ARCH}] trained feats={len(names)} test={len(te)} [{time.time()-t0:.0f}s]",
      flush=True)

import shap
rng = np.random.RandomState(0)
sel = rng.choice(len(te), min(1500, len(te)), replace=False)
Xs = X.iloc[te].reset_index(drop=True).iloc[sel]
ex = shap.TreeExplainer(m)
sh = np.asarray(ex.shap_values(Xs))
mean_abs = np.abs(sh).mean(0)
order_all = np.argsort(-mean_abs)

with open(os.path.join(ROOT, "results", f"shap_ranking_{ARCH}.csv"),
          "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "feature", "mean_abs_shap", "provenance", "group"])
    for r, j in enumerate(order_all, 1):
        b = base(names[j])
        w.writerow([r, names[j], round(float(mean_abs[j]), 6),
                    provenance(b), group(b)])
print(f"[{ARCH}] ranking csv written ({len(names)} features)", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOPB, TOPK = 15, 25
# beeswarm
Xv = np.zeros((len(Xs), len(names)))
for j, c in enumerate(names):
    col = Xs[c]
    Xv[:, j] = (pd.factorize(col.astype(str))[0] if c in CAT
                else pd.to_numeric(col, errors="coerce").fillna(0).values)
fig, ax = plt.subplots(figsize=(10, 8))
oo = order_all[:TOPB][::-1]
rng2 = np.random.default_rng(0)
cmap = plt.get_cmap("coolwarm")
for yi, j in enumerate(oo):
    v = sh[:, j]
    x = Xv[:, j]
    lo, hi = np.nanpercentile(x, [5, 95])
    c = np.full_like(v, 0.5) if hi - lo < 1e-12 else np.clip((x - lo) / (hi - lo), 0, 1)
    yy = yi + rng2.normal(0, 0.12, size=len(v))
    idx = np.argsort(c)
    ax.scatter(v[idx], yy[idx], c=c[idx], cmap=cmap, s=6, alpha=0.7,
               linewidths=0, vmin=0, vmax=1)
labels = [names[j] + (" †" if provenance(base(names[j])) == "literature" else "")
          for j in oo]
ax.set_yticks(range(len(oo)))
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("SHAP value (impact on PCE %)")
ax.set_title(f"{ARCH} SHAP beeswarm — top {TOPB} († = LLM-literature predictor)")
ax.grid(alpha=0.2, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", f"shap_beeswarm_{ARCH}.png"), dpi=140)
plt.close(fig)

# bar
fig, ax = plt.subplots(figsize=(9, 8))
ob = order_all[:TOPK][::-1]
vals = mean_abs[ob]
cols = ["#8e44ad" if provenance(base(names[j])) == "literature" else "#2471a3"
        for j in ob]
ax.barh(range(len(ob)), vals, color=cols)
ax.set_yticks(range(len(ob)))
ax.set_yticklabels([names[j] for j in ob], fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("mean(|SHAP|)")
ax.set_title(f"{ARCH} mean|SHAP| — top {TOPK} (purple = LLM-literature)")
ax.grid(alpha=0.25, axis="x")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", f"shap_bar_{ARCH}.png"), dpi=140)
plt.close(fig)
print(f"[{ARCH}] plots done [{time.time()-t0:.0f}s]", flush=True)
