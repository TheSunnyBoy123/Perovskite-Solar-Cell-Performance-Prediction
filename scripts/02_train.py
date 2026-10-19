import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA    = Path(__file__).parents[1] / "data"
RESULTS = Path(__file__).parents[1] / "results"

TARGET  = "final_pce"
NO_DOI  = "__NO_DOI__"

FORBIDDEN = {
    "Ref_ID", "Ref_DOI_number", TARGET, "Perovskite_composition_short_form", "doi_group",
    "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma", "Ref_journal",
    "Ref_publication_date", "publication_year", "Stability_measured", "EQE_measured",
    "JV_hysteresis_index", "Cell_area_measured_cm2", "Stabilised_performance_measured",
    "pub_date", "author_group", "partition", "lx_covered", "qty_covered",
}
CATEGORICALS = [
    "Cell_architecture", "ETL_deposition_method",
    "HTL_deposition_method", "Perovskite_deposition_method",
]

MAT_COLS = {
    f"{layer}_{p}"
    for layer in ("ETL", "HTL", "Perovskite")
    for p in ("band_gap", "LUMO", "HOMO", "electron_mobility", "hole_mobility",
              "dielectric_constant", "min_atomic_mass", "max_atomic_mass",
              "avg_atomic_mass", "min_radius", "max_radius", "avg_radius",
              "en_min", "en_max", "en_avg")
} | {"perovskite_tolerance_factor", "perovskite_octahedral_factor",
     "perovskite_tau_descriptor", "perovskite_cation_radius_ratio"}

REC_PREFIX = "rc_"


def feature_frame(df, tier):
    drop = set()
    if tier == "t1":
        drop = MAT_COLS | {c for c in df.columns if c.startswith(REC_PREFIX)}
    elif tier == "t2":
        drop = {c for c in df.columns if c.startswith(REC_PREFIX)}
    cols = [c for c in df.columns
            if c not in FORBIDDEN and c not in drop
            and (pd.api.types.is_numeric_dtype(df[c]) or c in CATEGORICALS)]
    return df[cols]


class Preprocessor:
    def fit(self, X):
        self.keep_ = [c for c in X.columns
                      if c in CATEGORICALS or X[c].isnull().mean() <= 0.8]
        num = [c for c in self.keep_ if c not in CATEGORICALS]
        self.na_cols_ = [c for c in num if X[c].isnull().any()]
        self.median_  = {c: float(X[c].median()) for c in num}
        return self

    def transform(self, X):
        Z = X[self.keep_].copy()
        flags = pd.DataFrame({f"{c}_na": Z[c].isnull().astype(float)
                               for c in self.na_cols_}, index=Z.index)
        for c in self.na_cols_:
            Z[c] = Z[c].fillna(self.median_[c])
        Z = pd.concat([Z, flags], axis=1)
        for c in CATEGORICALS:
            if c in Z:
                Z[c] = Z[c].fillna("unknown").astype(str)
        return Z


def fit_catboost(Xtr, ytr, Xva, yva, iterations, seed=42):
    from catboost import CatBoostRegressor
    ci = [i for i, c in enumerate(Xtr.columns) if c in CATEGORICALS]
    m = CatBoostRegressor(
        iterations=iterations, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
        subsample=0.8, colsample_bylevel=0.7, min_data_in_leaf=5,
        random_seed=seed, loss_function="RMSE", verbose=0,
    )
    m.fit(Xtr, ytr, cat_features=ci,
          eval_set=(Xva, yva), early_stopping_rounds=100)
    return m


def run(tier="t2", n_splits=5, iterations=1500):
    df = pd.read_parquet(DATA / "prepared.parquet")
    y  = df[TARGET].values.astype(float)
    groups = df["doi_group"].values

    Xraw = feature_frame(df, tier)
    print(f"Tier {tier}: {Xraw.shape[1]} features, {len(df):,} records")

    RESULTS.mkdir(exist_ok=True)
    runs_file = RESULTS / f"runs_{tier}.jsonl"
    records = []

    outer = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    for k, (tr_va, te) in enumerate(outer.split(np.arange(len(df)), groups=groups), 1):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42 + k)
        tr, va = next(inner.split(tr_va, groups=groups[tr_va]))
        tr, va = tr_va[tr], tr_va[va]

        pre = Preprocessor().fit(Xraw.iloc[tr])
        Xtr = pre.transform(Xraw.iloc[tr])
        Xva = pre.transform(Xraw.iloc[va])
        Xte = pre.transform(Xraw.iloc[te])

        t0    = time.time()
        model = fit_catboost(Xtr, y[tr], Xva, y[va], iterations)
        preds = model.predict(Xte)

        rec = dict(split=k, tier=tier, n_train=len(tr), n_val=len(va), n_test=len(te),
                   rmse=float(np.sqrt(mean_squared_error(y[te], preds))),
                   mae=float(mean_absolute_error(y[te], preds)),
                   r2=float(r2_score(y[te], preds)),
                   best_iter=int(model.get_best_iteration()),
                   elapsed_s=round(time.time() - t0, 1))
        records.append(rec)
        print(f"  split {k}/{n_splits}  RMSE={rec['rmse']:.3f}  MAE={rec['mae']:.3f}"
              f"  R²={rec['r2']:.3f}  iter={rec['best_iter']}  ({rec['elapsed_s']:.0f}s)")

    with runs_file.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nMean RMSE ({tier}): {np.mean([r['rmse'] for r in records]):.3f}")
    print(f"Results → {runs_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="t2", choices=["t1", "t2", "t3"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(tier=a.tier, n_splits=2 if a.smoke else 5, iterations=300 if a.smoke else 1500)
