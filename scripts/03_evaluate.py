"""
Step 3 — Summarise cross-validation results and run SHAP analysis.

Reads results/runs_<tier>.jsonl files produced by 02_train.py,
prints a summary table, and (optionally) plots SHAP feature importances
for the final split of a given tier.

Usage:
    python scripts/03_evaluate.py
    python scripts/03_evaluate.py --shap --tier t2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parents[1] / "results"
DATA    = Path(__file__).parents[1] / "data"
TARGET  = "final_pce"


def summarise():
    rows = []
    for f in sorted(RESULTS.glob("runs_*.jsonl")):
        tier = f.stem.replace("runs_", "")
        records = [json.loads(l) for l in f.read_text().splitlines() if l]
        rmses = [r["rmse"] for r in records]
        maes  = [r["mae"]  for r in records]
        r2s   = [r["r2"]   for r in records]
        rows.append({"tier": tier, "n_splits": len(records),
                     "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
                     "MAE_mean":  np.mean(maes),
                     "R2_mean":   np.mean(r2s)})
    if not rows:
        print("No results found. Run 02_train.py first.")
        return
    df = pd.DataFrame(rows).sort_values("RMSE_mean")
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\nCross-validation summary\n" + "=" * 50)
    print(df.to_string(index=False))


def shap_plot(tier="t2"):
    import shap
    from catboost import CatBoostRegressor
    from scripts._train_utils import feature_frame, Preprocessor  # reuse helpers

    df = pd.read_parquet(DATA / "prepared.parquet")
    y  = df[TARGET].values.astype(float)

    # use all data as train for SHAP (no held-out test; this is for interpretation only)
    Xraw = feature_frame(df, tier)
    pre  = Preprocessor().fit(Xraw)
    X    = pre.transform(Xraw)

    CATEGORICALS = ["Cell_architecture", "ETL_deposition_method",
                    "HTL_deposition_method", "Perovskite_deposition_method"]
    ci = [i for i, c in enumerate(X.columns) if c in CATEGORICALS]
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=8, random_seed=42,
        loss_function="RMSE", verbose=0,
    )
    model.fit(X, y, cat_features=ci)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap.summary_plot(shap_vals, X, max_display=20, show=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--tier", default="t2")
    a = ap.parse_args()
    summarise()
    if a.shap:
        shap_plot(a.tier)
