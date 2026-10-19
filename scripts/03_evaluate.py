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
        tier    = f.stem.replace("runs_", "")
        records = [json.loads(l) for l in f.read_text().splitlines() if l]
        rows.append({
            "tier":      tier,
            "n_splits":  len(records),
            "RMSE_mean": np.mean([r["rmse"] for r in records]),
            "RMSE_std":  np.std( [r["rmse"] for r in records]),
            "MAE_mean":  np.mean([r["mae"]  for r in records]),
            "R2_mean":   np.mean([r["r2"]   for r in records]),
        })
    if not rows:
        print("No results found. Run 02_train.py first.")
        return
    pd.set_option("display.float_format", "{:.3f}".format)
    print(pd.DataFrame(rows).sort_values("RMSE_mean").to_string(index=False))


def shap_plot(tier="t2"):
    import shap
    from catboost import CatBoostRegressor

    df   = pd.read_parquet(DATA / "prepared.parquet")
    y    = df[TARGET].values.astype(float)

    from scripts._02_train import feature_frame, Preprocessor, CATEGORICALS
    pre  = Preprocessor().fit(feature_frame(df, tier))
    X    = pre.transform(feature_frame(df, tier))

    ci    = [i for i, c in enumerate(X.columns) if c in CATEGORICALS]
    model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8,
                               random_seed=42, loss_function="RMSE", verbose=0)
    model.fit(X, y, cat_features=ci)
    shap.summary_plot(shap.TreeExplainer(model).shap_values(X), X, max_display=20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--tier", default="t2")
    a = ap.parse_args()
    summarise()
    if a.shap:
        shap_plot(a.tier)
