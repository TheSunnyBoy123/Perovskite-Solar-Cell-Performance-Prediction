"""
Step 4 — Predict PCE for new devices using a saved model bundle.

Loads models/v4_t3_bundle.pkl (the final M3 ensemble trained on the full
dataset) and writes per-device PCE predictions to a CSV file.

Usage:
    python scripts/04_predict.py --input my_devices.csv --out preds.csv

Input schema:
    Same columns as data/dataset_json_enriched.csv (PDP device + processing +
    composition columns + curated material-property columns) plus Ref_DOI_number.
    Recipe flags are looked up from data/lit_v4.json by DOI; devices without a
    matching DOI are predicted using the M2 feature set (recipe block missing).
"""
import argparse
import pickle
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).parents[1]
MODELS = ROOT / "models"
DATA   = ROOT / "data"

NO_DOI = "__NO_DOI__"


def normalize_doi(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DOI
    s = str(x).replace("﻿", "").strip().lower()
    if not s or s == "nan":
        return NO_DOI
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s


def load_recipe_flags(doi_group):
    lit_json = DATA / "lit_v4.json"
    if not lit_json.exists():
        return pd.Series(dtype=float)
    papers = json.loads(lit_json.read_text())["papers"]
    p = papers.get(doi_group, {})
    return pd.Series(p.get("recipe_flags", {}))


def predict(model_path, input_csv, out_csv):
    if not Path(model_path).exists():
        sys.exit(f"Model not found: {model_path}\n"
                 "Run the full training pipeline or obtain v4_t3_bundle.pkl from the authors.")

    df = pd.read_csv(input_csv, low_memory=False)
    df["doi_group"] = (df["Ref_DOI_number"].map(normalize_doi)
                       if "Ref_DOI_number" in df
                       else NO_DOI)

    # attach recipe flags from lit_v4.json
    lit_json = DATA / "lit_v4.json"
    if lit_json.exists():
        papers = json.loads(lit_json.read_text())["papers"]
        rec = (
            pd.DataFrame({doi: p.get("recipe_flags", {}) for doi, p in papers.items()})
            .T.rename_axis("doi_group").reset_index()
        )
        df = df.merge(rec.add_prefix("rc_").rename(columns={"rc_doi_group": "doi_group"}),
                      on="doi_group", how="left")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    preds = np.stack([m.predict(bundle["preprocessors"][i].transform(df))
                      for i, m in enumerate(bundle["models"])], axis=0)
    pce_mean = preds.mean(axis=0)
    pce_std  = preds.std(axis=0)

    out = df[[c for c in ("Ref_ID", "Ref_DOI_number") if c in df]].copy()
    out["pce_pred"]    = pce_mean
    out["pce_pred_sd"] = pce_std
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} predictions → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "v4_t3_bundle.pkl"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--out",   required=True)
    a = ap.parse_args()
    predict(a.model, a.input, a.out)
