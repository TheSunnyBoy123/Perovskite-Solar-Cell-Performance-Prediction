import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).parents[1]
MODELS = ROOT / "models"
NO_DOI = "__NO_DOI__"


def normalize_doi(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DOI
    s = str(x).replace("﻿", "").strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or NO_DOI


def predict(model_path, input_csv, out_csv):
    if not Path(model_path).exists():
        sys.exit(f"Model not found: {model_path}")

    df = pd.read_csv(input_csv, low_memory=False)
    df["doi_group"] = (df["Ref_DOI_number"].map(normalize_doi)
                       if "Ref_DOI_number" in df else NO_DOI)

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    preds = np.stack([m.predict(bundle["preprocessors"][i].transform(df))
                      for i, m in enumerate(bundle["models"])], axis=0)

    out = df[[c for c in ("Ref_ID", "Ref_DOI_number") if c in df]].copy()
    out["pce_pred"]    = preds.mean(axis=0)
    out["pce_pred_sd"] = preds.std(axis=0)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out):,} predictions → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "v4_t2_bundle.pkl"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--out",   required=True)
    a = ap.parse_args()
    predict(a.model, a.input, a.out)
