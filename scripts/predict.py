#!/usr/bin/env python3
"""Predict PCE (%) for PDP-schema device records with a saved bundle. No LLM / network access needed.

  python scripts/predict.py --model models/v4_t3_bundle.pkl --input devices.csv --out preds.csv

devices.csv: same columns as data/dataset_json_enriched.csv (PDP device / processing / composition columns plus the
curated material-property columns) and `Ref_DOI_number`. Recipe features are looked up in data/lit_v4.json by DOI;
alternatively pass rc_* columns (see psc.v4.REC_GROUPS). A DOI without an entry is predicted with the recipe block missing.
"""
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import normalize_doi
from psc.v4model import PSCModel

ap = argparse.ArgumentParser(); ap.add_argument("--model", default=str(ROOT / "models/v4_t3_bundle.pkl"))
ap.add_argument("--input", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
df = pd.read_csv(a.input, low_memory=False)
df["doi_group"] = df["Ref_DOI_number"].map(normalize_doi) if "Ref_DOI_number" in df else "__NO_DOI__"
m = PSCModel.load(a.model)
P = m.predict(df, per_seed=True)
out = df[[c for c in ("Ref_ID", "Ref_DOI_number") if c in df]].copy()
out["pce_pred"] = P.mean(0); out["pce_pred_sd_seeds"] = P.std(0)
out.to_csv(a.out, index=False); print(f"wrote {len(out)} predictions -> {a.out}")
