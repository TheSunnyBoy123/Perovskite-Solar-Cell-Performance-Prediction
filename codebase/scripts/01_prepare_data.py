"""
Step 1 — Load and prepare the dataset.

Reads data/dataset_json_enriched.csv, normalises DOI grouping keys,
joins publication metadata and recipe flags, and writes a cleaned
parquet file ready for modelling.

Usage:
    python scripts/01_prepare_data.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parents[1] / "data"
OUT  = DATA / "prepared.parquet"

TARGET  = "final_pce"
DOI_COL = "Ref_DOI_number"
NO_DOI  = "__NO_DOI__"


def normalize_doi(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DOI
    s = str(x).replace("﻿", "").strip().lower()
    if not s or s == "nan":
        return NO_DOI
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s


def load(arch="overall"):
    csv = DATA / "dataset_json_enriched.csv"
    if not csv.exists():
        sys.exit(f"Missing: {csv}\nSee data/README.md for how to obtain it.")

    df = pd.read_csv(csv, low_memory=False)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 1.0].reset_index(drop=True)   # drop PCE < 1 % (likely failures)

    if arch != "overall":
        df = df[df["Cell_architecture"].str.lower() == arch].reset_index(drop=True)

    df["doi_group"] = df[DOI_COL].map(normalize_doi)

    # publication metadata (lead author, date) — used for stricter split protocols
    meta_csv = DATA / "pdp_ref_metadata.csv"
    if meta_csv.exists():
        meta = pd.read_csv(meta_csv).drop_duplicates("Ref_ID")
        df = df.merge(meta, on="Ref_ID", how="left")
        df["pub_date"] = pd.to_datetime(df.get("Ref_publication_date"), errors="coerce")

    # recipe flags from lit_v4.json (M3 feature block)
    lit_json = DATA / "lit_v4.json"
    if lit_json.exists():
        papers = json.loads(lit_json.read_text())["papers"]
        rec = (
            pd.DataFrame({doi: p.get("recipe_flags", {}) for doi, p in papers.items()})
            .T.rename_axis("doi_group").reset_index()
        )
        df = df.merge(rec.add_prefix("rc_").rename(columns={"rc_doi_group": "doi_group"}),
                      on="doi_group", how="left")

    print(f"Loaded {len(df):,} records, {df['doi_group'].nunique():,} publications")
    df.to_parquet(OUT, index=False)
    print(f"Saved → {OUT}")
    return df


if __name__ == "__main__":
    load()
