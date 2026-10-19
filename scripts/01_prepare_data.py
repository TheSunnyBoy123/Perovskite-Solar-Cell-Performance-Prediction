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
        sys.exit(f"Missing: {csv}\nSee data/README.md")

    df = pd.read_csv(csv, low_memory=False)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 1.0].reset_index(drop=True)

    if arch != "overall":
        df = df[df["Cell_architecture"].str.lower() == arch].reset_index(drop=True)

    df["doi_group"] = df[DOI_COL].map(normalize_doi)

    meta_csv = DATA / "pdp_ref_metadata.csv"
    if meta_csv.exists():
        meta = pd.read_csv(meta_csv).drop_duplicates("Ref_ID")
        df = df.merge(meta, on="Ref_ID", how="left")
        df["pub_date"] = pd.to_datetime(df.get("Ref_publication_date"), errors="coerce")

    print(f"Loaded {len(df):,} records, {df['doi_group'].nunique():,} publications")
    df.to_parquet(OUT, index=False)
    print(f"Saved → {OUT}")
    return df


if __name__ == "__main__":
    load()
