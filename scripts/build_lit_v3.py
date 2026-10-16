#!/usr/bin/env python3
"""Build data/lit_v3.csv (schema lx3.0 quantity-level features) from the cached extraction JSONs."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import normalize_doi
from psc import litfeat3
json_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "json")
df = litfeat3.build_table(json_dir, normalize_doi)
df.to_csv(ROOT / "data" / "lit_v3.csv")
cov = df.notna().mean().sort_values(ascending=False)
print(len(df), "papers", df.shape[1], "cols")
print(cov.round(3).to_string())
