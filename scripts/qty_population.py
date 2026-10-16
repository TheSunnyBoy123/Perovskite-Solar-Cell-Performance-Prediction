#!/usr/bin/env python3
"""Define the quantity-covered population and the predeclared quantity feature list (availability only; no PCE / model output).

Eligibility: publication has >=1 non-missing value among the lx_qty_* columns other than back-contact ones.
Selected features: eligible-population paper-level coverage >= 10%.
Writes data/lx3_qty_selected.json and prints the population table."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from psc.data import load_dataset, doi_partition  # noqa: E402

df = load_dataset(ROOT / "data/dataset_json_enriched.csv", "overall", subset="lx_covered")
q = [c for c in df if c.startswith("lx_qty_")]
core = [c for c in q if "contact" not in c]
paper = df.drop_duplicates("doi_group").set_index("doi_group")[q]
elig = paper.index[paper[core].notna().any(axis=1)]
cov = paper.loc[elig].notna().mean().sort_values(ascending=False)
sel = [c for c in cov.index if cov[c] >= 0.10]
rec = df[df.doi_group.isin(elig)]
out = dict(rule="eligible = >=1 non-missing non-contact lx_qty_* column; selected = eligible-paper coverage >= 0.10",
           n_papers=int(len(elig)), n_records=int(len(rec)), selected=sel,
           coverage_eligible=cov.round(3).to_dict(), n_lx_covered_papers=int(len(paper)), n_lx_covered_records=int(len(df)))
(ROOT / "data/lx3_qty_selected.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "coverage_eligible"}, indent=1))
