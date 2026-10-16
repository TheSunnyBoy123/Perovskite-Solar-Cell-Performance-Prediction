#!/usr/bin/env python3
"""Audit of the two literature-coverage masks (read-only; no training). Writes results/coverage_mask_audit.{json,csv}.
Mask A (earlier 44-predictor block): record's publication has >=1 non-missing of the 32 base columns (results/feature_audit.json).
Mask B (recipe flags / lx_covered): normalised DOI occurs among the 2,835 extraction records (data/lit_v4.json, lit_v2.csv)."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, LIT32, NO_DOI_GROUP
df = load_dataset(); lit = sorted(LIT32)
A = set(df[df[lit].notna().any(axis=1)].doi_group)
P = json.load(open(ROOT / "data/lit_v4.json"))["papers"]
B = (set(df.doi_group) - {NO_DOI_GROUP}) & set(P)
only_b = sorted(B - A); only_a = sorted(A - B)
rec = df.groupby("doi_group").size()
rows = [{"doi_group": d, "n_filtered_records": int(rec[d]), "n_materials_extracted": P[d]["recipe"].get("n_materials"),
         "n_tokens": len(P[d]["tokens"]), "n_numeric": len(P[d]["numeric"]), "n_recipe_flags_set": sum(1 for k, v in P[d]["recipe"].items() if k != "n_materials" and not k.startswith("s_") and v),
         "any_of_32_base_columns_nonmissing": bool(df[df.doi_group == d][lit].notna().any().any())} for d in only_b]
pd.DataFrame(rows).to_csv(ROOT / "results/coverage_mask_audit.csv", index=False)
zero = [d for d, p in P.items() if p["recipe"].get("n_materials", 0) == 0]
out = {"mask_A_records": int(df.doi_group.isin(A).sum()), "mask_A_publications": len(A), "mask_B_records": int(df.doi_group.isin(B).sum()), "mask_B_publications": len(B),
       "A_minus_B_publications": len(only_a), "B_minus_A_publications": len(only_b), "B_minus_A_records": int(df.doi_group.isin(only_b).sum()),
       "extraction_records_total": len(P), "extraction_records_with_zero_materials": len(zero),
       "zero_material_records_matching_filtered_data": len(set(zero) & set(df.doi_group)), "zero_material_in_B_minus_A": len(set(zero) & set(only_b)),
       "B_minus_A_all_have_zero_materials": all(P[d]["recipe"].get("n_materials", 0) == 0 for d in only_b)}
json.dump(out, open(ROOT / "results/coverage_mask_audit.json", "w"), indent=1); print(json.dumps(out, indent=1))
