#!/usr/bin/env python3
"""Proxy validation of LLM-extracted literature features against independent PDP curation.
No hand-annotated gold set exists; the PDP columns/stack strings are curated by other people
from the same papers, so agreement is an (imperfect) accuracy estimate. -> results/extraction_validation.json
Numeric: LLM paper-level value vs the PDP device values of the same DOI (median). Flags: DOI-level
presence of a material in any PDP stack of that DOI vs the LLM flag."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import NO_DOI_GROUP, load_dataset  # noqa
df = load_dataset(); df = df[df.doi_group != NO_DOI_GROUP]
have = df[[c for c in df.columns if c in ("has_BCP",)]].notna().any(axis=1)
llm_doi = df.groupby("doi_group")["has_BCP"].apply(lambda s: s.notna().any())
dois = llm_doi[llm_doi].index; sub = df[df.doi_group.isin(dois)]
out = {"n_dois_with_llm_features": int(len(dois))}
num = {"perovskite_thickness_nm": "Perovskite_thickness_nm", "etl_thickness_nm": "ETL_thickness_nm",
       "htl_thickness_nm": "HTL_thickness_nm", "contact_thickness_nm": "Backcontact_thickness_nm",
       "perov_anneal_temp_C": "Perov_anneal_temp_C", "perov_anneal_time_min": "Perov_anneal_time_min"}
res = {}
for l, pcol in num.items():
    g = sub.groupby("doi_group").agg(llm=(l, "first"), pdp=(pcol, "median")).dropna()
    g = g[(g.llm > 0) & (g.pdp > 0)]
    if len(g) < 20: res[l] = {"n_dois": int(len(g))}; continue
    rel = (g.llm - g.pdp).abs() / g.pdp
    res[l] = {"n_dois": int(len(g)), "spearman": float(spearmanr(g.llm, g.pdp)[0]),
              "within_10pct": float((rel <= .10).mean()), "within_25pct": float((rel <= .25).mean()),
              "median_rel_err": float(rel.median())}
for l, pcol in num.items():  # record-level redundancy with the PDP column that t1 already contains
    both = df[[l, pcol]].dropna()
    res[l].update({"records_both": int(len(both)), "exact_equal_where_both": float((both[l] == both[pcol]).mean()),
                   "llm_value_where_pdp_missing_frac_of_records": float((df[l].notna() & df[pcol].isna()).mean())})
out["numeric"] = res
# flags vs PDP stack strings
stacks = sub[["doi_group", "ETL", "HTL", "Perovskite"]].astype(str)
pats = {"has_SnO2": ("ETL", "SnO2"), "has_TiO2": ("ETL", "TiO2"), "has_ZnO": ("ETL", "ZnO"), "has_C60": ("ETL", "C60"),
        "has_PCBM": ("ETL", "PCBM"), "has_BCP": ("ETL", "BCP"), "has_LiF": ("ETL", "LiF"), "has_Spiro": ("HTL", "Spiro"),
        "has_PEDOT": ("HTL", "PEDOT"), "has_NiOx": ("HTL", "NiO"), "has_PTAA": ("HTL", "PTAA")}
fl = {}
for f, (col, pat) in pats.items():
    truth = stacks.assign(t=stacks[col].str.contains(pat, case=False, regex=False)).groupby("doi_group").t.any()
    pred = sub.groupby("doi_group")[f].first().fillna(0).astype(float) > 0.5
    t, p = truth.reindex(pred.index).fillna(False), pred
    tp, fp, fn, tn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum())
    n = tp + fp + fn + tn; po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    fl[f] = {"n_dois": n, "prevalence_pdp": float(t.mean()), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
             "accuracy": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
out["flags"] = fl
json.dump(out, open(ROOT / "results" / "extraction_validation.json", "w"), indent=1)
print(pd.DataFrame(res).T.round(3).to_string()); print(pd.DataFrame(fl).T.round(3).to_string())
