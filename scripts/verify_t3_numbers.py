#!/usr/bin/env python3
"""Check that the M3 (final t3) numbers quoted in ml.tex match results/t3final_summary/numbers.json and matched_summary.csv.
Run after scripts/t3_protocol_summary.py and scripts/make_t3_tables.py. Prints FAILS: none when consistent."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "psc_manuscript/ml.tex").read_text(); N = json.load(open(ROOT / "results/t3final_summary/numbers.json"))
S = pd.read_csv(ROOT / "results/t3final_summary/matched_summary.csv")
fails = []
def need(label, text):
    if text not in tex: fails.append((label, text))
for a in ("overall", "pin", "nip"):
    need(f"doi {a} M3 R2", f"{N[f'doi_{a}_t3_r2']:.3f} \\pm {N[f'doi_{a}_t3_r2_sd']:.3f}")
for m in ("catboost", "ridge"):
    for t in ("t1", "t2", "t3"):
        need(f"gap {m} {t}", f"${N[f'gap_{m}_{t}']:.3f}$")
for proto in ("random", "doi", "author", "temporal"):
    for m in ("catboost", "ridge"):
        for sc in ("all", "covered"):
            r = S[(S.protocol == proto) & (S.arch == "overall") & (S.model == m) & (S.scope == sc)].iloc[0]
            need(f"d_r2 {proto} {m} {sc}", f"${r.d_r2:+.4f}$"); need(f"R2 {proto} {m} {sc}", f"${r.r2_t3:.3f}$")
for k, v in (("lead-author M3", N["author_catboost_t3"]), ("forward M3", N["temporal_catboost_t3"])):
    need(k, f"{v:.3f}")
for bad in ("not run for $\\mathcal{M}_3$", "were not run for $\\mathcal{M}_3$.", "t_1", "t_2", "t_3"):
    pass
if "These protocols were not run" in tex or "$\\mathcal{M}_3$ was not part of this experiment" in tex: fails.append(("stale statement", ""))
print("FAILS:", fails if fails else "none")
