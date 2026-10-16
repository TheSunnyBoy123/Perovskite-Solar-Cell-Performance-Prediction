#!/usr/bin/env python3
"""Machine-readable audit of the schema-lx2.0 literature features -> results/lit_v2_audit.json (+ .csv)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
from psc.features import base_frame
from psc.litfeat import SCHEMA_VERSION

df = load_dataset(ROOT / "data/dataset_json_enriched.csv", "overall")
cov = df[df.lx_covered == 1]
t2 = base_frame(df, "t2")
t2num = t2.select_dtypes("number").loc[cov.index]
t2num = t2num[[c for c in t2num.columns if not c.endswith("_na") and t2num[c].nunique() > 2]]
lx = [c for c in df.columns if c.startswith("lx_") and c != "lx_covered"]
paper = cov.drop_duplicates("doi_group")
rows = []
for c in lx:
    s = cov[c]; p = paper[c]
    nn = p.notna()
    red = {}
    if nn.sum() > 30 and s.nunique() > 1:
        cc = t2num.corrwith(s, method="spearman").abs().dropna().sort_values(ascending=False)
        red = {"max_abs_spearman_with_t2": float(cc.iloc[0]) if len(cc) else None, "most_similar_t2_col": cc.index[0] if len(cc) else None}
    rho = float(pd.Series(s).corr(cov[TARGET], method="spearman")) if s.nunique() > 1 else None
    grp = c.split("_")[1]
    rows.append(dict(feature=c, group=grp, level="paper (constant within DOI)",
                     coverage_records_all=float(df[c].notna().mean()), coverage_records_in_covered=float(s.notna().mean()),
                     coverage_papers_in_covered=float(nn.mean()), n_papers_nonmissing=int(nn.sum()),
                     binary=bool(set(p.dropna().unique()) <= {0.0, 1.0}), prevalence_or_median=float(p.mean() if set(p.dropna().unique()) <= {0.0, 1.0} else p.median()) if nn.any() else None,
                     spearman_with_target_descriptive=rho, **red,
                     provenance=f"{SCHEMA_VERSION}: deterministic parse of stored LLM JSON ({'completeness proxy' if grp=='meta' else 'post-fabrication characterisation' if grp=='char' else 'process/chemistry/context'})",
                     outcome_derived=False,
                     proposed_action={"meta": "ablation-only (reporting completeness; not a physical condition)",
                                      "char": "ablation-only (not known before device fabrication)"}.get(grp, "keep (tested in controlled ablation)")
                     if nn.sum() >= 30 else "drop (<30 papers non-missing)"))
for r in rows:
    if (r.get("max_abs_spearman_with_t2") or 0) >= 0.9 and r["group"] in ("proc", "chem", "ctx"):
        r["proposed_action"] = "redundant with t2 (|rho|>=0.9 with %s); keep only to document redundancy" % r["most_similar_t2_col"]
a = pd.DataFrame(rows)
a.round(4).to_csv(ROOT / "results/lit_v2_audit.csv", index=False)
(ROOT / "results/lit_v2_audit.json").write_text(json.dumps(dict(schema=SCHEMA_VERSION, n_papers_with_json=int(paper.shape[0]),
    n_records_covered=int(len(cov)), frac_records_covered=float(len(df.pipe(lambda d: d[d.lx_covered == 1])) / len(df)), features=rows), indent=1, default=float))
print(a[["feature", "coverage_records_in_covered", "n_papers_nonmissing", "max_abs_spearman_with_t2", "proposed_action"]].round(3).to_string())
