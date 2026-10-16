#!/usr/bin/env python3
"""Data/feature audit -> results/feature_audit.json. Run from ml_paper/."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import LIT32, MAT49, TARGET, NO_DOI_GROUP, load_dataset  # noqa
from psc.features import base_frame  # noqa

df = load_dataset(); out = {"n_records": len(df), "n_doi_groups_incl_no_doi": int(df.doi_group.nunique()),
                            "n_records_no_doi": int((df.doi_group == NO_DOI_GROUP).sum())}
lit = sorted(LIT32)
cov = {c: float(df[c].notna().mean()) for c in lit}
out["lit_coverage_per_feature"] = cov
anylit = df[lit].notna().any(axis=1)
out["records_with_any_lit_feature"] = float(anylit.mean()); out["dois_with_any_lit_feature"] = int(df[anylit].doi_group.nunique())
# paper-level constancy: fraction of (DOI, feature) cells whose non-null values are constant within the DOI
const = {}
for c in lit:
    g = df[[c, "doi_group"]].dropna().groupby("doi_group")[c].nunique()
    multi = df[[c, "doi_group"]].dropna().groupby("doi_group").size() > 1
    const[c] = float((g[multi] == 1).mean()) if multi.any() else None
out["lit_within_doi_constant_fraction"] = const
# material property coverage
out["mat_coverage_mean"] = float(np.mean([df[c].notna().mean() for c in MAT49 if c in df]))
for L in ("ETL", "HTL", "Perovskite"):
    out[f"{L}_band_gap_coverage"] = float(df[f"{L}_band_gap"].notna().mean())
# between-paper share of target variance (intraclass correlation, one-way)
g = df.groupby("doi_group")[TARGET]; k = g.size(); gm = g.mean(); n = len(df)
msb = (k * (gm - df[TARGET].mean()) ** 2).sum() / (len(k) - 1); msw = ((df[TARGET] - df.doi_group.map(gm)) ** 2).sum() / (n - len(k))
k0 = (n - (k ** 2).sum() / n) / (len(k) - 1)
out["target_icc_by_doi"] = float((msb - msw) / (msb + (k0 - 1) * msw))
out["records_per_doi_median"] = float(k.median()); out["records_per_doi_max"] = int(k.max())
# duplicate predictor configurations (t3 raw predictors) with target spread
import pandas as pd
X = base_frame(df, "t3"); key = pd.util.hash_pandas_object(X, index=False)
dup = key.duplicated(keep=False)
out["records_in_duplicate_t3_configs"] = int(dup.sum())
if dup.any():
    gg = df[dup].groupby(key[dup])
    out["duplicate_config_groups"] = int(gg.ngroups)
    out["duplicate_groups_spanning_multiple_dois"] = int((gg.doi_group.nunique() > 1).sum())
    out["duplicate_group_target_sd_median"] = float(gg[TARGET].std().median())
out["pce_gt_27"] = {"records": int((df[TARGET] > 27).sum()), "dois": int(df[df[TARGET] > 27].doi_group.nunique())}
json.dump(out, open(ROOT / "results" / "feature_audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or len(v) < 4}, indent=1))
print("lit coverage min/median/max", min(cov.values()), np.median(list(cov.values())), max(cov.values()))
print("lit within-doi constant median", np.nanmedian([v for v in const.values() if v is not None]))
