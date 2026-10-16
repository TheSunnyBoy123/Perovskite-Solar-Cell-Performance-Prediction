#!/usr/bin/env python3
"""Aggregate runs.jsonl -> summary tables (CSV + LaTeX). Run from ml_paper/.
Usage: python scripts/aggregate.py results/ablation_v1
SDs are across the repeated holdout splits (ddof=1); they describe split-to-split
variability, not a confidence interval (test sets overlap across repeats)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from psc.results import load_ablation  # noqa: E402
r = load_ablation() if d.name == "ablation_v1" else pd.DataFrame([json.loads(l) for l in open(d / "runs.jsonl")])
r = r.drop(columns=["info"])
g = r.groupby(["protocol", "arch", "model", "tier"])
s = g.agg(n=("n_test", "mean"), n_features=("n_features", "mean"), splits=("split", "count"),
          r2=("r2", "mean"), r2_sd=("r2", "std"), mae=("mae", "mean"), mae_sd=("mae", "std"),
          rmse=("rmse", "mean"), rmse_sd=("rmse", "std")).reset_index()
s.to_csv(d / "summary.csv", index=False)

# paired tier differences per split
rows = []
for (p, a, m), x in r.groupby(["protocol", "arch", "model"]):
    w = x.pivot(index="split", columns="tier", values=["r2", "mae", "rmse"])
    for hi, lo in (("t2", "t1"), ("t3", "t2"), ("t3", "t1")):
        row = dict(protocol=p, arch=a, model=m, contrast=f"{hi}-{lo}", splits=len(w))
        for k in ("r2", "mae", "rmse"):
            dd = (w[(k, hi)] - w[(k, lo)]).dropna()
            row[f"d_{k}"], row[f"d_{k}_sd"] = dd.mean(), dd.std()
            row[f"{k}_better_splits"] = int(((dd > 0) if k == "r2" else (dd < 0)).sum())
        rows.append(row)
pd.DataFrame(rows).to_csv(d / "paired_tier_diffs.csv", index=False)

# record-level minus DOI-grouped gap (same tier/model/arch), per split pairing by index
rows = []
for (a, m, t), x in r.groupby(["arch", "model", "tier"]):
    w = x.pivot(index="split", columns="protocol", values=["r2", "mae"])
    if ("r2", "doi") in w and ("r2", "random") in w:
        rows.append(dict(arch=a, model=m, tier=t, r2_doi=w[("r2", "doi")].mean(),
                         r2_random=w[("r2", "random")].mean(),
                         gap_r2=(w[("r2", "random")] - w[("r2", "doi")]).mean(),
                         mae_doi=w[("mae", "doi")].mean(), mae_random=w[("mae", "random")].mean()))
pd.DataFrame(rows).to_csv(d / "protocol_gap.csv", index=False)
print(s.round(4).to_string()); print(pd.DataFrame(rows).round(4).to_string())
