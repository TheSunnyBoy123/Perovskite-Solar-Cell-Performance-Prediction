#!/usr/bin/env python3
"""Compare a reproduction run with the original: python scripts/compare_runs.py results/repro_cleanenv results/ablation_v1
Prints the number of matched cells and the maximum absolute differences in R2/MAE/RMSE and writes <run>/comparison.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
a, b = (pd.DataFrame([json.loads(l) for l in open(Path(p) / "runs.jsonl")]) for p in sys.argv[1:3])
k = ["protocol", "arch", "tier", "model", "split"]
m = a.merge(b, on=k, suffixes=("_new", "_ref"))
out = {"cells_new": len(a), "cells_ref": len(b), "matched": len(m)}
for c in ("r2", "mae", "rmse"):
    d = (m[f"{c}_new"] - m[f"{c}_ref"]).abs(); out[f"max_abs_diff_{c}"] = float(d.max()); out[f"n_diff_gt_1e-6_{c}"] = int((d > 1e-6).sum())
worst = m.assign(d=(m.r2_new - m.r2_ref).abs()).sort_values("d", ascending=False).head(5)[k + ["r2_new", "r2_ref", "d"]]
out["worst5"] = worst.round(6).to_dict("records")
json.dump(out, open(Path(sys.argv[1]) / "comparison.json", "w"), indent=1); print(json.dumps(out, indent=1))
