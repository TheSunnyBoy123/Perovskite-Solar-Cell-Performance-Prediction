#!/usr/bin/env python3
"""Equal-budget random search of CatBoost configs for t2 and t3 (= t2 + recipe block) on v4dev only (DOI-grouped 5-fold)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from psc.data import load_dataset, TARGET, CAT_ALL
from psc.features import base_frame, TrainFitPreprocessor
from psc.splits import make_splits
from psc import v4
import importlib.util
spec = importlib.util.spec_from_file_location("v4_dev", ROOT / "scripts" / "v4_dev.py")
src = (ROOT / "scripts" / "v4_dev.py").read_text().rsplit("main()", 1)[0]
ns = {"__file__": str(ROOT / "scripts" / "v4_dev.py")}; exec(compile(src, "v4_dev", "exec"), ns)
fit_eval, blocks = ns["fit_eval"], ns["blocks"]

N_CFG = 24
rng = np.random.default_rng(2024)
CFGS = [dict(depth=int(rng.choice([4, 5, 6, 7, 8, 9, 10])), learning_rate=float(rng.choice([0.03, 0.05, 0.08])),
             l2_leaf_reg=float(rng.choice([1, 3, 10, 30])), colsample_bylevel=float(rng.choice([0.4, 0.55, 0.7, 0.85, 1.0])),
             min_data_in_leaf=int(rng.choice([1, 5, 10, 20, 40])), subsample=float(rng.choice([0.6, 0.8, 1.0])),
             random_strength=float(rng.choice([0.5, 1, 2]))) for _ in range(N_CFG)]
CFGS[0] = {}   # the matched default is always among the candidates
df = load_dataset(); df["v4part"] = df["doi_group"].map(v4.v4_partition); df = df[df.v4part == "v4dev"].reset_index(drop=True)
y = df[TARGET].values.astype(float); g = df["doi_group"].values
X2 = base_frame(df, "t2"); B = blocks(df)
splits = list(make_splits(g, "kfold", 5, 0.2, 0.15, 7))
res = {"t2": [], "t3": []}
out = ROOT / "results" / "v4" / "tune.json"
for ci, cfg in enumerate(CFGS):
    for tier in ("t2", "t3"):
        mses = []
        for k, (tr, va, te) in enumerate(splits):
            pre = TrainFitPreprocessor().fit(X2.iloc[tr]); Z = pre.transform(X2)
            p = dict(cfg)
            if tier == "t3":
                Z = pd.concat([Z, B["rec"]], axis=1); p["pen_lit"] = 0.5
            pr = fit_eval(Z, y, tr, va, te, set(CAT_ALL), p)
            mses.append(float(((pr - y[te]) ** 2).mean()))
        res[tier].append({"cfg": cfg, "mse": float(np.mean(mses)), "fold_mse": mses})
        print(ci, tier, cfg, round(np.mean(mses), 4), flush=True)
    out.write_text(json.dumps(res, indent=1))
