#!/usr/bin/env python3
"""Re-run selected stored experiment cells and compare with results/ablation_v1/runs.jsonl.
Usage: python scripts/repro_check.py   (about 2 minutes). Exit code 1 if any |dR2| > 0.01."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import TARGET, load_dataset  # noqa: E402
from psc.evaluate import metrics  # noqa: E402
from psc.features import TrainFitPreprocessor, base_frame  # noqa: E402
from psc.models import fit_predict  # noqa: E402
from psc.splits import make_splits  # noqa: E402
stored = {(r["protocol"], r["arch"], r["tier"], r["model"], r["split"]): r
          for r in map(json.loads, open(ROOT / "results/ablation_v1/runs.jsonl"))}
CELLS = [("doi", "overall", "t3", "catboost", 1), ("doi", "overall", "t1", "ridge", 2), ("doi", "nip", "t2", "catboost", 3)]
bad = 0
import catboost, sklearn, pandas
print("catboost", catboost.__version__, "sklearn", sklearn.__version__, "pandas", pandas.__version__)
for proto, arch, tier, model, k in CELLS:
    df = load_dataset(arch=arch); y = df[TARGET].values.astype(float)
    tr, va, te = list(make_splits(df["doi_group"].values, proto, 5, 0.2, 0.15, 42))[k - 1]
    X = base_frame(df, tier); pre = TrainFitPreprocessor().fit(X.iloc[tr])
    a, b, c = (pre.transform(X.iloc[i]) for i in (tr, va, te))
    p, _ = fit_predict(model, a, y[tr], b, y[va], c)
    m = metrics(y[te], p); s = stored[(proto, arch, tier, model, k)]
    d = m["r2"] - s["r2"]; bad += abs(d) > 0.01
    print(f"{proto} {arch} {tier} {model} s{k}: rerun R2={m['r2']:.4f} stored={s['r2']:.4f} diff={d:+.4f}")
sys.exit(int(bad > 0))
