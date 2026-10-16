#!/usr/bin/env python3
"""Fit deployment bundles (t2 and t3, tuned configs from results/v4/final/tuned_cfg.json) on ALL records; no evaluation here."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset, TARGET
from psc.v4model import PSCModel
cfg = json.loads((ROOT / "results/v4/final/tuned_cfg.json").read_text())
df = load_dataset(); y = df[TARGET].values.astype(float)
for t in ("t2", "t3"):
    PSCModel(t, cfg[t], seeds=range(5)).fit(df, y, df.doi_group.values).save(ROOT / f"models/v4_{t}_bundle.pkl"); print("saved", t)
