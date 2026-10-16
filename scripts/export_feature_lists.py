#!/usr/bin/env python3
"""Write docs/FEATURES.md + results/feature_lists.json: exact predictors per tier.
Column set is shown for a preprocessor fitted on the full filtered data (documentation only);
in experiments the >80%-missing filter / _na columns are re-derived from each training partition."""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from psc.data import load_dataset  # noqa
from psc.features import TrainFitPreprocessor, base_frame  # noqa
from psc.groups import group  # noqa
df = load_dataset(); out = {}; md = ["# Feature lists per tier (pooled data, full-data fit; documentation only)\n"]
for t in ("t1", "t2", "t3"):
    Z = TrainFitPreprocessor().fit_transform(base_frame(df, t)); cols = list(Z.columns)
    out[t] = {c: group(c) for c in cols}
    counts = pd.Series([group(c) for c in cols]).value_counts().to_dict()
    md.append(f"\n## {t}: {len(cols)} features {counts}\n")
    prev = set(out["t1" if t == "t2" else "t2"]) if t != "t1" else set()
    for g in ("process", "device", "composition", "material", "literature"):
        gc = [c for c in cols if group(c) == g]
        if gc: md.append(f"- **{g}** ({len(gc)}): " + ", ".join(f"`{c}`" for c in gc))
json.dump(out, open(ROOT / "results" / "feature_lists.json", "w"), indent=1)
(ROOT / "docs" / "FEATURES.md").write_text("\n".join(md) + "\n")
print({t: len(v) for t, v in out.items()})
assert set(out["t1"]) < set(out["t2"]) < set(out["t3"])
