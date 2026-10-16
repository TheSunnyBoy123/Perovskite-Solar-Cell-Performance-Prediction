#!/usr/bin/env python3
"""Build data/lit_v4.json (tokens, numeric aggregates, layer records per paper) from the cached extraction JSONs."""
import argparse, json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psc.data import normalize_doi
from psc import litfeat4
ap = argparse.ArgumentParser(); ap.add_argument("--json-dir", default=str(ROOT.parent / "json"))
a = ap.parse_args()
res = litfeat4.build(a.json_dir, normalize_doi)
(ROOT / "data" / "lit_v4.json").write_text(json.dumps({"schema": litfeat4.SCHEMA_VERSION, "papers": res}))
df = collections.Counter(t for p in res.values() for t in p["tokens"])
nm = collections.Counter(k for p in res.values() for k in p["numeric"])
print(len(res), "papers;", len(df), "tokens;", sum(1 for v in df.values() if v >= 25), "tokens DF>=25;", len(nm), "numeric cols")
print(df.most_common(40)); print(nm.most_common(25))
