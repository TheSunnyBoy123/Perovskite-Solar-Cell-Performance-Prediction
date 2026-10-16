#!/usr/bin/env python3
"""Hash every source/config/script file -> results/code_manifest.json (the package is not a git
commit; this records exactly which code produced the results). Run from ml_paper/."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
files = sorted(p for d in ("src", "scripts", "configs", "tests") for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".sh"})
m = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
m["_data"] = hashlib.sha256((ROOT / "data" / "dataset_json_enriched.csv").read_bytes()).hexdigest()
(ROOT / "results" / "code_manifest.json").write_text(json.dumps(m, indent=1)); print(len(m), "files hashed")
