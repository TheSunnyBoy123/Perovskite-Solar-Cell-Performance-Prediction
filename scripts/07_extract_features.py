"""
Step 7 — Build data/lit_v4.json from per-paper extraction JSONs.

Reads all *.json files in extractions/ (produced by 06_manual_queue_app.py)
and emits the paper-level recipe flags, numeric aggregates and token vocabulary
that become the M3 feature block.

Usage:
    python scripts/07_extract_features.py --extractions extractions/ --out data/lit_v4.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ALIASES = [
    ("spiro",  r"spiro|omeTAD|ometad|meotad"),
    ("ptaa",   r"\bptaa\b|polytriarylamine"),
    ("pedot",  r"pedot|pss"),
    ("pcbm",   r"pcbm|pc61bm|pc71bm|phenyl-c"),
    ("c60",    r"\bc60\b|fullerene"),
    ("bcp",    r"\bbcp\b|bathocuproine"),
    ("tio2",   r"tio2|tiox|titan"),
    ("sno2",   r"sno2|snox|tin oxide"),
    ("zno",    r"\bzno|zinc oxide"),
    ("litfsi", r"li-?tfsi|lithium bis"),
    ("tbp",    r"\btbp\b|tert-?butyl ?pyridine"),
    ("pea",    r"\bpe(a)?i\b|phenethyl"),
    ("mai",    r"\bmai\b|methylammonium iodide"),
    ("fai",    r"\bfai\b|formamidinium iodide"),
    ("csi",    r"\bcsi\b|cesium iodide"),
    ("dmf",    r"\bdmf\b|dimethylformamide"),
    ("dmso",   r"\bdmso\b|dimethyl.?sulfoxide"),
    ("cb",     r"\bcb\b|chlorobenzene"),
    ("ipa",    r"\bipa\b|isopropan"),
    ("au",     r"^au\b|\bgold\b"),
    ("ag",     r"^ag\b|\bsilver\b"),
    ("fto",    r"\bfto\b|sno2:f|fluorine"),
    ("ito",    r"\bito\b|indium tin"),
]
_ALIAS_RE = [(n, re.compile(p, re.I)) for n, p in ALIASES]

ROLE_MAP = {
    "antisolvent": "anti", "solvent": "solv", "additive": "add",
    "etl": "etl", "htl": "htl", "contact": "ctc",
    "perov": "perov", "precursor": "prec", "other": "oth",
}
OUTCOME_RE = re.compile(r"pce|efficiency|voc|jsc|fill.factor|ff|stability|eqe|hysteresis", re.I)

RECIPE = {
    "has_BCP":               r"\bbcp\b|bathocuproine",
    "has_LiF":               r"\blif\b|lithium fluoride",
    "has_PEAI":              r"\bpe(a)?i\b|phenethyl",
    "has_SAM":               r"\bsam\b|self-?assembled monolayer",
    "has_C60":               r"\bc60\b|fullerene",
    "has_PCBM":              r"pcbm|pc61bm|pc71bm",
    "has_Spiro":             r"spiro|ometad|meotad",
    "has_PEDOT":             r"pedot|pss",
    "has_NiOx":              r"\bnio|nickel oxide",
    "has_SnO2":              r"sno2|tin oxide",
    "has_TiO2":              r"tio2|tiox",
    "has_ZnO":               r"\bzno|zinc oxide",
    "has_PTAA":              r"\bptaa\b",
    "has_fullerene":         r"fullerene|\bc60\b|pcbm",
    "has_passivation_layer": r"passivat|interlayer|surface treatment",
    "is_flexible_y":         r"\bpet\b|\bpen\b|flexible|polyimide",
    "used_antisolvent":      r"anti-?solvent|quench",
    "used_glovebox":         r"glove.?box|nitrogen|\bn2\b|argon",
    "used_vacuum_evaporation": r"vacuum|evaporat|vapou?r",
}
_RECIPE_RE = {k: re.compile(v, re.I) for k, v in RECIPE.items()}


def _canon(s):
    s = str(s).strip().lower()
    for name, rx in _ALIAS_RE:
        if rx.search(s):
            return name
    return None


def _role(m: dict) -> str:
    t = str(m.get("type", "")).lower()
    for k, v in ROLE_MAP.items():
        if k in t:
            return v
    return "oth"


def material_records(doc: dict) -> list:
    recs = []
    for m in doc.get("materials", []):
        if not isinstance(m, dict):
            continue
        r = _role(m)
        name = _canon(str(m.get("formula", "")) + " " + str(m.get("common_name", "")))
        nums = {}
        for k, v in m.items():
            kl = str(k).lower()
            if OUTCOME_RE.search(kl) or kl in ("formula", "common_name", "type"):
                continue
            try:
                x = float(re.search(r"[-+]?\d*\.?\d+", str(v)).group())
            except (AttributeError, ValueError):
                continue
            if re.search(r"spin.*(speed|rpm)", kl):
                nums.setdefault("spin_rpm", []).append(x)
            elif re.search(r"anneal.*temp|temperature", kl):
                nums.setdefault("anneal_C", []).append(x)
            elif re.search(r"anneal.*time|time", kl):
                nums.setdefault("anneal_min", []).append(x)
            elif re.search(r"thick", kl):
                nums.setdefault("thick_nm", []).append(x)
        all_text = " ".join(
            str(v) for k, v in m.items()
            if not OUTCOME_RE.search(str(k)) and isinstance(v, (str, int, float))
        ).lower()
        recs.append({"role": r, "name": name, "nums": nums, "text": all_text})
    return recs


def paper_tokens(recs):
    return sorted({f"{r['role']}:{r['name']}" for r in recs if r["name"]})


def paper_numeric(recs):
    out = {}
    for role in ("perov", "etl", "htl", "add", "anti", "solv"):
        for kind in ("spin_rpm", "anneal_C", "anneal_min", "thick_nm"):
            vals = [x for r in recs if r["role"] == role for x in r["nums"].get(kind, [])]
            if vals:
                a = np.array(vals)
                out[f"{role}_{kind}_med"] = float(np.median(a))
                out[f"{role}_{kind}_min"] = float(a.min())
                out[f"{role}_{kind}_max"] = float(a.max())
                out[f"{role}_{kind}_n"]   = float(len(np.unique(np.round(a, 4))))
    return out


def paper_recipe(recs):
    all_text = " ".join(r["text"] for r in recs)
    return {k: float(bool(rx.search(all_text))) for k, rx in _RECIPE_RE.items()}


def normalize_doi(x):
    if not x:
        return "__NO_DOI__"
    s = str(x).replace("﻿", "").strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s or "__NO_DOI__"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", default="extractions/")
    ap.add_argument("--out",         default="data/lit_v4.json")
    a = ap.parse_args()

    files = sorted(Path(a.extractions).glob("*.json"))
    if not files:
        print(f"No JSON files in {a.extractions}")
        return

    papers = {}
    for f in files:
        try:
            doc     = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            doi_raw = doc.get("source", f.stem.replace("_", "/", 1))
            doi_raw = re.sub(r"^doi:\s*", "", str(doi_raw), flags=re.I).strip()
            doi_key = normalize_doi(doi_raw)
            recs    = material_records(doc)
            papers[doi_key] = {
                "tokens":       paper_tokens(recs),
                "numeric":      paper_numeric(recs),
                "recipe_flags": paper_recipe(recs),
            }
        except Exception as e:
            print(f"  skip {f.name}: {e}")

    Path(a.out).write_text(json.dumps({"schema": "lx4.0", "papers": papers}, indent=2))
    tok_counts = Counter(t for p in papers.values() for t in p["tokens"])
    print(f"{len(papers)} papers  •  {len(tok_counts)} unique tokens  →  {a.out}")


if __name__ == "__main__":
    main()
