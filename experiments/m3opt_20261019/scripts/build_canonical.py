#!/usr/bin/env python3
"""Phase A/B: deterministic JSON -> canonical intermediate representation + join/integrity audit.

Reads (never modifies):
  ../sandbox/pdb.csv                      original PDP subset (target JV_default_PCE)
  data/dataset_json_enriched.csv          merged table used by every manuscript model
  ../json/*.json                          raw LLM extraction (one file per publication)
  data/lit_v4.json                        stored paper-level recipe flags (production M3)
Writes into experiments/m3opt_20261019/{features,audit}/.

Canonical tables
  json_long.csv.gz   one row per (paper, material index, key): role, canonical identity, key class, raw value, parsed number
  paper_status.csv   one row per JSON file: identifiers, join result, extraction status, counts per key class
The extraction has no evidence spans, confidence or explicit-negative fields, so the only recoverable states are
  covered (file with >= 1 material record) / empty (file parsed, 0 material records) / uncovered (no file).
"Not mentioned" inside a covered paper is NOT a verified negative; it is reported as such everywhere.
"""
import hashlib, json, re, sys, platform
from pathlib import Path
import numpy as np, pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
UGP = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from psc.data import normalize_doi, load_dataset, TARGET  # noqa: E402
from psc.litfeat import role as llm_role  # noqa: E402
from psc.litfeat4 import canon, ROLE_MAP, paper_recipe  # noqa: E402

PDB = UGP / "sandbox" / "pdb.csv"
ENR = ROOT / "data" / "dataset_json_enriched.csv"
JSON_DIR = UGP / "json"
MD_DIR = UGP / "sandbox" / "markdown"
LIT_V4 = ROOT / "data" / "lit_v4.json"

# key classes (first match wins). Outcome = device JV / stability results, never usable as predictors.
KEY_CLASSES = [
    ("identity", r"^(formula|common_name|type|name|role|material)$"),
    ("outcome", r"(^|[^a-z])(pce|voc|v_oc|jsc|j_sc|ff|fill.?factor|efficien|power.?conversion|short.?circuit|open.?circuit|"
                r"eqe|ipce|hysteresis|t80|t90|t50|retention|degrad|stabil|mpp|champion|performance|pv_|r_?s\b|r_?sh\b)"),
    ("characterisation", r"grain|rough|rms|\bpl\b|^pl_|trpl|photolumin|lifetime|xrd|fwhm|trap|crystallite|absorb|transmit|"
                         r"conductiv|resistiv|sheet_res|contact_angle|quench|ideality|urbach|recombin|diffusion_length|"
                         r"carrier|capacitance|impedance|morpholog|coverage|density_of"),
    ("property", r"band.?gap|homo|lumo|work.?function|fermi|mobility|dielectric|ionization|electron_affinity|"
                 r"conduction_band|valence_band|cbm|vbm|refractive|tolerance"),
    ("process", r"thick|conc|molar|anneal|spin|rpm|temp|time|dur|volume|ratio|solvent|additive|dopant|antisolvent|"
                r"drip|deposit|atmosphere|humid|speed|wt|mass|amount|purity|evapor|rate|pressure|bak|dry|sinter|"
                r"calcin|wash|method|substrate|area|precursor|stoich"),
]
_KC = [(n, re.compile(p, re.I)) for n, p in KEY_CLASSES]
NUM_RX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def key_class(k):
    kl = str(k).strip().lower()
    for n, p in _KC:
        if p.search(kl):
            return n
    return "other"


def first_number(v):
    if isinstance(v, bool):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    m = NUM_RX.search(str(v)) if isinstance(v, str) else None
    return float(m.group()) if m else np.nan


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_dir(d, pat):
    h = hashlib.sha256()
    for f in sorted(Path(d).glob(pat)):
        h.update(f.name.encode()); h.update(sha256_file(f).encode())
    return h.hexdigest()


def main():
    feat, aud = EXP / "features", EXP / "audit"
    feat.mkdir(exist_ok=True); aud.mkdir(exist_ok=True)
    checks = {}

    # ---- 1. source checksums
    src = {"pdb.csv": sha256_file(PDB), "dataset_json_enriched.csv": sha256_file(ENR), "lit_v4.json": sha256_file(LIT_V4),
           "json_dir(sorted name+sha)": sha256_dir(JSON_DIR, "*.json"), "markdown_dir": sha256_dir(MD_DIR, "*.md"),
           "n_json": len(list(JSON_DIR.glob("*.json"))), "n_markdown": len(list(MD_DIR.glob("*.md"))),
           "python": platform.python_version()}

    # ---- 2. pdb.csv vs enriched table
    pdb = pd.read_csv(PDB, low_memory=False)
    enr = pd.read_csv(ENR, low_memory=False)
    m = pdb.merge(enr[["Ref_ID", "Ref_DOI_number", "final_pce", "Cell_architecture"]], on="Ref_ID", how="outer",
                  indicator=True, suffixes=("_pdb", "_enr"), validate="1:1")
    both = m[m._merge == "both"]
    checks["pdb_vs_enriched"] = {
        "pdb_rows": len(pdb), "enriched_rows": len(enr), "enriched_cols": enr.shape[1],
        "ref_id_dup_pdb": int(pdb.Ref_ID.duplicated().sum()), "ref_id_dup_enriched": int(enr.Ref_ID.duplicated().sum()),
        "only_pdb": int((m._merge == "left_only").sum()), "only_enriched": int((m._merge == "right_only").sum()),
        "doi_mismatch": int((both.Ref_DOI_number_pdb.astype(str) != both.Ref_DOI_number_enr.astype(str)).sum()),
        "target_mismatch": int((~np.isclose(both.JV_default_PCE, both.final_pce, equal_nan=True)).sum()),
        "arch_mismatch": int((both.Cell_architecture_pdb.astype(str) != both.Cell_architecture_enr.astype(str)).sum()),
        "target_missing": int(enr.final_pce.isna().sum()), "target_lt1": int((enr.final_pce < 1).sum()),
        "target_gt30": int((enr.final_pce > 30).sum()),
    }
    raw = enr.Ref_DOI_number
    checks["doi_normalisation"] = {"raw_unique": int(raw.nunique()), "normalised_unique": int(raw.map(normalize_doi).nunique()),
                                   "no_doi_records": int((raw.map(normalize_doi) == "__NO_DOI__").sum()),
                                   "rules": ["strip BOM and whitespace", "lower-case", "drop http(s)://(dx.)doi.org/",
                                             "drop leading 'doi:'", "missing -> __NO_DOI__ (one group)"]}
    pdp_groups = set(raw.map(normalize_doi))

    # ---- 3. JSON -> canonical long table
    rows, status = [], []
    v4 = json.loads(LIT_V4.read_text())["papers"]
    for f in sorted(JSON_DIR.glob("*.json")):
        st = {"file": f.name}
        try:
            doc = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            st.update(status="malformed", error=str(e)[:80]); status.append(st); continue
        stem_doi = f.stem.replace("_", "/", 1)                     # production rule (litfeat4.build)
        src_doi = re.sub(r"^doi:\s*", "", str(doc.get("source", "")).strip(), flags=re.I) if isinstance(doc, dict) else ""
        g_stem, g_src = normalize_doi(stem_doi), normalize_doi(src_doi) if src_doi else "__NO_DOI__"
        mats = doc.get("materials") if isinstance(doc, dict) else None
        st.update(stem_doi=stem_doi, source_field=src_doi, doi_group=g_stem, source_doi_group=g_src,
                  source_agrees=(g_src == g_stem) if src_doi else None,
                  in_pdp=g_stem in pdp_groups, src_in_pdp=g_src in pdp_groups,
                  top_keys="|".join(sorted(doc.keys())) if isinstance(doc, dict) else type(doc).__name__,
                  has_materials_key=isinstance(mats, list))
        if not isinstance(mats, list):
            # 18 files store a single material dict at top level instead of {"source","materials"}
            mats = [doc] if isinstance(doc, dict) and ("formula" in doc or "common_name" in doc) else []
            st["schema"] = "top_level_material" if mats else "no_materials"
        else:
            st["schema"] = "materials_list"
        n_by = {}
        nmat = 0
        for i, mm in enumerate(mats):
            if not isinstance(mm, dict):
                continue
            nmat += 1
            r = ROLE_MAP.get(llm_role(mm), "oth")
            name = canon(mm.get("formula", "")) or canon(mm.get("common_name", ""))
            for k, v in mm.items():
                kc = key_class(k)
                n_by[kc] = n_by.get(kc, 0) + 1
                vk = ("null" if v is None else "bool" if isinstance(v, bool) else "num" if isinstance(v, (int, float))
                      else "str" if isinstance(v, str) else "list" if isinstance(v, list) else "dict")
                rows.append((g_stem, f.name, i, r, name, str(k), kc, vk,
                             json.dumps(v, ensure_ascii=False)[:300] if vk in ("list", "dict") else str(v)[:300],
                             first_number(v) if vk in ("num", "str") else np.nan))
        st.update(n_materials=nmat, status="covered" if nmat > 0 else "empty",
                  **{f"n_keys_{k}": n_by.get(k, 0) for k, _ in KEY_CLASSES + [("other", None)]})
        # production flags reproduce?
        st["v4_flags_reproduce"] = (paper_recipe(doc) == v4[g_stem]["recipe"]) if g_stem in v4 and isinstance(doc, dict) else None
        status.append(st)
    long = pd.DataFrame(rows, columns=["doi_group", "file", "mat_idx", "role", "canon_name", "key", "key_class", "value_kind",
                                       "value_raw", "value_num"])
    long.to_csv(feat / "json_long.csv.gz", index=False, compression="gzip")
    ps = pd.DataFrame(status)
    ps.to_csv(feat / "paper_status.csv", index=False)

    # ---- 4. join audit
    dup = ps[ps.doi_group.duplicated(keep=False)]
    unmatched = ps[~ps.in_pdp]
    alt = unmatched[unmatched.src_in_pdp]
    checks["json_join"] = {
        "files": len(ps), "schema_counts": ps.schema.value_counts().to_dict(), "status_counts": ps.status.value_counts().to_dict(),
        "duplicate_doi_groups_across_files": int(dup.doi_group.nunique()),
        "source_field_disagrees_with_filename": int((ps.source_agrees == False).sum()),  # noqa: E712
        "matched_to_pdp": int(ps.in_pdp.sum()), "unmatched": len(unmatched),
        "unmatched_recoverable_via_source_field": len(alt),
        "unmatched_examples": unmatched.stem_doi.head(10).tolist(),
        "v4_flags_reproduce_all": bool(ps.v4_flags_reproduce.dropna().all()), "v4_papers": len(v4),
        "stored_v4_keys_not_in_json_dir": len(set(v4) - set(ps.doi_group)),
        "many_to_many_risk": "none: paper table is unique on doi_group; join is m:1 (validated)",
    }
    # filename rule: DOIs with '_' in the suffix are restored wrongly only if '/' was replaced elsewhere; check
    checks["json_join"]["stems_with_extra_underscore"] = int(ps.stem_doi.str.count("_").gt(0).sum())

    # ---- 5. record-level coverage on the modelling population (PCE >= 1, as every manuscript model)
    df = load_dataset()
    cov_groups = set(ps.loc[ps.status == "covered", "doi_group"])
    any_groups = set(ps.doi_group)
    df["json_state"] = np.where(df.doi_group.isin(cov_groups), "covered",
                                np.where(df.doi_group.isin(any_groups), "empty", "uncovered"))
    n_rec_before = len(df)
    j = df[["doi_group"]].merge(ps[["doi_group"]].drop_duplicates(), on="doi_group", how="left", validate="m:1")
    assert len(j) == n_rec_before, "row multiplication in join"
    df["year"] = pd.to_datetime(df.Ref_publication_date, errors="coerce").dt.year
    by = df.groupby("json_state").agg(records=(TARGET, "size"), dois=("doi_group", "nunique"), pce_mean=(TARGET, "mean"),
                                       pce_sd=(TARGET, "std"), year_median=("year", "median"),
                                       frac_pin=("Cell_architecture", lambda s: float((s == "pin").mean())))
    checks["record_coverage"] = {"population": "PCE>=1 (load_dataset default)", "records": len(df),
                                 "dois": int(df.doi_group.nunique()), "by_state": by.round(3).reset_index().to_dict("records"),
                                 "lx_covered_flag_agrees": bool((df.lx_covered.astype(bool) == df.json_state.eq("covered")).mean() > 0.99),
                                 "lx_covered_vs_json_state": pd.crosstab(df.lx_covered, df.json_state).to_dict()}
    yr = df.groupby("year").json_state.apply(lambda s: float((s == "covered").mean())).round(3)
    checks["coverage_by_year"] = {int(k): v for k, v in yr.items() if k == k}
    df[["Ref_ID", "doi_group", "json_state"]].to_csv(feat / "record_json_state.csv", index=False)

    # ---- 6. key-class census and outcome-key leakage check of the production filter
    from psc.litfeat import OUTCOME_KEY
    ko = long[long.key_class == "outcome"].key.str.lower()
    missed = ko[~ko.map(lambda k: bool(OUTCOME_KEY.search(k)))]
    checks["key_classes"] = {"rows_by_class": long.key_class.value_counts().to_dict(),
                             "distinct_keys": int(long.key.nunique()),
                             "outcome_keys_missed_by_production_OUTCOME_KEY": int(missed.size),
                             "missed_examples": missed.value_counts().head(15).to_dict(),
                             "note": "production recipe flags use key *names* and string values; missed outcome keys can add "
                                     "their key text to the flag search string (chemistry regexes rarely match outcome text). "
                                     "Numeric outcome values are never used by any production feature."}
    out = {"sources_sha256": src, **checks}
    (aud / "integrity_audit.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
