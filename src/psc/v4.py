"""lx4 feature blocks on top of the fixed t2 matrix; label-free construction, NaN kept native (CatBoost)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT
from .litfeat4 import canon, NUM_KINDS

LIT_V4 = ROOT / "data" / "lit_v4.json"
DF_MIN_TOK = 25          # token must occur in >= 25 covered papers (label-free)
COV_MIN_NUM = 0.03       # numeric column must be reported in >= 3% of covered papers


def v4_partition(doi, dev_frac=0.65):
    """Fresh DOI hash split (own salt): 'v4dev' for all development, 'v4final' reserved for one final scoring."""
    h = int(hashlib.sha256(("psc-v4|" + doi).encode()).hexdigest(), 16) % 1000
    return "v4dev" if h < dev_frac * 1000 else "v4final"


_CACHE = {}


def papers():
    if "p" not in _CACHE:
        _CACHE["p"] = json.loads(LIT_V4.read_text())["papers"]
    return _CACHE["p"]


def vocab():
    P = papers()
    c = {}
    for p in P.values():
        for t in p["tokens"]:
            c[t] = c.get(t, 0) + 1
    toks = sorted(t for t, n in c.items() if n >= DF_MIN_TOK)
    cols = {}
    n = len(P)
    for p in P.values():
        for k in p["numeric"]:
            cols[k] = cols.get(k, 0) + 1
    nums = sorted(k for k, v in cols.items() if v / n >= COV_MIN_NUM)
    return toks, nums


def paper_table(kind):
    """Paper-level frame indexed by doi_group."""
    P = papers(); toks, nums = vocab()
    if kind == "tok":
        rows = {d: {f"tk_{t}": float(t in set(p["tokens"])) for t in toks} for d, p in P.items()}
    elif kind == "rec":
        rows = {d: {f"rc_{k}": v for k, v in p["recipe"].items() if k != "n_materials" and not k.startswith("s_")} for d, p in P.items()}
    elif kind == "rec2":   # scoped perovskite flags replace the unscoped ones; HTL/ETL/context flags unchanged
        rows = {}
        for d, p in P.items():
            r = {k: v for k, v in p["recipe"].items() if k != "n_materials"}
            rows[d] = {f"rc_{k}": (r["s_" + k] if "s_" + k in r else v) for k, v in r.items() if not k.startswith("s_")}
    elif kind == "num":
        rows = {d: {f"nm_{k}": p["numeric"].get(k, np.nan) for k in nums} for d, p in P.items()}
    else:
        raise ValueError(kind)
    return pd.DataFrame.from_dict(rows, orient="index")


def attach(df, T):
    """Left-join paper-level table T onto records (NaN for uncovered papers)."""
    return T.reindex(df["doi_group"].values).set_index(df.index)


DEV_FILL = {"Perovskite_thickness_nm": ("perov", "thick_nm"), "ETL_thickness_nm": ("etl", "thick_nm"),
            "HTL_thickness_nm": ("htl", "thick_nm"), "Backcontact_thickness_nm": ("ctc", "thick_nm"),
            "Perov_anneal_temp_C": ("perov", "anneal_C"), "Perov_anneal_time_min": ("perov", "anneal_min")}


def _lay(layers, role, name):
    sel = [l for l in layers if l["role"] == role and name and l["name"] == name]
    return sel


def _med(sel, kind):
    v = [x for l in sel for x in l["nums"].get(kind, [])]
    return float(np.median(v)) if v else np.nan


def device_block(df):
    """Per-record features obtained by matching the PDP ETL / HTL / back contact to JSON layer records."""
    P = papers()
    etl = df["ETL"].map(lambda s: canon(s) if isinstance(s, str) else None).values
    htl = df["HTL"].map(lambda s: canon(s) if isinstance(s, str) else None).values
    bc = df["Backcontact_stack_sequence"].map(lambda s: canon(str(s).split("|")[0].replace("|", "")) if isinstance(s, str) else None).values \
        if "Backcontact_stack_sequence" in df else [None] * len(df)
    out = {k: np.full(len(df), np.nan) for k in
           ["dv_etl_found", "dv_htl_found", "dv_etl_thick", "dv_etl_anneal_C", "dv_etl_anneal_min", "dv_etl_conc",
            "dv_htl_thick", "dv_htl_conc", "dv_htl_anneal_C", "dv_ctc_thick", "dv_n_etl_layers", "dv_n_htl_layers"]}
    docs = df["doi_group"].values
    for i in range(len(df)):
        p = P.get(docs[i])
        if p is None:
            continue
        L = p["layers"]
        e = _lay(L, "etl", etl[i]); h = _lay(L, "htl", htl[i]); c = _lay(L, "ctc", bc[i])
        out["dv_etl_found"][i] = float(bool(e)); out["dv_htl_found"][i] = float(bool(h))
        out["dv_n_etl_layers"][i] = float(len([l for l in L if l["role"] == "etl"]))
        out["dv_n_htl_layers"][i] = float(len([l for l in L if l["role"] == "htl"]))
        out["dv_etl_thick"][i] = _med(e, "thick_nm"); out["dv_etl_anneal_C"][i] = _med(e, "anneal_C")
        out["dv_etl_anneal_min"][i] = _med(e, "anneal_min"); out["dv_etl_conc"][i] = _med(e, "conc_M")
        out["dv_htl_thick"][i] = _med(h, "thick_nm"); out["dv_htl_conc"][i] = _med(h, "conc_mgml")
        out["dv_htl_anneal_C"][i] = _med(h, "anneal_C"); out["dv_ctc_thick"][i] = _med(c, "thick_nm")
    return pd.DataFrame(out, index=df.index)


def fill_block(df, Tn):
    """PDP value where reported, else the paper-level LLM value (`provenance` flag kept as a separate column)."""
    out = {}
    for col, (ro, kind) in DEV_FILL.items():
        k = f"nm_{ro}_{kind}_med"
        if col not in df or k not in Tn:
            continue
        pdp = pd.to_numeric(df[col], errors="coerce"); llm = Tn[k]
        out[f"fl_{col}"] = pdp.where(pdp.notna(), llm)
        out[f"fl_src_{col}"] = np.where(pdp.notna(), 0.0, np.where(llm.notna(), 1.0, np.nan))
    return pd.DataFrame(out, index=df.index)


REC_GROUPS = {
    "P": ["cl_source", "acetate", "scn", "snf2", "ava", "hx_acid", "alkali", "cs", "pea_family", "lewis_base", "water_add", "polymer", "carbon", "dio"],
    "H": ["htl_tbp", "htl_litfsi", "htl_fk", "htl_f4tcnq", "htl_cu", "htl_borate"],
    "E": ["etl_ticl4", "etl_dopant"],
    "R": ["antisolvent", "vacuum", "two_step", "hot_cast", "solvent_anneal", "n2_box", "ambient"],
    "C": ["flexible", "tandem", "lead_free", "inorganic", "mesoscopic"],
    "U": ["undoped", "interface_pass"],
}


def rec_group_cols(groups):
    return [f"rc_{c}" for g in groups for c in REC_GROUPS[g]]
