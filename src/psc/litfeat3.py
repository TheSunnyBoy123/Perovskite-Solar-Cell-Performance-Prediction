"""Quantity-level literature features (schema v3) from the cached LLM-extraction JSON.

Builds on `litfeat` (v2 flags / process numbers) and adds what v2 collapsed into 0/1 flags:
per-species precursor concentrations and stoichiometric ratios, additive / dopant amounts,
solvent ratios, antisolvent volume, HTL dopant amounts, back-contact deposition conditions, and
chemistry-level categorical descriptors. Rules as in `litfeat`: outcome keys are dropped,
values are only taken from the stored extraction, units are normalised, not-reported stays NaN
(never 0). No reporting-completeness counts are produced here.

Groups (second token of the column name, so `features._lx_cols` can select them):
  lx_qty_*  normalised quantities (M, uL, mg/mL, nm, ratios)
  lx_cat_*  0/1 chemistry descriptors derived from species identity (NaN if no source text)
"""
import re
from pathlib import Path
import json

import numpy as np

from .litfeat import OUTCOME_KEY, _scalar, role, _text, ADDITIVES, SOLVENTS

SCHEMA_VERSION = "lx3.0"
SPECIES = {"PbI2": r"pbi2", "PbBr2": r"pbbr2", "FAI": r"\bfai\b|formamidinium iodide", "MAI": r"\bmai\b|methylammonium iodide|methyl ?ammonium iodide",
           "MABr": r"\bmabr\b|methylammonium bromide", "CsI": r"\bcsi\b|cesium iodide|caesium iodide"}


def _vals(v):
    """Flatten a stored value (scalar / {'value','unit'} / list of those) to [(number, unit)]."""
    if isinstance(v, list):
        return [x for e in v for x in _vals(e)]
    n, u = _scalar(v)
    return [(n, u)] if n is not None else []


def conc_M(v):
    """Concentration -> mol/L; NaN for mass/percent units (kept in mg/mL by conc_mgml)."""
    out = []
    for n, u in _vals(v):
        u = u.replace(" ", "")
        if u.startswith(("mm", "mmol")) and not u.startswith("mmhg"):
            n /= 1000
        elif u.startswith(("µ", "μ", "u")) and "m" in u[:3]:
            n /= 1e6
        elif u.startswith(("m", "mol", "")) and not re.match(r"^(mg|wt|%|vol|g|ml|l)", u):
            pass
        else:
            continue
        if 0.005 <= n <= 6:
            out.append(n)
    return float(np.median(out)) if out else np.nan


def conc_mgml(v):
    out = []
    for n, u in _vals(v):
        u = u.replace(" ", "")
        if u.startswith("mg/ml") or u.startswith("mgml"):
            pass
        elif u.startswith("g/l"):
            pass
        elif u.startswith("µg") or u.startswith("μg"):
            n /= 1000
        else:
            continue
        if 0.01 <= n <= 500:
            out.append(n)
    return float(np.median(out)) if out else np.nan


def volume_uL(v):
    out = []
    for n, u in _vals(v):
        u = u.replace(" ", "").replace("μ", "u").replace("µ", "u")
        if u.startswith("ul") or u.startswith("uL".lower()):
            pass
        elif u.startswith("ml"):
            n *= 1000
        else:
            continue
        if 0.5 <= n <= 20000:
            out.append(n)
    return float(np.median(out)) if out else np.nan


def thickness_nm(v):
    out = []
    for n, u in _vals(v):
        u = u.replace(" ", "").replace("μ", "u").replace("µ", "u")
        if u.startswith("um"):
            n *= 1000
        elif u.startswith("mm"):
            n *= 1e6
        elif u.startswith("a") and not u.startswith("au"):
            n /= 10
        if 1 <= n <= 5000:
            out.append(n)
    return float(np.median(out)) if out else np.nan


def ratio_val(v):
    """'4:1' style or bare numeric molar ratio -> first/second (float)."""
    out = []
    for e in (v if isinstance(v, list) else [v]):
        s = json.dumps(e) if isinstance(e, (dict, list)) else str(e)
        m = re.search(r"(\d*\.?\d+)\s*[:/]\s*(\d*\.?\d+)", s)
        if m and float(m.group(2)) > 0:
            out.append(float(m.group(1)) / float(m.group(2)))
        else:
            n, _ = _scalar(e)
            if n is not None:
                out.append(n)
    out = [x for x in out if 0 < x < 100]
    return float(np.median(out)) if out else np.nan


def _items(m, pats):
    for k, v in m.items():
        kl = str(k).lower()
        if OUTCOME_KEY.search(kl):
            continue
        if any(re.search(p, kl) for p in pats):
            yield kl, v


def _med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else np.nan


def paper_features(doc):
    mats = [m for m in doc.get("materials", []) if isinstance(m, dict)]
    byrole = {}
    for m in mats:
        byrole.setdefault(role(m), []).append(m)
    out = {}
    # --- per-species precursor concentration (M): explicit "<X>_concentration" keys or a record of species X
    for name, pat in SPECIES.items():
        vals = []
        for m in mats:
            ident = f"{m.get('formula', '')} {m.get('common_name', '')}".lower()
            if re.search(pat, ident) and role(m) in ("precursor", "perov", "other", "additive"):
                vals += [conc_M(v) for _, v in _items(m, [r"conc", r"molar"])]
            for _, v in _items(m, [rf"^(?:{pat})_(?:conc|molar)"]):
                vals.append(conc_M(v))
        out[f"lx_qty_conc_{name}"] = _med(vals)
    pb = np.nansum([out["lx_qty_conc_PbI2"], out["lx_qty_conc_PbBr2"]]) if (out["lx_qty_conc_PbI2"] == out["lx_qty_conc_PbI2"] or out["lx_qty_conc_PbBr2"] == out["lx_qty_conc_PbBr2"]) else np.nan
    org = np.nansum([out["lx_qty_conc_FAI"], out["lx_qty_conc_MAI"], out["lx_qty_conc_MABr"]]) if any(out[f"lx_qty_conc_{k}"] == out[f"lx_qty_conc_{k}"] for k in ("FAI", "MAI", "MABr")) else np.nan
    out["lx_qty_pb_total_M"] = pb
    out["lx_qty_org_over_pb"] = org / pb if pb == pb and org == org and pb > 0 else np.nan
    out["lx_qty_pbbr_frac"] = (out["lx_qty_conc_PbBr2"] / pb if pb == pb and pb > 0 and out["lx_qty_conc_PbBr2"] == out["lx_qty_conc_PbBr2"]
                               else (0.0 if out["lx_qty_conc_PbI2"] == out["lx_qty_conc_PbI2"] else np.nan))
    # --- solvents
    out["lx_qty_dmso_pb_ratio"] = _med([ratio_val(v) for m in mats for _, v in _items(m, [r"dmso.*(pb|ratio)|(pb).*dmso"])
                                        if not isinstance(v, dict) or True])
    out["lx_qty_solvent_vol_ratio"] = _med([ratio_val(v) for m in byrole.get("solvent", []) for _, v in _items(m, [r"volume_ratio|^ratio"])])
    out["lx_qty_solvent_vol_uL"] = _med([volume_uL(v) for m in byrole.get("solvent", []) for _, v in _items(m, [r"^volume"])])
    # --- antisolvent
    out["lx_qty_anti_vol_uL"] = _med([volume_uL(v) for m in byrole.get("antisolvent", []) for _, v in _items(m, [r"volume"])])
    # --- additives / dopants (passivation agents) : amount, not just identity
    ad = byrole.get("additive", [])
    out["lx_qty_add_conc_M"] = _med([conc_M(v) for m in ad for _, v in _items(m, [r"conc", r"molar"])])
    out["lx_qty_add_conc_mgml"] = _med([conc_mgml(v) for m in ad for _, v in _items(m, [r"conc"])])
    out["lx_qty_add_vol_uL"] = _med([volume_uL(v) for m in ad for _, v in _items(m, [r"volume"])])
    out["lx_qty_n_add_species"] = float(len({str(m.get("formula", "")).lower() for m in ad if m.get("formula")})) or np.nan
    # HTL doping package amounts (LiTFSI, tBP, FK209) wherever they are recorded
    for name, pat in (("litfsi", ADDITIVES["LiTFSI"]), ("tbp", ADDITIVES["tBP"]), ("fk209", ADDITIVES["FK209"])):
        sel = [m for m in mats if re.search(pat, f"{m.get('formula', '')} {m.get('common_name', '')}".lower())]
        out[f"lx_qty_dop_{name}_vol_uL"] = _med([volume_uL(v) for m in sel for _, v in _items(m, [r"volume"])])
        out[f"lx_qty_dop_{name}_conc"] = _med([conc_M(v) for m in sel for _, v in _items(m, [r"conc"])])
    out["lx_qty_htl_conc_mgml"] = _med([conc_mgml(v) for m in byrole.get("htl", []) for _, v in _items(m, [r"conc"])])
    out["lx_qty_htl_conc_M"] = _med([conc_M(v) for m in byrole.get("htl", []) for _, v in _items(m, [r"conc"])])
    out["lx_qty_etl_conc_M"] = _med([conc_M(v) for m in byrole.get("etl", []) for _, v in _items(m, [r"conc"])])
    out["lx_qty_etl_conc_mgml"] = _med([conc_mgml(v) for m in byrole.get("etl", []) for _, v in _items(m, [r"conc"])])
    # --- back contact deposition
    ct = byrole.get("contact", [])
    out["lx_qty_contact_thick_nm"] = _med([thickness_nm(v) for m in ct for _, v in _items(m, [r"thick"])])
    out["lx_qty_contact_rate"] = _med([_scalar(v)[0] for m in ct for _, v in _items(m, [r"(deposition|evaporation).*rate"])])
    p = [_scalar(v)[0] for m in ct for _, v in _items(m, [r"(deposition|evaporation).*pressure"])]
    p = [np.log10(x) for x in p if x and x > 0]
    out["lx_qty_contact_log_pressure"] = _med(p)
    # --- chemistry descriptors from species identity (NaN when there is no source text for that role)
    solv = " ".join(_text(m) for m in byrole.get("solvent", []) + byrole.get("perov", []) + byrole.get("precursor", []))
    has = lambda k: bool(re.search(SOLVENTS[k], solv))
    if any(has(k) for k in ("DMF", "DMSO", "GBL")):
        d, s, g = has("DMF"), has("DMSO"), has("GBL")
        out["lx_cat_solv_dmf_dmso"] = float(d and s)
        out["lx_cat_solv_dmf_only"] = float(d and not s and not g)
        out["lx_cat_solv_gbl"] = float(g)
        out["lx_cat_solv_dmso_only"] = float(s and not d and not g)
    else:
        out.update({"lx_cat_solv_dmf_dmso": np.nan, "lx_cat_solv_dmf_only": np.nan, "lx_cat_solv_gbl": np.nan, "lx_cat_solv_dmso_only": np.nan})
    addt = " ".join(_text(m) for m in ad)
    if addt.strip():
        classes = [k for k, pat in ADDITIVES.items() if re.search(pat, addt)]
        out["lx_cat_n_additive_classes"] = float(len(classes))
        out["lx_cat_has_2d_spacer"] = float(bool(re.search(r"\bpe(a)?i\b|phenethyl|\bbai\b|butylammonium|\boai\b|octylammonium|\bnh4|ammonium", addt)))
        out["lx_cat_has_interlayer"] = float(any(re.search(r"interlayer|passivat|modif", f"{m.get('type', '')}".lower()) for m in ad))
    else:
        out.update({"lx_cat_n_additive_classes": np.nan, "lx_cat_has_2d_spacer": np.nan, "lx_cat_has_interlayer": np.nan})
    htl = " ".join(_text(m) for m in byrole.get("htl", []) + [m for m in mats if re.search(ADDITIVES["LiTFSI"] + "|" + ADDITIVES["tBP"], f"{m.get('formula', '')} {m.get('common_name', '')}".lower())])
    if re.search(ADDITIVES["LiTFSI"] + "|" + ADDITIVES["tBP"] + "|" + ADDITIVES["FK209"], htl):   # a doping package is documented
        out["lx_cat_htl_litfsi_tbp"] = float(bool(re.search(ADDITIVES["LiTFSI"], htl)) and bool(re.search(ADDITIVES["tBP"], htl)))
        out["lx_cat_htl_cobalt"] = float(bool(re.search(ADDITIVES["FK209"], htl)))
    else:
        out["lx_cat_htl_litfsi_tbp"] = out["lx_cat_htl_cobalt"] = np.nan
    return out


def build_table(json_dir, normalize_doi):
    import pandas as pd
    rows = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        doi = f.stem.replace("_", "/", 1)
        if "10." not in doi:
            doi = re.sub(r"^doi:\s*", "", str(doc.get("source", "")).strip(), flags=re.I)
        rows[normalize_doi(doi)] = paper_features(doc)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "doi_group"
    return df
