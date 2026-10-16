"""Deterministic literature-feature builder (schema v2) from the raw LLM-extraction JSON.

The raw JSON (one file per paper, `materials` list of free-form dicts) was produced by the
original extraction run. This module re-reads it with role-aware, unit-normalising parsers and
builds *process/chemistry* features the PDP does not hold in this form (solvent system,
antisolvent, additive classes, spin conditions, precursor concentration, atmosphere, ...).

Rules
* Outcome keys (PCE/Voc/Jsc/FF/efficiency/stability/hysteresis ...) are dropped before any
  aggregation (`OUTCOME_KEY`): reported results of the paper never enter a feature.
* Values come only from the stored extraction; nothing is imputed here (missing stays NaN).
* Feature groups (prefix): lx_proc_*  fabrication process,  lx_chem_* chemistry / identities,
  lx_ctx_* measurement & environment context,  lx_char_* post-fabrication film characterisation
  (grain size, roughness ...; NOT available before a device is made),  lx_meta_* reporting
  completeness (kept separate so it can be ablated).
All features are paper-level (constant within a DOI); device-level matching needs full text.
"""
import json
import re
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "lx2.0"

OUTCOME_KEY = re.compile(
    r"(^|_)(pce|voc|jsc|ff|efficiency|eff|power_conversion|fill_factor|short_circuit|open_circuit|"
    r"stabili|hysteresis|t80|t90|retention|degrad|lifetime|eqe|ipce|rs|rsh|r_s|r_sh|mpp|performance)(_|$)")

TIME_S = {"s": 1, "sec": 1, "min": 60, "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600, "ms": 1e-3}
LEN_NM = {"nm": 1, "um": 1e3, "μm": 1e3, "µm": 1e3, "mm": 1e6, "å": 0.1, "a": 0.1}

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _scalar(v):
    """-> (number, unit-string) from str / number / {'value':..,'unit':..} / list; (None,'') if absent."""
    if v is None:
        return None, ""
    if isinstance(v, bool):
        return None, ""
    if isinstance(v, (int, float)):
        return float(v), ""
    if isinstance(v, dict):
        for k in ("value", "Value", "val", "Type"):
            if k in v:
                n, u = _scalar(v[k])
                return n, (str(v.get("unit", "")) or u)
        return None, ""
    if isinstance(v, list):
        return _scalar(v[0]) if v else (None, "")
    s = str(v).replace("−", "-").replace("–", "-").replace("×10", "e").replace(",", "")
    m = re.findall(_NUM, s)
    if not m:
        return None, ""
    nums = [float(x) for x in m[:2]]
    # "a-b" range -> mean (only if the dash really separates two numbers)
    if len(m) >= 2 and re.search(_NUM + r"\s*(?:-|to)\s*" + _NUM, s) and nums[1] > 0:
        n = (nums[0] + abs(nums[1])) / 2
    else:
        n = nums[0]
    return n, s[s.find(m[0]) + len(m[0]):].strip().lower()


def num(v, kind):
    """Parse v to a canonical unit. kind in temp_C | time_s | time_min | len_nm | rpm | conc_M |
    area_cm2 | pct | mw_cm2. Returns NaN if unparsable or physically implausible."""
    n, u = _scalar(v)
    if n is None:
        return np.nan
    u = u.replace(" ", "")
    if kind == "temp_C":
        if re.match(r"^(k)\b", u):
            n -= 273.15
        return n if -20 <= n <= 1000 else np.nan
    if kind in ("time_s", "time_min"):
        f = 1.0
        for k, mult in sorted(TIME_S.items(), key=lambda t: -len(t[0])):
            if re.match(rf"^{k}(?![a-z])", u):
                f = mult
                break
        else:
            f = 60.0 if kind == "time_min" else 1.0   # bare numbers: unit unknown -> reject below by plausibility
        sec = n * f
        out = sec if kind == "time_s" else sec / 60
        lim = (0.1, 1e5) if kind == "time_s" else (0.01, 5000)
        return out if lim[0] <= out <= lim[1] else np.nan
    if kind == "len_nm":
        f = 1.0
        for k, mult in sorted(LEN_NM.items(), key=lambda t: -len(t[0])):
            if u.startswith(k):
                f = mult
                break
        out = n * f
        return out if 0.5 <= out <= 5e6 else np.nan
    if kind == "layer_nm":      # layer thickness: bare numbers assumed nm; 2 nm - 3 um plausible
        f = 1.0
        for k, mult in sorted(LEN_NM.items(), key=lambda t: -len(t[0])):
            if u.startswith(k):
                f = mult
                break
        out = n * f
        return out if 2 <= out <= 3000 else np.nan
    if kind == "rpm":
        return n if 100 <= n <= 20000 else np.nan
    if kind == "conc_M":
        if re.match(r"^(wt|mg|g|%|vol|ml|l\b|µ|μ|u)", u):
            return np.nan
        if u.startswith(("mm", "mmol")):
            n /= 1000
        return n if 0.01 <= n <= 5 else np.nan
    if kind == "area_cm2":
        if u.startswith("mm"):
            n /= 100
        return n if 0.001 <= n <= 100 else np.nan
    if kind == "mw_cm2":
        return n if 1 <= n <= 5000 else np.nan
    if kind == "pct":
        return n if 0 <= n <= 100 else np.nan
    raise ValueError(kind)


# ---- role of a material record -------------------------------------------------------------
def role(m):
    t = f"{m.get('type', '')} {m.get('common_name', '')}".lower()
    if re.search(r"anti-?solvent|quench", t): return "antisolvent"
    if re.search(r"solvent", t) and not re.search(r"perovskite", t): return "solvent"
    if re.search(r"additive|dopant|passivat|interlayer|modif|spacer", t): return "additive"
    if re.search(r"\betl\b|electron.transport|electron.selective|\bblocking|compact|mesoporous|scaffold", t): return "etl"
    if re.search(r"\bhtl\b|\bhtm\b|hole.transport|hole.selective|hole.conduct", t): return "htl"
    if re.search(r"electrode|contact|\bau\b|\bag\b|cathode|anode", t) and not re.search(r"transparent", t): return "contact"
    if re.search(r"substrate|\bfto\b|\bito\b|glass|transparent", t): return "substrate"
    if re.search(r"perovskite|absorber|photoactive|active.layer|light.harvest", t) and not re.search(r"precursor", t): return "perov"
    if re.search(r"precursor", t): return "precursor"
    return "other"


def _first(m, pats):
    """First value whose (lower-cased) key matches any regex in pats and is not an outcome key."""
    for k, v in m.items():
        kl = str(k).lower()
        if OUTCOME_KEY.search(kl):
            continue
        if any(re.search(p, kl) for p in pats):
            yield kl, v


def _agg(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.median(vals)) if vals else np.nan


def _text(m):
    return " ".join(str(x) for x in [m.get("formula", ""), m.get("common_name", ""), m.get("type", "")]
                    + [v for k, v in m.items() if re.search(r"solvent|additive|dopant|antisolvent|atmosphere", str(k).lower())
                       and not isinstance(v, (dict, list))]).lower()


SOLVENTS = {"DMF": r"\bdmf\b|dimethylformamide|n,n-dimethylformamide", "DMSO": r"\bdmso\b|dimethyl ?sulfoxide",
            "GBL": r"\bgbl\b|butyrolactone", "NMP": r"\bnmp\b|methyl-?2-?pyrrolidone",
            "2ME": r"2-?me\b|methoxyethanol", "ACN": r"\bacn\b|acetonitrile",
            "CB": r"\bcb\b|chlorobenzene", "toluene": r"toluene", "ether": r"diethyl ether|ethyl ether|\bether\b",
            "EA": r"ethyl acetate|\bea\b", "IPA": r"\bipa\b|isopropan|2-propanol", "ethanol": r"ethanol|\betoh\b",
            "chloroform": r"chloroform", "water": r"\bwater\b|h2o|deionized|di water"}
ANTISOLVENTS = {k: SOLVENTS[k] for k in ("CB", "toluene", "ether", "EA", "chloroform", "IPA")}
ADDITIVES = {
    "LiTFSI": r"li-?tfsi|lithium bis|li salt|li-tfsi",
    "tBP": r"\btbp\b|t-bp|tert-?butyl ?pyridine|4-?tert",
    "FK209": r"fk-?209|co\(iii\)|cobalt\(iii\)|cobalt\s*\(iii\)",
    "ammonium_halide": r"\bpe(a)?i\b|phenethyl|\bbai\b|butylammonium|\boai\b|octylammonium|\bgai\b|guanidin|\bnh4(cl|i|br)|"
                       r"ammonium (iodide|chloride|bromide)|\bfai\b.*excess",
    "chloride_source": r"macl|pbcl2|nh4cl|methylammonium chloride|\bkcl\b|cscl",
    "alkali_halide": r"\bki\b|\bcsi\b|\brbi\b|\bnai\b|\bli[ib]r?\b|potassium iodide|cesium iodide|rubidium",
    "thiocyanate": r"scn|thiocyanate",
    "lewis_base": r"urea|thiourea|pyridine|\bdmso\b|\bphosph|\bsulfoxide",
    "polymer": r"\bpmma\b|\bpeg\b|\bpvp\b|\bptaa\b|poly\(|polymer|\bpvdf\b",
    "fullerene": r"c60|c70|pcbm|\bicba\b|fulleren",
    "f4tcnq": r"f4-?tcnq", "dio": r"diiodooctane|\bdio\b", "hydrohalic": r"\bhi\b|\bhcl\b|hydroiodic|hypophosph|hpa",
}
ATMOS = {"N2": r"\bn2\b|nitrogen|glove ?box|argon|\bar\b", "dry_air": r"dry air|\bcda\b",
         "ambient": r"ambient|\bair\b|humid|open air|room", "O2": r"\bo2\b|oxygen"}


def _flags(text, table, prefix):
    """0/1 identity flags; NaN for every flag when the source text is empty (not reported != absent)."""
    if not text.strip():
        return {f"{prefix}{k}": np.nan for k in table}
    return {f"{prefix}{k}": float(bool(re.search(p, text))) for k, p in table.items()}


def paper_features(doc):
    """doc = parsed JSON dict -> dict of lx_* features (NaN for not reported)."""
    mats = [m for m in doc.get("materials", []) if isinstance(m, dict)]
    out = {}
    byrole = {}
    for m in mats:
        byrole.setdefault(role(m), []).append(m)

    def collect(roles, pats, kind):
        vals = []
        for r in roles:
            for m in byrole.get(r, []):
                for _, v in _first(m, pats):
                    vals.append(num(v, kind))
        return _agg(vals)

    # --- fabrication process (lx_proc_*)
    P = ["perov", "precursor"]
    out["lx_proc_perov_spin_rpm"] = collect(P, [r"spin.*(speed|rpm)|^rpm$|rotation"], "rpm")
    out["lx_proc_perov_spin_s"] = collect(P, [r"spin.*time|spin.*duration"], "time_s")
    out["lx_proc_perov_anneal_C"] = collect(P, [r"anneal.*temp|temp.*anneal|annealing_temperature|heat.*temp|bak(e|ing).*temp"], "temp_C")
    out["lx_proc_perov_anneal_min"] = collect(P, [r"anneal.*time|time.*anneal|bak(e|ing).*time|heat.*time"], "time_min")
    out["lx_proc_perov_dry_C"] = collect(P, [r"dry.*temp|drying_temperature"], "temp_C")
    out["lx_proc_etl_spin_rpm"] = collect(["etl"], [r"spin.*(speed|rpm)|^rpm$"], "rpm")
    out["lx_proc_etl_anneal_C"] = collect(["etl"], [r"(anneal|sinter|heat|bak).*temp|temp.*(anneal|sinter)"], "temp_C")
    out["lx_proc_etl_anneal_min"] = collect(["etl"], [r"(anneal|sinter|heat|bak).*time"], "time_min")
    out["lx_proc_htl_spin_rpm"] = collect(["htl"], [r"spin.*(speed|rpm)|^rpm$"], "rpm")
    out["lx_proc_htl_anneal_C"] = collect(["htl"], [r"(anneal|heat|bak|dry).*temp"], "temp_C")
    out["lx_proc_htl_thickness_nm"] = collect(["htl"], [r"thickness"], "layer_nm")
    out["lx_proc_etl_thickness_nm"] = collect(["etl"], [r"thickness"], "layer_nm")
    out["lx_proc_substrate_thickness_nm"] = collect(["substrate"], [r"thickness"], "layer_nm")
    out["lx_proc_n_spin_steps"] = float(max(
        [sum(1 for k in m if re.search(r"spin.*(speed|rpm)|step", str(k).lower()) and not OUTCOME_KEY.search(str(k).lower()))
         for m in byrole.get("perov", []) + byrole.get("precursor", [])] or [0])) or np.nan
    out["lx_proc_precursor_conc_M"] = collect(P, [r"precursor_conc|^concentration|molarity|conc_?m\b|concentration_m"], "conc_M")
    out["lx_proc_pbi2_excess_pct"] = collect(P + ["additive"], [r"excess"], "pct")
    out["lx_proc_perov_thickness_nm"] = collect(["perov"], [r"thickness"], "layer_nm")
    # --- chemistry (lx_chem_*)
    txt_all = " ".join(_text(m) for m in mats)
    solv = " ".join(_text(m) for m in byrole.get("solvent", []) + byrole.get("perov", []) + byrole.get("precursor", []))
    anti = " ".join(_text(m) for m in byrole.get("antisolvent", [])) + " " + " ".join(
        str(v) for m in mats for k, v in m.items() if re.search(r"anti.?solvent", str(k).lower()))
    addt = " ".join(_text(m) for m in byrole.get("additive", []))
    out.update(_flags(solv, SOLVENTS, "lx_chem_solv_"))
    out["lx_chem_n_solvents"] = float(sum(v for k, v in out.items() if k.startswith("lx_chem_solv_") and k != "lx_chem_solv_water")) or np.nan
    anti_any = bool(anti.strip()) or bool(re.search(r"anti-?solvent|quench", txt_all))
    out.update(_flags(anti, ANTISOLVENTS, "lx_chem_anti_"))
    out["lx_chem_antisolvent_any"] = float(anti_any)
    out.update(_flags(addt + " " + txt_all, ADDITIVES, "lx_chem_add_"))
    out["lx_chem_n_additive_records"] = float(len(byrole.get("additive", [])))
    out["lx_chem_n_precursor_records"] = float(len(byrole.get("precursor", [])))
    # --- measurement / environment context (lx_ctx_*)
    out["lx_ctx_humidity_pct"] = _agg([num(v, "pct") for m in mats for k, v in m.items()
                                       if re.search(r"humid", str(k).lower()) and not re.search(r"stabil|aging|storage", str(k).lower())])
    atm = " ".join(str(v) for m in mats for k, v in m.items() if re.search(r"atmosph|environment|glove", str(k).lower()))
    out.update(_flags(atm, ATMOS, "lx_ctx_atm_"))
    out["lx_ctx_light_mw_cm2"] = _agg([num(v, "mw_cm2") for m in mats for k, v in m.items()
                                       if re.search(r"light_intensity|illumination|irradi", str(k).lower())])
    out["lx_ctx_active_area_cm2"] = _agg([num(v, "area_cm2") for m in mats for k, v in m.items()
                                          if re.search(r"active_area|device_area|cell_area|^area", str(k).lower())])
    # --- film characterisation, not known before the device exists (lx_char_*)
    out["lx_char_grain_nm"] = _agg([num(v, "len_nm") for m in byrole.get("perov", []) for _, v in _first(m, [r"grain"])])
    out["lx_char_rms_nm"] = _agg([num(v, "len_nm") for m in mats for _, v in _first(m, [r"rms|roughness"])])
    out["lx_char_substrate_sheet_res"] = _agg([_scalar(v)[0] for m in byrole.get("substrate", []) for _, v in _first(m, [r"sheet_resistance"])])
    # --- reporting completeness (lx_meta_*)
    keys = [str(k).lower() for m in mats for k in m if not OUTCOME_KEY.search(str(k).lower())]
    out["lx_meta_n_materials"] = float(len(mats))
    out["lx_meta_n_fields"] = float(len(keys))
    out["lx_meta_n_distinct_fields"] = float(len(set(keys)))
    out["lx_meta_n_roles"] = float(len(byrole))
    return out


GROUPS = ("lx_proc_", "lx_chem_", "lx_ctx_", "lx_char_", "lx_meta_")


def build_table(json_dir, normalize_doi):
    """Parse all JSON files -> DataFrame indexed by normalised DOI (one row per paper)."""
    import pandas as pd
    rows = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        doi = f.stem.replace("_", "/", 1)
        if "10." not in doi:
            src = re.sub(r"^doi:\s*", "", str(doc.get("source", "")).strip(), flags=re.I)
            doi = src
        rows[normalize_doi(doi)] = {"json_file": f.name, **paper_features(doc)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "doi_group"
    return df
