"""Schema lx4.0: richer, label-free representation of the cached LLM-extraction JSON.

Three paper-level views (plus device-level matching, see `v4.py`), all built only from the stored
extraction. Outcome keys (PCE/Voc/Jsc/FF/stability ...) are dropped exactly as in `litfeat`.
Nothing is imputed here: not reported stays NaN.

  tokens   role-tagged canonical chemistry identities (`add:litfsi`, `solv:dmso`, `anti:cb`, ...),
           synonym-merged through ALIASES. Presence is label-free; vocabulary selection by document
           frequency happens in `v4.py`.
  numeric  role x quantity-kind aggregates (median / min / max / n values) in canonical units,
           so multi-value (multi-stage / multi-condition) information is not collapsed to one median.
  layers   per-material records (role, canonical name, thickness, anneal T/t, concentration,
           spin rpm, ...) used for matching to the PDP ETL / HTL / back-contact of each device.
"""
import json
import re
from pathlib import Path

import numpy as np

from .litfeat import OUTCOME_KEY, _scalar, num, role
from .litfeat3 import conc_M, conc_mgml, volume_uL

SCHEMA_VERSION = "lx4.0"

# canonical name <- regex on the lower-cased "formula | common_name" string (first match wins)
ALIASES = [
    ("spiro", r"spiro|omeTAD|ometad|meotad|2,2',7,7'-tetrakis"),
    ("ptaa", r"\bptaa\b|polytriarylamine|poly\[bis\(4-phenyl\)"),
    ("pedot", r"pedot|pss"),
    ("p3ht", r"p3ht|poly\(3-hexylthiophene"),
    ("nio", r"\bnio|nickel oxide|ni[ -]?ox"),
    ("cuscn", r"cuscn|copper thiocyanate"),
    ("cui", r"\bcui\b|copper iodide"),
    ("moo3", r"moo3|moox|molybdenum"),
    ("polytpd", r"poly-?tpd"),
    ("pcbm", r"pcbm|pc61bm|pc71bm|pc60bm|phenyl-c"),
    ("icba", r"icba"),
    ("c60", r"\bc60\b|fullerene"),
    ("bcp", r"\bbcp\b|bathocuproine"),
    ("tio2", r"tio2|tiox|titan|\btio\b"),
    ("sno2", r"sno2|snox|tin oxide|tin\(iv\) oxide|snO2"),
    ("zno", r"\bzno|zinc oxide"),
    ("al2o3", r"al2o3|alumin"),
    ("zro2", r"zro2|zircon"),
    ("litfsi", r"li-?tfsi|lithium bis|li salt|lithium bis\(trifluoromethane"),
    ("tbp", r"\btbp\b|t-bp|tert-?butyl ?pyridine|4-?tert|c10h15n"),
    ("fk209", r"fk-?209|co\(iii\)|cobalt"),
    ("pea", r"\bpe(a)?i\b|phenethyl|phenylethyl|pea\b"),
    ("ba", r"\bbai\b|butylammonium|\bba\b"),
    ("oa", r"\boai\b|octylammonium"),
    ("gua", r"\bgai\b|guanidin|\bgacl|\bgasc"),
    ("avai", r"5-?ava|aminovaleric|valeric"),
    ("nh4", r"nh4|ammonium (chloride|iodide)"),
    ("macl", r"macl|methylammonium chloride|ch3nh3cl"),
    ("pbcl2", r"pbcl2|lead chloride"),
    ("pbscn", r"pb\(scn\)2|lead thiocyanate|thiocyanate|\bscn"),
    ("pbi2", r"pbi2|lead iodide|lead\(ii\) iodide"),
    ("pbbr2", r"pbbr2"),
    ("mai", r"\bmai\b|ch3nh3i\b|methylammonium iodide|methylamine hydroiodide|methyl ?ammonium iodide"),
    ("mabr", r"mabr|ch3nh3br|methylammonium bromide"),
    ("fai", r"\bfai\b|hc\(nh2\)2i|formamidinium iodide"),
    ("csi", r"\bcsi\b|cesium iodide|caesium iodide"),
    ("csbr", r"csbr"),
    ("rbi", r"\brbi\b|rubidium"),
    ("ki", r"\bki\b|potassium"),
    ("snf2", r"snf2|tin\(ii\) fluoride"),
    ("snI2", r"sni2|snii2"),
    ("urea", r"urea"),
    ("pmma", r"pmma|polymethyl"),
    ("peg", r"\bpeg\b|polyethylene glycol"),
    ("lif", r"\blif\b|lithium fluoride"),
    ("f4tcnq", r"f4-?tcnq"),
    ("dmf", r"\bdmf\b|dimethylformamide"),
    ("dmso", r"\bdmso\b|dimethyl ?sulfoxide"),
    ("gbl", r"\bgbl\b|butyrolactone"),
    ("nmp", r"\bnmp\b|pyrrolidone"),
    ("cb", r"\bcb\b|chlorobenzene|c6h5cl|\bcbz\b"),
    ("dcb", r"dichlorobenzene|\bdcb\b"),
    ("toluene", r"toluene|c7h8\b|\btol\b"),
    ("ether", r"diethyl ether|ethoxyethane|\bdee\b|\bether\b|c4h10o"),
    ("ea", r"ethyl acetate|\bea\b"),
    ("chloroform", r"chloroform|chcl3"),
    ("ipa", r"\bipa\b|isopropan|2-propanol|c3h8o"),
    ("ethanol", r"ethanol|\betoh\b"),
    ("acn", r"acetonitrile|\bacn\b"),
    ("anisole", r"anisole"),
    ("water", r"\bwater\b|h2o|deionized"),
    ("au", r"^au\b|\bgold\b"),
    ("ag", r"^ag\b|\bsilver\b"),
    ("al", r"^al\b|\baluminum\b|aluminium"),
    ("carbon", r"^c$|carbon|graphite|cnt"),
    ("cu", r"^cu\b|\bcopper\b"),
    ("fto", r"\bfto\b|sno2:f|snO2:f|f:sno2|fluorine"),
    ("ito", r"\bito\b|in2o3|indium tin"),
]
_ALIAS = [(n, re.compile(p, re.I)) for n, p in ALIASES]


def canon(s):
    s = str(s).strip().lower()
    if not s or s in ("...", "<chemical formula>"):
        return None
    for n, p in _ALIAS:
        if p.search(s):
            return n
    return None


ROLE_MAP = {"antisolvent": "anti", "solvent": "solv", "additive": "add", "etl": "etl", "htl": "htl",
            "contact": "ctc", "substrate": "sub", "perov": "perov", "precursor": "prec", "other": "oth"}

KIND_KEYS = [
    ("spin_rpm", r"spin.*(speed|rpm)|^rpm$|rotation|spin_coating_speed"),
    ("spin_s", r"spin.*(time|dur)"),
    ("anneal_C", r"(anneal|bak|heat|sinter|dry|calcin|temperature).*temp|temp.*(anneal|bak|sinter|dry)|^annealing_temperature|^temperature"),
    ("anneal_min", r"(anneal|bak|heat|sinter|dry|calcin).*time|^annealing_time|^time$"),
    ("thick_nm", r"thick"),
    ("conc", r"conc|molar"),
    ("vol_uL", r"volume"),
    ("ratio", r"ratio"),
]
_KIND = [(k, re.compile(p)) for k, p in KIND_KEYS]


def _key_kind(kl):
    for k, p in _KIND:
        if p.search(kl):
            return k
    return None


def _flat(v):
    if isinstance(v, list):
        return [x for e in v for x in _flat(e)]
    return [v]


def _parse(kind, v):
    """-> list of canonical-unit floats for one stored value (list-aware)."""
    out = []
    for e in _flat(v):
        if kind == "spin_rpm":
            x = num(e, "rpm")
        elif kind == "spin_s":
            x = num(e, "time_s")
        elif kind == "anneal_C":
            x = num(e, "temp_C")
        elif kind == "anneal_min":
            x = num(e, "time_min")
        elif kind == "thick_nm":
            x = num(e, "layer_nm")
        elif kind == "conc":
            x = conc_M(e)
            if x != x:
                x = conc_mgml(e)
                x = np.log10(x) + 10 if x == x else np.nan  # mg/mL kept on a separate offset-log scale; split below
        elif kind == "vol_uL":
            x = volume_uL(e)
        else:
            x = np.nan
        if x == x:
            out.append(float(x))
    return out


def material_records(doc):
    """One dict per material: role tag, canonical name(s) and parsed numeric kinds (outcome keys dropped)."""
    recs = []
    for m in doc.get("materials", []):
        if not isinstance(m, dict):
            continue
        r = ROLE_MAP.get(role(m), "oth")
        names = [canon(m.get("formula", "")), canon(m.get("common_name", ""))]
        name = next((n for n in names if n), None)
        txt = f"{m.get('formula', '')} | {m.get('common_name', '')}".lower()
        nums = {}
        for k, v in m.items():
            kl = str(k).lower()
            if OUTCOME_KEY.search(kl) or kl in ("formula", "common_name", "type"):
                continue
            kind = _key_kind(kl)
            if kind is None:
                continue
            if kind == "conc":   # molar vs mass concentration kept apart
                molar = [x for x in (conc_M(e) for e in _flat(v)) if x == x]
                mass = [x for x in (conc_mgml(e) for e in _flat(v)) if x == x]
                if molar:
                    nums.setdefault("conc_M", []).extend(molar)
                if mass:
                    nums.setdefault("conc_mgml", []).extend(mass)
                continue
            vals = _parse(kind, v)
            if vals:
                nums.setdefault(kind, []).extend(vals)
        extra = " ".join(str(v) for k, v in m.items() if re.search(r"additive|dopant|antisolvent|solvent", str(k).lower())
                         and not isinstance(v, (dict, list)))
        recs.append({"role": r, "name": name, "txt": txt, "type": str(m.get("type", "")).lower(),
                     "nums": nums, "extra": extra.lower()})
    return recs


def paper_tokens(recs):
    toks = set()
    for r in recs:
        if r["name"]:
            toks.add(f"{r['role']}:{r['name']}")
        # identities named inside solvent / additive / antisolvent fields of any record
        for n, p in _ALIAS:
            if r["extra"] and p.search(r["extra"]):
                toks.add(f"x:{n}")
        for w in re.findall(r"[a-z]{4,}", r["type"]):
            if w in ("passivation", "interlayer", "additive", "dopant", "scaffold", "mesoporous", "compact", "blocking",
                     "doping", "modifier", "spacer", "nanoparticle", "quantum", "graphene", "interface", "buffer",
                     "encapsulant", "polymer", "reference"):
                toks.add(f"ty:{r['role']}:{w}")
    return toks


NUM_KINDS = ["spin_rpm", "spin_s", "anneal_C", "anneal_min", "thick_nm", "conc_M", "conc_mgml", "vol_uL"]
NUM_ROLES = ["perov", "prec", "etl", "htl", "add", "anti", "solv", "ctc", "sub", "oth"]


def paper_numeric(recs):
    """role x kind -> {med, min, max, n}. n counts distinct reported values (multi-stage / multi-condition)."""
    out = {}
    for ro in NUM_ROLES:
        for kind in NUM_KINDS:
            vals = [x for r in recs if r["role"] == ro for x in r["nums"].get(kind, [])]
            if vals:
                a = np.array(vals)
                u = np.unique(np.round(a, 6))
                out[f"{ro}_{kind}_med"] = float(np.median(a))
                out[f"{ro}_{kind}_min"] = float(a.min())
                out[f"{ro}_{kind}_max"] = float(a.max())
                out[f"{ro}_{kind}_n"] = float(len(u))
    return out


# --- recipe flags aligned with the PDP additive / solvent / quench ontology (schema alignment only, no labels) ---------
RECIPE = {
    "cl_source": r"pbcl2|macl\b|nh4cl|ch3nh3cl|methylammonium chloride|lead chloride|cscl|chloride|mapbi3-?x?cl|i3-?xclx|\bcl\b",
    "acetate": r"acetate|pb\(ac\)|pbac2|\bmaac\b|\bpbac|ch3coo",
    "scn": r"scn|thiocyanate",
    "snf2": r"snf2|tin\(ii\) fluoride|tin fluoride",
    "ava": r"5-?ava|aminovaleric|5-aminovaleric",
    "hx_acid": r"\bhi\b|hydroiodic|hydrobromic|\bhbr\b|\bhcl\b|hypophosph|\bhpa\b|formic acid|acetic acid",
    "alkali": r"\brbi\b|rubidium|\bki\b|potassium|\bnai\b|\bli[ib]r?\b",
    "cs": r"\bcsi\b|cesium|caesium|\bcsbr\b|cspb",
    "pea_family": r"\bpe(a)?i\b|phenethyl|phenylethyl|\bbai\b|butylammonium|\boai\b|octylammonium|\bgai\b|guanidin|\bnh4i|ammonium",
    "lewis_base": r"urea|thiourea|pyridine|phosph|sulfoxide|formamide(?!.*iodide)",
    "water_add": r"\bh2o\b|deionized|di water",
    "polymer": r"\bpeg\b|\bpvp\b|\bpmma\b|\bps\b|polystyrene|polyethylene|\bpei\b|polymer",
    "carbon": r"graphene|\brgo\b|cnt|nanotube|fullerene|\bc60\b|pcbm",
    "dio": r"diiodooctane|\bdio\b|1,8-diiodo",
    "htl_tbp": r"\btbp\b|t-bp|tert-?butyl ?pyridine|4-?tert",
    "htl_litfsi": r"li-?tfs[ia]|lithium bis|li salt|lin\(cf3so2\)2n|litfsi",
    "htl_fk": r"fk-?(209|102|269|211)|co\(iii\)|cobalt",
    "htl_f4tcnq": r"f4-?tcnq|tcnq",
    "htl_cu": r"cu\(tfsi\)2|cu\(oac\)2|cu:|copper",
    "htl_borate": r"tpfb|borate|\bbcf\b|b\(c6f5\)3",
    "etl_ticl4": r"ticl4",
    "etl_dopant": r"\bnb\b|\bmg:|mg-?doped|zn-?doped|ni-?doped|\bli-doped|y-doped|nb-?doped|sn-doped|doped",
    "undoped": r"undoped|pristine|without (additive|dopant)|dopant-free|dopant free",
    "interface_pass": r"passivat|interlayer|surface treatment|post-?treatment|modif",
    "antisolvent": r"anti-?solvent|quench",
    "vacuum": r"vacuum|evaporat|co-?evaporat|vapou?r|sublim",
    "two_step": r"two-?step|sequential|dipping|\bdip\b",
    "hot_cast": r"hot-?cast",
    "solvent_anneal": r"solvent anneal|dmf vapou?r|vapou?r anneal",
    "flexible": r"\bpet\b|\bpen\b|flexible|polyimide",
    "tandem": r"tandem|silicon|\bsi\b",
    "lead_free": r"snI3|sni2|bismuth|\bbii3|antimony|germanium|tin-?based|lead-?free",
    "inorganic": r"cspbi|cspbbr|cssni|inorganic",
    "mesoscopic": r"mesoporous|mp-?tio2|scaffold|carbon electrode",
    "n2_box": r"glove ?box|nitrogen|\bn2\b|argon",
    "ambient": r"ambient|humid|\bair\b",
}
_RECIPE = {k: re.compile(p, re.I) for k, p in RECIPE.items()}
RECIPE_SCOPE = {"htl_": ("htl", "add", "oth"), "etl_": ("etl", "add", "oth")}
PEROV_SCOPE = ("perov", "prec", "add", "solv", "anti")
PEROV_FLAGS = ("cl_source", "acetate", "scn", "snf2", "ava", "hx_acid", "alkali", "cs", "pea_family", "lewis_base",
               "water_add", "polymer", "carbon", "dio", "antisolvent", "vacuum", "two_step", "hot_cast", "solvent_anneal")


def paper_recipe(doc):
    """0/1 recipe-identity flags over the non-outcome string content of the extraction (1 = mentioned)."""
    recs = []
    for m in doc.get("materials", []):
        if not isinstance(m, dict):
            continue
        r = ROLE_MAP.get(role(m), "oth")
        parts = []
        for k, v in m.items():
            kl = str(k).lower()
            if OUTCOME_KEY.search(kl):
                continue
            parts.append(kl.replace("_", " "))
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
            elif isinstance(v, list):
                parts += [str(x) for x in v if isinstance(x, (str, int, float))]
        recs.append((r, " ".join(parts).lower()))
    out = {}
    for name, rx in _RECIPE.items():
        scope = next((sc for pre, sc in RECIPE_SCOPE.items() if name.startswith(pre)), None)
        out[name] = float(any(rx.search(t) for r, t in recs if scope is None or r in scope))
    for name in PEROV_FLAGS:   # scoped variant: only text of perovskite-forming records (precision)
        out["s_" + name] = float(any(_RECIPE[name].search(t) for r, t in recs if r in PEROV_SCOPE))
    out["n_materials"] = float(len(recs))
    return out


def build(json_dir, normalize_doi):
    """-> {doi_group: {"tokens": [...], "numeric": {...}, "layers": [records]}}"""
    res = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        doi = f.stem.replace("_", "/", 1)
        if "10." not in doi:
            doi = re.sub(r"^doi:\s*", "", str(doc.get("source", "")).strip(), flags=re.I)
        recs = material_records(doc)
        res[normalize_doi(doi)] = {"tokens": sorted(paper_tokens(recs)), "numeric": paper_numeric(recs), "recipe": paper_recipe(doc),
                                   "layers": [{k: r[k] for k in ("role", "name", "nums")}
                                              for r in recs if r["role"] in ("etl", "htl", "ctc", "perov", "add")]}
    return res
