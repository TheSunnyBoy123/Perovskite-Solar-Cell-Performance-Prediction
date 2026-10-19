"""Shared helpers for the recipe-flag source audit (17-paper pilot).

Everything here is deterministic: source text is read from the cached Markdown
files, flags are recomputed from the cached extraction JSON with the production
functions in src/psc/litfeat4.py, and nothing calls an LLM.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent                        # ml_paper/audit/recipe_flag_pilot
ML_PAPER = PILOT.parent.parent             # ml_paper
UGP = ML_PAPER.parent                      # .../UGP
sys.path.insert(0, str(ML_PAPER / "src"))

from psc.data import normalize_doi  # noqa: E402
from psc.litfeat4 import RECIPE, _RECIPE, paper_recipe  # noqa: E402

COMBINED_JSON = UGP / "opencode_sandbox" / "combined_extracted_features.json"
MARKDOWN_DIR = UGP / "sandbox" / "markdown"
PDF_DIR = UGP / "sandbox" / "pdfs"
LIT_V4 = ML_PAPER / "data" / "lit_v4.json"
OUT = PILOT

FLAGS = list(RECIPE)  # the 36 M3 recipe flags, in production order
assert len(FLAGS) == 36

REASONS = {
    "LITERATURE": "mention in cited literature / introduction only",
    "REFERENCE": "text is inside the reference list or an author/title string",
    "ARTIFACT": "regex matched a non-target string (e.g. MA inside 'methylammonium', 'pristine' film, FTO)",
    "SOLVENT": "solvent or diluent only, not an additive or layer",
    "CLEANING": "substrate cleaning or rinse only",
    "ELECTRODE": "metal top/back electrode evaporated; excluded from vacuum/evaporation flag by rule",
    "TEST_CONDITION": "atmosphere/humidity during stability or characterisation, not fabrication",
    "GENERIC": "generic word (e.g. 'modification', 'organic-inorganic', 'air') without a qualifying element",
    "LIGAND_NAME": "term occurs only inside a chemical name (e.g. the FK209 cobalt ligand)",
    "CHEMISTRY": "reagent or product in an unrelated chemical context (e.g. HTM synthesis, decomposition)",
    "SPIN_PROGRAM": "two-speed spin programme, not a two-step/sequential perovskite deposition",
    "PROCESS_DRYING": "vacuum-assisted drying step, not vacuum deposition of a layer",
    "COMPUTATIONAL": "appears only in DFT / simulation text",
    "FILM_ONLY": "characterised as a film; no device in which it is used",
    "AFFILIATION": "author affiliation or supplier name",
    "ETCH": "acid used to etch patterned FTO, not a device layer or recipe ingredient",
    "NO_SOURCE_TERM": "no regex hit anywhere in the full text",
    "DEFINITION_AMBIGUOUS": "flag definition or device role cannot be confirmed from the text; needs human adjudication",
    "PLANAR_DEVICE": "mesoporous/scaffold mentioned only as a comparison; device is planar",
}


def canon(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def flatten(text):
    return re.sub(r"\s+", " ", text)


def load_combined():
    return json.loads(COMBINED_JSON.read_text())


def frame():
    """Markdown files that have an extraction record in the combined JSON.

    Returns (rows, excluded). rows: dicts with doi, md_path, stem.
    excluded: Markdown files with no matching extraction record.
    """
    J = load_combined()
    jmap = {canon(normalize_doi(k)): normalize_doi(k) for k in J}
    rows, excluded = [], []
    for md in sorted(MARKDOWN_DIR.glob("*.md")):
        stem = md.stem
        doi = jmap.get(canon(stem))
        if doi is None:
            excluded.append({"stem": stem, "md_path": str(md)})
        else:
            rows.append({"doi": doi, "stem": stem, "md_path": md})
    return rows, excluded


def source_text(md_path):
    return md_path.read_text(encoding="utf8", errors="ignore")


def find_anchor(flat, anchor):
    """Index of the first occurrence of anchor in whitespace-normalised text, or -1."""
    return flat.find(anchor)


def window(flat, idx, length=None, before=120, after=120):
    if idx < 0:
        return ""
    return flat[max(0, idx - before): idx + (length or 0) + after].strip()


def heading_before(flat, idx):
    """Nearest preceding markdown heading (approximate; the Markdown has no page numbers)."""
    best = ""
    for m in re.finditer(r"#{1,6} ([^#]{2,90}?)(?= [A-Z0-9*_\[(]|$| \*|#)", flat[:idx]):
        best = m.group(1).strip()
    return best


def regex_hits(flat):
    """Per-flag count and first-hit (index, text) on the full source text, using production regexes."""
    out = {}
    for k in FLAGS:
        ms = list(_RECIPE[k].finditer(flat))
        out[k] = {"n": len(ms), "first": ms[0].start() if ms else -1,
                  "first_match": ms[0].group(0) if ms else ""}
    return out


def json_flags(doi, J):
    """Recompute the 36 flags for a DOI from the combined JSON via the production function."""
    rc = paper_recipe(J[doi])
    return {k: int(rc[k] == 1.0) for k in FLAGS}, rc


def json_matches(doi, J):
    """For each flag predicted 1: the first extraction record whose text triggers it (scope as in litfeat4)."""
    from psc.litfeat4 import OUTCOME_KEY, RECIPE_SCOPE, PEROV_SCOPE, ROLE_MAP, role  # noqa: F401
    recs = []
    for m in J[doi].get("materials", []):
        if not isinstance(m, dict):
            continue
        r = ROLE_MAP.get(role(m), "oth")
        parts = []
        for k, v in m.items():
            if OUTCOME_KEY.search(str(k).lower()):
                continue
            parts.append(str(k).lower().replace("_", " "))
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
            elif isinstance(v, list):
                parts += [str(x) for x in v if isinstance(x, (str, int, float))]
        recs.append((r, " ".join(parts).lower(), m))
    out = {}
    for k in FLAGS:
        scope = next((sc for pre, sc in RECIPE_SCOPE.items() if k.startswith(pre)), None)
        for r, t, m in recs:
            if scope is not None and r not in scope:
                continue
            mt = _RECIPE[k].search(t)
            if mt:
                name = m.get("formula") or m.get("common_name") or "?"
                out[k] = {"record": str(name), "role": r, "matched": mt.group(0)}
                break
    return out


# Alias families for the deterministic-conversion check (does the JSON text contain the
# material under another name?). Lower-case substrings, checked against the whole JSON text.
JSON_ALIASES = {
    "htl_tbp": ["tert-butylpyridine", "tert-butyl pyridine", "tbp", "t-bp", "butylpyridine"],
    "htl_litfsi": ["litfsi", "li-tfsi", "tfsi", "bis(trifluoromethyl", "lithium bis", "li-tfs"],
    "htl_fk": ["fk209", "fk 209", "fk-209", "cobalt"],
    "etl_ticl4": ["ticl4", "ticl 4", "titanium tetrachloride"],
    "etl_dopant": ["doped", "dope", "er-yb", "sn-doped"],
    "hx_acid": ["hydroiodic", "hydrochloric", "hydrobromic", "hydroiodide", "hbr", "hcl", "hi)"],
    "cs": ["cs0", "cs+", "cesium", "caesium", "csi", "cspb", "cs("],
    "lewis_base": ["dmso", "sulfoxide", "pyridine", "urea", "thiourea"],
    "carbon": ["pc71bm", "pc61bm", "pcbm", "fullerene", "c60", "graphene"],
    "n2_box": ["glove", "nitrogen", "argon", "n2", "inert"],
    "two_step": ["two-step", "two step", "sequential", "2-step"],
    "mesoscopic": ["mesopor", "scaffold", "paste"],
    "vacuum": ["vacuum", "evaporat", "sublim"],
    "polymer": ["pedot", "polymer", "pvp", "polystyrene"],
    "ambient": ["air", "ambient", "humid"],
    "antisolvent": ["antisolvent", "anti-solvent", "quench", "diethyl ether", "chlorobenzene"],
    "interface_pass": ["passiv", "interlayer", "interfacial", "surface treatment", "post-treatment"],
    "undoped": ["dopant-free", "dopant free", "undoped", "pristine"],
    "alkali": ["rubidium", "rb0", "rb+", "potassium", "lithium", "sodium"],
    "inorganic": ["inorganic", "cspbi", "all-inorganic"],
    "lead_free": ["tin", "sn", "germanium", "bismuth", "antimony"],
    "acetate": ["acetate"],
    "pea_family": ["phenethyl", "phenylethyl", "pea", "butylammonium", "guanidin", "bai", "ammonium", "aminovaleric"],
    "cl_source": ["pbcl", "chlor", "facl", "macl"],
    "ava": ["aminovaleric", "5-ava", "ava"],
    "solvent_anneal": ["solvent anneal", "vapor anneal", "vapour anneal"],
    "flexible": ["flexible", "polyimide", " pet", " pen"],
    "tandem": ["tandem", "silicon", "perovskite/silicon"],
}


def json_text_lower(J, doi):
    return json.dumps(J[doi], ensure_ascii=False).lower()


def json_alias_present(flag, jtext):
    """True if an alias of the flag's material occurs in the JSON text.

    Short aliases (<= 4 characters) must sit on word-like boundaries so that e.g. 'hcl'
    does not match inside 'adahcl'.
    """
    for a in JSON_ALIASES.get(flag, []):
        if len(a) <= 4:
            if re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", jtext):
                return True
        elif a in jtext:
            return True
    return False
