"""Dataset loading, DOI normalisation and feature-tier definitions."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "dataset_json_enriched.csv"
REF_META = ROOT / "data" / "pdp_ref_metadata.csv"   # Ref_ID, lead author, publication date (PDP export)
TARGET = "final_pce"
DOI_COL = "Ref_DOI_number"
NO_DOI_GROUP = "__NO_DOI__"
NO_AUTHOR_GROUP = "__NO_AUTHOR__"

PROPS = ["band_gap", "LUMO", "HOMO", "electron_mobility", "hole_mobility",
         "dielectric_constant", "min_atomic_mass", "max_atomic_mass",
         "avg_atomic_mass", "min_radius", "max_radius", "avg_radius",
         "en_min", "en_max", "en_avg"]
# material-property block: 45 layer properties + 4 perovskite geometry descriptors
MAT49 = ({f"{L}_{p}" for L in ("ETL", "HTL", "Perovskite") for p in PROPS}
         | {"perovskite_tolerance_factor", "perovskite_octahedral_factor",
            "perovskite_tau_descriptor", "perovskite_cation_radius_ratio"})
# LLM / full-text derived block (paper-level)
LIT32 = {"has_BCP", "has_LiF", "has_PEAI", "has_SAM", "has_C60", "has_PCBM",
         "has_Spiro", "has_PEDOT", "has_NiOx", "has_SnO2", "has_TiO2", "has_ZnO",
         "has_PTAA", "has_fullerene", "has_passivation_layer", "is_flexible_y",
         "is_encapsulated", "has_stability_measurement", "used_antisolvent",
         "used_glovebox", "used_vacuum_evaporation", "json_numeric_values_found",
         "spin_speed_rpm", "perov_concentration", "etl_thickness_nm",
         "htl_thickness_nm", "etl_anneal_temp_C", "htl_anneal_temp_C",
         "perov_anneal_temp_C", "perov_anneal_time_min",
         "perovskite_thickness_nm", "contact_thickness_nm"}
# columns removed from the t1 / t2 / t3 representations (nested by construction)
# reporting-completeness style literature flags (suspected paper-identity proxies)
LIT_META = {"json_numeric_values_found", "has_stability_measurement"}
TIER_DROP = {"t1": MAT49 | LIT32, "t2": LIT32, "t3": set(),
             # feature-group controls (not part of the nested main ablation)
             "t1_lit": MAT49,                  # database + literature, no material properties
             "t3_nometa": LIT_META,            # t3 without reporting-completeness flags
             "t2_noise": LIT32}                # t2 + 32 pure-noise columns (see features.py)
NOISE_COLS = 32

# identifiers, outcome-adjacent measurements and free text never used as predictors
FORBIDDEN = ["Ref_ID", DOI_COL, TARGET, "Perovskite_composition_short_form",
             "doi_loo_mean", "doi_loo_std", "doi_count", "doi_sigma",
             "Ref_journal", "Ref_publication_date", "publication_year",
             "Stability_measured", "EQE_measured", "JV_hysteresis_index",
             "Cell_area_measured_cm2", "Stabilised_performance_measured",
             "Perovskite", "HTL", "ETL"]     # ETL/HTL/Perovskite names are re-admitted only by the *_id control tiers
CAT_BASE = ["Cell_architecture", "ETL_deposition_method",
            "HTL_deposition_method", "Perovskite_deposition_method"]
CAT_ID = ["ETL", "HTL", "Perovskite"]          # material identities (control 6.4: t1_id / t2_id tiers only)
CAT = list(CAT_BASE)                          # categorical columns of every tier
CAT_ALL = CAT_BASE + CAT_ID                   # + identities (only present in *_id tiers)
TIER_DROP.update({"t1_id": MAT49 | LIT32, "t2_id": LIT32})
ARCHS = {"overall": None, "pin": "pin", "nip": "nip"}
LIT_V2 = ROOT / "data" / "lit_v2.csv"   # schema lx2.0 paper-level features built by scripts/build_lit_v2.py
# tier -> (columns of lit_v2 kept by prefix, drop original LIT32?) ; t2-based tiers keep LIT32 out
LX_TIERS = {"t3x": ("proc", "chem", "ctx"), "t3x_proc": ("proc",), "t3x_chem": ("chem",),
            "t3x_ctx": ("ctx",), "t3x_char": ("proc", "chem", "ctx", "char"),
            "t3x_meta": ("meta",), "t3x_all": ("proc", "chem", "ctx", "char", "meta"),
            "t3b": ("proc", "chem", "ctx")}          # t3b = t3 (original LIT32) + t3x
# t3q = t3x + quantity-level (lx3.0) block; *_new = lx3.0 columns only; no reporting-completeness columns anywhere
LX_TIERS.update({"t3q": ("proc", "chem", "ctx", "qty", "cat"), "t3q_new": ("qty", "cat"),
                 "t3q_qty": ("qty",), "t3q_cat": ("cat",), "t3q_nochem": ("proc", "ctx", "qty", "cat")})
LIT_V3 = ROOT / "data" / "lit_v3.csv"
QTY_SEL = ROOT / "data" / "lx3_qty_selected.json"   # predeclared quantity features + eligible population (scripts/qty_population.py)
QTY_TIERS = {"t3qq": ("t2", False, True),       # tier -> (base tier, keep original LIT32?, add cat cols?)  ; selected qty always added
             "t3qq_cat": ("t2", False, True), "t3_qq": ("t3", True, False)}
LX_CONTROLS = ("lxperm", "lxnoise")                  # t2_lxperm_s<k> / t2_lxnoise_s<k>
PART_DEV_FRAC = 0.5


def doi_partition(doi):
    """Deterministic, data-independent dev/confirm assignment of a publication (hash of its DOI key)."""
    import hashlib
    h = int(hashlib.sha256(("psc-part-v1|" + doi).encode()).hexdigest(), 16) % 1000
    return "dev" if h < PART_DEV_FRAC * 1000 else "confirm"


def qty_partition(doi):
    """Fresh 70/30 DOI hash (own salt) over the quantity-covered population; 'qconf' is reserved for one final scoring."""
    import hashlib
    h = int(hashlib.sha256(("psc-qty-v1|" + doi).encode()).hexdigest(), 16) % 1000
    return "qdev" if h < 700 else "qconf"


def normalize_doi(x):
    """Canonical DOI key: strip BOM/space, lowercase, drop doi.org / doi: prefixes.

    Returns NO_DOI_GROUP for missing values. Non-DOI URLs are kept verbatim
    (lower-cased) as their own key.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NO_DOI_GROUP
    s = str(x).replace("﻿", "").strip().lower()
    if not s or s == "nan":
        return NO_DOI_GROUP
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s


def normalize_author(x):
    """Lead-author key: lower-case surname without 'et al.'; missing -> one explicit block."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or not str(x).strip():
        return NO_AUTHOR_GROUP
    return re.sub(r"\s+et\.? al\.?$", "", str(x).strip().lower()).strip()


def load_dataset(path=DEFAULT_DATA, arch="overall", min_pce=1.0, subset=None, partition=None):
    """Load, filter (missing/PCE<min_pce) and subset by architecture.

    Adds `doi_group` (normalised DOI; one explicit NO_DOI_GROUP block for
    records lacking a DOI).
    """
    df = pd.read_csv(path, low_memory=False)
    n0 = len(df)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= min_pce].reset_index(drop=True)
    if ARCHS[arch] is not None:
        df = df[df["Cell_architecture"].str.lower() == ARCHS[arch]].reset_index(drop=True)
    df = df.copy()  # defragment the wide frame before adding a column
    df["doi_group"] = df[DOI_COL].map(normalize_doi)
    if REF_META.exists():  # lead author (laboratory proxy) and publication date for stricter protocols
        meta = pd.read_csv(REF_META).drop_duplicates("Ref_ID")
        df = df.merge(meta, on="Ref_ID", how="left", validate="m:1")
        df["author_group"] = df["Ref_lead_author"].map(normalize_author)
        # one author key per publication (modal), so DOI-disjointness is implied by author-disjointness
        has = df["doi_group"] != NO_DOI_GROUP
        modal = df[has].groupby("doi_group")["author_group"].agg(lambda x: x.value_counts().index[0])
        df.loc[has, "author_group"] = df.loc[has, "doi_group"].map(modal)
        df.loc[~has, "author_group"] = NO_DOI_GROUP  # records without DOI stay one block here too
        df["pub_date"] = pd.to_datetime(df["Ref_publication_date"], errors="coerce")
    if LIT_V2.exists():   # schema-v2 literature block, joined on the normalised DOI (paper-level)
        lx = pd.read_csv(LIT_V2).drop(columns=["json_file"], errors="ignore").drop_duplicates("doi_group")
        df = df.merge(lx, on="doi_group", how="left", validate="m:1")
        df["lx_covered"] = df["doi_group"].isin(set(lx["doi_group"])).astype(int)
    if LIT_V3.exists():   # schema-v3 quantity-level block, same paper-level join
        l3 = pd.read_csv(LIT_V3).drop_duplicates("doi_group")
        df = df.merge(l3, on="doi_group", how="left", validate="m:1")
    df["partition"] = df["doi_group"].map(doi_partition)
    if subset == "lx_covered":     # identical record subset for every tier being compared
        df = df[df["lx_covered"] == 1]
    if QTY_SEL.exists():
        sel = json.loads(QTY_SEL.read_text())
        paper = df.drop_duplicates("doi_group").set_index("doi_group")
        core = [c for c in paper.columns if c.startswith("lx_qty_") and "contact" not in c]
        elig = set(paper.index[paper[core].notna().any(axis=1)]) if core else set()
        df["qty_covered"] = df["doi_group"].isin(elig).astype(int)
        df["qpartition"] = df["doi_group"].map(qty_partition)
    if subset == "qty_covered":
        df = df[df["qty_covered"] == 1]
    if partition in ("qdev", "qconf"):
        df = df[df["qpartition"] == partition]
    elif partition:
        df = df[df["partition"] == partition]
    df = df.reset_index(drop=True)
    df.attrs["n_raw"] = n0
    return df
