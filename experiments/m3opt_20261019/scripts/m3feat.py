"""M3 feature representations built from the raw LLM JSON (label-free unless stated) for the m3opt experiment.

Everything here is additive to the fixed M2 (t2) matrix; M1/M2 are never changed.

Paper-level tables (index = doi_group):
  flags(variant)      the 36 production recipe flags (`litfeat4.paper_recipe`)
      'v4'            stored data/lit_v4.json, production DOI join (filename, first '_' -> '/'), empty papers = 0
      'fixed'         same flags, DOI resolution fixed (see resolve_doi) and empty extractions -> NaN (not negatives)
      'strict'        'fixed' + extended outcome-key filter before the regex search (OUTCOME_STRICT)
  text()              non-outcome, non-characterisation extraction text per paper (identity / process / property)
  jprop()             paper-reported intrinsic properties (perovskite band gap, HTL HOMO, ETL LUMO, work functions)
  design()            study-design counts (n material records, distinct perovskite / ETL / HTL identities)

Label-using blocks (fit on the training pool of a fold only; out-of-fold inside the pool):
  knn_te(...)         similarity-weighted mean paper-PCE of the k nearest training papers in TF-IDF space
  tok_te(...)         m-estimate target encoding of canonical tokens, aggregated per paper (mean / max / min / n)
"""
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
UGP = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from psc.data import normalize_doi  # noqa: E402
from psc import litfeat4 as L4  # noqa: E402
from psc.litfeat import role as llm_role  # noqa: E402

JSON_DIR = UGP / "json"
LIT_V4 = ROOT / "data" / "lit_v4.json"

FLAG_NAMES = [k for k in L4.RECIPE]                       # 36 flags, production order
GROUPS = {"P": ["cl_source", "acetate", "scn", "snf2", "ava", "hx_acid", "alkali", "cs", "pea_family", "lewis_base",
                "water_add", "polymer", "carbon", "dio"],
          "H": ["htl_tbp", "htl_litfsi", "htl_fk", "htl_f4tcnq", "htl_cu", "htl_borate"],
          "E": ["etl_ticl4", "etl_dopant"],
          "R": ["antisolvent", "vacuum", "two_step", "hot_cast", "solvent_anneal", "n2_box", "ambient"],
          "C": ["flexible", "tandem", "lead_free", "inorganic", "mesoscopic"],
          "U": ["undoped", "interface_pass"]}
assert sorted(sum(GROUPS.values(), [])) == sorted(FLAG_NAMES)

OUTCOME_STRICT = re.compile(
    r"(^|[^a-z])(pce|voc|v_oc|jsc|j_sc|ff|fill.?factor|efficien|power.?conversion|short.?circuit|open.?circuit|eqe|ipce|"
    r"hysteresis|t80|t90|t50|retention|degrad|stabil|mpp|champion|performance|pv_)", re.I)
CHAR_KEY = re.compile(r"grain|rough|rms|trpl|photolumin|lifetime|xrd|fwhm|trap|crystallite|absorb|transmit|conductiv|"
                      r"resistiv|contact_angle|ideality|urbach|recombin|diffusion_length|impedance|morpholog", re.I)


# ---------------------------------------------------------------- DOI resolution (deterministic, recorded rules)
def resolve_doi(stem, source, known):
    """Rules, first hit in `known` (PDP doi_groups) wins:
    R1 filename, first '_' -> '/' (production rule)      R2 filename, every '_' -> '/' (IOP-style DOIs)
    R3 `source` field if it is a DOI                      else R1 (kept, unjoined)."""
    c1 = normalize_doi(stem.replace("_", "/", 1))
    c2 = normalize_doi(stem.replace("_", "/"))
    s = re.sub(r"^doi:\s*", "", str(source or "").strip(), flags=re.I)
    c3 = normalize_doi(s) if s.lower().startswith("10.") else None
    for rule, c in (("R1", c1), ("R2", c2), ("R3", c3)):
        if c and c in known:
            return c, rule
    return c1, "unjoined"


@lru_cache(maxsize=1)
def _docs():
    return {f.stem: json.loads(f.read_text()) for f in sorted(JSON_DIR.glob("*.json"))}


def _materials(doc):
    m = doc.get("materials") if isinstance(doc, dict) else None
    if isinstance(m, list):
        return [x for x in m if isinstance(x, dict)]
    return [doc] if isinstance(doc, dict) and ("formula" in doc or "common_name" in doc) else []


PLACEHOLDER = re.compile(r"^\s*<[^>]*>\s*$")
EVAL_WORDS = re.compile(r"\b(best|champion\w*|stab\w*|degrad\w*|enhanc\w*|improv\w*|efficien\w*|perform\w*|record|excellent|"
                        r"superior|loss\w*|high\w*|low\w*|good|poor|better|worse)\b")


def _clean_materials(doc):
    """Drop prompt-template records (formula / common_name is a '<...>' placeholder) and '...' template keys."""
    out = []
    for m in _materials(doc):
        if any(isinstance(m.get(k), str) and PLACEHOLDER.match(m[k]) for k in ("formula", "common_name")):
            continue
        m = {k: v for k, v in m.items() if str(k).strip() != "..."}
        if m:
            out.append(m)
    return out


@lru_cache(maxsize=4)
def resolution(known_key):
    known = _KNOWN[known_key]
    rows = []
    for stem, doc in _docs().items():
        g, rule = resolve_doi(stem, doc.get("source") if isinstance(doc, dict) else None, known)
        rows.append((stem, g, rule, len(_materials(doc))))
    r = pd.DataFrame(rows, columns=["stem", "doi_group", "rule", "n_materials"])
    # one file per doi_group (assert: no many-to-one collapse)
    assert not r[r.rule != "unjoined"].doi_group.duplicated().any(), "two JSON files resolve to one publication"
    return r


_KNOWN = {}


def set_known(groups):
    """Register the PDP publication keys used for resolution (call once with df.doi_group)."""
    _KNOWN["pdp"] = frozenset(groups)
    resolution.cache_clear()


def _recipe_strict(doc):
    """paper_recipe with the extended outcome-key filter (keys and their values skipped)."""
    stripped = []
    for m in _materials(doc):
        stripped.append({k: v for k, v in m.items() if not OUTCOME_STRICT.search(str(k))})
    return L4.paper_recipe({"materials": stripped})


@lru_cache(maxsize=8)
def flags(variant):
    if variant == "v4":
        P = json.loads(LIT_V4.read_text())["papers"]
        T = pd.DataFrame({d: {k: p["recipe"][k] for k in FLAG_NAMES} for d, p in P.items()}).T
        return T.astype(float)
    res = resolution("pdp")
    docs = _docs()
    rows = {}
    for stem, g, rule, nm in res.itertuples(index=False):
        if rule == "unjoined":
            continue
        if variant == "clean":                              # strict + template records removed
            mats = _clean_materials(docs[stem])
            rows[g] = ({k: np.nan for k in FLAG_NAMES} if not mats else
                       {k: v for k, v in _recipe_strict({"materials": mats}).items() if k in FLAG_NAMES})
            continue
        if nm == 0:
            rows[g] = {k: np.nan for k in FLAG_NAMES}      # extraction failure: unknown, not negative
            continue
        r = L4.paper_recipe({"materials": _materials(docs[stem])}) if variant == "fixed" else _recipe_strict(docs[stem])
        rows[g] = {k: r[k] for k in FLAG_NAMES}
    return pd.DataFrame.from_dict(rows, orient="index").astype(float)


# ---------------------------------------------------------------- paper text / properties / design
@lru_cache(maxsize=4)
def text(variant="v1"):
    """Paper text: role-tagged canonical identities + identity strings + process key names / string values.
    Outcome keys (strict filter), characterisation keys and all numbers removed.
      v1     as above (E1/E2)
      clean  template records removed, evaluative words (best/champion/stability/high/...) removed
      chem   clean records, identity fields only (role tag, canonical name, formula, common_name, type)"""
    res = resolution("pdp"); docs = _docs(); out = {}
    for stem, g, rule, nm in res.itertuples(index=False):
        if rule == "unjoined" or nm == 0:
            continue
        mats = _materials(docs[stem]) if variant == "v1" else _clean_materials(docs[stem])
        if not mats:
            continue
        toks = []
        for m in mats:
            r = L4.ROLE_MAP.get(llm_role(m), "oth")
            nm_ = L4.canon(m.get("formula", "")) or L4.canon(m.get("common_name", ""))
            if nm_:
                toks.append(f"R_{r}_{nm_}")
            if variant == "chem":
                toks += [str(m.get(k, "")).lower() for k in ("formula", "common_name", "type") if isinstance(m.get(k), str)]
                continue
            for k, v in m.items():
                ks = str(k)
                if OUTCOME_STRICT.search(ks) or CHAR_KEY.search(ks):
                    continue
                toks.append(re.sub(r"[^a-z]+", " ", ks.lower()))
                if isinstance(v, str):
                    toks.append(v.lower())
                elif isinstance(v, list):
                    toks += [str(x).lower() for x in v if isinstance(x, str)]
        s = " ".join(toks)
        if variant != "v1":
            s = EVAL_WORDS.sub(" ", s)
        s = re.sub(r"[-+]?\d*\.?\d+", " ", s)
        out[g] = re.sub(r"\s+", " ", s)
    return pd.Series(out)


PROP_KEYS = {"pv_bandgap": ("perov", r"band.?gap|^eg\b|e_g"), "htl_homo": ("htl", r"homo|valence|vbm|ionization"),
             "etl_lumo": ("etl", r"lumo|conduction|cbm|electron_affinity"), "htl_wf": ("htl", r"work.?function"),
             "etl_wf": ("etl", r"work.?function")}
PROP_RANGE = {"pv_bandgap": (1.1, 3.2), "htl_homo": (-7.0, -4.0), "etl_lumo": (-5.5, -2.5), "htl_wf": (-6.5, -3.5),
              "etl_wf": (-6.0, -3.0)}


def _nums(v):
    if isinstance(v, bool):
        return []
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, str):
        return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", v)[:3]]
    if isinstance(v, list):
        return [x for e in v for x in _nums(e)]
    return []


@lru_cache(maxsize=1)
def jprop():
    res = resolution("pdp"); docs = _docs(); out = {}
    for stem, g, rule, nm in res.itertuples(index=False):
        if rule == "unjoined" or nm == 0:
            continue
        vals = {k: [] for k in PROP_KEYS}
        for m in _materials(docs[stem]):
            r = L4.ROLE_MAP.get(llm_role(m), "oth")
            for k, v in m.items():
                kl = str(k).lower()
                for name, (role, rx) in PROP_KEYS.items():
                    if (r == role or (role == "perov" and r in ("perov", "prec"))) and re.search(rx, kl):
                        lo, hi = PROP_RANGE[name]
                        for x in _nums(v):
                            x2 = -abs(x) if name != "pv_bandgap" else x      # energies reported with either sign
                            if lo <= x2 <= hi:
                                vals[name].append(x2)
        out[g] = {f"jp_{k}": (float(np.median(v)) if v else np.nan) for k, v in vals.items()}
    return pd.DataFrame.from_dict(out, orient="index")


@lru_cache(maxsize=1)
def design():
    res = resolution("pdp"); docs = _docs(); out = {}
    for stem, g, rule, nm in res.itertuples(index=False):
        if rule == "unjoined" or nm == 0:
            continue
        recs = L4.material_records({"materials": _materials(docs[stem])})
        nm_by = lambda ro: len({r["name"] for r in recs if r["role"] == ro and r["name"]})  # noqa: E731
        out[g] = {"ds_n_records": float(len(recs)), "ds_n_perov_ids": float(len({r["txt"] for r in recs if r["role"] in ("perov",)})),
                  "ds_n_etl": float(nm_by("etl")), "ds_n_htl": float(nm_by("htl")), "ds_n_add": float(nm_by("add")),
                  "ds_n_roles": float(len({r["role"] for r in recs}))}
    return pd.DataFrame.from_dict(out, orient="index")


@lru_cache(maxsize=1)
def tokens():
    """lx4 canonical role-tagged tokens per paper, recomputed under the fixed DOI resolution."""
    res = resolution("pdp"); docs = _docs(); out = {}
    for stem, g, rule, nm in res.itertuples(index=False):
        if rule == "unjoined" or nm == 0:
            continue
        out[g] = sorted(L4.paper_tokens(L4.material_records({"materials": _materials(docs[stem])})))
    return out


# ---------------------------------------------------------------- record-level attachment and representations
def attach(groups, T):
    return T.reindex(np.asarray(groups)).reset_index(drop=True)


def permute_papers(T, seed):
    """DOI-level control: paper rows shuffled among the covered papers (keeps marginals + mask, kills association)."""
    rng = np.random.default_rng(seed)
    v = T.values[rng.permutation(len(T))]
    return pd.DataFrame(v, index=T.index, columns=T.columns)


def represent(F, how, tr_idx=None):
    """Record-level flag matrix F (NaN = uncovered/failed) -> representation."""
    F = F.copy()
    cov = F.notna().any(axis=1).astype(float)
    if how == "nan":
        return F
    if how == "zero_cov":                    # explicit missingness: 0 for not-mentioned and uncovered, plus coverage flag
        return F.fillna(0.0).assign(rc_covered=cov)
    if how == "cat3":                        # three-state categorical: '1' / '0' / 'na'
        return F.apply(lambda c: c.map({1.0: "1", 0.0: "0"}).fillna("na"))
    if how in ("groups", "flags_groups"):
        G = pd.DataFrame({f"rcg_{g}": F[[f"rc_{c}" for c in cs]].sum(axis=1, min_count=1) for g, cs in GROUPS.items()})
        G["rcg_all"] = F.sum(axis=1, min_count=1)
        return G if how == "groups" else pd.concat([F, G], axis=1)
    if how == "prev":                        # drop flags with training covered-record prevalence < 3% or > 97%
        p = F.iloc[tr_idx].mean()
        return F[p[(p >= 0.03) & (p <= 0.97)].index]
    raise ValueError(how)


# ---------------------------------------------------------------- label-using blocks (fold-internal)
def _paper_target(groups, y, idx):
    s = pd.Series(y[idx]).groupby(np.asarray(groups)[idx]).agg(["mean", "size"])
    return s


def _inner_folds(papers, k, seed):
    rng = np.random.default_rng(seed)
    p = np.array(sorted(papers)); rng.shuffle(p)
    return {g: i % k for i, g in enumerate(p)}


def knn_te(groups, y, pool_idx, Xtxt, k=25, n_inner=5, seed=0, prefix="kn"):
    """Xtxt: L2-normalised sparse TF-IDF rows indexed by paper (dict paper->row id) -> record frame.
    Pool papers get out-of-fold values (inner paper folds); all other papers use the whole pool."""
    M, pid = Xtxt
    tgt = _paper_target(groups, y, pool_idx)
    pool = [g for g in tgt.index if g in pid]
    fold = _inner_folds(pool, n_inner, seed)
    pool_rows = np.array([pid[g] for g in pool]); pool_y = tgt.loc[pool, "mean"].values
    pool_f = np.array([fold[g] for g in pool])
    queries = sorted(set(np.asarray(groups)) & set(pid))
    S = (M[[pid[g] for g in queries]] @ M[pool_rows].T).toarray()          # cosine similarity
    out = {}
    for i, g in enumerate(queries):
        s = S[i].copy()
        if g in fold:
            s[pool_f == fold[g]] = -np.inf                                   # OOF: drop own inner fold (incl. itself)
        top = np.argpartition(-s, k)[:k]
        top = top[np.isfinite(s[top])]
        w = np.clip(s[top], 0, None)
        out[g] = {f"{prefix}_mean": float(np.average(pool_y[top], weights=w)) if w.sum() > 0 else np.nan,
                  f"{prefix}_top1": float(pool_y[top[np.argmax(s[top])]]) if len(top) else np.nan,
                  f"{prefix}_maxsim": float(s[top].max()) if len(top) else np.nan,
                  f"{prefix}_meansim": float(s[top].mean()) if len(top) else np.nan}
    return attach(groups, pd.DataFrame.from_dict(out, orient="index"))


def tok_te(groups, y, pool_idx, toks, m=20.0, min_df=10, n_inner=5, seed=0, prefix="tt"):
    """Token target encoding (m-estimate toward the pool mean), out-of-fold for pool papers, aggregated per paper."""
    tgt = _paper_target(groups, y, pool_idx)
    pool = [g for g in tgt.index if g in toks]
    fold = _inner_folds(pool, n_inner, seed)
    mu = float(tgt.loc[pool, "mean"].mean())

    def table(papers):
        s, n = {}, {}
        for g in papers:
            v = tgt.at[g, "mean"]
            for t in toks[g]:
                s[t] = s.get(t, 0.0) + v; n[t] = n.get(t, 0) + 1
        return {t: (s[t] + m * mu) / (n[t] + m) for t in s if n[t] >= min_df}
    tabs = {f: table([g for g in pool if fold[g] != f]) for f in range(n_inner)}
    full = table(pool)
    out = {}
    for g in sorted(set(np.asarray(groups)) & set(toks)):
        tab = tabs[fold[g]] if g in fold else full
        v = [tab[t] - mu for t in toks[g] if t in tab]
        out[g] = {f"{prefix}_mean": float(np.mean(v)) if v else 0.0, f"{prefix}_max": float(np.max(v)) if v else 0.0,
                  f"{prefix}_min": float(np.min(v)) if v else 0.0, f"{prefix}_n": float(len(v))}
    return attach(groups, pd.DataFrame.from_dict(out, orient="index"))


def tfidf(papers_text, fit_papers, max_features=4000, min_df=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z_][a-z0-9_]{1,}\b")
    vec.fit(papers_text.loc[[p for p in fit_papers if p in papers_text.index]].values)
    M = vec.transform(papers_text.values)
    return M, {g: i for i, g in enumerate(papers_text.index)}, vec


def svd_block(groups, Xtxt, fit_rows, k=16, seed=0, prefix="sv"):
    from sklearn.decomposition import TruncatedSVD
    M, pid = Xtxt
    s = TruncatedSVD(n_components=k, random_state=seed).fit(M[fit_rows])
    Z = s.transform(M)
    T = pd.DataFrame(Z, index=list(pid.keys()), columns=[f"{prefix}_{i}" for i in range(k)])
    return attach(groups, T)
