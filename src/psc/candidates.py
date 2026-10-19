"""Covered-population literature representations C1-C3 for the development selection.

Each candidate is a record-level block built from a paper-level table. The paper tables use only the
LLM extraction (`data/lit_v4.json`) and never the outcome. Every statistic that is learned from data
(medians, prevalence filters) is computed on the training rows of the fold passed to `build`.

Penalty targets are the candidate's own columns: names begin with `nm_c1` (C1), `tk_` (C2), `rc_s_` or
`ix_` (C3). `CANDIDATE_PREFIXES` checks this; the selection script passes an explicit penalty vector.
"""
import numpy as np
import pandas as pd

from . import v4

PEN = 0.5                                   # first-feature-use penalty, as for the existing flag block
ROLES = ("perov", "etl", "htl", "ctc")
KINDS = ("thick_nm", "anneal_C", "anneal_min", "spin_rpm")
TOK_PREFIX = ("add:", "solv:", "anti:")
MIN_PAPERS = 85                             # numeric key reported in >= 3% of the 2,835 extracted papers
MIN_PREV = 0.03                             # training-record prevalence bounds for binary columns
MAX_PREV = 0.97
TOK_DF_MIN = 25                             # token in >= 25 extracted papers (same as v4.vocab)
ORGANIC_HTL = "spiro|ptaa|p3ht|tpd|poly"
OXIDE_ETL = "tio2|sno2|zno"
GATED = {"htl": ["htl_tbp", "htl_litfsi", "htl_fk", "htl_cu"], "etl": ["etl_ticl4", "etl_dopant"]}
CANDIDATE_PREFIXES = ("nm_c1", "tk_", "rc_s_", "ix_")


def paper_tables():
    """Label-free paper-level tables indexed by doi_group. `REC` supplies the unscoped flags used for gating."""
    P = v4.papers()
    nkeys = [f"{r}_{k}_med" for r in ROLES for k in KINDS]
    keep = [k for k in nkeys if sum(k in p["numeric"] for p in P.values()) >= MIN_PAPERS]
    C1 = pd.DataFrame.from_dict({d: {f"nm_c1_{k}": p["numeric"].get(k, np.nan) for k in keep} for d, p in P.items()}, orient="index")
    tok_count = {}
    for p in P.values():
        for t in set(p["tokens"]):
            tok_count[t] = tok_count.get(t, 0) + 1
    toks = sorted(t for t, n in tok_count.items() if n >= TOK_DF_MIN and t.startswith(TOK_PREFIX))
    C2 = pd.DataFrame.from_dict({d: {f"tk_{t}": float(t in set(p["tokens"])) for t in toks} for d, p in P.items()}, orient="index")
    C3 = pd.DataFrame.from_dict({d: {f"rc_{k}": float(v) for k, v in p["recipe"].items() if k.startswith("s_")} for d, p in P.items()}, orient="index")
    return {"C1": C1, "C2": C2, "C3": C3, "REC": v4.paper_table("rec")}


def permute_block(T, seed):
    """Paper-level control: within each column, the non-missing values are shuffled among the papers that have a value.
    The missingness pattern and the marginal distribution are unchanged; the association with papers is removed."""
    rng = np.random.default_rng(seed)
    out = T.copy()
    for c in out.columns:
        v = out[c].to_numpy(dtype=float).copy()
        m = ~np.isnan(v)
        v[m] = v[m][rng.permutation(int(m.sum()))]
        out[c] = v
    return out


def build(kind, tables, df, tr):
    """Record-level block for candidate `kind` over the rows of `df`, fitted on training rows `tr` only."""
    R = v4.attach(df, tables[kind]).reset_index(drop=True)
    trR = R.iloc[tr]
    if kind == "C1":
        out = {}
        for c in R.columns:
            med = trR[c].median()
            out[c] = R[c].fillna(med if np.isfinite(med) else 0.0)
            out[c.replace("nm_c1_", "nm_c1i_")] = R[c].notna().astype(float)   # per-key coverage indicator
        return pd.DataFrame(out)
    if kind == "C2":
        keep = [c for c in R.columns if MIN_PREV <= trR[c].mean() <= MAX_PREV]
        return R[keep].fillna(0.0)
    if kind == "C3":
        keep = [c for c in R.columns if trR[c].mean() >= MIN_PREV]
        out = R[keep].fillna(0.0)
        G = v4.attach(df, tables["REC"]).reset_index(drop=True)
        htl_gate = df["HTL"].fillna("").str.lower().str.contains(ORGANIC_HTL).astype(float).reset_index(drop=True)
        etl_gate = df["ETL"].fillna("").str.lower().str.contains(OXIDE_ETL).astype(float).reset_index(drop=True)
        gated = {}
        for f in GATED["htl"]:
            gated[f"ix_{f}_x_organicHTL"] = G[f"rc_{f}"].fillna(0.0) * htl_gate
        for f in GATED["etl"]:
            gated[f"ix_{f}_x_oxideETL"] = G[f"rc_{f}"].fillna(0.0) * etl_gate
        gated = pd.DataFrame(gated)
        gated = gated[[c for c in gated.columns if gated[c].iloc[tr].mean() >= MIN_PREV]]
        return pd.concat([out, gated], axis=1)
    raise ValueError(kind)


def combine(blocks):
    """C4: the union of the passing candidate blocks (names are already disjoint across candidates)."""
    return pd.concat(blocks, axis=1)


def penalty_vector(cols, cand_cols):
    """Penalty for each feature column: PEN on candidate columns only, 0 on the base M2 columns."""
    s = set(cand_cols)
    return [PEN if c in s else 0.0 for c in cols]


def assert_prefixes(cols):
    bad = [c for c in cols if not c.startswith(CANDIDATE_PREFIXES)]
    if bad:
        raise AssertionError(f"candidate columns without a penalty prefix: {bad[:5]}")
