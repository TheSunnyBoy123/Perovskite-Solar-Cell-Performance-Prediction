"""Train-fitted feature preprocessing (no statistic is learned from val/test)."""
import numpy as np
import pandas as pd

import re
from .data import QTY_SEL, QTY_TIERS, CAT, CAT_ALL, CAT_BASE, CAT_ID, FORBIDDEN, LX_CONTROLS, LX_TIERS, NOISE_COLS, TIER_DROP

MAX_MISSING = 0.8


def _lx_cols(df, groups):
    return [c for c in df.columns if c.startswith("lx_") and c.split("_")[1] in groups]


def _qty_cols(df):
    import json
    sel = json.loads(QTY_SEL.read_text())["selected"]
    return [c for c in sel if c in df.columns]


def _doi_level_control(df, kind, seed, cols):
    """DOI-level control block: one vector per publication, propagated to its records.
    lxperm : real lx vectors permuted across covered DOIs (keeps marginals + availability mask, kills association)
    lxnoise: N(0,1) per DOI, same availability mask as the real block (per column)."""
    one = df.drop_duplicates("doi_group").set_index("doi_group")[cols]
    rng = np.random.default_rng(seed)
    if kind == "lxperm":
        cov = one.index[df.drop_duplicates("doi_group").set_index("doi_group")["lx_covered"] == 1]
        vals = one.loc[cov].values.copy()
        vals = vals[rng.permutation(len(vals))]
        out = pd.DataFrame(vals, index=cov, columns=cols).reindex(one.index)
    else:
        out = pd.DataFrame(rng.normal(size=one.shape), index=one.index, columns=cols).where(one.notna())
    out.columns = [f"ctl_{c}" for c in cols]
    return df[["doi_group"]].join(out, on="doi_group").drop(columns="doi_group")


def base_frame(df, tier):
    """Raw (unimputed) predictor frame for a tier; row-wise, learns nothing."""
    ctl = None
    m = re.match(r"^(t2)_(lx|lxq|lxc|lxs)(perm|noise)_s(\d+)$", tier)
    lx_keep = []
    if m:   # lxq* controls mimic the full t3q block (matched columns and availability mask)
        cols = (_qty_cols(df) if m.group(2) == "lxs" else
                _lx_cols(df, LX_TIERS[{"lx": "t3x", "lxq": "t3q", "lxc": "t3q_cat"}[m.group(2)]]))
        ctl = _doi_level_control(df, "lx" + m.group(3), int(m.group(4)), cols)
        key = "t2"
    elif tier in QTY_TIERS:   # predeclared quantity features (+ chemistry descriptors for t3qq_cat), optionally on top of original t3
        key, _, _ = QTY_TIERS[tier]
        lx_keep = _qty_cols(df) + (_lx_cols(df, ("cat",)) if tier == "t3qq_cat" else [])
    elif re.match(r"^t1_(t3\w+)$", tier) and tier[3:] in LX_TIERS:   # PDP + literature, no material properties
        key = "t1"
        lx_keep = _lx_cols(df, LX_TIERS[tier[3:]])
    else:
        key = tier
        if tier in LX_TIERS:
            lx_keep = _lx_cols(df, LX_TIERS[tier])
            key = "t3" if tier == "t3b" else "t2"
    drop = set(TIER_DROP[key]) | {c for c in df.columns if c.startswith("lx_")} - set(lx_keep)
    d = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    if ctl is not None:
        d = pd.concat([d, ctl.set_index(d.index)], axis=1)
    for name, a, b in (("HTL_hole_barrier", "HTL_HOMO", "Perovskite_HOMO"),
                       ("ETL_electron_barrier", "ETL_LUMO", "Perovskite_LUMO")):
        if a in d.columns and b in d.columns:
            d[name] = np.abs(pd.to_numeric(d[a], errors="coerce")
                             - pd.to_numeric(d[b], errors="coerce"))
        else:
            d[name] = np.nan
    forb = [c for c in FORBIDDEN if not (tier.endswith("_id") and c in CAT_ID)]
    X = d.drop(columns=forb + ["partition", "lx_covered", "qty_covered", "qpartition", "doi_group", "author_group", "pub_date", "Ref_lead_author", "Ref_publication_date"], errors="ignore")
    X = X.rename(columns={"Perovskite_deposition_quenching_induced_crystallisation":
                          "Perov_quenching_crystallisation"})
    X = X.drop(columns=[c for c in X.columns
                        if not pd.api.types.is_numeric_dtype(X[c]) and c not in CAT_ALL])
    for c in CAT_BASE:
        if c not in X.columns:
            X[c] = np.nan
    if tier.endswith("_noise"):  # negative control: uninformative N(0,1) columns, fixed seed
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(rng.normal(size=(len(X), NOISE_COLS)), index=X.index,
                             columns=[f"noise_{i}" for i in range(NOISE_COLS)])
        X = pd.concat([X, noise], axis=1)
    return X


class TrainFitPreprocessor:
    """Drop >80%-missing columns (except predeclared lx_qty_*), median-impute, add `_na` flags; fit on train only."""

    def fit(self, X):
        # predeclared lx_qty_* columns are exempt from the missingness filter (train-median imputation + _na flag instead)
        self.keep_ = [c for c in X.columns if c in CAT_ALL or c.startswith(("lx_qty_", "ctl_lx_qty_")) or X[c].isnull().mean() <= MAX_MISSING]
        num = [c for c in self.keep_ if c not in CAT_ALL]
        self.na_cols_ = [c for c in num if X[c].isnull().any()]
        self.median_ = {c: float(X[c].median()) for c in num if X[c].isnull().any()}
        # train medians for every numeric column: guards against NaNs that appear only in val/test
        self.median_all_ = {c: float(X[c].median()) for c in num}
        self.cat_ = [c for c in CAT_ALL if c in self.keep_]
        return self

    def transform(self, X):
        Z = X[self.keep_].copy()
        flags = pd.DataFrame({f"{c}_na": Z[c].isnull().astype(float)
                              for c in self.na_cols_}, index=Z.index)
        for c in self.na_cols_:
            Z[c] = Z[c].fillna(self.median_[c])
        Z = pd.concat([Z, flags], axis=1)
        for c, m in self.median_all_.items():  # no-op unless a column has NaN only outside train
            if Z[c].isnull().any():
                Z[c] = Z[c].fillna(m)
        for c in self.cat_:
            Z[c] = Z[c].fillna("unknown").astype(str)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)
