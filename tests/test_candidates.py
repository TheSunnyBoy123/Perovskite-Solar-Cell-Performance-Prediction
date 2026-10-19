"""Safeguards for the covered-population candidate representations (no LLM data needed)."""
import numpy as np
import pandas as pd

from psc import candidates as cd
from psc.splits import make_splits


def _df(n_docs=6, per=4):
    docs = [f"10.1/d{i}" for i in range(n_docs) for _ in range(per)]
    return pd.DataFrame({"doi_group": docs,
                         "HTL": ["spiro-OMeTAD"] * (len(docs) // 2) + ["NiOx"] * (len(docs) - len(docs) // 2),
                         "ETL": ["TiO2"] * len(docs)})


def test_c1_medians_use_training_rows_only():
    df = _df()
    T = pd.DataFrame({"nm_c1_perov_thick_nm_med": [400.0, 410, 420, 430, 440, 450]},
                     index=[f"10.1/d{i}" for i in range(6)])
    tr = np.arange(0, 16)          # docs 0-3
    te = np.arange(16, 24)         # docs 4-5
    base = cd.build("C1", {"C1": T}, df, tr)
    T2 = T.copy(); T2.iloc[4:] = 1e6          # change only the test-paper values
    changed = cd.build("C1", {"C1": T2}, df, tr)
    pd.testing.assert_frame_equal(base.iloc[tr].reset_index(drop=True), changed.iloc[tr].reset_index(drop=True))
    assert not np.allclose(base.iloc[te]["nm_c1_perov_thick_nm_med"], changed.iloc[te]["nm_c1_perov_thick_nm_med"])


def test_c2_prevalence_is_fitted_on_training_rows():
    df = _df()
    docs = [f"10.1/d{i}" for i in range(6)]
    # token present only in papers 4-5 (test): training prevalence 0, so the column must be dropped
    T = pd.DataFrame({"tk_add:litfsi": [0, 0, 0, 0, 1, 1], "tk_solv:dmso": [1, 0, 1, 0, 1, 0]}, index=docs, dtype=float)
    tr = np.arange(0, 16)
    out = cd.build("C2", {"C2": T}, df, tr)
    assert "tk_add:litfsi" not in out.columns
    assert "tk_solv:dmso" in out.columns


def test_c3_gating_and_penalty_prefixes():
    df = _df()
    docs = [f"10.1/d{i}" for i in range(6)]
    S = pd.DataFrame({"rc_s_cs": [1, 1, 0, 0, 1, 1]}, index=docs, dtype=float)
    R = pd.DataFrame({f"rc_{f}": [1, 1, 1, 1, 0, 0] for f in cd.GATED["htl"] + cd.GATED["etl"]}, index=docs, dtype=float)
    tables = {"C3": S, "REC": R}
    tr = np.arange(0, 16)
    out = cd.build("C3", tables, df, tr)
    org = df["HTL"].str.lower().str.contains(cd.ORGANIC_HTL).astype(float).values
    # HTL gating: flag x organic-HTL indicator (spiro-OMeTAD rows only)
    assert set(np.unique(out["ix_htl_tbp_x_organicHTL"].values)) <= {0.0, 1.0}
    np.testing.assert_allclose(out["ix_htl_tbp_x_organicHTL"].values, R["rc_htl_tbp"].reindex(df.doi_group).values * org)
    cd.assert_prefixes(out.columns)
    pen = cd.penalty_vector(list(out.columns) + ["t2_base"], out.columns)
    assert pen[-1] == 0.0 and all(p == cd.PEN for p in pen[:-1])


def test_permutation_control_keeps_missingness_and_values():
    T = pd.DataFrame({"a": [1.0, np.nan, 3.0, 7.0, np.nan, 9.0], "b": [0.0, 1, 1, 0, 0, 1]})
    P = cd.permute_block(T, seed=3)
    assert (P.isna() == T.isna()).all().all()
    np.testing.assert_array_equal(np.sort(P["a"].dropna().values), np.sort(T["a"].dropna().values))
    np.testing.assert_array_equal(np.sort(P["b"].values), np.sort(T["b"].values))


def test_kfold_folds_are_doi_disjoint():
    groups = np.repeat([f"g{i}" for i in range(40)], 5)
    for tr, va, te in make_splits(groups, "kfold", 5, 0.2, 0.15, 42):
        assert not set(groups[tr]) & set(groups[te])
        assert not set(groups[va]) & set(groups[te])
        assert not set(groups[tr]) & set(groups[va])
