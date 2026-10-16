import numpy as np
import pandas as pd
import pytest

from psc.data import (CAT, DEFAULT_DATA, FORBIDDEN, LIT32, MAT49, NO_DOI_GROUP,
                      TARGET, TIER_DROP, load_dataset, normalize_doi)
from psc.features import TrainFitPreprocessor, base_frame
from psc.splits import make_splits


def test_normalize_doi_variants():
    a = "10.1039/C3EE43619E"
    assert normalize_doi(a) == normalize_doi("10.1039/c3ee43619e")
    assert normalize_doi("﻿10.1016/j.x.1") == "10.1016/j.x.1"
    assert normalize_doi("https://doi.org/10.1/ABC") == "10.1/abc"
    assert normalize_doi("doi: 10.1/abc") == "10.1/abc"
    assert normalize_doi(np.nan) == NO_DOI_GROUP == normalize_doi("")


def synth(n_doi=60, per=8, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"10.1/{i}" for i in range(n_doi)], per).astype(object)
    g[:5] = NO_DOI_GROUP
    return g


@pytest.mark.parametrize("protocol", ["doi", "random"])
def test_split_partition_and_reproducible(protocol):
    g = synth()
    s1 = list(make_splits(g, protocol, 4, 0.2, 0.15, 42))
    s2 = list(make_splits(g, protocol, 4, 0.2, 0.15, 42))
    for (a, b, c), (a2, b2, c2) in zip(s1, s2):
        assert (a == a2).all() and (b == b2).all() and (c == c2).all()
        allidx = np.concatenate([a, b, c])
        assert len(allidx) == len(set(allidx)) == len(g)  # exact partition


def test_doi_split_disjoint_including_no_doi_block():
    g = synth()
    for tr, va, te in make_splits(g, "doi", 5, 0.2, 0.15, 7):
        s = [set(g[i]) for i in (tr, va, te)]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
        # all missing-DOI records stay together
        holders = [k for k, x in enumerate(s) if NO_DOI_GROUP in x]
        assert len(holders) == 1


def test_random_split_does_mix_groups():
    g = synth()
    tr, va, te = next(make_splits(g, "random", 1, 0.2, 0.15, 1))
    assert set(g[tr]) & set(g[te])


def test_preprocessor_fit_on_train_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    X.loc[::3, "a"] = np.nan
    for c in CAT:
        X[c] = "x"
    tr, te = X.iloc[:60], X.iloc[60:].copy()
    te.loc[:, "a"] = 1e6  # extreme test values must not affect train statistics
    pre = TrainFitPreprocessor().fit(tr)
    assert pre.median_["a"] == pytest.approx(tr["a"].median())
    assert pre.keep_ == TrainFitPreprocessor().fit(tr).keep_
    Z = pre.transform(te)
    assert Z["a_na"].sum() == 0 and "a_na" in Z


def test_tiers_nested_and_clean():
    assert TIER_DROP["t3"] <= TIER_DROP["t2"] <= TIER_DROP["t1"]
    assert len(MAT49) == 49 and len(LIT32) == 32 and not (MAT49 & LIT32)
    assert TARGET in FORBIDDEN and "Ref_DOI_number" in FORBIDDEN


@pytest.mark.skipif(not DEFAULT_DATA.exists(), reason="dataset not present")
def test_real_data_features_nested_no_target_leak():
    df = load_dataset()
    cols = {}
    for t in ("t1", "t2", "t3"):
        X = base_frame(df, t)
        cols[t] = set(X.columns)
        assert TARGET not in X.columns and "doi_group" not in X.columns
        assert not any(c.lower().startswith("final_pce") for c in X.columns)
        assert X.index.equals(df.index)
    assert cols["t1"] < cols["t2"] < cols["t3"]
    # repeated rows: none identical incl. target
    assert df.drop(columns=["doi_group"]).duplicated().sum() == 0
    # normalisation merges DOI variants (hidden leakage under raw strings)
    raw = df["Ref_DOI_number"].dropna().astype(str).nunique()
    assert df["doi_group"].nunique() - 1 < raw


@pytest.mark.skipif(not DEFAULT_DATA.exists(), reason="dataset not present")
def test_real_data_doi_splits_no_overlap():
    df = load_dataset()
    g = df["doi_group"].values
    for tr, va, te in make_splits(g, "doi", 5, 0.2, 0.15, 42):
        s = [set(g[i]) for i in (tr, va, te)]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
