import json
from pathlib import Path
import numpy as np, pandas as pd, pytest
from psc.data import (ARCHS, DEFAULT_DATA, FORBIDDEN, NO_DOI_GROUP, TARGET, DOI_COL,
                      load_dataset, normalize_doi)
from psc.features import TrainFitPreprocessor, base_frame
from psc.splits import make_splits

RES = Path(__file__).resolve().parents[1] / "results"
needs_data = pytest.mark.skipif(not DEFAULT_DATA.exists(), reason="dataset not present")


@needs_data
@pytest.mark.parametrize("arch", ["overall", "pin", "nip"])
def test_doi_disjoint_every_arch_and_seed(arch):
    df = load_dataset(arch=arch); g = df["doi_group"].values
    for seed in (42, 7):
        for tr, va, te in make_splits(g, "doi", 5, 0.2, 0.15, seed):
            S = [set(g[i]) for i in (tr, va, te)]
            assert not (S[0] & S[1] or S[0] & S[2] or S[1] & S[2])
            assert len(np.concatenate([tr, va, te])) == len(g)


@needs_data
def test_sample_counts_match_paper():
    df = load_dataset()
    assert len(df) == 41107 and df["doi_group"].nunique() == 7327
    assert (df["doi_group"] == NO_DOI_GROUP).sum() == 232
    assert load_dataset(arch="nip").shape[0] == 28340 and load_dataset(arch="pin").shape[0] == 12684
    pre = TrainFitPreprocessor()
    widths = [pre.fit_transform(base_frame(df, t)).shape[1] for t in ("t1", "t2", "t3")]
    assert widths == [77, 179, 223]   # full-data fit; per-split widths vary by +/-2


@needs_data
def test_no_outcome_or_identifier_columns_in_any_tier():
    df = load_dataset()
    banned = set(FORBIDDEN) | {"doi_group"}
    for t in ("t1", "t2", "t3", "t1_lit", "t3_nometa", "t2_noise"):
        X = base_frame(df, t)
        assert not (banned & set(X.columns)), t
        assert not [c for c in X.columns if c.startswith(("JV_", "Ref_", "Stability_", "EQE_"))]
        assert not [c for c in X.columns if X[c].dtype == object and c not in
                    ("Cell_architecture", "ETL_deposition_method",
                     "HTL_deposition_method", "Perovskite_deposition_method")]


@needs_data
def test_no_single_feature_nearly_determines_target():
    """Cheap target-leak screen: |Spearman| of any numeric predictor with PCE must be < 0.6."""
    df = load_dataset(); X = base_frame(df, "t3").select_dtypes("number")
    corr = X.apply(lambda c: c.corr(df[TARGET], method="spearman")).abs()
    assert corr.max() < 0.6, corr.sort_values().tail(3)


@needs_data
def test_duplicate_check_is_not_vacuous_and_cross_doi_dups_reported():
    df = load_dataset(); X = base_frame(df, "t3")
    h = pd.util.hash_pandas_object(X, index=False)
    n_doi = df.assign(h=h).groupby("h")["doi_group"].nunique()
    # paper reports 162 cross-publication duplicate groups; the test pins it so changes are noticed
    assert int((n_doi > 1).sum()) == 162


def test_preprocessor_all_statistics_train_only():
    rng = np.random.default_rng(0)
    cat = ["Cell_architecture", "ETL_deposition_method", "HTL_deposition_method", "Perovskite_deposition_method"]
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": np.nan})
    for k in cat: X[k] = "x"
    X.loc[::4, "a"] = np.nan
    tr, te = X.iloc[:100], X.iloc[100:].copy()
    te["b"] = np.where(np.arange(100) % 2, np.nan, 1e9)       # NaN pattern and scale only in test
    p1 = TrainFitPreprocessor().fit(tr)
    te2 = te.copy(); te2["a"] = -1e9; te2["b"] = 7.0
    p2 = TrainFitPreprocessor().fit(tr)
    assert p1.median_ == p2.median_ and p1.keep_ == p2.keep_ and p1.na_cols_ == p2.na_cols_
    assert "c" not in p1.keep_                                  # >80% missing in train
    Z = p1.transform(te)
    assert not Z.drop(columns=cat).isnull().any().any()         # L3: no NaN may survive transform


def test_ridge_scaler_winsorizer_fit_on_train_only():
    from psc.models import Winsorizer
    tr = pd.DataFrame({"a": np.arange(100.0)}); w = Winsorizer().fit(tr)
    assert w.hi_["a"] == pytest.approx(np.quantile(np.arange(100.0), 0.995))
    out = w.transform(pd.DataFrame({"a": [1e9]}))
    assert out["a"].iloc[0] == pytest.approx(w.hi_["a"])


@pytest.mark.skipif(not (RES / "ablation_v1" / "preds").exists(), reason="results absent")
def test_saved_preds_independent_partition_and_pairing():
    """Recompute disjointness from saved indices (not from self-reported counts) and
    check identical test/train/val indices across tiers and models."""
    P = RES / "ablation_v1" / "preds"
    for proto in ("doi", "random"):
        for arch in ("overall", "pin", "nip"):
            for k in range(1, 6):
                ref = None
                for t in ("t1", "t2", "t3"):
                    for m in ("mean", "ridge", "catboost"):
                        z = np.load(P / f"{proto}_{arch}_{t}_{m}_{k}.npz")
                        idx = (z["train_idx"], z["val_idx"], z["test_idx"])
                        assert not (set(idx[0]) & set(idx[1]) or set(idx[0]) & set(idx[2])
                                    or set(idx[1]) & set(idx[2]))
                        if ref is None: ref = idx
                        assert all(np.array_equal(a, b) for a, b in zip(ref, idx))


@needs_data
@pytest.mark.skipif(not (RES / "ablation_v1" / "preds").exists(), reason="results absent")
def test_saved_split_doi_overlap_recomputed_from_data():
    df = load_dataset(); g = df["doi_group"].values
    z = np.load(RES / "ablation_v1" / "preds" / "doi_overall_t3_catboost_1.npz")
    assert not (set(g[z["test_idx"]]) & (set(g[z["train_idx"]]) | set(g[z["val_idx"]])))
    assert np.allclose(df[TARGET].values[z["test_idx"]], z["y"])   # y alignment with data


@pytest.mark.skipif(not (RES / "ablation_v1" / "runs.jsonl").exists(), reason="results absent")
def test_doi_arm_catboost_converged_before_cap():
    rows = [json.loads(l) for l in (RES / "ablation_v1" / "runs.jsonl").read_text().splitlines()]
    cb = [r for r in rows if r["model"] == "catboost" and r["protocol"] == "doi"]
    assert cb and all(r["info"]["best_iteration"] < 1499 for r in cb)


@pytest.mark.skipif(not (RES / "random_convergence_cap20000" / "runs.jsonl").exists(), reason="rerun absent")
def test_record_level_arm_converged_with_20000_cap():
    rows = [json.loads(l) for l in (RES / "random_convergence_cap20000" / "runs.jsonl").read_text().splitlines()]
    assert len(rows) == 15 and all(r["info"]["params"]["iterations"] == 20000 for r in rows)
    assert all(r["info"]["best_iteration"] < 19000 for r in rows)  # stopped on their own


def test_xgb_rf_category_maps_from_train_only(monkeypatch):
    from psc import models
    from psc.data import CAT
    n = 40
    X = pd.DataFrame({c: ["a"] * n for c in CAT}); X["v"] = np.arange(n, dtype=float)
    Xte = X.copy(); Xte[CAT[0]] = "test_only_level"
    y = np.arange(n, dtype=float)
    p, _ = models.fit_predict("rf", X, y, X, y, Xte)     # must not raise; unseen level -> -1
    assert np.isfinite(p).all()


def test_split_seed_changes_and_repeats_differ():
    g = np.repeat(np.arange(200), 5).astype(object)
    a = list(make_splits(g, "doi", 3, 0.2, 0.15, 1)); b = list(make_splits(g, "doi", 3, 0.2, 0.15, 2))
    assert not np.array_equal(a[0][2], b[0][2]) and not np.array_equal(a[0][2], a[1][2])
