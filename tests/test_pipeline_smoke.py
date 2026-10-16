import inspect
import numpy as np
import pandas as pd
import pytest

from psc import models
from psc.data import CAT, NO_DOI_GROUP, TARGET, load_dataset
from psc.evaluate import metrics
from psc.features import TrainFitPreprocessor, base_frame
from psc.splits import make_splits


def synthetic_csv(path, n_doi=80, per=10, seed=0):
    rng = np.random.default_rng(seed)
    n = n_doi * per
    doi = np.repeat([f"10.9/{i}" for i in range(n_doi)], per).astype(object)
    doi[:4] = np.nan
    paper_effect = np.repeat(rng.normal(size=n_doi), per)
    df = pd.DataFrame({
        "Ref_ID": np.arange(n), "Ref_DOI_number": doi,
        "Cell_architecture": rng.choice(["nip", "pin"], n),
        "ETL_deposition_method": rng.choice(["spin", "evap", None], n),
        "HTL_deposition_method": rng.choice(["spin", "evap"], n),
        "Perovskite_deposition_method": rng.choice(["spin", "evap"], n),
        "ETL_LUMO": rng.normal(size=n), "Perovskite_LUMO": rng.normal(size=n),
        "HTL_HOMO": rng.normal(size=n), "Perovskite_HOMO": rng.normal(size=n),
        "ETL_band_gap": rng.normal(size=n), "has_BCP": rng.integers(0, 2, n).astype(float),
        "spin_speed_rpm": np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)),
        "x1": rng.normal(size=n)})
    df[TARGET] = 10 + 2 * df.x1 + paper_effect + rng.normal(size=n)
    df.to_csv(path, index=False)
    return path


def test_end_to_end_smoke(tmp_path):
    f = synthetic_csv(tmp_path / "s.csv")
    df = load_dataset(f)
    assert df["doi_group"].eq(NO_DOI_GROUP).sum() == 4
    y = df[TARGET].values
    for tier in ("t1", "t2", "t3"):
        X = base_frame(df, tier)
        tr, va, te = next(make_splits(df["doi_group"].values, "doi", 1, 0.2, 0.15, 1))
        pre = TrainFitPreprocessor().fit(X.iloc[tr])
        a, b, c = (pre.transform(X.iloc[i]) for i in (tr, va, te))
        assert list(a.columns) == list(b.columns) == list(c.columns)
        assert not a.drop(columns=CAT).isnull().any().any()
        for m, p in (("mean", None), ("ridge", None), ("catboost", {"iterations": 30})):
            pred, _ = models.fit_predict(m, a, y[tr], b, y[va], c, p)
            assert pred.shape == (len(te),) and np.isfinite(pred).all()
        assert set(metrics(y[te], pred)) == {"r2", "mae", "rmse"}


def test_fit_predict_never_receives_test_targets():
    assert list(inspect.signature(models.fit_predict).parameters) == \
        ["name", "Xtr", "ytr", "Xva", "yva", "Xte", "params"]


def test_early_stopping_uses_validation_not_test(tmp_path, monkeypatch):
    f = synthetic_csv(tmp_path / "s.csv")
    df = load_dataset(f); y = df[TARGET].values
    X = base_frame(df, "t3")
    tr, va, te = next(make_splits(df["doi_group"].values, "doi", 1, 0.2, 0.15, 1))
    pre = TrainFitPreprocessor().fit(X.iloc[tr])
    a, b, c = (pre.transform(X.iloc[i]) for i in (tr, va, te))
    import catboost
    seen = {}
    orig = catboost.CatBoostRegressor.fit

    def spy(self, X_, y_=None, **kw):
        seen["eval"] = kw["eval_set"]
        return orig(self, X_, y_, **kw)
    monkeypatch.setattr(catboost.CatBoostRegressor, "fit", spy)
    models.fit_predict("catboost", a, y[tr], b, y[va], c, {"iterations": 10})
    ex, ey = seen["eval"]
    assert ex.equals(b) and np.array_equal(ey, y[va])
    assert len(ex) != len(c) or not ex.equals(c)


def test_prediction_invariant_to_test_targets(tmp_path):
    """Changing test-set targets cannot change predictions (no target leakage path)."""
    f = synthetic_csv(tmp_path / "s.csv")
    df = load_dataset(f); y = df[TARGET].values
    X = base_frame(df, "t3")
    tr, va, te = next(make_splits(df["doi_group"].values, "doi", 1, 0.2, 0.15, 1))
    pre = TrainFitPreprocessor().fit(X.iloc[tr])
    a, b, c = (pre.transform(X.iloc[i]) for i in (tr, va, te))
    p1, _ = models.fit_predict("ridge", a, y[tr], b, y[va], c)
    y2 = y.copy(); y2[te] += 100
    p2, _ = models.fit_predict("ridge", a, y2[tr], b, y2[va], c)
    assert np.allclose(p1, p2)


def test_run_records_have_zero_doi_overlap():
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "results" / "ablation_v1" / "runs.jsonl"
    if not p.exists():
        pytest.skip("full results not present")
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    for r in rows:
        if r["protocol"] == "doi":
            assert r["doi_overlap_test_train"] == r["doi_overlap_val_train"] == r["doi_overlap_test_val"] == 0
    assert len(rows) == 270
