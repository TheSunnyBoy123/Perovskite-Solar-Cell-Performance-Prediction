"""Deployable t2 / t3 model bundles (CatBoost seed ensembles + train-fitted preprocessing + lx4 recipe lookup).

Inference needs no LLM: the recipe flags are looked up from the versioned artifact data/lit_v4.json by DOI,
or supplied directly as `rc_*` columns; papers without an entry get NaN (the model handles missing natively).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .data import CAT_ALL, normalize_doi
from .features import TrainFitPreprocessor, base_frame
from .splits import make_splits
from . import v4

DEFAULT_PARAMS = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, subsample=0.8,
                      colsample_bylevel=0.7, min_data_in_leaf=5, loss_function="RMSE")


def rec_columns(df):
    """Recipe flag matrix aligned to df rows: explicit rc_* columns win, else DOI lookup, else NaN."""
    T = v4.paper_table("rec")
    if "doi_group" not in df:
        key = df["Ref_DOI_number"].map(normalize_doi) if "Ref_DOI_number" in df else pd.Series(["__NO_DOI__"] * len(df), index=df.index)
        df = df.assign(doi_group=key)
    R = v4.attach(df, T)
    for c in T.columns:
        if c in df:
            R[c] = pd.to_numeric(df[c], errors="coerce").where(df[c].notna(), R[c])
    return R


def _penalty(cols, pen):
    return [pen if c.startswith("rc_") else 0.0 for c in cols]


class PSCModel:
    def __init__(self, tier="t3", params=None, pen_lit=0.5, seeds=(0, 1, 2, 3, 4)):
        assert tier in ("t1", "t2", "t3")
        self.tier, self.params, self.pen_lit, self.seeds = tier, dict(params or {}), pen_lit, tuple(seeds)

    def _base(self):
        return "t1" if self.tier == "t1" else "t2"   # recipe flags (t3) are appended to the t2 matrix

    def _matrix(self, df, pre):
        Z = pre.transform(base_frame(df, self._base()))
        if self.tier == "t3":
            Z = pd.concat([Z, rec_columns(df)], axis=1)
        return Z

    def fit(self, df, y, groups, val_size=0.15):
        """df: PDP-schema records (dataset_json_enriched columns incl. doi_group). One model per seed, each with its
        own DOI-disjoint validation split used only for early stopping; the preprocessor is fit on that seed's train part."""
        self.models_, self.pres_, self.cols_ = [], [], None
        X = base_frame(df, self._base())
        R = rec_columns(df) if self.tier == "t3" else None
        for s in self.seeds:
            tr, va, _ = next(make_splits(groups, "doi", 1, 0.001, val_size, 1000 + s))
            tr = np.concatenate([tr, _])  # the 0.1% outer 'test' sliver is returned to training
            pre = TrainFitPreprocessor().fit(X.iloc[tr])
            Z = pre.transform(X)
            if R is not None:
                Z = pd.concat([Z, R], axis=1)
            p = {**DEFAULT_PARAMS, **self.params, "random_seed": s}
            if R is not None and self.pen_lit:
                p["first_feature_use_penalties"] = _penalty(Z.columns, self.pen_lit); p["penalties_coefficient"] = 1.0
            ci = [i for i, c in enumerate(Z.columns) if c in set(CAT_ALL)]
            m = CatBoostRegressor(**p, verbose=0, thread_count=-1)
            m.fit(Z.iloc[tr], y[tr], cat_features=ci, eval_set=(Z.iloc[va], y[va]), early_stopping_rounds=100, verbose=False)
            self.models_.append(m); self.pres_.append(pre); self.cols_ = list(Z.columns)
        return self

    def predict(self, df, per_seed=False):
        P = []
        for m, pre in zip(self.models_, self.pres_):
            P.append(m.predict(self._matrix(df, pre)[self.cols_]))
        P = np.array(P)
        return P if per_seed else P.mean(0)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
