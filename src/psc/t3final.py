"""Final t3 = t2 + the 36 V4 recipe flags (looked up by DOI from data/lit_v4.json) for the protocol re-runs.

The flag block is appended to the train-fitted t2 matrix. CatBoost keeps missing flags native and applies the documented
first-feature-use penalty (0.5) to the flag columns; estimators that cannot take missing values (ridge, MLP, GPR, random
forest) receive train-mean imputed flags plus one coverage indicator (`rc_covered`)."""
import numpy as np
import pandas as pd

from .features import TrainFitPreprocessor, base_frame
from . import v4

PEN = 0.5
_REC = {}


def rec_block(df):
    if "T" not in _REC:
        _REC["T"] = v4.paper_table("rec")
    return v4.attach(df, _REC["T"])


def matrices(df, Xraw, tr, va, te, tier, impute=False, R=None):
    """(Xtr, Xva, Xte, penalties|None, n_flags). Preprocessing statistics (and flag means) use the training records only."""
    pre = TrainFitPreprocessor().fit(Xraw.iloc[tr])
    Z = [pre.transform(Xraw.iloc[i]) for i in (tr, va, te)]
    if tier != "t3f":
        return (*Z, None, 0)
    R = rec_block(df) if R is None else R
    parts = [R.iloc[i].reset_index(drop=True) for i in (tr, va, te)]
    if impute:
        mu = parts[0].mean()                                    # nan-skipping mean over training records
        cov = [p.notna().any(axis=1).astype(float).rename("rc_covered") for p in parts]
        parts = [p.fillna(mu).fillna(0.0).join(c) for p, c in zip(parts, cov)]
    out = [pd.concat([z.reset_index(drop=True), p], axis=1) for z, p in zip(Z, parts)]
    pen = [PEN if c.startswith("rc_") else 0.0 for c in out[0].columns]
    return (*out, pen, R.shape[1])
