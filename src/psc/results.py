"""Load experiment records. Record-level CatBoost fits come from the raised-cap rerun when present
(the 1,500-iteration cap truncated 31/45 record-level fits; DOI-grouped fits all stop earlier)."""
import json
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results"


def read(name):
    f = RES / name / "runs.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(f)]) if f.exists() else None


def load_ablation(use_cap_rerun=True):
    A = read("ablation_v1")
    C = read("ablation_random_cap6000") if use_cap_rerun else None
    if C is not None and len(C) == 45:
        A = pd.concat([A[~((A.protocol == "random") & (A.model == "catboost"))], C], ignore_index=True)
    V = read("random_convergence_cap20000") if use_cap_rerun else None
    if V is not None and len(V) == 15:  # converged pooled record-level fits (best iteration 7.2k-11.7k, never capped)
        A = pd.concat([A[~((A.protocol == "random") & (A.model == "catboost") & (A.arch == "overall"))], V], ignore_index=True)
    return A
