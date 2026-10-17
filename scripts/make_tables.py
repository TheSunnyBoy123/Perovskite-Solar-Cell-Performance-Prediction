#!/usr/bin/env python3
"""Generate LaTeX tables + a numbers file for the manuscript from results/*/runs.jsonl.
Run from ml_paper/ after the experiments:  python scripts/make_tables.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"; T = ROOT / "psc_manuscript" / "tables"; T.mkdir(exist_ok=True)


def load(name):
    f = R / name / "runs.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(f)]) if f.exists() else None


def write_rows(path, rows):
    """Row files omit the final row terminator; ml.tex writes `\\input{...}\\\\` (a row-ending
    backslash pair directly before \\bottomrule at the end of an \\input file breaks the build)."""
    t = "\n".join(rows).rstrip()
    assert t.endswith("\\\\"), path
    Path(path).write_text(t[:-2].rstrip() + "\n")


def pm(m, s, p=3):
    return f"${m:.{p}f} \\pm {s:.{p}f}$"


TIERN = {"t1": "Database only ($\\mathcal{M}_1$)", "t2": "+ material ($\\mathcal{M}_2$)", "t3": "+ material + LLM ($\\mathcal{M}_3$)",
         "t1_lit": "$\\mathcal{M}_1$ + LLM (no material)", "t3_nometa": "$\\mathcal{M}_3$ minus 2 completeness flags",
         "t2_noise": "$\\mathcal{M}_2$ + 32 noise columns"}
ARCHN = {"overall": "Pooled", "pin": "p-i-n", "nip": "n-i-p"}
import sys; sys.path.insert(0, str(ROOT / "src"))
from psc.results import load_ablation  # noqa: E402
A = load_ablation(); nums = {}

# ---- Table: tiers (CatBoost, DOI-grouped)
x = A[(A.protocol == "doi") & (A.model == "catboost")]
rows = []
for arch in ("overall", "pin", "nip"):
    for t in ("t1", "t2", "t3"):
        g = x[(x.arch == arch) & (x.tier == t)]
        rows.append(f"{ARCHN[arch]} & {TIERN[t]} & {int(g.n_test.mean()):,} & {int(round(g.n_features.mean()))} & "
                    f"{pm(g.r2.mean(), g.r2.std())} & {pm(g.mae.mean(), g.mae.std())} & {pm(g.rmse.mean(), g.rmse.std())} \\\\")
        nums[f"r2_{arch}_{t}"] = g.r2.mean(); nums[f"nfeat_{arch}_{t}"] = g.n_features.mean()
    if arch != "nip": rows.append("\\midrule")
write_rows(T / "tiers.tex", rows)

# ---- Table: paired differences
b = pd.read_csv(R / "ablation_v1" / "paired_bootstrap_doi_catboost.csv")
rows = []
for arch in ("overall", "pin", "nip"):
    cells = []
    for c in ("t2-t1", "t3-t2"):
        y = b[(b.arch == arch) & (b.contrast == c)]
        cells.append(f"${y.d_r2.mean():+.3f}$ & ${y.d_rmse.mean():+.3f}$ & {int((y.d_r2 > 0).sum())}/5 & "
                     f"{int(((y.r2_lo > 0) | (y.r2_hi < 0)).sum())}/5")
    rows.append(f"{ARCHN[arch]} & " + " & ".join(cells) + " \\\\")
write_rows(T / "delta.tex", rows)

# ---- Table: random vs DOI (CatBoost + ridge)
rows = []
for m, mn in (("catboost", "CatBoost"), ("ridge", "Ridge")):
    for t in ("t1", "t2", "t3"):
        d = A[(A.arch == "overall") & (A.model == m) & (A.tier == t)]
        a = d[d.protocol == "doi"].sort_values("split"); r = d[d.protocol == "random"].sort_values("split")
        gap = r.r2.values - a.r2.values
        rows.append(f"{mn} & {TIERN[t]} & {pm(r.r2.mean(), r.r2.std())} & {pm(a.r2.mean(), a.r2.std())} & "
                    f"${gap.mean():.3f}$ & ${r.mae.mean():.2f}$ & ${a.mae.mean():.2f}$ \\\\")
        nums[f"gap_{m}_{t}"] = gap.mean()
    if m == "catboost": rows.append("\\midrule")
write_rows(T / "split.tex", rows)

# ---- Table: protocol facts
d = x[(x.arch == "overall") & (x.tier == "t3")]
nums.update(n_train=d.n_train.mean(), n_val=d.n_val.mean(), n_test=d.n_test.mean(), doi_train=d.n_doi_train.mean(),
            doi_val=d.n_doi_val.mean(), doi_test=d.n_doi_test.mean(),
            max_overlap=int(A[A.protocol == "doi"][["doi_overlap_test_train", "doi_overlap_val_train", "doi_overlap_test_val"]].max().max()))

# ---- Table: feature-group controls
G = load("feature_groups_v1")
if G is not None:
    rows = []
    for arch in ("overall", "pin", "nip"):
        for t in ("t1_lit", "t3_nometa", "t2_noise"):
            g = G[(G.arch == arch) & (G.tier == t)]
            if len(g) == 5:
                rows.append(f"{ARCHN[arch]} & {TIERN[t]} & {int(round(g.n_features.mean()))} & {pm(g.r2.mean(), g.r2.std())} & "
                            f"{pm(g.mae.mean(), g.mae.std())} & {pm(g.rmse.mean(), g.rmse.std())} \\\\")
    write_rows(T / "groups.tex", rows)

# ---- Table: baselines (pooled, DOI-grouped)
Bl = load("baselines_v1")
parts = []
base = A[(A.protocol == "doi") & (A.arch == "overall")]
for m, mn in (("mean", "Training-mean predictor"), ("ridge", "Ridge regression"), ("catboost", "CatBoost (matched)")):
    for t in ("t1", "t3"):
        g = base[(base.model == m) & (base.tier == t)]
        parts.append((mn, t, g))
if Bl is not None:
    for m, mn in (("xgb_default", "XGBoost (library defaults)"), ("rf", "Random forest"), ("mlp", "MLP (val. early stopping)"),
                  ("gpr_sub", "GPR (3,000-record subsample)")):
        for t in ("t1", "t3"):
            g = Bl[(Bl.model == m) & (Bl.tier == t)]
            if len(g) == 5: parts.append((mn, t, g))
LAB = {'t1': '$\\mathcal{M}_1$', 't3': '$\\mathcal{M}_3$'}
rows = [f"{mn} & {LAB[t]} & {pm(g.r2.mean(), g.r2.std())} & {pm(g.mae.mean(), g.mae.std())} & {pm(g.rmse.mean(), g.rmse.std())} \\\\"
        for mn, t, g in parts]
write_rows(T / "baselines.tex", rows)

# ---- Table: model development
M = load("model_dev_v1")
if M is not None:
    rows = []
    for arch in ("overall", "pin", "nip"):
        for t in ("t1", "t2", "t3"):
            g = M[(M.arch == arch) & (M.tier == t)]
            if len(g) == 5:
                rows.append(f"{ARCHN[arch]} & {TIERN[t]} & " + " & ".join(
                    pm(g[f"{k}_r2"].mean(), g[f"{k}_r2"].std()) for k in ("cb_tuned", "cb_bag", "xgb", "ens")) + " \\\\")
                nums[f"dev_ens_r2_{arch}_{t}"] = g["ens_r2"].mean()
                nums[f"dev_ens_mae_{arch}_{t}"] = g["ens_mae"].mean()
                nums[f"dev_ens_rmse_{arch}_{t}"] = g["ens_rmse"].mean()
    write_rows(T / "model_dev.tex", rows)
json.dump({k: float(v) for k, v in nums.items()}, open(R / "manuscript_numbers.json", "w"), indent=1)
print(json.dumps({k: round(float(v), 4) for k, v in nums.items()}, indent=0)[:1500])

# ---- Table: strict validation (pooled): DOI-grouped vs lead-author-grouped vs forward-in-time
S = load("strict_validation_v1")
if S is not None and len(S) >= 216:
    D = A[(A.protocol == "doi") & (A.arch == "overall") & (A.model.isin(["mean", "ridge", "catboost"]))]
    allp = pd.concat([D, S[S.arch == "overall"]], ignore_index=True)
    rows = []
    for proto, pn in (("random", "Record-level"), ("doi", "DOI-grouped"), ("author", "Lead-author-grouped"), ("temporal", "Forward in time")):
        for m, mn in (("catboost", "CatBoost"), ("ridge", "Ridge"), ("mean", "Mean")):
            x = allp[(allp.protocol == proto) & (allp.model == m)] if proto != "random" else A[(A.protocol == "random") & (A.arch == "overall") & (A.model == m)]
            if x.empty: continue
            cells = [pm(x[x.tier == t].r2.mean(), x[x.tier == t].r2.std()) for t in ("t1", "t2", "t3")]
            rows.append(f"{pn} & {mn} & {int(round(x[x.tier == 't3'].n_test.mean())):,} & {int(x[x.tier == 't3'].split.nunique())} & " + " & ".join(cells) + f" & ${x[x.tier == 't3'].mae.mean():.2f}$ \\\\")
    write_rows(T / "strict.tex", rows)
    # per-window temporal detail (CatBoost, pooled)
    tw = S[(S.arch == "overall") & (S.protocol == "temporal") & (S.model == "catboost")]
    trows = []
    for k in sorted(tw.split.unique()):
        g = tw[tw.split == k]; r = g.iloc[0]
        trows.append(f"{k} & {r.train_date_max} & {r.test_date_min} & {int(r.n_train):,} & {int(r.n_test):,} & " + " & ".join(f"${g[g.tier == t].r2.iloc[0]:.3f}$" for t in ("t1", "t2", "t3")) + " \\\\")
    write_rows(T / "temporal.tex", trows)
    # paired t2-t1, t3-t2 under author / temporal
    for proto in ("author", "temporal"):
        for m in ("catboost",):
            w = S[(S.arch == "overall") & (S.protocol == proto) & (S.model == m)].pivot(index="split", columns="tier", values="r2")
            nums[f"d_{proto}_t2t1"] = (w.t2 - w.t1).mean(); nums[f"d_{proto}_t3t2"] = (w.t3 - w.t2).mean()
            nums[f"r2_{proto}_catboost_t3"] = w.t3.mean()
    json.dump({k: float(v) for k, v in nums.items()}, open(R / "manuscript_numbers.json", "w"), indent=1)
