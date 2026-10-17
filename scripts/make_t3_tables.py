#!/usr/bin/env python3
"""Row blocks for the manuscript tables that now include the final t3 (t2 + 36 recipe flags).
Reads results/ablation_v1 (+ converged record-level reruns), strict_validation_v1 and results/t3final_matched_v1 and
results/t3final_summary/matched_pairs.csv; writes results/t3final_summary/table_rows.json (label -> list of LaTeX rows,
without the final row terminator on the last row) and prints them. scripts/patch_t3_tables.py inserts them into ml.tex."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from psc.results import load_final  # noqa: E402

M1, M2, M3 = r"$\mathcal{M}_1$", r"$\mathcal{M}_2$", r"$\mathcal{M}_3$"
TIER = {"t1": f"Database only ({M1})", "t2": f"+ material ({M2})", "t3": f"+ recipe flags ({M3})"}
pm = lambda x, p=3: f"${x.mean():.{p}f} \\pm {x.std():.{p}f}$"
ARCHN = {"overall": "Pooled", "pin": "p-i-n", "nip": "n-i-p"}
A = load_final()
S = pd.DataFrame([json.loads(l) for l in open(R / "strict_validation_v1/runs.jsonl")]).drop(columns=["info"])
T = pd.DataFrame([json.loads(l) for l in open(R / "t3final_matched_v1/runs.jsonl")]).drop(columns=["info"])
P = pd.read_csv(R / "t3final_summary/matched_pairs.csv")
out = {}; nums = {}

# repeated DOI-grouped holdouts (tab:tiers)
x = A[(A.protocol == "doi") & (A.model == "catboost")]; rows = []
for arch in ARCHN:
    for t in ("t1", "t2", "t3"):
        g = x[(x.arch == arch) & (x.tier == t)]
        rows.append(f"{ARCHN[arch]} & {TIER[t]} & {int(g.n_test.mean()):,} & {int(round(g.n_features.mean()))} & {pm(g.r2)} & {pm(g.mae)} & {pm(g.rmse)} \\\\")
        for k in ("r2", "mae", "rmse"): nums[f"doi_{arch}_{t}_{k}"] = float(g[k].mean()); nums[f"doi_{arch}_{t}_{k}_sd"] = float(g[k].std())
    if arch != "nip": rows.append("\\midrule")
out["tab:tiers"] = rows

# paired differences (tab:delta): t1->t2 from the stored bootstrap file, t3-t2 from matched_pairs.csv (scope all)
b = pd.read_csv(R / "ablation_v1/paired_bootstrap_doi_catboost.csv"); rows = []
for arch in ARCHN:
    y = b[(b.arch == arch) & (b.contrast == "t2-t1")]
    c1 = f"${y.d_r2.mean():+.3f}$ & ${y.d_rmse.mean():+.3f}$ & {int((y.d_r2 > 0).sum())}/5 & {int(((y.r2_lo > 0) | (y.r2_hi < 0)).sum())}/5"
    z = P[(P.protocol == "doi") & (P.arch == arch) & (P.model == "catboost") & (P.scope == "all")]
    c2 = f"${z.d_r2.mean():+.3f}$ & ${z.d_rmse.mean():+.3f}$ & {int((z.d_r2 > 0).sum())}/5 & {int(((z.lo > 0) | (z.hi < 0)).sum())}/5"
    rows.append(f"{ARCHN[arch]} & {c1} & {c2} \\\\"); nums[f"doi_{arch}_d32"] = float(z.d_r2.mean())
out["tab:delta"] = rows

# record-level vs DOI-grouped (tab:split)
rows = []
for m, mn in (("catboost", "CatBoost"), ("ridge", "Ridge")):
    for t in ("t1", "t2", "t3"):
        d = A[(A.arch == "overall") & (A.model == m) & (A.tier == t)]
        a = d[d.protocol == "doi"].sort_values("split"); r = d[d.protocol == "random"].sort_values("split")
        gap = r.r2.values - a.r2.values
        rows.append(f"{mn} & {TIER[t]} & {pm(r.r2)} & {pm(a.r2)} & ${gap.mean():.3f}$ & ${r.mae.mean():.2f}$ & ${a.mae.mean():.2f}$ \\\\")
        nums[f"gap_{m}_{t}"] = float(gap.mean()); nums[f"rand_{m}_{t}"] = float(r.r2.mean()); nums[f"rand_{m}_{t}_mae"] = float(r.mae.mean())
    if m == "catboost": rows.append("\\midrule")
out["tab:split"] = rows

# strict validation (tab:strict)
allp = pd.concat([A[(A.protocol.isin(["doi"])) & (A.arch == "overall")], S[S.arch == "overall"],
                  T[(T.arch == "overall") & (T.protocol.isin(["author", "temporal"]))].assign(tier="t3")], ignore_index=True)
rows = []
for proto, pn in (("random", "Record-level"), ("doi", "DOI-grouped"), ("author", "Lead-author-grouped"), ("temporal", "Forward in time")):
    for m, mn in (("catboost", "CatBoost"), ("ridge", "Ridge"), ("mean", "Mean")):
        x = A[(A.protocol == "random") & (A.arch == "overall") & (A.model == m)] if proto == "random" else allp[(allp.protocol == proto) & (allp.model == m)]
        if m == "mean":
            x = pd.concat([x[x.tier == "t1"], x[x.tier == "t2"]]) if proto != "random" else x
            cells = [pm(x[x.tier == "t1"].r2)] * 1 + [pm(x[x.tier == "t2"].r2)] + ["---"]; mae = f"${x[x.tier == 't2'].mae.mean():.2f}$"
            ns = int(round(x[x.tier == "t2"].n_test.mean())); nsp = int(x[x.tier == "t2"].split.nunique())
            rows.append(f"{pn} & {mn} & {ns:,} & {nsp} & " + " & ".join(cells) + f" & {mae} & --- \\\\"); continue
        cells = [pm(x[x.tier == t].r2) for t in ("t1", "t2", "t3")]
        rows.append(f"{pn} & {mn} & {int(round(x[x.tier == 't2'].n_test.mean())):,} & {int(x[x.tier == 't2'].split.nunique())} & " + " & ".join(cells) +
                    f" & ${x[x.tier == 't2'].mae.mean():.2f}$ & ${x[x.tier == 't3'].mae.mean():.2f}$ \\\\")
        for t in ("t1", "t2", "t3"): nums[f"{proto}_{m}_{t}"] = float(x[x.tier == t].r2.mean()); nums[f"{proto}_{m}_{t}_sd"] = float(x[x.tier == t].r2.std())
        for t in ("t2", "t3"): nums[f"{proto}_{m}_{t}_mae"] = float(x[x.tier == t].mae.mean())
out["tab:strict"] = rows

# forward windows (tab:temporal)
tw = allp[(allp.protocol == "temporal") & (allp.arch == "overall") & (allp.model == "catboost")]
rows = []
for k in sorted(tw.split.unique()):
    g = tw[tw.split == k]; r = g[g.tier == "t2"].iloc[0]
    rows.append(f"{k} & {r.train_date_max} & {r.test_date_min} & {int(r.n_train):,} & {int(r.n_test):,} & " +
                " & ".join(f"${g[g.tier == t].r2.iloc[0]:.3f}$" for t in ("t1", "t2", "t3")) + " \\\\")
out["tab:temporal"] = rows

# paired t3 - t2 across protocols (new table), CatBoost and ridge, pooled
rows = []
for proto, pn in (("random", "Record-level"), ("doi", "DOI-grouped"), ("author", "Lead-author-grouped"), ("temporal", "Forward in time")):
    for m, mn in (("catboost", "CatBoost"), ("ridge", "Ridge")):
        for scope, sn in (("all", "all"), ("covered", "covered")):
            z = P[(P.protocol == proto) & (P.arch == "overall") & (P.model == m) & (P.scope == scope)]
            n = len(z)
            rows.append(f"{pn if (m, scope) == ('catboost', 'all') else ''} & {mn if scope == 'all' else ''} & {sn} & {int(round(z.n.mean())):,} & "
                        f"${z.r2_a.mean():.3f}$ & ${z.r2_b.mean():.3f}$ & ${z.d_r2.mean():+.4f}$ & ${z.d_mae.mean():+.3f}$ & {int((z.d_r2 > 0).sum())}/{n} & {int((z.lo > 0).sum())}/{int((z.hi < 0).sum())} \\\\")
            nums[f"p_{proto}_{m}_{scope}_d"] = float(z.d_r2.mean())
    if proto != "temporal": rows.append("\\midrule")
out["tab:t3protocols"] = rows
json.dump(out, open(R / "t3final_summary/table_rows.json", "w"), indent=1)
json.dump(nums, open(R / "t3final_summary/numbers.json", "w"), indent=1)
for k, v in out.items(): print("==", k); print("\n".join(v))
