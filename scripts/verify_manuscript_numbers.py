#!/usr/bin/env python3
"""Read-only check: key numbers typed into psc_manuscript/ml.tex agree with the authoritative result files."""
import json, re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
tex = (ROOT / "psc_manuscript/ml.tex").read_text(); fails = []
def need(s, why):
    if s not in tex: fails.append(f"missing in ml.tex: {s!r} ({why})")
M = pd.read_csv(ROOT / "results/v4/final_t1/metrics_3tier.csv"); C = json.load(open(ROOT / "results/v4/final_t1/comparisons_3tier.json"))
N = ROOT / "results/v4/final"; V = pd.read_csv(N / "metrics.csv").set_index(["scope", "arm"]); K = json.load(open(N / "comparisons.json"))
sg = lambda x: f"${x:+.4f}$"
for _, r in M.iterrows():
    row = f"{r.n:,.0f} & {r.dois:,.0f} & {r.r2:.3f} [{r.r2_lo:.3f}, {r.r2_hi:.3f}] & {r.mae:.3f} & {r.rmse:.3f}"
    need(row, f"three-tier table {r.scope} {r.tier}")
for sc in C:
    for k, v in C[sc].items():
        need(f"{sg(v['dR2'])} [{sg(v['lo'])}, {sg(v['hi'])}]".replace("[$", "[$"), f"paired table {sc} {k}")
        need(f"{v['seeds_pos']}/{v['n_seeds']}", "seeds")
# V4 config table
rows = {("covered", "Default"): ("t2_default", "t3_default", "A_matched_default (t3 vs t2)"), ("covered", "Each tuned"): ("t2_tuned", "t3_tuned", "B_each_tuned (t3_tuned vs t2_tuned)"),
        ("covered", "t3cfg"): ("t3cfg_on_t2", "t3_tuned", "A_matched_tuned_t3cfg (t3 vs t2 under t3's config)"), ("covered", "t2cfg"): ("t2_tuned", "t2cfg_on_t3", "A_matched_tuned_t2cfg (t3 vs t2 under t2's config)")}
for sc in ("covered", "full"):
    for key, (a, b, ck) in {"Default": ("t2_default", "t3_default", "A_matched_default (t3 vs t2)"), "Each tuned": ("t2_tuned", "t3_tuned", "B_each_tuned (t3_tuned vs t2_tuned)"),
                            "t3": ("t3cfg_on_t2", "t3_tuned", "A_matched_tuned_t3cfg (t3 vs t2 under t3's config)"), "t2": ("t2_tuned", "t2cfg_on_t3", "A_matched_tuned_t2cfg (t3 vs t2 under t2's config)")}.items():
        v = K[sc][ck]; label = {"Default": "Default", "Each tuned": "Each tuned", "t3": "$\\mathcal{M}_3$'s tuned config.", "t2": "$\\mathcal{M}_2$'s tuned config."}[key]
        pre = ("Covered" if sc == "covered" else "All") + f" & {label} & {V.loc[(sc, a), 'r2']:.3f} & {V.loc[(sc, b), 'r2']:.3f} & {sg(v['dR2'])} & $[{v['lo']:+.4f}, {v['hi']:+.4f}]$ & {v['seeds_pos']}/{v['n_seeds']}"
        need(pre, f"config table {sc} {key}")
# t2/t3 default arms of the new table equal the published V4 arms
for sc in ("covered", "full"):
    for t in ("t2", "t3"):
        a = M[(M.scope == sc) & (M.tier == t)].r2.item(); b = V.loc[(sc, f"{t}_default"), "r2"]
        if abs(a - b) > 1e-12: fails.append(f"{sc} {t} default R2 differs between final_t1 and final: {a} vs {b}")
# five-holdout tables from run records
from psc.results import load_ablation
A = load_ablation(); d = A[(A.protocol == "doi") & (A.model == "catboost")]
for arch, t, ex in (("overall", "t1", "0.368 \\pm 0.024"), ("overall", "t2", "0.408 \\pm 0.032"), ("pin", "t2", "0.291 \\pm 0.065"), ("nip", "t2", "0.404 \\pm 0.013")):
    x = d[(d.arch == arch) & (d.tier == t)].r2; got = f"{x.mean():.3f} \\pm {x.std():.3f}"
    if got != ex: fails.append(f"{arch} {t}: {got} != {ex}")
    need(ex, "five-holdout")
for s in ("215", "77 / 179 / 215", "15,027", "2,731", "14,896", "2,699"): need(s, "counts")
for bad in (r"\mathcal{M}_3^{\mathrm{orig}}", "\\input{tables", "TODO", "??"): 
    if bad in tex: fails.append(f"stale string present: {bad}")
print("FAILS:", fails if fails else "none"); sys.exit(1 if fails else 0)
