#!/usr/bin/env python3
"""Inexpensive integrity checks for the handoff package. Read-only; writes VERIFICATION_LOG.md."""
import json, re, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; ROOT = H.parent; R = ROOT / "results"; T = H / "tables"
sys.path.insert(0, str(ROOT / "src"))
log, fails = [], []
def check(name, ok, detail=""):
    log.append(f"- [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok: fails.append(name)
close = lambda a, b, tol=5e-5: abs(a - b) <= tol

# 1. dataset counts
raw = pd.read_csv(ROOT / "data/dataset_json_enriched.csv", low_memory=False)
check("raw 43,398 x 156, 7,499 DOI strings", raw.shape == (43398, 156) and raw.Ref_DOI_number.nunique() == 7499)
check("1,002 missing PCE; 1,289 PCE<1", raw.final_pce.isna().sum() == 1002 and (raw.final_pce < 1).sum() == 1289)
import psc.data as pdt, psc.v4 as v4m
df = pdt.load_dataset(); a = df.Cell_architecture.value_counts()
check("41,107 filtered records = 43,398-1,002-1,289", len(df) == 41107 == 43398 - 1002 - 1289)
check("7,327 groups incl. 232-record no-DOI block (7,326 publications)", df.doi_group.nunique() == 7327 and (df.doi_group == "__NO_DOI__").sum() == 232)
check("arch 28,340 / 12,684 / 83 other", a["nip"] == 28340 and a["pin"] == 12684 and len(df) - a["nip"] - a["pin"] == 83)
y = df[pdt.TARGET]; check("PCE mean 12.42, median 13.0, range 1.0-36.2; 17 records >27 (6 DOIs)", close(y.mean(), 12.42, 5e-3) and y.median() == 13.0 and y.min() == 1.0 and y.max() == 36.2 and (y > 27).sum() == 17 and df[y > 27].doi_group.nunique() == 6)
vc = df[df.doi_group != "__NO_DOI__"].doi_group.value_counts(); check("max records per real publication = 127; median per group 4", vc.max() == 127 and df.doi_group.value_counts().median() == 4)
df["part"] = df.doi_group.map(v4m.v4_partition); fin = df[df.part == "v4final"]; dev = df[df.part == "v4dev"]
check("v4dev 26,982/4,708; v4final 14,125/2,619; covered 5,460/1,005", len(dev) == 26982 and dev.doi_group.nunique() == 4708 and len(fin) == 14125 and fin.doi_group.nunique() == 2619 and fin.lx_covered.sum() == 5460 and fin[fin.lx_covered == 1].doi_group.nunique() == 1005)
check("coverage 15,027 records / 2,731 DOIs (36.6%)", df.lx_covered.sum() == 15027 and df[df.lx_covered == 1].doi_group.nunique() == 2731 and close(df.lx_covered.mean(), 0.3656, 5e-4))
fa = json.load(open(R / "feature_audit.json")); check("original-block coverage 36.2% / 14,896 / 2,699", close(fa["records_with_any_lit_feature"], 0.3624, 5e-4) and round(fa["records_with_any_lit_feature"] * fa["n_records"]) == 14896 and fa["dois_with_any_lit_feature"] == 2699)
check("ICC 0.692", close(fa["target_icc_by_doi"], 0.692, 5e-4))
check("36 V4 flags", len([c for c in v4m.paper_table("rec").columns]) == 36 and sum(len(v) for v in v4m.REC_GROUPS.values()) == 36)

# 2. V4 values vs docs/V4_REPORT.md
V = pd.read_csv(T / "v4_main_results.csv"); cmp_ = V[V.row_type == "paired comparison"]
def g(scope, lab):
    return cmp_[cmp_.scope.str.startswith(scope) & (cmp_.comparison == lab)].iloc[0]
exp = {("all", "A: same default settings"): (0.0025, -0.0011, 0.0062), ("covered", "A: same default settings"): (0.0071, -0.0007, 0.0157),
       ("covered", "B: each tier with its own tuned config"): (0.0094, -0.0016, 0.0219), ("all", "B: each tier with its own tuned config"): (0.0015, -0.0039, 0.0067),
       ("covered", "A: both under t3-tuned config"): (0.0066, -0.0021, 0.0162), ("all", "A: both under t3-tuned config"): (0.0026, -0.0015, 0.0067),
       ("covered", "A: both under t2-tuned config"): (0.0024, -0.0077, 0.0131), ("all", "A: both under t2-tuned config"): (-0.0016, -0.0064, 0.0030)}
for (s, lab), (d, lo, hi) in exp.items():
    r = g(s, lab); check(f"V4_REPORT {s} / {lab}", close(r.delta_r2, d) and close(r.ci_lo, lo) and close(r.ci_hi, hi), f"{r.delta_r2:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]")
cb = g("covered", "B: each tier with its own tuned config"); check("absolute covered R2 t2 0.361 / t3 0.371", close(cb.r2_reference, 0.361, 5e-4) and close(cb.r2_comparison, 0.371, 5e-4))
check("delta R2 equals difference of arm R2 (all comparison rows, tol 1e-9)", all(abs(r.r2_comparison - r.r2_reference - r.delta_r2) < 1e-9 for r in cmp_.itertuples()))
ctl = cmp_[cmp_.comparison.str.startswith("control")]
pc, pf = ctl[ctl.scope.str.startswith("covered") & ctl.arm.str.contains("perm")], ctl[ctl.scope.str.startswith("all") & ctl.arm.str.contains("perm")]
check("permuted-control ranges covered -0.0049..+0.0015, all -0.0028..+0.0007", close(pc.delta_r2.min(), -0.0049) and close(pc.delta_r2.max(), 0.0015) and close(pf.delta_r2.min(), -0.0028) and close(pf.delta_r2.max(), 0.0007))
check("coverage-flag control +0.0015 / +0.0011", close(ctl[ctl.arm.str.contains("coverflag") & ctl.scope.str.startswith("covered")].delta_r2.iloc[0], 0.0015) and close(ctl[ctl.arm.str.contains("coverflag") & ctl.scope.str.startswith("all")].delta_r2.iloc[0], 0.0011))
M = pd.read_csv(R / "v4/final/metrics.csv"); check("metrics.csv n/DOIs: covered 5460/1005, full 14125/2619", set(map(tuple, M[["scope", "n", "dois"]].drop_duplicates().values.tolist())) == {("full", 14125, 2619), ("covered", 5460, 1005)})
# dev diagnostics
d = pd.read_csv(R / "v4/pen_05.csv"); p = d.pivot_table(index=["seed", "fold"], columns="tier", values="mse"); v = d.groupby(["seed", "fold"])["var"].first(); x = (p["t2"] - p["t2+rec"]) / v
check("dev +0.0070, 12/15 folds (pen_05.csv)", close(x.mean(), 0.0070, 5e-4) and (x > 0).sum() == 12)
tn = json.load(open(R / "v4/tune.json")); check("t3 tuning MSE lower than t2 in 6/24 configs", sum(a_["mse"] > b_["mse"] for a_, b_ in zip(tn["t2"], tn["t3"])) == 6)
# 3. tier / headline numbers
O = pd.read_csv(T / "original_tier_ablation.csv"); o = O[(O.row_type == "tier metrics") & (O.protocol == "doi") & (O.estimator == "catboost") & (O.subset == "pooled")].set_index("tier")
check("original pooled R2 0.368/0.408/0.407 (+-0.024/0.032/0.031)", close(o.r2_mean["t1"], 0.368, 5e-4) and close(o.r2_mean["t2"], 0.408, 5e-4) and close(o.r2_mean["t3"], 0.407, 5e-4) and close(o.r2_sd["t3"], 0.031, 5e-4))
M2 = pd.read_csv(T / "model_development.csv"); e = M2[(M2.subset == "pooled") & M2.estimator.str.startswith("mean of")].set_index("tier")
check("model-dev pooled ensemble 0.377/0.416/0.418", close(e.r2_mean["t1"], 0.377, 5e-4) and close(e.r2_mean["t2"], 0.416, 5e-4) and close(e.r2_mean["t3"], 0.418, 5e-4))
check("model-dev table has 36 rows (3 subsets x 3 tiers x 4 models)", len(M2) == 36)
# raw check: model_dev table vs manuscript-input table fragment (read-only)
frag = ROOT / "psc_manuscript/tables/model_dev.tex"
if frag.exists():
    t = frag.read_text(); m = re.findall(r"\$([0-9.]+) \\pm ([0-9.]+)\$", t)
    ref = [(r.r2_mean, r.r2_sd) for r in M2.itertuples() if r.subset == "pooled" and r.tier == "t1"]
    check("model_dev.tex pooled t1 row matches CSV", [tuple(map(float, x)) for x in m[:4]] == [(round(a_, 3), round(b_, 3)) for a_, b_ in [(M2[(M2.subset == "pooled") & (M2.tier == "t1")].iloc[i].r2_mean, M2[(M2.subset == "pooled") & (M2.tier == "t1")].iloc[i].r2_sd) for i in range(4)]])
E = pd.read_csv(T / "evaluation_protocols.csv")
def pool(proto, est, tier): return E[E.protocol.str.startswith(proto) & (E.estimator == est) & (E.subset == "pooled") & (E.tier == tier)].r2_mean.iloc[0]
check("record-level CatBoost 0.587/0.618/0.627 (converged)", close(pool("record", "catboost", "t1"), 0.587, 5e-4) and close(pool("record", "catboost", "t2"), 0.618, 5e-4) and close(pool("record", "catboost", "t3"), 0.627, 5e-4))
check("lead-author t3 0.395; forward t3 0.335", close(pool("lead", "catboost", "t3"), 0.395, 5e-4) and close(pool("forward", "catboost", "t3"), 0.335, 5e-4))
# 4. representation labels
bad = []
for f_ in ("original_tier_ablation.csv", "model_development.csv", "evaluation_protocols.csv", "baseline_studies.csv"):
    d_ = pd.read_csv(T / f_); d_ = d_[d_.tier.astype(str) == "t3"] if "tier" in d_ else d_.iloc[0:0]
    bad += [f_ for r in d_.itertuples() if "original" not in str(r.feature_representation).lower()]
check("every 't3' row outside v4_main_results is labelled ORIGINAL block", not bad, str(set(bad)))
a4 = V[V.row_type == "arm"]; check("every V4 't3' arm labelled V4 recipe flags", all("V4" in str(r.feature_representation) for r in a4.itertuples() if str(r.tier) == "t3"))
check("no V4 entries invented in model_development.csv", (M2.v4_entry.astype(str).str.startswith("NA")).all())
# 5. CSV hygiene
for f_ in sorted(glob.glob(str(T / "*.csv"))):
    d_ = pd.read_csv(f_); check(f"opens: {Path(f_).name}", len(d_) > 0 and not d_.columns.duplicated().any(), f"{len(d_)} rows x {len(d_.columns)} cols")
rd = (H / "DATA_DICTIONARY.md").read_text(); allcols = {c for f_ in glob.glob(str(T / "*.csv")) for c in pd.read_csv(f_).columns}
check("every CSV column documented in DATA_DICTIONARY.md", all(f"`{c}`" in rd for c in allcols))
# 6. referenced files exist
rep = (H / "RESULTS_REPORT.md").read_text() + (H / "README.md").read_text() + (H / "figures/FIGURE_MANIFEST.md").read_text()
toks = sorted(set(re.findall(r"`([A-Za-z0-9_./*{},\-]+\.(?:csv|json|jsonl|md|py|npz|pdf|png|tex|sh))`", rep)))
miss = []
for t in toks:
    if any(ch in t for ch in "*{}"): continue
    cands = [ROOT / t, ROOT / "docs" / t, ROOT / "results" / t, ROOT / "scripts" / t, ROOT.parent / t.lstrip("./"), H / t, H / "tables" / t, H / "figures" / t, H / "figures/scripts" / t, ROOT / "results/v4" / t, ROOT / "results/v4/final" / t,
             ROOT / "results/ablation_v1" / t, ROOT / "results/shap" / t, ROOT / "src/psc" / t, ROOT / "audit" / t, ROOT / "legacy" / t, ROOT / "configs" / t, ROOT / "data" / t, ROOT / "psc_manuscript" / t, ROOT / "psc_manuscript/figure_scripts" / t, ROOT / "results/qty_holdout" / t, ROOT / "results/model_dev_v1" / t, ROOT / "results/identity_vs_t1id.csv"]
    if not any(c.exists() for c in cands): miss.append(t)
check("all file paths cited in report/README/manifest exist (glob/brace patterns skipped)", not miss, "missing: " + ", ".join(miss) if miss else f"{len(toks)} paths checked")
(H / "VERIFICATION_LOG.md").write_text("# Verification log\n\nRun by `verify_handoff.py` (read-only; no training, no LLM).\n\n" + "\n".join(log) + f"\n\n**{len(log)-len(fails)} passed, {len(fails)} failed.**\n")
print("\n".join(log)); print("FAILS:", fails)
