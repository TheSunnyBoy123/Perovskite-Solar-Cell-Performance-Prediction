#!/usr/bin/env python3
"""Derive the handoff CSV tables from existing result artifacts. READ-ONLY with respect to the project:
nothing under results/, data/, configs/ or psc_manuscript/ is modified; outputs go to results_handoff/tables/.
No model is trained and no LLM is called. Every value is copied or aggregated (mean, sd with ddof=1 over splits)
from the cited source file; blank cells mean 'not available in the artifacts'.
Run from anywhere:  python build_handoff_tables.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
ROOT = H.parent
R = ROOT / "results"
T = H / "tables"
sys.path.insert(0, str(ROOT / "src"))
rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
ARCH = {"overall": "pooled", "pin": "p-i-n", "nip": "n-i-p"}
ORIG = "t2 + earlier 44-predictor literature representation E (superseded; 32 base columns + indicators; paper-level; 36.2% of records)"
rjl = lambda n: pd.DataFrame([json.loads(l) for l in open(R / n / "runs.jsonl")])
sd = lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan

# ======================================================================= v4_main_results.csv
M = pd.read_csv(R / "v4/final/metrics.csv")
C = json.load(open(R / "v4/final/comparisons.json"))
CFG = json.load(open(R / "v4/final/tuned_cfg.json"))
met = {(r.scope, r.arm): r for r in M.itertuples()}
ARM_DESC = {
 "t2_default": ("t2", "matched default CatBoost"), "t3_default": ("t3", "matched default CatBoost"),
 "t2_tuned": ("t2", "t2-tuned CatBoost config"), "t3_tuned": ("t3", "t3-tuned CatBoost config"),
 "t3cfg_on_t2": ("t2", "t3-tuned config applied to t2"), "t2cfg_on_t3": ("t3", "t2-tuned config applied to t3"),
 **{f"ctl_perm{j}_default": ("t2 + DOI-permuted recipe block", "matched default CatBoost") for j in range(5)},
 "ctl_coverflag_default": ("t2 + coverage indicator only", "matched default CatBoost")}
CIM = "publication-cluster bootstrap (2000 resamples, seed 0) of paired squared error on seed-averaged predictions; percentile 95%"
PART = ("v4final: reserved 35% of publications (salted DOI hash 'psc-v4'); models trained on all v4dev (65%) with DOI-disjoint 15% validation for early stopping; "
        "reserved DOIs were seen in earlier experiments with other feature sets (fresh for this representation, not untouched)")
rows = []
for (scope, arm), r in met.items():
    tier, cfg = ARM_DESC[arm]
    rows.append(dict(row_type="arm", scope="covered (papers with V4 flags)" if scope == "covered" else "all records", arm=arm, tier=tier,
        feature_representation="V4 recipe flags (36)" if tier.startswith("t3") else ("t2 (no LLM block)" if tier == "t2" else "control"),
        estimator_config=cfg, n_test_records=int(r.n), n_test_dois=int(r.dois), n_seed_models=5, r2=r.r2, mae=r.mae, rmse=r.rmse,
        data_split=PART, source=rel(R / "v4/final/metrics.csv")))
CMP = {"A_matched_default (t3 vs t2)": ("t2_default", "t3_default", "A: same default settings"),
       "A_matched_tuned_t3cfg (t3 vs t2 under t3's config)": ("t3cfg_on_t2", "t3_tuned", "A: both under t3-tuned config"),
       "A_matched_tuned_t2cfg (t3 vs t2 under t2's config)": ("t2_tuned", "t2cfg_on_t3", "A: both under t2-tuned config"),
       "B_each_tuned (t3_tuned vs t2_tuned)": ("t2_tuned", "t3_tuned", "B: each tier with its own tuned config"),
       "tuning_effect_t2 (t2_tuned vs t2_default)": ("t2_default", "t2_tuned", "tuning effect on t2 (not an LLM comparison)"),
       "tuning_effect_t3 (t3_tuned vs t3_default)": ("t3_default", "t3_tuned", "tuning effect on t3 (not an LLM comparison)")}
for scope in ("covered", "full"):
    for k, v in C[scope].items():
        if k in CMP: a, b, lab = CMP[k]
        else: a, b, lab = "t2_default", k.split(" vs ")[0], "control: " + ("DOI-permuted recipe block" if "perm" in k else "coverage indicator only") + " vs t2 (default config)"
        ra, rb = met[(scope, a)], met[(scope, b)]
        rows.append(dict(row_type="paired comparison", scope="covered (papers with V4 flags)" if scope == "covered" else "all records",
            comparison=lab, arm=f"{a} -> {b}", r2_reference=ra.r2, r2_comparison=rb.r2, delta_r2=v["dR2"], ci_lo=v["lo"], ci_hi=v["hi"], ci_method=CIM,
            delta_mae_derived=rb.mae - ra.mae, delta_rmse_derived=rb.rmse - ra.rmse, mae_reference=ra.mae, mae_comparison=rb.mae,
            rmse_reference=ra.rmse, rmse_comparison=rb.rmse, seeds_positive=v["seeds_pos"], n_seeds=v["n_seeds"],
            per_seed_delta_r2=";".join(f"{x:.6f}" for x in v["per_seed"]), n_test_records=int(ra.n), n_test_dois=int(ra.dois),
            data_split=PART, source=rel(R / "v4/final/comparisons.json") + " ; " + rel(R / "v4/final/metrics.csv"),
            notes="delta_mae/rmse_derived = comparison minus reference from metrics.csv (negative = lower error); no CI is stored for MAE/RMSE"))
pd.DataFrame(rows).to_csv(T / "v4_main_results.csv", index=False)

# ======================================================================= original_tier_ablation.csv
A = pd.read_csv(R / "ablation_v1/summary.csv")
rows = []
for r in A.itertuples():
    rows.append(dict(row_type="tier metrics", protocol=r.protocol, subset=ARCH[r.arch], tier=r.tier, feature_representation=("t1" if r.tier == "t1" else "t2" if r.tier == "t2" else ORIG),
        estimator=r.model, n_splits=int(r.splits), mean_n_test_records=r.n, mean_n_features=r.n_features, r2_mean=r.r2, r2_sd=r.r2_sd, mae_mean=r.mae, mae_sd=r.mae_sd,
        rmse_mean=r.rmse, rmse_sd=r.rmse_sd, source=rel(R / "ablation_v1/summary.csv"),
        notes="record-level CatBoost rows in this file are the truncated 1,500-iteration run; see evaluation_protocols.csv for the converged reruns" if (r.protocol == "random" and r.model == "catboost") else ""))
D = pd.read_csv(R / "ablation_v1/paired_tier_diffs.csv"); B = pd.read_csv(R / "ablation_v1/paired_bootstrap_summary_doi_catboost.csv")
Bk = {(b.arch, b.contrast): b for b in B.itertuples()}
for r in D.itertuples():
    b = Bk.get((r.arch, r.contrast)) if (r.protocol == "doi" and r.model == "catboost") else None
    rows.append(dict(row_type="paired difference", protocol=r.protocol, subset=ARCH[r.arch], tier=r.contrast, estimator=r.model, n_splits=int(r.splits),
        delta_r2_mean=r.d_r2, delta_r2_sd=r.d_r2_sd, splits_r2_better=r.r2_better_splits, delta_mae_mean=r.d_mae, delta_mae_sd=r.d_mae_sd,
        delta_rmse_mean=r.d_rmse, delta_rmse_sd=r.d_rmse_sd, splits_ci_excludes_zero=(b.r2_ci_excl0 if b is not None else np.nan),
        source=rel(R / "ablation_v1/paired_tier_diffs.csv") + (" ; " + rel(R / "ablation_v1/paired_bootstrap_summary_doi_catboost.csv") if b is not None else ""),
        notes="per-split DOI-cluster bootstrap CI (1000 resamples); t2+E = t2 plus the earlier 44-predictor literature representation E"))
pd.DataFrame(rows).to_csv(T / "original_tier_ablation.csv", index=False)
# split-level n for the main DOI-grouped catboost cells
runs = rjl("ablation_v1"); sp = runs[(runs.protocol == "doi") & (runs.model == "catboost")].copy()
sp["subset"] = sp.arch.map(ARCH)
sp[["subset", "tier", "split", "n_train", "n_val", "n_test", "n_doi_train", "n_doi_val", "n_doi_test", "n_features", "doi_overlap_test_train", "doi_overlap_val_train", "doi_overlap_test_val", "r2", "mae", "rmse"]].to_csv(T / "original_tier_ablation_per_split.csv", index=False)

# ======================================================================= model_development.csv
md = rjl("model_dev_v1"); rows = []
for (arch, tier), g in md.groupby(["arch", "tier"]):
    for k, name in (("cb_tuned", "CatBoost tuned (depth/L2 grid, 6 points, by validation RMSE)"), ("cb_bag", "CatBoost 3-seed bag of selected config"),
                    ("xgb", "XGBoost (early-stopped)"), ("ens", "mean of CatBoost bag and XGBoost")):
        rows.append(dict(subset=ARCH[arch], tier=tier, feature_representation=("t1" if tier == "t1" else "t2" if tier == "t2" else ORIG), estimator=name, n_splits=len(g),
            mean_n_test_records=g.n_test.mean(), mean_n_features=g.n_features.mean(), r2_mean=g[f"{k}_r2"].mean(), r2_sd=sd(g[f"{k}_r2"]),
            mae_mean=g[f"{k}_mae"].mean(), mae_sd=sd(g[f"{k}_mae"]), rmse_mean=g[f"{k}_rmse"].mean(), rmse_sd=sd(g[f"{k}_rmse"]),
            selected_configs_by_split=json.dumps([x for x in g.selected]), data_split="5 DOI-grouped holdouts (GroupShuffleSplit, test 20%, inner DOI-disjoint validation 15%, seed 42)",
            source=rel(R / "model_dev_v1/runs.jsonl"), v4_entry="NA (no V4 t1/Bag/XGB/Ens results exist)"))
pd.DataFrame(rows).to_csv(T / "model_development.csv", index=False)

# ======================================================================= evaluation_protocols.csv
from psc.results import load_ablation
AA = load_ablation(); SV = rjl("strict_validation_v1"); rows = []
def agg(df, protocol_label, source, extra=""):
    for (arch, model, tier), g in df.groupby(["arch", "model", "tier"]):
        rows.append(dict(protocol=protocol_label, subset=ARCH[arch], estimator=model, tier=tier,
            feature_representation=("t1" if tier == "t1" else "t2" if tier == "t2" else ORIG), n_splits=len(g), mean_n_test_records=g.n_test.mean(),
            mean_n_test_dois=g.n_doi_test.mean() if "n_doi_test" in g else np.nan, r2_mean=g.r2.mean(), r2_sd=sd(g.r2), mae_mean=g.mae.mean(), mae_sd=sd(g.mae),
            rmse_mean=g.rmse.mean(), rmse_sd=sd(g.rmse), source=source, notes=extra))
agg(AA[AA.protocol == "doi"], "DOI-grouped (5 repeated holdouts)", rel(R / "ablation_v1/runs.jsonl"))
agg(AA[AA.protocol == "random"], "record-level random (5 splits)", rel(R / "ablation_v1/runs.jsonl") + " + ablation_random_cap6000 + random_convergence_cap20000",
    "CatBoost: pooled from the 20,000-iteration converged rerun (random_convergence_cap20000); p-i-n/n-i-p from the 6,000-iteration rerun (ablation_random_cap6000); ridge/mean from ablation_v1 (see psc/results.py load_ablation)")
agg(SV[SV.protocol == "author"], "lead-author-grouped (5 splits)", rel(R / "strict_validation_v1/runs.jsonl"))
tw = SV[SV.protocol == "temporal"]
agg(tw, "forward-in-time (3 windows)", rel(R / "strict_validation_v1/runs.jsonl"), "3 non-overlapping windows; sd is over windows")
pd.DataFrame(rows).to_csv(T / "evaluation_protocols.csv", index=False)
tw[tw.model == "catboost"][["arch", "tier", "split", "train_date_max", "test_date_min", "n_train", "n_test", "r2", "mae", "rmse"]].to_csv(T / "evaluation_protocols_temporal_windows.csv", index=False)
pg = pd.read_csv(R / "ablation_v1/protocol_gap.csv"); pg["source"] = rel(R / "ablation_v1/protocol_gap.csv")
pg["notes"] = "ablation_v1 protocol_gap.csv: record-level side uses the (possibly truncated) ablation_v1 record-level fits unless noted; compare with converged values in evaluation_protocols.csv"
pg.to_csv(T / "evaluation_protocols_gap_ablation_v1.csv", index=False)

# ======================================================================= feature_coverage.csv
FA = json.load(open(R / "feature_audit.json")); LV = json.load(open(ROOT / "data/lit_v4.json"))["papers"]
rows = []
for k, v in FA["lit_coverage_per_feature"].items():
    rows.append(dict(representation="earlier literature block E", feature=k, coverage_fraction_of_41107_records=v, coverage_fraction_of_covered_papers="", n_papers_with_value="", source=rel(R / "feature_audit.json")))
flag_names = [k for k in next(iter(LV.values()))["recipe"] if k != "n_materials" and not k.startswith("s_")]
for k in flag_names:
    vals = pd.Series([p["recipe"][k] for p in LV.values()])
    rows.append(dict(representation="V4 recipe flags (rc_*)", feature=f"rc_{k}", coverage_fraction_of_covered_papers=1.0, n_papers_with_value=int(len(vals)),
        flag_prevalence_among_2835_papers=float(vals.mean()), source=rel(ROOT / "data/lit_v4.json"), notes="flag is 0/1 for every covered paper; NaN for records of papers without extraction"))
try:
    sel = json.load(open(ROOT / "data/lx3_qty_selected.json"))
    for c in (sel if isinstance(sel, list) else sel.get("columns", sel)):
        rows.append(dict(representation="quantity-level exploration (lx3), 15 predeclared columns", feature=str(c), source=rel(ROOT / "data/lx3_qty_selected.json"),
                         notes="coverage >=10% of the 9,441-record / 1,680-DOI eligible population (docs/LX3_RESULTS.md)"))
except Exception as e:
    print("qty selected not read:", e)
import psc.data as pdata, psc.v4 as v4m
df = pdata.load_dataset(); df["part"] = df.doi_group.map(v4m.v4_partition)
summ = [dict(representation="dataset counts", feature="all records", n_records=len(df), n_dois_incl_no_doi_block=df.doi_group.nunique(), source="psc.data.load_dataset()"),
        dict(representation="V4 (lx_covered)", feature="covered records", n_records=int(df.lx_covered.sum()), n_dois_incl_no_doi_block=df[df.lx_covered == 1].doi_group.nunique(), coverage_fraction_of_41107_records=float(df.lx_covered.mean()), source="psc.data (lx_covered) ; psc.v4"),
        dict(representation="earlier literature block E (feature_audit)", feature="records with any lit feature", n_records=int(round(FA["records_with_any_lit_feature"] * FA["n_records"])), n_dois_incl_no_doi_block=FA["dois_with_any_lit_feature"], coverage_fraction_of_41107_records=FA["records_with_any_lit_feature"], source=rel(R / "feature_audit.json"),
             notes="differs from lx_covered (36.6%, 2,731 DOIs): see RESULTS_REPORT section I, unresolved"),]
for p, g in df.groupby("part"):
    summ.append(dict(representation=f"V4 partition {p}", feature="records", n_records=len(g), n_dois_incl_no_doi_block=g.doi_group.nunique(), n_covered_records=int(g.lx_covered.sum()), n_covered_dois=g[g.lx_covered == 1].doi_group.nunique(), source="psc.v4.v4_partition (salt psc-v4, dev_frac 0.65)"))
pd.DataFrame(summ + rows).to_csv(T / "feature_coverage.csv", index=False)
df[["doi_group"]].assign(n=1).groupby("doi_group").n.sum().describe().to_csv(T / "_records_per_doi_describe.csv")

# ======================================================================= control_experiments.csv
rows = []
def addc(experiment, fr, partition, r, src, note="", **kw):
    rows.append(dict(experiment=experiment, feature_representation=fr, partition=partition, source=src, notes=note, **r, **kw))
for scope in ("covered", "full"):
    for k, v in C[scope].items():
        if k.startswith("ctl_"):
            addc("V4 negative control", "t2 + " + ("DOI-permuted V4 recipe block" if "perm" in k else "coverage indicator only"), f"v4final {scope}",
                 dict(reference="t2_default", delta_r2=v["dR2"], ci_lo=v["lo"], ci_hi=v["hi"], seeds_positive=f'{v["seeds_pos"]}/{v["n_seeds"]}', n_records=int(met[(scope, "t2_default")].n), n_dois=int(met[(scope, "t2_default")].dois)),
                 rel(R / "v4/final/comparisons.json"), "matched default config; CI = publication-cluster bootstrap")
for sc in ("covered", "full"):
    pass
FG = rjl("feature_groups_v1")
for (arch, tier), g in FG.groupby(["arch", "tier"]):
    addc("feature-group control (5 DOI-grouped holdouts)", {"t1_lit": "t1 + earlier literature representation E (no material properties)", "t3_nometa": "t2+E minus 2 reporting-completeness flags", "t2_noise": "t2 + 32 N(0,1) noise columns"}[tier], f"ablation holdouts {ARCH[arch]}",
         dict(reference="", r2_mean=g.r2.mean(), r2_sd=sd(g.r2), mae_mean=g.mae.mean(), rmse_mean=g.rmse.mean(), n_splits=len(g), n_features=g.n_features.mean(), n_records=g.n_test.mean()), rel(R / "feature_groups_v1/runs.jsonl"))
def csvrows(path, experiment, fr, part, cols):
    d = pd.read_csv(R / path)
    for r in d.to_dict("records"):
        addc(experiment, fr + " : " + str(r.get("tier", r.get("block", ""))), part, {c2: r.get(c1, "") for c1, c2 in cols.items()}, rel(R / path))
std = {"ref": "reference", "d_r2": "delta_r2", "ci_lo_mean": "ci_lo_dmse", "ci_hi_mean": "ci_hi_dmse", "seeds_positive": "seeds_positive", "folds_win": "folds_positive", "n_rec": "n_records", "n_doi": "n_dois", "r2_ref": "r2_reference", "r2_tier": "r2_tier", "n_seeds": "n_seeds"}
for o in ("overall", "nip", "pin"):
    csvrows(f"ablE_vs_t2_{o}.csv", "full-data single-stage ablation E2-E6 (DOI-grouped 5-fold x 3 seeds, matched config)", "alternative literature blocks vs t2 (see docs/LX_RESULTS.md)", f"full data {ARCH[o]}", std)
csvrows("identity_vs_t1id.csv", "material-identity control (DOI-grouped 5-fold x 3 seeds)", "t1/t2 vs t1+identity", "full data pooled", std)
csvrows("lx_dev_vs_t2.csv", "covered-dev single-stage (lx2 block), DOI-grouped 5-fold", "re-parsed lx2 blocks vs t2", "covered dev (7,640 rec / 1,368 DOIs)", std)
for fnm, part in (("lx_stack_dev_summary.csv", "covered dev"), ("lx_stack_confirm_summary.csv", "covered confirm")):
    csvrows(fnm, "lx2 two-stage residual design vs matched controls", "lx2 61-column block (x) / chem", part,
            {"ref": "reference", "d_r2": "delta_r2", "ci_lo": "ci_lo_dmse", "ci_hi": "ci_hi_dmse", "seeds_pos": "seeds_positive", "folds_win": "folds_positive", "n_rec": "n_records", "n_doi": "n_dois", "n_seeds": "n_seeds"})
csvrows("qty_dev_summary.csv", "quantity-covered experiment dev (qdev, DOI-grouped 5-fold x 5 seeds)", "lx3 quantity/chemistry blocks vs t2", "qdev 6,677 rec / 1,180 DOIs", std)
for r in json.load(open(R / "qty_holdout/summary.json")):
    addc("quantity-covered experiment held-out (train qdev, score qconf)", "lx3 quantity/chemistry blocks vs t2: " + r["tier"], "qconf 2,764 rec / 500 DOIs",
         dict(r2_tier=r["r2"], mae_tier=r["mae"], rmse_tier=r["rmse"], delta_r2=r["d_r2"], ci_lo=r["d_r2_ci"][0], ci_hi=r["d_r2_ci"][1], seeds_positive=r["seeds_pos"]), rel(R / "qty_holdout/summary.json"), "exploratory; population selected by availability")
pd.DataFrame(rows).to_csv(T / "control_experiments.csv", index=False)

# ======================================================================= baseline_studies.csv
BL = rjl("baselines_v1"); rows = []
src_b = rel(R / "baselines_v1/runs.jsonl")
for (model, tier), g in BL.groupby(["model", "tier"]):
    rows.append(dict(study="this study (in-framework reference model)", target="PCE", estimator=model, tier=tier, feature_representation="t1" if tier == "t1" else ORIG, split_protocol="5 DOI-grouped holdouts",
        n_splits=len(g), r2_mean=g.r2.mean(), r2_sd=sd(g.r2), mae_mean=g.mae.mean(), mae_sd=sd(g.mae), rmse_mean=g.rmse.mean(), rmse_sd=sd(g.rmse), units="PCE percentage points", source=src_b))
ab = AA[(AA.protocol == "doi") & (AA.arch == "overall") & (AA.model.isin(["mean", "ridge", "catboost"])) & (AA.tier.isin(["t1", "t3"]))]
for (model, tier), g in ab.groupby(["model", "tier"]):
    rows.append(dict(study="this study (in-framework reference model)", target="PCE", estimator=model + (" (matched)" if model == "catboost" else ""), tier=tier, feature_representation="t1" if tier == "t1" else ORIG,
        split_protocol="5 DOI-grouped holdouts", n_splits=len(g), r2_mean=g.r2.mean(), r2_sd=sd(g.r2), mae_mean=g.mae.mean(), mae_sd=sd(g.mae), rmse_mean=g.rmse.mean(), rmse_sd=sd(g.rmse), units="PCE percentage points", source=rel(R / "ablation_v1/runs.jsonl")))
rows += [
 dict(study="Roberts et al., Adv. Phys. Res. 3, 2400060 (2024)", target="Jsc (mA cm-2)", estimator="XGBoost", split_protocol="80/10/10 train/validation/test; duplicates averaged (pseudo-target)", r2_mean=0.35, rmse_mean=3.58, units="mA cm-2 (RMSE); NOT comparable with PCE metrics", source="docs/LIT_VERIFICATION.md (verified against ../Lit/roberts.pdf)"),
 dict(study="Fukasawa et al., Energy Adv. 3, 812 (2024)", target="PCE", estimator="random forest (best model)", split_protocol="8:2 train/test (no grouping by publication described); 25% of training used for validation", r2_mean=0.7177, units="R2 as reported (Table 2, zero-complement)", source="docs/LIT_VERIFICATION.md (verified against ../Lit/fukasawa.pdf)"),
 dict(study="Fukasawa et al., Energy Adv. 3, 812 (2024)", target="PCE", estimator="random forest (best model)", split_protocol="294 newly registered records (as of 24 Aug 2023)", r2_mean=0.45, mae_mean=3.20, units="PCE percent (MAE as reported)", source="docs/LIT_VERIFICATION.md", notes="mean-model R2 on the same set -0.33, MAE 5.40"),
 dict(study="Hussain et al., RSC Adv. (2023)", target="PCE", estimator="", split_protocol="not stated in the article", source="docs/LIT_VERIFICATION.md", notes="values not recorded in project artifacts")]
pd.DataFrame(rows).to_csv(T / "baseline_studies.csv", index=False)
print("tables written:", sorted(p.name for p in T.glob("*.csv")))


# ======================================================================= V4 common-split three-tier table (t1/t2/t3 on identical reserved publications)
import json as _json
M3 = pd.read_csv(R / "v4/final_t1/metrics_3tier.csv"); C3 = _json.load(open(R / "v4/final_t1/comparisons_3tier.json"))
rows3 = []
for r in M3.itertuples():
    rows3.append(dict(row_type="tier metrics", population={"covered": "LLM-covered", "full": "all reserved"}[r.scope], tier=r.tier, comparison="", feature_representation={"t1": "t1 (PDP features)", "t2": "t2 (t1 + material properties)", "t3": "t3 (t2 + V4 36 recipe flags)"}[r.tier],
        estimator_config="matched default CatBoost", n_seeds=5, n_test_records=r.n, n_test_dois=r.dois, r2=r.r2, r2_ci_lo=r.r2_lo, r2_ci_hi=r.r2_hi, mae=r.mae, rmse=r.rmse,
        delta_r2="", delta_ci_lo="", delta_ci_hi="", seeds_positive="", split="v4final (reserved, DOI-disjoint from v4dev)", source="results/v4/final_t1/metrics_3tier.csv (t1 new; t2/t3 = results/v4/final default arms)"))
for sc, pop in (("covered", "LLM-covered"), ("full", "all reserved")):
    m = M3[M3.scope == sc].set_index("tier")
    for k, (a_, b_) in {"t2_minus_t1": ("t1", "t2"), "t3_minus_t2": ("t2", "t3"), "t3_minus_t1": ("t1", "t3")}.items():
        v = C3[sc][k]
        rows3.append(dict(row_type="paired difference", population=pop, tier="", comparison=k.replace("_minus_", " - "), feature_representation="", estimator_config="matched default CatBoost", n_seeds=5,
            n_test_records=int(m.n.iloc[0]), n_test_dois=int(m.dois.iloc[0]), r2="", r2_ci_lo="", r2_ci_hi="", mae=m.loc[b_, "mae"] - m.loc[a_, "mae"], rmse=m.loc[b_, "rmse"] - m.loc[a_, "rmse"],
            delta_r2=v["dR2"], delta_ci_lo=v["lo"], delta_ci_hi=v["hi"], seeds_positive=f"{v['seeds_pos']}/{v['n_seeds']}", split="v4final (reserved, DOI-disjoint from v4dev)",
            source="results/v4/final_t1/comparisons_3tier.json (mae/rmse columns = difference of point metrics, derived)"))
pd.DataFrame(rows3).to_csv(T / "v4_three_tier_common_split.csv", index=False)

# ======================================================================= relabel the earlier 44-predictor block (run-file tier code 't3') as t2+E
EARLY_MAP = {"t3": "t2+E", "t3-t2": "(t2+E)-t2", "t3-t1": "(t2+E)-t1"}
for fn in ("original_tier_ablation.csv", "original_tier_ablation_per_split.csv", "model_development.csv", "baseline_studies.csv", "evaluation_protocols.csv",
           "evaluation_protocols_gap_ablation_v1.csv", "evaluation_protocols_temporal_windows.csv"):
    d = pd.read_csv(T / fn)
    if "run_file_tier" not in d.columns:
        d.insert(list(d.columns).index("tier") + 1, "run_file_tier", d["tier"])
    d["tier"] = d["run_file_tier"].replace(EARLY_MAP)
    d.to_csv(T / fn, index=False)
d = pd.read_csv(T / "control_experiments.csv")
d["feature_representation"] = d["feature_representation"].astype(str).str.replace(" : t3", " : E (earlier 44-predictor block; run-file code t3)", regex=False).str.replace("alternative literature blocks vs t2", "earlier alternative literature blocks vs t2", regex=False).replace("nan", "")
d.to_csv(T / "control_experiments.csv", index=False)
