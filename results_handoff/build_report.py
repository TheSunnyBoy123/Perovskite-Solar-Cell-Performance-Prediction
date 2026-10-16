#!/usr/bin/env python3
"""Assemble RESULTS_REPORT.md from the CSV tables and source artifacts (read-only; no training, no LLM).
Every number printed in a table is read from a file; hand-typed numbers in prose are checked by verify_handoff.py."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; ROOT = H.parent; R = ROOT / "results"; T = H / "tables"
sys.path.insert(0, str(ROOT / "src"))

def f(x, n=3):
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "": return "NA"
    return f"{x:.{n}f}"
def pm(m, s, n=3): return f"{m:.{n}f} ± {s:.{n}f}"
def sg(x, n=4): return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{n}f}"
def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)

V = pd.read_csv(T / "v4_main_results.csv"); O = pd.read_csv(T / "original_tier_ablation.csv"); MD = pd.read_csv(T / "model_development.csv")
EP = pd.read_csv(T / "evaluation_protocols.csv"); CE = pd.read_csv(T / "control_experiments.csv"); BS = pd.read_csv(T / "baseline_studies.csv")
FC = pd.read_csv(T / "feature_coverage.csv"); PS = pd.read_csv(T / "original_tier_ablation_per_split.csv")
meta = json.load(open(R / "ablation_v1/meta.json"))

# ---------- V4 tables
cmp_ = V[V.row_type == "paired comparison"]
def v4rows(scope_key, keys):
    rows = []
    for lab in keys:
        r = cmp_[(cmp_.scope.str.startswith(scope_key)) & (cmp_.comparison == lab)].iloc[0]
        rows.append([lab, r.arm, f(r.r2_reference), f(r.r2_comparison), sg(r.delta_r2), f"[{sg(r.ci_lo)}, {sg(r.ci_hi)}]", f"{int(r.seeds_positive)}/{int(r.n_seeds)}",
                     f(r.mae_reference), f(r.mae_comparison), sg(r.delta_mae_derived, 3), f(r.rmse_reference), f(r.rmse_comparison), sg(r.delta_rmse_derived, 3)])
    return rows
HEAD = ["comparison", "arms (reference -> comparison)", "R² t2", "R² t3", "ΔR²", "95% CI", "seeds ΔR²>0", "MAE t2", "MAE t3", "ΔMAE*", "RMSE t2", "RMSE t3", "ΔRMSE*"]
KEYS = ["A: same default settings", "A: both under t3-tuned config", "A: both under t2-tuned config", "B: each tier with its own tuned config"]
TT = pd.read_csv(T / "v4_three_tier_common_split.csv")
tt_tab = md([[r.population, r.tier, int(r.n_test_records), int(r.n_test_dois), f"{r.r2:.4f} [{r.r2_ci_lo:.4f}, {r.r2_ci_hi:.4f}]", f(r.mae), f(r.rmse)] for r in TT[TT.row_type == "tier metrics"].itertuples()],
            ["population", "tier", "test records", "test DOIs", "R² [95% publication-cluster CI]", "MAE", "RMSE"])
tp_tab = md([[r.population, r.comparison, sg(r.delta_r2), f"[{sg(r.delta_ci_lo)}, {sg(r.delta_ci_hi)}]", r.seeds_positive, sg(r.mae, 3), sg(r.rmse, 3)] for r in TT[TT.row_type == "paired difference"].itertuples()],
            ["population", "comparison", "ΔR²", "95% paired CI", "seeds ΔR²>0", "ΔMAE (derived)", "ΔRMSE (derived)"])
cov_tab = md(v4rows("covered", KEYS), HEAD); full_tab = md(v4rows("all", KEYS), HEAD)
tun = md([[r.scope, r.comparison, sg(r.delta_r2), f"[{sg(r.ci_lo)}, {sg(r.ci_hi)}]", f"{int(r.seeds_positive)}/{int(r.n_seeds)}"] for r in cmp_[cmp_.comparison.str.startswith("tuning effect")].itertuples()],
         ["scope", "comparison", "ΔR²", "95% CI", "seeds>0"])
ctl = md([[r.scope, r.arm.replace("t2_default -> ", ""), sg(r.delta_r2), f"[{sg(r.ci_lo)}, {sg(r.ci_hi)}]", f"{int(r.seeds_positive)}/{int(r.n_seeds)}"] for r in cmp_[cmp_.comparison.str.startswith("control")].itertuples()],
         ["scope", "control arm (vs t2 default)", "ΔR²", "95% CI", "seeds>0"])
arms = V[V.row_type == "arm"]
arm_tab = md([[r.scope, r.arm, r.tier, r.estimator_config, f(r.r2, 4), f(r.mae, 3), f(r.rmse, 3), int(r.n_test_records), int(r.n_test_dois)] for r in arms.itertuples()],
             ["scope", "arm", "tier", "estimator config", "R²", "MAE", "RMSE", "n test records", "n test DOIs"])
cfg = json.load(open(R / "v4/final/tuned_cfg.json")); cfg_tab = md([[k] + [cfg["t2"][k], cfg["t3"][k]] for k in cfg["t2"]], ["CatBoost parameter", "t2 tuned", "t3 tuned"])

# ---------- original ablation tables
od = O[(O.row_type == "tier metrics") & (O.protocol == "doi") & (O.estimator == "catboost")]
nd = PS.groupby("subset").agg(n_test=("n_test", "mean"), n_doi_test=("n_doi_test", "mean"), n_train=("n_train", "mean"), n_val=("n_val", "mean"), n_doi_train=("n_doi_train", "mean"), n_doi_val=("n_doi_val", "mean"))
abl_rows = []
for sub in ("pooled", "p-i-n", "n-i-p"):
    for t in ("t1", "t2", "t2+E"):
        r = od[(od.subset == sub) & (od.tier == t)].iloc[0]
        abl_rows.append([sub, "t2+E = t2 + earlier literature representation E" if t == "t2+E" else t, f"{r.mean_n_features:.1f}", f"{r.mean_n_test_records:.0f}", f"{nd.loc[sub,'n_doi_test']:.0f}",
                         pm(r.r2_mean, r.r2_sd), pm(r.mae_mean, r.mae_sd), pm(r.rmse_mean, r.rmse_sd)])
abl_tab = md(abl_rows, ["subset", "tier", "mean features", "mean test records/split", "mean test DOIs/split", "R² (mean ± sd, 5 splits)", "MAE", "RMSE"])
pdiff = O[(O.row_type == "paired difference") & (O.protocol == "doi") & (O.estimator == "catboost")]
pd_tab = md([[r.subset, r.tier, sg(r.delta_r2_mean, 3), f(r.delta_r2_sd), f"{int(r.splits_r2_better)}/5", f"{int(r.splits_ci_excludes_zero)}/5", sg(r.delta_rmse_mean, 3)] for r in pdiff.itertuples()],
            ["subset", "contrast", "ΔR² mean", "ΔR² sd", "splits R² better", "splits CI excludes 0", "ΔRMSE mean"])
ctrl = CE[CE.experiment.str.startswith("feature-group")]
fg_tab = md([[r.partition, r.feature_representation, f(r.n_features, 1), pm(r.r2_mean, r.r2_sd), f(r.mae_mean), f(r.rmse_mean)] for r in ctrl.itertuples()], ["subset", "variant", "mean features", "R² mean ± sd", "MAE", "RMSE"])

# ---------- model development
mdt = md([[r.subset, r.tier + (" (t2 + earlier representation E)" if r.tier == "t2+E" else ""), r.estimator.split(" (")[0].replace("mean of CatBoost bag and XGBoost", "Ensemble (mean of Bag and XGB)"), pm(r.r2_mean, r.r2_sd), f(r.mae_mean), f(r.rmse_mean)] for r in MD.itertuples()],
         ["subset", "tier", "model", "R² mean ± sd (5 splits)", "MAE", "RMSE"])
sel = MD[(MD.subset == "pooled") & MD.estimator.str.startswith("CatBoost tuned")][["tier", "selected_configs_by_split"]]
sel_tab = md([[r.tier, r.selected_configs_by_split] for r in sel.itertuples()], ["tier (pooled)", "selected {depth, l2_leaf_reg} per split 1-5"])

# ---------- protocols
def ep(proto, est, sub="pooled"):
    d = EP[(EP.protocol.str.startswith(proto)) & (EP.estimator == est) & (EP.subset == sub)]
    return {r.tier: r for r in d.itertuples()}
proto_rows = []
for label, key in (("DOI-grouped", "DOI-grouped"), ("record-level random", "record-level"), ("lead-author-grouped", "lead-author"), ("forward-in-time (3 windows)", "forward")):
    for est in ("catboost", "ridge", "mean"):
        d = ep(key, est)
        proto_rows.append([label, est, int(d["t2+E"].n_splits), f"{d['t2+E'].mean_n_test_records:.0f}"] + [pm(d[t].r2_mean, d[t].r2_sd) for t in ("t1", "t2", "t2+E")] + [f(d["t2+E"].mae_mean, 2)])
proto_tab = md(proto_rows, ["protocol", "model", "splits", "mean n test", "R² t1", "R² t2", "R² t2+E (earlier representation)", "MAE t2+E"])
gap = {est: {t: ep("record-level", est)[t].r2_mean - ep("DOI-grouped", est)[t].r2_mean for t in ("t1", "t2", "t2+E")} for est in ("catboost", "ridge")}
temporal = pd.read_csv(T / "evaluation_protocols_temporal_windows.csv"); temporal = temporal[temporal.arch == "overall"]
tw_tab = md([[r.split, r.tier, r.train_date_max, r.test_date_min, int(r.n_train), int(r.n_test), f(r.r2), f(r.mae, 2)] for r in temporal.itertuples()], ["window", "tier", "train until", "test from", "n train", "n test", "R²", "MAE"])
bl = BS[BS.study.str.startswith("this study")]
bl_tab = md([[r.estimator, r.tier, pm(r.r2_mean, r.r2_sd), pm(r.mae_mean, r.mae_sd), pm(r.rmse_mean, r.rmse_sd)] for r in bl.itertuples()], ["model", "tier (t2+E = t2 + earlier representation E)", "R²", "MAE", "RMSE"])
prior = md([[r.study, r.target, r.estimator, r.split_protocol, f(r.r2_mean, 4) if r.r2_mean == r.r2_mean else "NA", f(r.rmse_mean, 2) if r.rmse_mean == r.rmse_mean else "NA", f(r.mae_mean, 2) if r.mae_mean == r.mae_mean else "NA", r.units if isinstance(r.units, str) else ""]
            for r in BS[~BS.study.str.startswith("this study")].itertuples()], ["study", "target", "model", "split", "R²", "RMSE", "MAE", "units / note"])
idc = CE[CE.experiment.str.startswith("material-identity")]
id_tab = md([[r.feature_representation.replace("t1/t2 vs t1+identity : ", ""), r.reference, f(r.r2_reference, 4), f(r.r2_tier, 4), sg(r.delta_r2), f"[{f(r.ci_lo_dmse,3)}, {f(r.ci_hi_dmse,3)}] (ΔMSE)", r.seeds_positive, r.folds_positive] for r in idc.itertuples()],
            ["tier", "reference", "R² ref", "R² tier", "ΔR² (tier − ref)", "95% CI", "seeds>0", "folds>0"])
ee = json.load(open(R / "extraction_validation.json"))
num_tab = md([[k, v["n_dois"], f(v["spearman"], 2), f(v["within_10pct"], 2), f(v["exact_equal_where_both"], 2), f(v["llm_value_where_pdp_missing_frac_of_records"], 3)] for k, v in ee["numeric"].items()],
             ["field", "DOIs", "Spearman", "within 10%", "exact equal where both exist", "LLM value where PDP missing (fraction of records)"])
flag_tab = md([[k, v["n_dois"], f(v["prevalence_pdp"], 3), f(v["precision"], 2), f(v["recall"], 2), f(v["kappa"], 2)] for k, v in ee["flags"].items()], ["flag (earlier representation E)", "DOIs", "PDP prevalence", "precision", "recall", "kappa"])
sg_ = pd.read_csv(R / "shap/shap_groups.csv")
shap_tab = md([[r.arch, r.group, r.n_features, f(r.sum_mean_abs_shap), f"{100*r.share:.1f}%"] for r in sg_.itertuples()], ["model", "group", "features", "Σ mean|SHAP|", "share"])
top = []
for a in ("overall", "pin", "nip"):
    d = pd.read_csv(R / f"shap/shap_ranking_{a}.csv").head(8)
    top.append([a, "; ".join(f"{r.feature} {r.mean_abs_shap:.3f}" for r in d.itertuples())])
top_tab = md(top, ["model", "top 8 features by mean |SHAP| (PCE points)"])
fl = FC[(FC.representation == "V4 recipe flags (rc_*)")]
GRP = {"P": "additive/precursor", "H": "HTL dopant", "E": "ETL dopant", "R": "process route", "C": "device context", "U": "undoped / interface passivation"}
import psc.v4 as v4m
g_of = {f"rc_{c}": GRP[g] for g, cs in v4m.REC_GROUPS.items() for c in cs}
flag4 = md([[r.feature, g_of[r.feature], f(r.flag_prevalence_among_2835_papers, 3)] for r in fl.itertuples()], ["flag", "group", "prevalence among 2,835 extracted papers"])
orig_cov = FC[(FC.representation == "original literature block")]

# ---------- verification of prediction-error claims from stored predictions
ys, ps = [], []
for k in range(1, 6):
    d = np.load(R / f"ablation_v1/preds/doi_overall_t3_catboost_{k}.npz"); ys.append(d["y"]); ps.append(d["pred"])
y1, p1 = ys[0], ps[0]; yA, pA = np.concatenate(ys), np.concatenate(ps)
hi1 = y1 >= 20; hiA = yA >= 20
hp = (f"split 1: n={len(y1)}, records with PCE ≥ 20%: {hi1.sum()}, mean(pred − exp) for those = {np.mean(p1[hi1]-y1[hi1]):+.2f} pp, max prediction {p1.max():.1f}%, "
      f"sd(pred)/sd(exp) = {p1.std()/y1.std():.2f}; five splits stacked: n={len(yA)}, ≥20%: {hiA.sum()}, mean(pred − exp) = {np.mean(pA[hiA]-yA[hiA]):+.2f} pp; "
      f"negative predictions: {(pA<0).sum()} of {len(pA)} (min {pA.min():.2f}%).")
nm = pd.read_csv(T / "_records_per_doi_describe.csv", index_col=0)

REP = f"""# RESULTS REPORT — perovskite solar-cell PCE prediction (handoff for manual manuscript writing)

Generated from existing artifacts only (no training, no LLM calls, no new experiments). Project root: `{ROOT}`. Every table below is produced by
`results_handoff/build_report.py` from the files named in section I; hand-typed numbers are asserted against sources by `verify_handoff.py`.
**Tier naming used in this report.** `t1` = PDP features; `t2` = t1 + material properties; `t3` = t2 + the **final V4 recipe flags (36)**.
The earlier **44-predictor literature representation (E) is never called t3 here**: in the run files and logs of the old five-split experiments the tier code `t3`
means t2 + that earlier block; in this report and in the CSV tables it is written **t2+E** (the original code is kept in the column `run_file_tier`). `t3` refers only to the final V4 recipe flags.
Results of the earlier representation and of V4 come from different protocols and are not directly comparable.

## A. Research objective and dataset

* **Objective.** Predict device power conversion efficiency (target `final_pce` = PDP `JV_default_PCE`, %, from `data/dataset_json_enriched.csv`) for unseen
  publications, and quantify how much LLM-derived paper-level features add beyond PDP and material-property features.
* **Source.** Perovskite Database Project export (Jacobsson et al., Nat. Energy 7, 107 (2022), CC BY 4.0), enriched with material properties and LLM features (`docs/DATA.md`).
* **Counts (verified by `verify_handoff.py`).** 43,398 raw device records × 156 columns; 7,499 raw DOI strings. Removed: 1,002 records with missing PCE and a further 1,289 with PCE < 1.0%.
  Filtered set: **41,107 device records** from **7,326 publications** (normalised DOIs) plus **one explicit block of 232 records without a DOI** (7,327 groups in all code; the 232-record block is never split).
  Raw-string DOIs in the filtered data: 7,371 → 7,326 after normalisation (45 case/prefix variants; `docs/DATA.md`, `docs/AUDIT_REPORT.md`).
* **Architecture.** 28,340 n-i-p, 12,684 p-i-n, 83 other/unspecified (52 Unknown, 20 Back contacted, 7 Front contacted, 3 Schottky, 1 Pn-heterojunction); the 83 appear only in pooled models. Architecture subsets are modelled independently.
* **PCE distribution (filtered).** mean 12.42, median 13.0, range 1.0–36.2%. 17 records (6 publications) exceed 27% PCE and are retained under the pre-declared filter (sensitivity: excluding them from test sets changes pooled t2+E R² by about +0.002, `docs/TODO_REGISTER.md` #8).
* **Records vs publications.** Median {nm.loc['50%','n']:.0f} records per DOI group; the maximum of 232 reported in `results/feature_audit.json` is the no-DOI block — the largest real publication has 127 records (the manuscript's "maximum 127" is the publication-level figure; both are correct, see section I). ICC of PCE by DOI = 0.692 (69% of variance between publications). 31,591 records belong to duplicated original-t3 predictor configurations (7,733 groups; only 162 groups span >1 DOI; median within-group sd of PCE 1.21).
* **Feature coverage.** Material properties: mean coverage 0.910 (ETL band gap 0.966, HTL 0.839, perovskite 0.948). LLM-derived blocks: see sections B and F — coverage is all-or-none per paper (about 36–37% of records).
* **Missing-data policy** (`src/psc/features.py`, `docs/AUDIT_REPORT.md`): all preprocessing fitted on the training partition of each split only; columns with >80% missing in training are dropped; numerical columns median-imputed with a binary `_na` indicator; missing categoricals labelled "unknown"; DOI, author, outcome-adjacent measurements (`Stability_measured`, `EQE_measured`, `JV_hysteresis_index`, …) and free text are never predictors. **V4 recipe flags are left as native NaN** for papers without extraction (CatBoost handles missing natively) with a first-feature-use penalty of 0.5 on these columns.

## B. Definitive definitions of t1, t2 and t3

| tier | content | features (pooled) | source |
|---|---|---|---|
| **t1** | PDP-derived device, process and composition features: 21 process, 9 device, 47 composition (incl. `_na` indicators) | 77 | `docs/FEATURES.md`, `src/psc/features.py` |
| **t2** | t1 + curated material-property block (102 columns incl. indicators: band gap, HOMO/LUMO, mobilities, dielectric constant, atomic mass/radius/electronegativity min/max/avg, tolerance/octahedral factors, barriers; first non-missing value per layer family; material-level reference values, not per-device measurements) | 179 | `docs/FEATURES.md` |
| **t3 (final, V4)** | t2 + **36 binary LLM-derived recipe flags** (`rc_*`, `data/lit_v4.json`, `src/psc/litfeat4.py: paper_recipe`, `scripts/build_lit_v4.py`) | 215 in `results/v4/full_rec.csv` (179 + 36); 213 in dev fold files for 177-feature t2 | `results/v4/full_rec.csv` |
| *t2+E* (earlier representation E, superseded; **not t3**) | t2 + 44 predictors = 32 base columns + `_na` indicators (`has_*` material flags, `used_*` process flags, `is_*`, thicknesses/anneal, `json_numeric_values_found`, `has_stability_measurement`, …), paper-level, from the first extraction parse | 223 | `docs/FEATURES.md`; `results/ablation_v1` |

**The 36 V4 flags.** Built from the cached LLM-extraction JSON (2,835 papers) **without any new LLM call**: for each paper, material records are role-mapped (perovskite / ETL / HTL / additive / precursor / solvent / antisolvent / other),
outcome-named keys (PCE, Voc, Jsc, FF, stability…) are dropped, the remaining string content is lower-cased and each flag is set to 1 if its regular expression is matched anywhere in the scope-appropriate records (HTL/ETL flags only in HTL/ETL/additive/other records) — i.e. **1 = "mentioned in the paper's extracted recipe", not "verified as used in this device"**. Groups (`src/psc/v4.py: REC_GROUPS`): additive/precursor identities (14), HTL dopants (6), ETL dopants (2), process route (7), device context (5), undoped/interface passivation (2).
Intended information content: recipe identities that the restricted PDP columns of t1/t2 do not contain (additives, precursors, dopants, route). Recorded development diagnostics (`docs/V4_REPORT.md`): giving t2 the PDP's own curated process columns for the same papers adds only +0.022 R² (additives +0.012; solvents/quench +0.003), which is why the identities were targeted. Flags are paper-level and cannot distinguish devices within a paper.

{flag4}

**Coverage of the V4 block.** Flags exist for papers with an extraction record: **15,027 of 41,107 records (36.6%), 2,731 publications** (`lx_covered`); NaN elsewhere. The 32 publications / 131 records by which this mask exceeds the earlier block's mask all have an extraction record with zero extracted materials (their flags are all 0, not missing); see `docs/COVERAGE_MASK_AUDIT.md`, `results/coverage_mask_audit.csv`. Reserved evaluation partition: 5,460 covered records / 1,005 DOIs of 14,125 / 2,619.
**Do not mix representations.** Results labelled "t2+E" in sections D, E and G (five-split ablation, model development, protocol comparisons, baselines, SHAP) were produced with the earlier 44-predictor representation E. A V4 t1 result on the reserved split was added after the first handoff (section C.0, `results/v4/final_t1/`); no V4 five-split ablation, no V4 Bag/XGB/Ensemble and no V4 SHAP exists.

## C. Main V4 results (reserved evaluation, `results/v4/final/`, `results/v4/final_t1/`)

### C.0 Common-split t1 / t2 / t3 (added after the first handoff; `scripts/v4_t1_baseline.py`)
Same reserved `v4final` partition, test records, matched default CatBoost configuration, preprocessing and 5-seed procedure for all three tiers. t1 was trained for this comparison (it was missing from `results/v4/final/`); t2 and t3 are the existing `t2_default` / `t3_default` predictions (not retrained; their t3−t2 deltas reproduce C.1/C.3). Feature columns after preprocessing: t1 77, t2 179, t3 215. t1 was scored after t2/t3 with identical code path, seeds and validation splits; no choice depended on it.
Source: `results/v4/final_t1/{{metrics_3tier.csv, comparisons_3tier.json}}`; table `tables/v4_three_tier_common_split.csv`.

{tt_tab}

Paired differences (bootstrap over publications, 2,000 resamples; ΔMAE/ΔRMSE are differences of the point metrics, not bootstrapped):

{tp_tab}

Reading: R², MAE and RMSE improve monotonically t1 → t2 → t3 in both populations. The t1 → t2 step has a CI excluding zero; the t2 → t3 step (+0.0071 covered, +0.0025 all) has a CI that **includes zero**, so it is not statistically established. These are separate from the separately tuned comparisons below (t1 was not tuned).

**Protocol.** (`scripts/v4_final.py`, `docs/V4_REPORT.md`) Publications assigned by SHA-256 of the normalised DOI with salt `psc-v4` (`psc.v4.v4_partition`): `v4dev` 65% (26,982 records / 4,708 DOIs; all design, tuning, ablations) and `v4final` 35% (**14,125 records / 2,619 DOIs; of which 5,460 records / 1,005 DOIs are covered by V4 flags**), scored once.
Final models: trained on all `v4dev` records, DOI-disjoint 15% validation split per seed for early stopping (100 rounds), **5 seed models averaged**, predictions scored on `v4final`. t2 = `psc.features.base_frame(df,'t2')`; t3 = t2 + 36 `rc_*` columns (NaN for uncovered papers, first-feature-use penalty 0.5). "Default" = the matched CatBoost configuration (depth 8, lr 0.05, L2 3, subsample 0.8, colsample 0.7, min_data_in_leaf 5, ≤1,500 iterations, `psc.v4model.DEFAULT_PARAMS`);
"tuned" = best of 24 random-search configurations per tier (default included) by DOI-grouped 5-fold mean squared error on `v4dev` (`scripts/v4_tune.py`, `results/v4/tune.json`; t3 has lower tuning MSE than t2 in only 6/24 shared configurations; best CV MSE t2 14.348, t3 14.387).
CI: publication-cluster bootstrap, 2,000 resamples (seed 0), of the paired squared-error difference of the seed-averaged predictions, divided by the test-set target variance; 95% percentile interval. "seeds ΔR²>0" counts the five per-seed paired differences.
**Held-out status:** the `v4final` DOIs were seen in earlier experiments with other feature representations (earlier representation E, lx2/lx3 blocks); they are fresh for the V4 representation but **not untouched data**; the flag block was selected after development comparisons on `v4dev`.

Tuned configurations (`results/v4/final/tuned_cfg.json`):

{cfg_tab}

### C.1–C.2 Covered records (papers with V4 flags; 5,460 records / 1,005 DOIs): same-settings (A) and separately tuned (B)
{cov_tab}

### C.3–C.4 All reserved records (14,125 records / 2,619 DOIs)
{full_tab}

\\* ΔMAE and ΔRMSE are **derived** here as (comparison − reference) from `metrics.csv`; the artifacts store no CI for MAE/RMSE. Negative = lower error for t3. "A both under t3-tuned/t2-tuned config" applies one tuned configuration to both tiers.
Matches `docs/V4_REPORT.md`: A default covered +0.0071 [−0.0007, +0.0157] (4/5), all +0.0025 [−0.0011, +0.0062] (4/5); B covered +0.0094 [−0.0016, +0.0219] (5/5), all +0.0015 [−0.0039, +0.0067] (3/5); absolute covered R² t2-tuned 0.361, t3-tuned 0.371. **All intervals include zero.**

### C.5 Tuning effects (not LLM comparisons) and negative controls
{tun}

{ctl}

Controls: five copies of the 36-column block with rows permuted across publications (DOI-level permutation, `np.random.default_rng(100+j)`) and a coverage-indicator-only block (`rc_covered`), all with the matched default config. Covered permuted range −0.0049…+0.0015, all-records −0.0028…+0.0007; coverage-only +0.0015 / +0.0011.
Interpretation (team, `docs/V4_REPORT.md`): tuning alone moves t2 by about +0.002…+0.003, comparable to the t3 effect, and the t3 gain shrinks under t2's tuned configuration (+0.0024 covered; −0.0016 all), so the tuned comparison B should not be read as an LLM gain by itself.

### C.6 Metrics for every arm (`results/v4/final/metrics.csv`)
{arm_tab}

### C.7 Development-stage evidence (v4dev only; DOI-grouped 5-fold × 3 seeds, 15 folds; recomputed from `results/v4/*.csv`)
* Block with first-feature-use penalty 0.5 (`pen_05.csv`, the final setting): covered-dev ΔR² = **+0.0070, 12/15 folds positive**; without penalty (`rec1.csv`): +0.0052, 12/15. (`docs/V4_REPORT.md` quotes +0.007, which corresponds to the penalised run; it is not a discrepancy.)
* Five DOI-permuted copies (`recctl.csv`, unpenalised): −0.0086, −0.0056, −0.0029, +0.0022, −0.0069 (matches the reported range −0.009…+0.002).
* Existing JSON content otherwise exhausted: token bags +0.004 (`blocks1.csv`: +0.0039), numeric quantities −0.0023, device-matched layer values −0.0012, PDP gap-filling −0.0030 (all from `blocks1.csv`); full-data all-record 5-fold with penalty-free block: +0.0027, 11/15 folds (`full_rec.csv`).
* Dropping any one flag group lowers the gain and no group alone reproduces it (`recabl.csv`: group-only −0.0053…+0.0005; drop-one +0.0010…+0.0030 vs +0.0052 for all) — a distributed signal near the noise floor. Role-scoped variant +0.001 (`rec2.csv`).
* These dev numbers are **development results** (the flag set and penalty were chosen on them) and must not be presented as held-out evidence.

## D. Earlier fixed-configuration ablation (earlier representation E; `results/ablation_v1`)

**Protocol.** Five repeated DOI-grouped holdouts (`GroupShuffleSplit`, test 20% of publications, seed 42; per-split inner DOI-disjoint validation = 15% of training publications, used only for early stopping). Config `configs/ablation.json`, data SHA-256 `{meta['data_sha256'][:16]}…`. One fixed CatBoost configuration for all tiers (depth 8, lr 0.05, L2 3, subsample 0.8, colsample 0.7, min leaf 5, seed 42, ≤1,500 iterations, 100-round early stopping on validation) — **hyperparameters inherited from earlier work, not tuned in this study**. Mean train/val/test sizes per split (pooled): ≈{nd.loc['pooled','n_train']:.0f} / {nd.loc['pooled','n_val']:.0f} / {nd.loc['pooled','n_test']:.0f} records; ≈{nd.loc['pooled','n_doi_train']:.0f} / {nd.loc['pooled','n_doi_val']:.0f} / {nd.loc['pooled','n_doi_test']:.0f} DOIs. Zero DOI overlap between train/validation/test in every split (recorded in `runs.jsonl`, `tests/test_protocol_audit.py`).
Test sets of the five splits overlap, so sd values describe split-to-split variability, not CIs. **t2+E in this section = t2 plus the earlier 44-predictor literature representation E (run-file code `t3`).** These values are unchanged by V4.

{abl_tab}

Paired differences (per-split cluster bootstrap over test publications, 1,000 resamples; `paired_tier_diffs.csv`, `paired_bootstrap_summary_doi_catboost.csv`):

{pd_tab}

Feature-group controls (earlier representation E; same protocol; `results/feature_groups_v1`):

{fg_tab}

Material-identity control (`docs/IDENTITY_CONTROL.md`, `results/identity_vs_t1id.csv`; full data, DOI-grouped 5-fold × seeds 42–44, matched config; 41,107 records / 7,327 DOI groups; **different protocol from the five holdouts**):

{id_tab}

Additional single-stage full-data results with other literature blocks (E2–E6, `results/ablE_vs_t2_*.csv`, matched config, DOI-grouped 5-fold × 3 seeds): earlier representation E −0.0024 (pooled), −0.0007 (n-i-p), −0.0002 (p-i-n) relative to t2 (t2 R² 0.4073 / 0.4246 / 0.3218); this is a **different protocol** from the five holdouts above (pooled t2+E − t2 = −0.0013) — both are null.
Per-split values, split sizes and feature counts: `tables/original_tier_ablation_per_split.csv`.

## E. Model-development comparison (`results/model_dev_v1`, **earlier representation E only**)

**Procedure** (`scripts/run_model_dev.py`, `configs/model_dev.json`): per (subset, tier, DOI-grouped split) — same 5 splits as section D — select among a predeclared CatBoost grid of six configurations
(depth/L2 ∈ {{(6,3),(8,3),(10,3),(8,10),(10,10),(8,1)}}; lr 0.05, ≤3,000 iterations, subsample 0.8, colsample 0.7, min leaf 5; 150-round early stopping) by **validation RMSE**; then (i) tuned CatBoost, (ii) 3-seed bag (seeds 42, 7, 123) of the selected configuration, (iii) XGBoost with early stopping on the validation set, (iv) mean of bag and XGBoost. The test partition is used only for final scores; identical grid and budget for every tier. 45 cells (3 subsets × 3 tiers × 5 splits).
**t2+E here = t2 plus the earlier representation E. No V4 entries exist; none were invented.**

{mdt}

Selected configurations change between splits (pooled):

{sel_tab}

Headline (team reading in `docs/AUDIT_REPORT.md`): pooled ensemble 0.418 ± 0.026, MAE 2.92 — 0.011 above matched CatBoost, produced by tuning/bagging/ensembling, not by features (t1 0.377 → t2 0.416 → t2+E 0.418).

## F. Feature engineering and extraction

* **Full-text retrieval and extraction coverage.** 7,499 raw DOI strings in the PDP; 3,299 PDFs retrieved → 3,141 Markdown conversions (OpenDataLoader PDF) → 2,835 structured JSON paper records (`json/`, not packaged) → 2,726 raw DOI strings = 2,699 normalised DOIs matched to the filtered data (14,896 records, 36.2%) for the earlier representation E; the V4 mask (`lx_covered`) is 15,027 records / 2,731 DOIs (36.6%). (Attrition counts: `docs/TODO_REGISTER.md` #6, `docs/AUDIT_REPORT.md`; the 3,299/3,141/2,835 counts come from the project logs/docs, not from a file in `results/`.)
* **LLM extraction.** Locally served Qwen (alias `qwen35`, vLLM, temperature 0.1, ≤16,384 output tokens, one request per document, zero-shot system prompt recovered verbatim in `docs/EXTRACTION.md`). **The exact checkpoint/quantisation is not recorded anywhere in the repository** (manuscript text says "Qwen3.5-9B" — unverifiable from artifacts). The number of few-shot examples cannot be verified; the recovered script is zero-shot.
* **Original literature block** (`src/psc/litfeat.py`, `docs/FEATURES.md`): 32 paper-level base columns (material-presence `has_*`, process `used_*`, thicknesses/anneal/spin/concentration, reporting-completeness `json_numeric_values_found`, `has_stability_measurement`) → 44 predictors with indicators; coverage 36.2% of records, constant within a paper. Per-feature coverage (`results/feature_audit.json`): flags 0.362; thicknesses 0.081–0.176; anneal 0.037–0.070; concentration 0.051; spin speed 0.100.
* **Quantity-level / re-parsed exploration (lx2, lx3; development-era, not final).** lx2.0: 61 publication-level columns (process, chemistry, context) used in a two-stage residual design on covered papers (dev +0.0022 [−0.0003, +0.0051], 5/5 seeds; confirm +0.0011 [−0.0015, +0.0036], 4/5; pre-registered CI-excludes-zero rule **not met**; gain concentrated in a minority of publications — removing the 25 most-helped publications turns it negative; `docs/LX_RESULTS.md`, `docs/PREREG_LX.md`). lx3.0: 39 columns (30 quantity `lx_qty_*`, 9 chemistry `lx_cat_*`); most quantity columns cover 2–15% of papers; only 2 of 30 survived the 80%-missingness filter in earlier runs. A quantity-covered experiment on the 9,441-record / 1,680-DOI eligible population (15 predeclared columns with ≥10% coverage; partition `qdev` 6,677 rec / 1,180 DOIs, `qconf` 2,764 / 500) gave dev ΔR² +0.0104 (t2 + quantities) and held-out +0.0026 [−0.010, +0.014], with the earlier representation E −0.0141 [−0.038, +0.006] on `qconf`; controls fluctuate −0.01…+0.005 (`docs/LX3_RESULTS.md`, `results/qty_holdout/summary.json`). Exploratory; population selected by availability. Table: `tables/control_experiments.csv`.
* **Why the V4 groups were retained** (evidence recorded in `docs/V4_REPORT.md` and section C.7): token bags, numeric quantities, device-matched layer values and PDP gap-filling gave ~0 or negative on development; recipe identities gave +0.007 on covered dev (12/15 folds) versus permuted controls −0.009…+0.002; no single group reproduces the gain; interpretation that the additive/recipe identities are the targeted missing information is the team's (see ceiling analysis, +0.022 for PDP's own curated columns).
* **Extraction validation (proxy only; no gold annotation exists; hallucination rate not measured).** Agreement with PDP's independent curation (`results/extraction_validation.json`, `scripts/validate_extraction.py`), earlier representation E, 2,699 DOIs:

{num_tab}

{flag_tab}

  `has_BCP`/`has_LiF` cannot be assessed (PDP stack strings rarely contain those layers). Numerical values equal the PDP value exactly in 59–82% of records where both exist and add a value where PDP is empty in only 0.8–5.3% of records — the earlier representation E is largely redundant with t1. **No validation of the V4 flags against any reference was run**; they are regexes over the same extraction JSON.
* **Original-block coverage vs V4 coverage** are different masks (see section I).

## G. Other scientific results (all on the earlier representation E where a literature tier appears)

### G.1–G.3 Evaluation protocols (pooled; matched CatBoost, ridge, training-mean; `tables/evaluation_protocols.csv`)
{proto_tab}

* Record-level vs DOI-grouped gap (R² difference, CatBoost): t1 {gap['catboost']['t1']:.3f}, t2 {gap['catboost']['t2']:.3f}, t2+E {gap['catboost']['t2+E']:.3f}; ridge: {gap['ridge']['t1']:.3f}, {gap['ridge']['t2']:.3f}, {gap['ridge']['t2+E']:.3f}. Record-level CatBoost for the pooled data = converged 20,000-iteration rerun (best iterations 7,207–11,702; the 6,000-cap run was 0.005–0.007 lower; the original 1,500-cap run truncated 31/45 fits). Record-level p-i-n/n-i-p values come from the 6,000-cap rerun. `tables/evaluation_protocols_gap_ablation_v1.csv` holds the gap computed inside `ablation_v1` (uses the truncated record-level fits — different from the converged values above; see section I).
* Lead-author grouping uses the modal PDP lead-author surname per publication (coarse laboratory proxy; large surnames over-group). Forward-in-time: publications dated by earliest PDP record; three windows with quantiles (0.55,0.70], (0.70,0.85], (0.85,1.00]; train strictly earlier, most recent 15% of train as validation; zero DOI/author overlap.

Forward windows (pooled CatBoost, `strict_validation_v1`):

{tw_tab}

### G.4 Reference models (pooled, 5 DOI-grouped holdouts; `results/baselines_v1`, `ablation_v1`)
{bl_tab}

XGBoost defaults: `tree_method=hist`, no tuning/early stopping; RF 300 trees; MLP two hidden layers with validation early stopping; GPR exact on a 3,000-record random training subsample. These are **not** re-implementations of the Roberts/Fukasawa pipelines.

### G.5 Prediction distribution and high-PCE error (recomputed from `results/ablation_v1/preds/doi_overall_t3_catboost_*.npz`, matched CatBoost, earlier representation E)
{hp}
Squared-error training regresses to the mean; predictions saturate near 21%.

### G.6 Material-property vs material-identity control
See section D table (identity-only t1_id R² 0.4062 vs t2 0.4073 vs t1 0.3698; t2 − t1_id = +0.0008 CI includes zero) — team interpretation: the t2 gain over t1 reflects knowing *which* materials are present; transferable physical information is not demonstrated (leave-material-out test not run).

### G.7 SHAP (`scripts/shap_analysis.py`, matched CatBoost trained on t2+E (earlier representation E, not t3) on DOI-grouped split 1; TreeExplainer on 1,500 random test records; attribution, not benefit)
{shap_tab}

{top_tab}

### G.8 Prior PDP-based studies (as verified against the PDFs in `../Lit/`, `docs/LIT_VERIFICATION.md`)
{prior}

Metrics are not comparable across targets (Roberts: J_SC in mA cm⁻²; ours: PCE, percentage points). Hussain et al.: split not stated. arXiv:2602.04914 and 2602.13312 (LLM–perovskite pipelines) were **not read**.

## H. Limitations and unresolved issues (no resolution attempted)
1. **Incremental V4 gain is not statistically established**: every 95% interval (covered/all × same-settings/tuned) includes zero; point estimates are small (+0.0015…+0.0094 R²) and configuration-dependent (shrinks to +0.0024 / −0.0016 under t2's tuned config); tuning alone moves t2 by +0.002…+0.003.
2. **Partial coverage**: V4 flags exist for 36.6% of records (covered-record R² ≈ 0.36 vs ≈ 0.40 overall; different populations); full-data benefit is diluted; the other ≈4,500 publications lack extraction and no local full texts exist.
3. **Reused evaluation publications**: `v4final` DOIs appeared in earlier experiments with other feature sets; the flag block was chosen after development comparisons; a single reserved split of ≈1,000 covered publications resolves R² differences to about ±0.01.
4. **Development vs held-out**: V4 development numbers (+0.007 etc.) are tuned-on-dev; only section C is held-out; the five-split ablation numbers are repeated overlapping holdouts, not independent CV.
5. **No external validation** on independently collected devices; DOI grouping does not separate same-laboratory papers; 1,277 records in 162 groups of identical t2+E predictor vectors span >1 DOI (1,277 is quoted from `docs/AUDIT_REPORT.md` / `audit/adversarial_review.md`; the 162 groups are in `feature_audit.json`; the 1,277 count is not stored in a results file).
6. **Representation mismatch**: a V4 t1/t2/t3 comparison exists only for the single reserved split (C.0); no V4 five-split ablation, no V4 Bag/XGB/Ensemble, no V4 p-i-n/n-i-p split, no V4 SHAP, no V4 reference models; all those results use the earlier representation (t2+E).
7. **Extraction**: no gold annotation, no hallucination rate, Qwen checkpoint unrecorded, V4 flags unvalidated, flags mean "mentioned" not "used in the device".
8. **Provenance of retrieved full texts (needs the authors' decision; see `docs/RETRIEVAL_PROVENANCE_DECISION.md`)**: `legacy/NOTES_retrieval_history.md` records that the recovered scripts that produced the historical corpus tried OpenAlex first and then Sci-Hub domains, with some manual downloads, and the per-PDF route was not logged; the released `scripts/retrieve_fulltext.py` is legal-route-only with a per-DOI log but did not produce the historical corpus. The manuscript now states this and makes no claim of exclusively legal retrieval. `docs/AUDIT_REPORT.md` states the corpus behind the reported features should be regenerated or checked against the legal-only policy before release. Not resolved here.
9. **Reproducibility**: `ml_paper/` was not under git when the experiments ran (`git_dirty: true` in meta files); it now has its own repository (baseline commit after the runs); original 270-fit ablation reproduced in a clean venv (max |ΔR²| 1.1e-4); other experiments run once; the extraction stage cannot be reproduced from the package.
10. Fixed CatBoost hyperparameters in the ablation were inherited, not tuned; V4 tuning used a 24-configuration random search.

## I. Source-of-truth index

| result group | authoritative source | configuration / split |
|---|---|---|
| V4 arms, comparisons, controls (section C) | `results/v4/final/metrics.csv`, `comparisons.json`, `tuned_cfg.json`; script `scripts/v4_final.py` | v4final reserved partition; trained on v4dev; 5 seed models |
| V4 tuning | `results/v4/tune.json`, `scripts/v4_tune.py` | 24 configs × DOI-grouped 5-fold on v4dev |
| V4 development diagnostics | `results/v4/{{rec1,pen_,pen_05,pen_20,recctl,recabl,rec2,blocks1,full_rec}}.csv`, `docs/V4_REPORT.md` | v4dev; DOI-grouped 5-fold × 3 seeds |
| Original ablation (section D) | `results/ablation_v1/{{runs.jsonl,summary.csv,paired_*}}`, `configs/ablation.json` | 5 DOI-grouped holdouts |
| Feature-group controls | `results/feature_groups_v1/runs.jsonl`, `configs/ablation_groups.json` | same 5 holdouts |
| Model development (section E) | `results/model_dev_v1/runs.jsonl`, `scripts/run_model_dev.py`, `configs/model_dev.json` | same 5 holdouts |
| Protocol comparisons (G.1–G.3) | `ablation_v1`, `ablation_random_cap6000` (6k iters), `random_convergence_cap20000` (pooled converged), `strict_validation_v1`; `psc/results.py: load_ablation` | see section G |
| Reference models | `results/baselines_v1/runs.jsonl`, `configs/baselines.json` | same 5 holdouts |
| SHAP | `results/shap/*.csv`, `scripts/shap_analysis.py` | split 1, 1,500 test records |
| Identity / E2–E6 / lx2 / lx3 / qty controls | `results/identity_vs_t1id.csv`, `ablE_vs_t2_*.csv`, `lx_*_summary.csv`, `lx_dev_vs_t2.csv`, `qty_dev_summary.csv`, `qty_holdout/summary.json`, docs `IDENTITY_CONTROL.md`, `LX_RESULTS.md`, `LX3_RESULTS.md` | DOI-grouped 5-fold or qdev/qconf; see CSV `partition` column |
| Extraction validation, coverage, ICC | `results/extraction_validation.json`, `results/feature_audit.json`, `results/coverage_mask_audit.json` | earlier block (validation); both masks (audit) |
| V4 common-split t1/t2/t3 (C.0) | `results/v4/final_t1/{{metrics_3tier.csv,comparisons_3tier.json}}`, `scripts/v4_t1_baseline.py` | v4final reserved partition; trained on v4dev; matched default CatBoost; 5 seed models |
| Prior work | `docs/LIT_VERIFICATION.md`, `../Lit/*.pdf` | as published |
| Earlier audit context | `audit/adversarial_review.md`, `audit/lit_audit.md`, `docs/AUDIT_REPORT.md`, `docs/REPRODUCIBILITY_REPORT.md`, `docs/TODO_REGISTER.md` | pre-V4 manuscript audit; its manuscript-specific findings are not restated here |

**Disagreements / apparent inconsistencies (unresolved unless stated; no preferred value chosen):**
1. *Original-block vs V4 coverage*: 36.2% / 14,896 records / 2,699 DOIs (`feature_audit.json`: any original lit feature) vs 36.6% / 15,027 records / 2,731 DOIs (`lx_covered`, used for V4 and lx2/lx3). Different masks (2,699 vs 2,731 DOIs) of the same 2,835 JSON files. **Explained (audit `docs/COVERAGE_MASK_AUDIT.md`)**: the 32 extra DOIs (131 records) all have an extraction record with zero extracted materials; the earlier block is entirely missing for them, the recipe flags are all 0. Why those extractions are empty is not recoverable (raw JSON/PDFs not in the repository). The masks are not identical.
2. *Maximum records per publication*: 232 (`feature_audit.json`) vs 127 (largest real DOI). Explained: 232 is the no-DOI block. Resolved (explained).
3. *Publication count*: 7,326 publications vs 7,327 groups: the extra group is the no-DOI block. Resolved (explained).
4. *Earlier-block (t2+E) − t2*: −0.0013 (five holdouts, section D) vs −0.0024 (single-stage full-data DOI-grouped 5-fold × 3 seeds, `ablE`); different protocols and seeds; both null; not reconciled.
5. *Dev gain*: `docs/V4_REPORT.md` +0.007 = penalised run (`pen_05.csv`); unpenalised +0.0052. Explained, not conflicting.
6. *Record-level R²*: `ablation_v1` internal record-level fits are the 1,500-iteration truncated ones; converged reruns give pooled 0.587/0.618/0.627; the gap in `protocol_gap.csv` uses the truncated fits. Use the converged table (G.1) — flagged, not silently chosen: both CSVs are in `tables/`.
7. *Retrieval route*: legal-only released script vs recovered scripts that also used Sci-Hub (limitation 8). Unresolved; authors' decision (`docs/RETRIEVAL_PROVENANCE_DECISION.md`).
8. *Qwen checkpoint*: manuscript says Qwen3.5-9B; the repository does not record it. Unverifiable.
9. *Attrition counts* 3,299 → 3,141 → 2,835 are from docs/logs, not from a results file; not independently re-counted here.
"""
(H / "RESULTS_REPORT.md").write_text(REP)
print("report written", len(REP))
