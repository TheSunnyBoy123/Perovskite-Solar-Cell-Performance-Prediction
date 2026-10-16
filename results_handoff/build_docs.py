"""Write DATA_DICTIONARY.md, figures/FIGURE_MANIFEST.md and README.md (static text + auto column check)."""
from pathlib import Path
import pandas as pd, glob
H = Path(__file__).resolve().parent
D = {
"row_type": "'arm' (one scored model) | 'comparison' | 'paired comparison' | 'tier metrics' | 'paired difference' (what the row represents)",
"scope": "V4 evaluation subset: 'covered (papers with V4 flags)' or 'all records' of the reserved v4final partition",
"arm": "V4 model arm (e.g. t2_default, t3_tuned, ctl_perm0_default) or 'reference -> comparison' arm pair",
"tier": "feature tier. 't1' = PDP features, 't2' = t1 + material properties, 't3' = t2 + final V4 36 recipe flags (only in v4_main_results.csv and v4_three_tier_common_split.csv). In the five-split ablation/model-dev/strict/baselines files the earlier 44-predictor literature representation is written 't2+E' (contrasts '(t2+E)-t2', '(t2+E)-t1'); its original run-file code 't3' is kept in `run_file_tier`.",
"feature_representation": "explicit name of the feature set behind the row (t1, t2, t2 + earlier representation E, V4 recipe flags (36), control description)",
"run_file_tier": "tier code exactly as stored in the original run files / logs (for the earlier representation E this is 't3', which must NOT be read as the final V4 t3)",
"population": "V4 evaluation population: 'LLM-covered' (reserved records whose publication has V4 extraction data) or 'all reserved'",
"comparison": "paired comparison label (V4 files) or contrast; differences are second-minus-first (e.g. 't3 - t2' = t3 minus t2)",
"r2_ci_lo": "lower 95% publication-cluster bootstrap bound of the tier R2", "r2_ci_hi": "upper 95% publication-cluster bootstrap bound of the tier R2",
"delta_ci_lo": "lower 95% paired publication-cluster bootstrap bound of delta R2", "delta_ci_hi": "upper 95% paired publication-cluster bootstrap bound of delta R2",
"split": "evaluation partition",
"estimator": "model/estimator name (catboost, ridge, mean, xgb_default, rf, mlp, gpr_sub, or a model-development variant)",
"estimator_config": "hyperparameter configuration of the arm (matched default / t2-tuned / t3-tuned / cross-applied)",
"comparison": "which paired comparison: A (same settings), B (each tuned), tuning effect, or control",
"protocol": "evaluation protocol (DOI-grouped, record-level random, lead-author-grouped, forward-in-time)",
"split_protocol": "split description for baseline/prior-work rows",
"data_split": "description of the partition and training/validation scheme for the row",
"partition": "population/partition label for control rows (e.g. v4final covered, covered dev, qconf)",
"experiment": "experiment name for control rows",
"subset": "pooled | n-i-p | p-i-n",
"arch": "architecture code in source files (overall=pooled, pin=p-i-n, nip=n-i-p)",
"model": "estimator code in temporal-window / gap files",
"study": "study identifier (this study or cited prior work)",
"target": "predicted quantity (PCE %, or Jsc mA cm-2 for Roberts et al.)",
"units": "units / comparability note",
"n_splits": "number of splits/windows aggregated",
"n_seeds": "number of seeds in the paired comparison",
"n_seed_models": "number of seed models averaged to form the predictions",
"split": "split index (1-5) or forward window index",
"n_records": "number of records in the evaluated population (records = device entries)",
"n_dois": "number of unique publications (normalised DOIs) in the population",
"n_test_records": "number of test records",
"n_test_dois": "number of unique test publications",
"mean_n_test_records": "mean test records per split",
"mean_n_test_dois": "mean test publications per split",
"mean_n_features": "mean feature count after preprocessing (includes _na indicators)",
"n_features": "feature count",
"n_train": "training records", "n_val": "validation records (early stopping)", "n_test": "test records",
"n_doi_train": "training publications", "n_doi_val": "validation publications", "n_doi_test": "test publications",
"doi_overlap_test_train": "number of DOIs shared between test and train (0 expected)", "doi_overlap_val_train": "DOIs shared between validation and train",
"doi_overlap_test_val": "DOIs shared between test and validation",
"train_date_max": "latest training-publication date (forward windows)", "test_date_min": "earliest test-publication date",
"r2": "coefficient of determination of one scored model", "mae": "mean absolute error (PCE percentage points)", "rmse": "root mean squared error (PCE percentage points)",
"r2_mean": "mean R2 over splits", "r2_sd": "sample sd (ddof=1) of R2 over splits (split variability, not a CI)",
"mae_mean": "mean MAE (pp)", "mae_sd": "sd of MAE over splits", "rmse_mean": "mean RMSE (pp)", "rmse_sd": "sd of RMSE over splits",
"r2_reference": "R2 of the reference arm (t2 side)", "r2_comparison": "R2 of the comparison arm (t3 side)", "r2_tier": "R2 of the tier in a control comparison",
"mae_reference": "MAE reference arm", "mae_comparison": "MAE comparison arm", "mae_tier": "MAE of the tier", "rmse_reference": "RMSE reference arm", "rmse_comparison": "RMSE comparison arm", "rmse_tier": "RMSE of the tier",
"reference": "reference arm/tier of a paired control comparison",
"delta_r2": "R2(comparison) - R2(reference); positive = comparison better",
"delta_r2_mean": "mean over splits of per-split paired delta R2", "delta_r2_sd": "sd over splits of per-split delta R2",
"delta_mae_mean": "mean per-split delta MAE (negative = better)", "delta_mae_sd": "sd of delta MAE", "delta_rmse_mean": "mean per-split delta RMSE", "delta_rmse_sd": "sd of delta RMSE",
"delta_mae_derived": "MAE(comparison) - MAE(reference) computed from metrics.csv (derived, no CI available)", "delta_rmse_derived": "RMSE(comparison) - RMSE(reference), derived",
"ci_lo": "lower end of 95% interval of delta R2", "ci_hi": "upper end of 95% interval of delta R2",
"ci_lo_dmse": "lower end of 95% interval of delta MSE (as stored in the source CSV; NOT delta R2)", "ci_hi_dmse": "upper end of 95% interval of delta MSE",
"ci_method": "how the interval was computed",
"seeds_positive": "number (or n/N) of seeds with positive delta", "folds_positive": "folds (of 15 or 25) with positive delta",
"splits_r2_better": "splits (of 5) in which R2 improved", "splits_ci_excludes_zero": "splits (of 5) whose per-split bootstrap CI excluded zero",
"per_seed_delta_r2": "semicolon-separated per-seed paired delta R2",
"gap_r2": "R2(record-level random) - R2(DOI-grouped)", "r2_doi": "R2 DOI-grouped", "r2_random": "R2 record-level random", "mae_doi": "MAE DOI-grouped", "mae_random": "MAE record-level",
"selected_configs_by_split": "JSON list of the CatBoost {depth,l2_leaf_reg} chosen on validation in splits 1-5",
"v4_entry": "explicit statement that no V4 value exists for model-development rows",
"representation": "feature_coverage.csv: which representation the row describes", "feature": "feature or summary label",
"n_dois_incl_no_doi_block": "unique DOI groups counting the 232-record no-DOI block as one group",
"coverage_fraction_of_41107_records": "fraction of all filtered records with a value", "coverage_fraction_of_covered_papers": "fraction of extracted papers with a value",
"n_papers_with_value": "papers with a value (flags are 0/1 for every extracted paper)", "flag_prevalence_among_2835_papers": "fraction of the 2,835 extracted papers with flag = 1",
"n_covered_records": "records of covered papers", "n_covered_dois": "covered publications",
"n": "(_records_per_doi_describe.csv) summary statistic of records per DOI group; first unnamed column = statistic name",
"Unnamed: 0": "(helper file) statistic name",
"source": "file(s) the value was read from, relative to ml_paper/", "notes": "caveats specific to the row; read before using the value",
}
cols = {}
for f in sorted(glob.glob(str(H / "tables/*.csv"))):
    for c in pd.read_csv(f).columns: cols.setdefault(c, []).append(Path(f).name)
missing = [c for c in cols if c not in D]
assert not missing, missing
lines = ["# Data dictionary for `tables/*.csv`", "",
 "Conventions: R² = coefficient of determination; MAE/RMSE in **PCE percentage points**; Jsc errors only for Roberts et al. (mA cm⁻²). 'sd' = sample standard deviation (ddof=1) over splits (split variability, **not** a confidence interval). "
 "Blank / `NA` = not available in the artifacts; nothing is estimated. 'records' = device entries; 'DOIs' = unique normalised publication DOIs (+ the 232-record no-DOI block where stated).",
 "**Tier codes.** `t1` = PDP features, `t2` = t1 + material properties, `t3` = t2 + the final V4 36 recipe flags (only in `v4_main_results.csv` and `v4_three_tier_common_split.csv`). The earlier 44-predictor literature representation (E) appears as `t2+E` in `original_tier_ablation*.csv`, `model_development.csv`, `evaluation_protocols*.csv` and `baseline_studies.csv`; the code used in the original run files (`t3`) is preserved in the column `run_file_tier`. Results of the two representations come from different protocols and must not be compared directly.", "",
 "| column | meaning | files |", "|---|---|---|"]
for c, fs in cols.items(): lines.append(f"| `{c}` | {D[c]} | {', '.join(fs)} |")
lines += ["", "## Files", "",
 "* `v4_three_tier_common_split.csv` — common-split t1/t2/t3 on the reserved v4final publications (matched default CatBoost): per-tier R² with CI, MAE, RMSE, sample sizes; paired t2−t1, t3−t2, t3−t1.",
 "* `v4_main_results.csv` — V4 reserved evaluation: 24 arm rows (12 arms × 2 scopes) + 24 paired-comparison rows (A, B, tuning effects, controls).",
 "* `original_tier_ablation.csv` (+ `_per_split.csv`) — five-split fixed-config ablation (t1, t2, t2+E = earlier representation) incl. record-level rows and paired differences.",
 "* `model_development.csv` — Tuned/Bag/XGB/Ensemble for every subset × tier (t2+E, earlier representation; no V4 entries).",
 "* `evaluation_protocols.csv` (+ `_temporal_windows.csv`, `_gap_ablation_v1.csv`) — DOI-grouped, record-level, lead-author, forward-in-time.",
 "* `feature_coverage.csv` — dataset counts, V4 partition counts, per-feature coverage of the earlier block E, V4 flag prevalence, quantity-level columns.",
 "* `control_experiments.csv` — V4 permuted/coverage controls, feature-group controls, material-identity control, E2–E6, lx2/lx3/qty experiments.",
 "* `baseline_studies.csv` — in-framework reference models and cited prior studies.",
 "* `_records_per_doi_describe.csv` — helper for the records-per-publication statistics."]
(H / "DATA_DICTIONARY.md").write_text("\n".join(lines) + "\n")

FM = """# Figure manifest

All files are **copies** of the figures currently produced for the manuscript (`psc_manuscript/figures/`); nothing was re-plotted or restyled for this handoff.
Generation scripts are copied into `figures/scripts/` for reference; to regenerate, run the originals in place (`python psc_manuscript/make_figures.py [tiers split diag dist shapgroups shap v4]`), because they locate `results/` relative to their own directory.
Shared style: `scripts/style.py` (Arial 8 pt, blue/green/gold tiers, dot/error-bar and hatched-bar design of the original 13 Oct figures).

| file | shows | representation / protocol | data source | script |
|---|---|---|---|---|
| fig_feature_tiers.png/.pdf | R² and RMSE of t1 / t2 (mean ± sd, 5 splits), pooled, p-i-n, n-i-p | t1 and t2 only; fixed CatBoost; 5 DOI-grouped holdouts (the earlier-block bars were removed from the manuscript figure) | `results/ablation_v1` | `scripts/fig_feature_tiers.py` |
| fig_split_effect.png/.pdf | record-level random vs DOI-grouped R², CatBoost and ridge, pooled | t1/t2 only (manuscript version); record-level CatBoost/ridge from converged/rerun files via `psc.results.load_ablation` | `results/ablation_v1`, `ablation_random_cap6000`, `random_convergence_cap20000` | `scripts/fig_split_effect.py` |
| fig_oof_diagnostics.png/.pdf | parity and residuals, 5 test sets stacked (42,142 predictions) | matched CatBoost, pooled, t2 | `results/ablation_v1/preds/doi_overall_t2_catboost_{1..5}.npz` | `scripts/fig_oof_diagnostics.py` |
| fig_pce_dist_trio.png/.pdf | experimental vs predicted PCE histograms, split 1: CatBoost, MLP, GPR | t1 (the tier for which all three models exist), pooled; reference-model scope only | `results/ablation_v1/preds`, `results/baselines_v1/preds` | `scripts/fig_pce_dist_trio.py` |
| fig_shap_groups.png/.pdf | share of mean abs SHAP by feature group | earlier representation E (t2+E) CatBoost, NOT t3, split 1, 1,500 test records | `results/shap/shap_groups.csv` | `scripts/fig_shap_groups.py` |
| shap.png/.pdf | top-12 features by mean abs SHAP, three models | earlier representation E (t2+E), NOT t3 | `results/shap/shap_ranking_*.csv` | `scripts/shap.py` |
| fig_three_tier.png/.pdf | R² per tier with 95% CIs and paired t2−t1, t3−t2 differences, covered vs all reserved records | FINAL V4 t1/t2/t3 on identical reserved publications, matched default CatBoost | `results/v4/final_t1/{metrics_3tier.csv,comparisons_3tier.json}` | `scripts/fig_three_tier.py` |
| fig_v4_recipe.png/.pdf | ΔR² (t3 − t2) with 95% publication-cluster CIs; same-default vs each-tuned; grey band = range of five permuted controls | FINAL V4 recipe flags; reserved v4final; covered vs all records | `results/v4/final/comparisons.json` | `scripts/fig_v4_recipe.py` |
| device_structure.png, ml_pipeline.png | device schematic; workflow diagram | not data figures | `psc_manuscript/tikz/`, `scripts/build_workflow_figure.py` | — |

Caveats: the files in `figures/` were refreshed to the current manuscript versions (final t1/t2/t3 labels; the earlier first-handoff copies are preserved in Git history). SHAP figures describe models with the earlier representation E, not t3. The MLP/GPR distribution figure is t1-only. `ml_pipeline.png` shows 77/179/215 predictors. Earlier bar-style versions of these figures (from `scripts/make_figures.py`) are in `psc_manuscript/figures_prev_backup/` and are not part of this package.
"""
(H / "figures/FIGURE_MANIFEST.md").write_text(FM)
README = """# Results handoff — PSC PCE prediction with LLM-derived recipe features

**Start with `RESULTS_REPORT.md`** (sections A–I). Machine-readable tables are in `tables/` (see `DATA_DICTIONARY.md`); figures and their manifest are in `figures/`; `VERIFICATION_LOG.md` lists the integrity checks that were run.

## Which results are which
| what | representation | where |
|---|---|---|
| **Final model (V4)** — t1 / t2 / t3 on identical reserved publications (`v4final`: 14,125 records / 2,619 DOIs; covered subset 5,460 / 1,005), matched default CatBoost; t3 = t2 + 36 recipe flags; plus separately tuned t2 vs t3 | **V4 recipe flags** | `RESULTS_REPORT.md` §C (C.0 three-tier); `tables/v4_three_tier_common_split.csv`, `tables/v4_main_results.csv`; `figures/fig_v4_recipe.png` |
| Earlier five-split fixed-configuration ablation (t1 / t2 / t2+E) | **earlier 44-predictor representation E** (code `t3` in old run files; written `t2+E` here, never t3) | §D; `tables/original_tier_ablation*.csv` |
| Model-development table (Tuned / Bag / XGB / Ensemble) | earlier representation E (t2+E) only; no V4 entries exist | §E; `tables/model_development.csv` |
| Record-level vs DOI-grouped, lead-author, forward-in-time, reference models, SHAP, prediction errors | earlier representation E (t2+E) for the literature tier; reference models at t1 | §G |
| Feature construction, coverage, extraction validation, lx2/lx3/quantity explorations | both (labelled) | §F; `tables/feature_coverage.csv`, `control_experiments.csv` |

## Authoritative sources (inside `ml_paper/`)
V4: `results/v4/final/{metrics.csv, comparisons.json, tuned_cfg.json}`, `results/v4/tune.json`, `docs/V4_REPORT.md`, `scripts/v4_final.py`. Original: `results/ablation_v1`, `model_dev_v1`, `baselines_v1`, `strict_validation_v1`, `feature_groups_v1`, `shap/`. Full index: `RESULTS_REPORT.md` §I.

## Key numbers needing interpretation (details in §C, §H, §I)
* All four V4 95% intervals include zero; gains are small and configuration-dependent (suggestive, not established).
* `v4final` publications were seen in earlier experiments with other feature sets; the flag block was chosen after development comparisons.
* The V4 and the five-split results use different partitions and protocols — absolute R² values are not comparable.
* Open items: coverage masks differ (36.2% vs 36.6%); retrieval-route provenance; Qwen checkpoint unrecorded; no V4 extraction validation; see §I.

## Regenerating this package (no training, no LLM)
`python build_handoff_tables.py && python build_report.py && python build_docs.py && python verify_handoff.py` (read-only on project artifacts).
"""
(H / "README.md").write_text(README)
print("docs written")
