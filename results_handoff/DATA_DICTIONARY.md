# Data dictionary for `tables/*.csv`

Conventions: R² = coefficient of determination; MAE/RMSE in **PCE percentage points**; Jsc errors only for Roberts et al. (mA cm⁻²). 'sd' = sample standard deviation (ddof=1) over splits (split variability, **not** a confidence interval). Blank / `NA` = not available in the artifacts; nothing is estimated. 'records' = device entries; 'DOIs' = unique normalised publication DOIs (+ the 232-record no-DOI block where stated).
**Tier codes.** `t3` has two meanings depending on the file: in `v4_main_results.csv` it is the FINAL V4 recipe-flag tier; in `original_tier_ablation*.csv`, `model_development.csv`, `evaluation_protocols*.csv`, `baseline_studies.csv` and feature-group rows it is the ORIGINAL 44-predictor literature block. The `feature_representation` column always says which.

| column | meaning | files |
|---|---|---|
| `Unnamed: 0` | (helper file) statistic name | _records_per_doi_describe.csv |
| `n` | (_records_per_doi_describe.csv) summary statistic of records per DOI group; first unnamed column = statistic name | _records_per_doi_describe.csv |
| `study` | study identifier (this study or cited prior work) | baseline_studies.csv |
| `target` | predicted quantity (PCE %, or Jsc mA cm-2 for Roberts et al.) | baseline_studies.csv |
| `estimator` | model/estimator name (catboost, ridge, mean, xgb_default, rf, mlp, gpr_sub, or a model-development variant) | baseline_studies.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `tier` | feature tier code. In files from the five-split ablation/model-dev/strict/baselines 't3' = ORIGINAL 44-predictor literature block; in v4_main_results.csv 't3' = final V4 recipe flags. Contrast labels such as 't3-t2' also appear here. | baseline_studies.csv, evaluation_protocols.csv, evaluation_protocols_gap_ablation_v1.csv, evaluation_protocols_temporal_windows.csv, model_development.csv, original_tier_ablation.csv, original_tier_ablation_per_split.csv, v4_main_results.csv |
| `feature_representation` | explicit name of the feature set behind the row (t1, t2, ORIGINAL block, V4 recipe flags (36), control description) | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv, v4_main_results.csv |
| `split_protocol` | split description for baseline/prior-work rows | baseline_studies.csv |
| `n_splits` | number of splits/windows aggregated | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `r2_mean` | mean R2 over splits | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `r2_sd` | sample sd (ddof=1) of R2 over splits (split variability, not a CI) | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `mae_mean` | mean MAE (pp) | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `mae_sd` | sd of MAE over splits | baseline_studies.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `rmse_mean` | mean RMSE (pp) | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `rmse_sd` | sd of RMSE over splits | baseline_studies.csv, evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `units` | units / comparability note | baseline_studies.csv |
| `source` | file(s) the value was read from, relative to ml_paper/ | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, evaluation_protocols_gap_ablation_v1.csv, feature_coverage.csv, model_development.csv, original_tier_ablation.csv, v4_main_results.csv |
| `notes` | caveats specific to the row; read before using the value | baseline_studies.csv, control_experiments.csv, evaluation_protocols.csv, evaluation_protocols_gap_ablation_v1.csv, feature_coverage.csv, original_tier_ablation.csv, v4_main_results.csv |
| `experiment` | experiment name for control rows | control_experiments.csv |
| `partition` | population/partition label for control rows (e.g. v4final covered, covered dev, qconf) | control_experiments.csv |
| `reference` | reference arm/tier of a paired control comparison | control_experiments.csv |
| `delta_r2` | R2(comparison) - R2(reference); positive = comparison better | control_experiments.csv, v4_main_results.csv |
| `ci_lo` | lower end of 95% interval of delta R2 | control_experiments.csv, v4_main_results.csv |
| `ci_hi` | upper end of 95% interval of delta R2 | control_experiments.csv, v4_main_results.csv |
| `seeds_positive` | number (or n/N) of seeds with positive delta | control_experiments.csv, v4_main_results.csv |
| `n_records` | number of records in the evaluated population (records = device entries) | control_experiments.csv, feature_coverage.csv |
| `n_dois` | number of unique publications (normalised DOIs) in the population | control_experiments.csv |
| `n_features` | feature count | control_experiments.csv, original_tier_ablation_per_split.csv |
| `ci_lo_dmse` | lower end of 95% interval of delta MSE (as stored in the source CSV; NOT delta R2) | control_experiments.csv |
| `ci_hi_dmse` | upper end of 95% interval of delta MSE | control_experiments.csv |
| `folds_positive` | folds (of 15 or 25) with positive delta | control_experiments.csv |
| `r2_reference` | R2 of the reference arm (t2 side) | control_experiments.csv, v4_main_results.csv |
| `r2_tier` | R2 of the tier in a control comparison | control_experiments.csv |
| `n_seeds` | number of seeds in the paired comparison | control_experiments.csv, v4_main_results.csv |
| `mae_tier` | MAE of the tier | control_experiments.csv |
| `rmse_tier` | RMSE of the tier | control_experiments.csv |
| `protocol` | evaluation protocol (DOI-grouped, record-level random, lead-author-grouped, forward-in-time) | evaluation_protocols.csv, original_tier_ablation.csv |
| `subset` | pooled | n-i-p | p-i-n | evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv, original_tier_ablation_per_split.csv |
| `mean_n_test_records` | mean test records per split | evaluation_protocols.csv, model_development.csv, original_tier_ablation.csv |
| `mean_n_test_dois` | mean test publications per split | evaluation_protocols.csv |
| `arch` | architecture code in source files (overall=pooled, pin=p-i-n, nip=n-i-p) | evaluation_protocols_gap_ablation_v1.csv, evaluation_protocols_temporal_windows.csv |
| `model` | estimator code in temporal-window / gap files | evaluation_protocols_gap_ablation_v1.csv |
| `r2_doi` | R2 DOI-grouped | evaluation_protocols_gap_ablation_v1.csv |
| `r2_random` | R2 record-level random | evaluation_protocols_gap_ablation_v1.csv |
| `gap_r2` | R2(record-level random) - R2(DOI-grouped) | evaluation_protocols_gap_ablation_v1.csv |
| `mae_doi` | MAE DOI-grouped | evaluation_protocols_gap_ablation_v1.csv |
| `mae_random` | MAE record-level | evaluation_protocols_gap_ablation_v1.csv |
| `split` | split index (1-5) or forward window index | evaluation_protocols_temporal_windows.csv, original_tier_ablation_per_split.csv |
| `train_date_max` | latest training-publication date (forward windows) | evaluation_protocols_temporal_windows.csv |
| `test_date_min` | earliest test-publication date | evaluation_protocols_temporal_windows.csv |
| `n_train` | training records | evaluation_protocols_temporal_windows.csv, original_tier_ablation_per_split.csv |
| `n_test` | test records | evaluation_protocols_temporal_windows.csv, original_tier_ablation_per_split.csv |
| `r2` | coefficient of determination of one scored model | evaluation_protocols_temporal_windows.csv, original_tier_ablation_per_split.csv, v4_main_results.csv |
| `mae` | mean absolute error (PCE percentage points) | evaluation_protocols_temporal_windows.csv, original_tier_ablation_per_split.csv, v4_main_results.csv |
| `rmse` | root mean squared error (PCE percentage points) | evaluation_protocols_temporal_windows.csv, original_tier_ablation_per_split.csv, v4_main_results.csv |
| `representation` | feature_coverage.csv: which representation the row describes | feature_coverage.csv |
| `feature` | feature or summary label | feature_coverage.csv |
| `n_dois_incl_no_doi_block` | unique DOI groups counting the 232-record no-DOI block as one group | feature_coverage.csv |
| `coverage_fraction_of_41107_records` | fraction of all filtered records with a value | feature_coverage.csv |
| `n_covered_records` | records of covered papers | feature_coverage.csv |
| `n_covered_dois` | covered publications | feature_coverage.csv |
| `coverage_fraction_of_covered_papers` | fraction of extracted papers with a value | feature_coverage.csv |
| `n_papers_with_value` | papers with a value (flags are 0/1 for every extracted paper) | feature_coverage.csv |
| `flag_prevalence_among_2835_papers` | fraction of the 2,835 extracted papers with flag = 1 | feature_coverage.csv |
| `mean_n_features` | mean feature count after preprocessing (includes _na indicators) | model_development.csv, original_tier_ablation.csv |
| `selected_configs_by_split` | JSON list of the CatBoost {depth,l2_leaf_reg} chosen on validation in splits 1-5 | model_development.csv |
| `data_split` | description of the partition and training/validation scheme for the row | model_development.csv, v4_main_results.csv |
| `v4_entry` | explicit statement that no V4 value exists for model-development rows | model_development.csv |
| `row_type` | 'arm' (one scored model) | 'comparison' | 'paired comparison' | 'tier metrics' | 'paired difference' (what the row represents) | original_tier_ablation.csv, v4_main_results.csv |
| `delta_r2_mean` | mean over splits of per-split paired delta R2 | original_tier_ablation.csv |
| `delta_r2_sd` | sd over splits of per-split delta R2 | original_tier_ablation.csv |
| `splits_r2_better` | splits (of 5) in which R2 improved | original_tier_ablation.csv |
| `delta_mae_mean` | mean per-split delta MAE (negative = better) | original_tier_ablation.csv |
| `delta_mae_sd` | sd of delta MAE | original_tier_ablation.csv |
| `delta_rmse_mean` | mean per-split delta RMSE | original_tier_ablation.csv |
| `delta_rmse_sd` | sd of delta RMSE | original_tier_ablation.csv |
| `splits_ci_excludes_zero` | splits (of 5) whose per-split bootstrap CI excluded zero | original_tier_ablation.csv |
| `n_val` | validation records (early stopping) | original_tier_ablation_per_split.csv |
| `n_doi_train` | training publications | original_tier_ablation_per_split.csv |
| `n_doi_val` | validation publications | original_tier_ablation_per_split.csv |
| `n_doi_test` | test publications | original_tier_ablation_per_split.csv |
| `doi_overlap_test_train` | number of DOIs shared between test and train (0 expected) | original_tier_ablation_per_split.csv |
| `doi_overlap_val_train` | DOIs shared between validation and train | original_tier_ablation_per_split.csv |
| `doi_overlap_test_val` | DOIs shared between test and validation | original_tier_ablation_per_split.csv |
| `scope` | V4 evaluation subset: 'covered (papers with V4 flags)' or 'all records' of the reserved v4final partition | v4_main_results.csv |
| `arm` | V4 model arm (e.g. t2_default, t3_tuned, ctl_perm0_default) or 'reference -> comparison' arm pair | v4_main_results.csv |
| `estimator_config` | hyperparameter configuration of the arm (matched default / t2-tuned / t3-tuned / cross-applied) | v4_main_results.csv |
| `n_test_records` | number of test records | v4_main_results.csv |
| `n_test_dois` | number of unique test publications | v4_main_results.csv |
| `n_seed_models` | number of seed models averaged to form the predictions | v4_main_results.csv |
| `comparison` | which paired comparison: A (same settings), B (each tuned), tuning effect, or control | v4_main_results.csv |
| `r2_comparison` | R2 of the comparison arm (t3 side) | v4_main_results.csv |
| `ci_method` | how the interval was computed | v4_main_results.csv |
| `delta_mae_derived` | MAE(comparison) - MAE(reference) computed from metrics.csv (derived, no CI available) | v4_main_results.csv |
| `delta_rmse_derived` | RMSE(comparison) - RMSE(reference), derived | v4_main_results.csv |
| `mae_reference` | MAE reference arm | v4_main_results.csv |
| `mae_comparison` | MAE comparison arm | v4_main_results.csv |
| `rmse_reference` | RMSE reference arm | v4_main_results.csv |
| `rmse_comparison` | RMSE comparison arm | v4_main_results.csv |
| `per_seed_delta_r2` | semicolon-separated per-seed paired delta R2 | v4_main_results.csv |

## Files

* `v4_main_results.csv` — V4 reserved evaluation: 24 arm rows (12 arms × 2 scopes) + 24 paired-comparison rows (A, B, tuning effects, controls).
* `original_tier_ablation.csv` (+ `_per_split.csv`) — five-split fixed-config ablation (t1, t2, ORIGINAL block) incl. record-level rows and paired differences.
* `model_development.csv` — Tuned/Bag/XGB/Ensemble for every subset × tier (ORIGINAL block; no V4 entries).
* `evaluation_protocols.csv` (+ `_temporal_windows.csv`, `_gap_ablation_v1.csv`) — DOI-grouped, record-level, lead-author, forward-in-time.
* `feature_coverage.csv` — dataset counts, V4 partition counts, per-feature coverage of the ORIGINAL block, V4 flag prevalence, quantity-level columns.
* `control_experiments.csv` — V4 permuted/coverage controls, feature-group controls, material-identity control, E2–E6, lx2/lx3/qty experiments.
* `baseline_studies.csv` — in-framework reference models and cited prior studies.
* `_records_per_doi_describe.csv` — helper for the records-per-publication statistics.
