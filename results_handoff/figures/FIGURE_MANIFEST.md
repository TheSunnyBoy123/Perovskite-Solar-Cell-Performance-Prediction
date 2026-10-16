# Figure manifest

All files are **copies** of the figures currently produced for the manuscript (`psc_manuscript/figures/`); nothing was re-plotted or restyled for this handoff.
Generation scripts are copied into `figures/scripts/` for reference; to regenerate, run the originals in place (`python psc_manuscript/make_figures.py [tiers split diag dist shapgroups shap v4]`), because they locate `results/` relative to their own directory.
Shared style: `scripts/style.py` (Arial 8 pt, blue/green/gold tiers, dot/error-bar and hatched-bar design of the original 13 Oct figures).

| file | shows | representation / protocol | data source | script |
|---|---|---|---|---|
| fig_feature_tiers.png/.pdf | R² and RMSE of t1 / t2 / t3 (mean ± sd, 5 splits), pooled, p-i-n, n-i-p | ORIGINAL block (legend says "original block"); fixed CatBoost; 5 DOI-grouped holdouts | `results/ablation_v1` | `scripts/fig_feature_tiers.py` |
| fig_split_effect.png/.pdf | record-level random vs DOI-grouped R², CatBoost and ridge, pooled | ORIGINAL block; record-level CatBoost/ridge from converged/rerun files via `psc.results.load_ablation` | `results/ablation_v1`, `ablation_random_cap6000`, `random_convergence_cap20000` | `scripts/fig_split_effect.py` |
| fig_oof_diagnostics.png/.pdf | parity and residuals, 5 test sets stacked (42,142 predictions) | ORIGINAL block, matched CatBoost, pooled | `results/ablation_v1/preds/doi_overall_t3_catboost_{1..5}.npz` | `scripts/fig_oof_diagnostics.py` |
| fig_pce_dist_trio.png/.pdf | experimental vs predicted PCE histograms, split 1: CatBoost, MLP, GPR | ORIGINAL block, pooled | `results/ablation_v1/preds`, `results/baselines_v1/preds` | `scripts/fig_pce_dist_trio.py` |
| fig_shap_groups.png/.pdf | share of mean abs SHAP by feature group | ORIGINAL block CatBoost, split 1, 1,500 test records | `results/shap/shap_groups.csv` | `scripts/fig_shap_groups.py` |
| shap.png/.pdf | top-12 features by mean abs SHAP, three models | ORIGINAL block | `results/shap/shap_ranking_*.csv` | `scripts/shap.py` |
| fig_v4_recipe.png/.pdf | ΔR² (V4 t3 − t2) with 95% publication-cluster CIs; same-default vs each-tuned; grey band = range of five permuted controls | FINAL V4 recipe flags; reserved v4final; covered vs all records | `results/v4/final/comparisons.json` | `scripts/fig_v4_recipe.py` |
| device_structure.png, ml_pipeline.png | device schematic; workflow diagram | not data figures | `psc_manuscript/tikz/`, `scripts/build_workflow_figure.py` | — |

Caveats: figures use tier code `t3` in axis/legend text for the ORIGINAL block except where stated; `ml_pipeline.png` predates V4 (its Panel 6 describes the matched CatBoost and the 77/179/223 feature counts of the ORIGINAL representation). Earlier bar-style versions of these figures (from `scripts/make_figures.py`) are in `psc_manuscript/figures_prev_backup/` and are not part of this package.
