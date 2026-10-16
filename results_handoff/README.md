# Results handoff — PSC PCE prediction with LLM-derived recipe features

**Start with `RESULTS_REPORT.md`** (sections A–I). Machine-readable tables are in `tables/` (see `DATA_DICTIONARY.md`); figures and their manifest are in `figures/`; `VERIFICATION_LOG.md` lists the integrity checks that were run.

## Which results are which
| what | representation | where |
|---|---|---|
| **Final model (V4)** — t3 = t2 + 36 recipe flags; reserved-publication evaluation (`v4final`: 14,125 records / 2,619 DOIs; covered subset 5,460 / 1,005), same-settings and separately tuned | **V4 recipe flags** | `RESULTS_REPORT.md` §C; `tables/v4_main_results.csv`; `figures/fig_v4_recipe.png` |
| Earlier five-split fixed-configuration ablation (t1 / t2 / original literature block) | **ORIGINAL 44-predictor block** (called `t3` in old files) | §D; `tables/original_tier_ablation*.csv` |
| Model-development table (Tuned / Bag / XGB / Ensemble) | ORIGINAL block only; no V4 entries exist | §E; `tables/model_development.csv` |
| Record-level vs DOI-grouped, lead-author, forward-in-time, reference models, SHAP, prediction errors | ORIGINAL block | §G |
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
