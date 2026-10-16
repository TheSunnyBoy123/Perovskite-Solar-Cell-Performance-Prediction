# Verification log

Run by `verify_handoff.py` (read-only; no training, no LLM).

- [PASS] raw 43,398 x 156, 7,499 DOI strings
- [PASS] 1,002 missing PCE; 1,289 PCE<1
- [PASS] 41,107 filtered records = 43,398-1,002-1,289
- [PASS] 7,327 groups incl. 232-record no-DOI block (7,326 publications)
- [PASS] arch 28,340 / 12,684 / 83 other
- [PASS] PCE mean 12.42, median 13.0, range 1.0-36.2; 17 records >27 (6 DOIs)
- [PASS] max records per real publication = 127; median per group 4
- [PASS] v4dev 26,982/4,708; v4final 14,125/2,619; covered 5,460/1,005
- [PASS] coverage 15,027 records / 2,731 DOIs (36.6%)
- [PASS] original-block coverage 36.2% / 14,896 / 2,699
- [PASS] ICC 0.692
- [PASS] 36 V4 flags
- [PASS] V4_REPORT all / A: same default settings — +0.0025 [-0.0011,+0.0062]
- [PASS] V4_REPORT covered / A: same default settings — +0.0071 [-0.0007,+0.0157]
- [PASS] V4_REPORT covered / B: each tier with its own tuned config — +0.0094 [-0.0016,+0.0219]
- [PASS] V4_REPORT all / B: each tier with its own tuned config — +0.0015 [-0.0039,+0.0067]
- [PASS] V4_REPORT covered / A: both under t3-tuned config — +0.0066 [-0.0021,+0.0162]
- [PASS] V4_REPORT all / A: both under t3-tuned config — +0.0026 [-0.0015,+0.0067]
- [PASS] V4_REPORT covered / A: both under t2-tuned config — +0.0024 [-0.0077,+0.0131]
- [PASS] V4_REPORT all / A: both under t2-tuned config — -0.0016 [-0.0064,+0.0030]
- [PASS] absolute covered R2 t2 0.361 / t3 0.371
- [PASS] delta R2 equals difference of arm R2 (all comparison rows, tol 1e-9)
- [PASS] permuted-control ranges covered -0.0049..+0.0015, all -0.0028..+0.0007
- [PASS] coverage-flag control +0.0015 / +0.0011
- [PASS] metrics.csv n/DOIs: covered 5460/1005, full 14125/2619
- [PASS] dev +0.0070, 12/15 folds (pen_05.csv)
- [PASS] t3 tuning MSE lower than t2 in 6/24 configs
- [PASS] original pooled R2 0.368/0.408/0.407 (+-0.024/0.032/0.031)
- [PASS] model-dev pooled ensemble 0.377/0.416/0.418
- [PASS] model-dev table has 36 rows (3 subsets x 3 tiers x 4 models)
- [PASS] model_dev.tex pooled t1 row matches CSV
- [PASS] record-level CatBoost 0.587/0.618/0.627 (converged)
- [PASS] lead-author t3 0.395; forward t3 0.335
- [PASS] every 't3' row outside v4_main_results is labelled ORIGINAL block — set()
- [PASS] every V4 't3' arm labelled V4 recipe flags
- [PASS] no V4 entries invented in model_development.csv
- [PASS] opens: _records_per_doi_describe.csv — 8 rows x 2 cols
- [PASS] opens: baseline_studies.csv — 18 rows x 16 cols
- [PASS] opens: control_experiments.csv — 106 rows x 26 cols
- [PASS] opens: evaluation_protocols.csv — 108 rows x 16 cols
- [PASS] opens: evaluation_protocols_gap_ablation_v1.csv — 27 rows x 10 cols
- [PASS] opens: evaluation_protocols_temporal_windows.csv — 27 rows x 10 cols
- [PASS] opens: feature_coverage.csv — 80 rows x 12 cols
- [PASS] opens: model_development.csv — 36 rows x 17 cols
- [PASS] opens: original_tier_ablation.csv — 108 rows x 25 cols
- [PASS] opens: original_tier_ablation_per_split.csv — 45 rows x 16 cols
- [PASS] opens: v4_main_results.csv — 48 rows x 31 cols
- [PASS] every CSV column documented in DATA_DICTIONARY.md
- [PASS] all file paths cited in report/README/manifest exist (glob/brace patterns skipped) — 101 paths checked

**49 passed, 0 failed.**
