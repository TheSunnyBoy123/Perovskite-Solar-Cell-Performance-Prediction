# Adversarial review: ml_paper (DOI-grouped PSC PCE prediction)

Scope: src/psc/*.py, scripts/{run_experiment,run_model_dev,shap_analysis,aggregate,paired_bootstrap}.py, configs, tests, results/{ablation_v1,baselines_v1,feature_groups_v1,model_dev_v1,shap}, psc_manuscript/ml.tex and tables. `python -m pytest tests -q` gives 14 passed. All recomputation used runs.jsonl, preds/*.npz and the raw CSV only. No files were modified apart from this report. model_dev_v1 is incomplete (21 of 30 expected rows: pin t2 has 1 split, pin t3 and nip are missing), so it is ignored below.

## 0. Verdict

The core experimental pipeline is sound. I found no test-to-train leakage, no DOI straddling and no test-based early stopping in the ablation, baseline or feature-group runs. Tier, protocol and model comparisons are properly paired. The serious problems are in the manuscript, not the code:

1. The Abstract and Conclusion are stale and contradict the Results.
2. Some literature claims are unverified.
3. A few interpretive statements go beyond the evidence.
4. The test suite is weaker than it looks (several tests are vacuous or circular).

## 1. Findings ranked by severity

### HIGH

**H1. Abstract (ml.tex:46-47) and Conclusion (ml.tex:~486) contradict the Results, and the abstract is flagged only by a TODO.**

| Stale claim | Where | Actual (recomputed) |
|---|---|---|
| Final pooled R2 0.41 +/- 0.01, MAE 2.91, RMSE 3.76 | Abstract | Matched t3: 0.4066 +/- 0.031, MAE 2.958, RMSE 3.787 (see M5 for the 0.413 / 2.91 provenance) |
| p-i-n 0.37, n-i-p 0.45 | Abstract | 0.294 and 0.405 |
| Conclusion p-i-n 0.365, n-i-p 0.452 | Conclusion | same |
| LLM features give "a further small but consistent improvement in every subset and every metric (+0.005 to +0.007)" | Abstract, Conclusion | t2->t3 is -0.0013 pooled, +0.0025 p-i-n, +0.0014 n-i-p. Pooled RMSE gets worse (+0.004). Bootstrap CI excludes 0 in 0/15 pairs. The Results say "no measurable change". |
| "Delta R2 = +0.038" is the material gain | Abstract, Conclusion | +0.038 is t3-t1. The t1->t2 gain is +0.0395, so the label is wrong. |
| Record-level overestimates R2 by 0.17-0.24 | Abstract, Conclusion | Pooled CatBoost 0.166-0.174; ridge 0.055-0.058; n-i-p 0.20-0.21; p-i-n 0.275-0.286. The range matches no stated subset. |
| LLM gain "0.074 under random, 0.004 under DOI" | Conclusion | 0.0069 and -0.0013 |
| "largest inflation for the literature-augmented model" | Abstract | Gaps are 0.170 / 0.166 / 0.174, which is not distinguishable |
| 2,726 full-text articles | Abstract, Conclusion | 2,726 raw DOI strings, but 2,699 publications after normalisation. The Introduction uses 2,699. |

The Introduction (ml.tex:93) already states "no measurable gain". The paper therefore tells opposite stories in different sections. The abstract also still says records are "typically" evaluated with random splits, which lit_audit.md item 1 asked to soften to "often".

**H2. Placeholder `%%DEVPARA%%` is still in ml.tex (~line 345).** The model-development paragraph is missing. tables/model_dev.tex has only 2 pooled rows (t1, t2) and is not wired in. The "final model" in the Abstract and Conclusion is undefined: the matched CatBoost, the dev ensemble (0.418 for t3), or something else.

**H3. 31 of 45 random-protocol CatBoost fits hit the 1,500-iteration cap (`best_iteration` >= 1499).**
- Median best iteration is 1499 under the record-level protocol. It is 337, 458 and 486 for t1, t2 and t3 under DOI-grouped splitting, with 0 capped.
- Random-split models are therefore not converged. The record-level R2 and the "gap" are lower bounds.
- "Identical training budgets" (ml.tex:~370) is true literally but hides this asymmetry. The paper never reports it.
- The random validation set also contains same-paper records, so early stopping on it is itself leaky. This is part of the design but should be stated.
- Fix: rerun the random arm with a higher cap (for example 5,000), or at least report the capped fraction.

**H4. Unverified literature claims are stated as fact.** audit/lit_audit.md says the Roberts journal paper and the Fukasawa full text could not be read. ml.tex still asserts:
- ml.tex:84: 42,459 -> 36,937 records, 248 variables, R2 0.7177, 294 new records, R2 0.45, MAE 3.20, 6,446 degenerate points / 20,695 vectors (also repeated at :88 and in the comparison table).
- ml.tex:88: "the two that describe their partitioning use record-level splits [Roberts, Fukasawa]". For Roberts only the thesis (80/10/10) is verified, not the journal paper. For Fukasawa only "split 8:2" is seen, and "random" is not stated.
- ml.tex:~443-445: "XGBoost library defaults (the setting of Roberts-style pipelines)" and "random forest (the best model of Fukasawa et al.)", plus 300 trees / min_samples_leaf=2 labelled "Fukasawa-style" (models.py:66-73). None of this is verified.

These must be checked against the full texts or hedged ("as reported in the abstract/snippets"). Because the novelty argument rests on "prior work used record splits", a reviewer who opens those papers is the main rejection risk.

### MEDIUM

**M1. Hyper-parameters are called "pre-declared" but come from legacy scripts that early-stopped on the test fold.**
- legacy/pipeline_original/{nna_train,train_tiers,oof_train}.py has `PARAMS = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, ...)`, identical to `CATBOOST_MATCHED` (models.py:26-28).
- The README itself says the legacy scripts used test-fold early stopping. If depth, learning rate or l2 were ever chosen from legacy test scores, a mild optimistic selection effect enters the absolute R2. It does not affect the paired tier comparison.
- Disclose where the values came from. model_dev (validation-selected grid) partially addresses this, but only for the pooled t1/t2 rows so far.

**M2. The "p-i-n is not intrinsically harder" claim (ml.tex:~314) is contradicted by the data.**
- The mean-predictor MAE is 3.838 for p-i-n and 4.106 for n-i-p (protocol_gap.csv).
- t3 MAE is 3.051 versus 2.964, so the relative MAE reduction is 20.5% (p-i-n) versus 27.8% (n-i-p).
- p-i-n is genuinely harder relative to baseline. Only "R2 is variance-dependent" is defensible. Ridge on p-i-n is worse still (0.196, below its own t1 and t2).

**M3. The ridge explanation of the gap is under-supported (ml.tex:~391).**
- "Ridge shows a much smaller gap (0.055-0.058), consistent with less capacity to exploit publication-specific structure" holds for the pooled subset only. For p-i-n the ridge gap is 0.114-0.191 and for n-i-p 0.060-0.071 (protocol_gap.csv).
- The explanation is also confounded by lower absolute performance and by the unconverged random arm (H3).
- A capacity-controlled test, for example CatBoost with depth 2-3 or a kNN, is needed before using the word "supported".
- The gap pairing by split index (aggregate.py:~38) pairs different test sets. This is harmless for means but the "per-split gap" is not a paired quantity.

**M4. Cross-publication duplicates straddle the DOI split.**
- 162 duplicate-vector groups span more than one DOI (1,277 records have an identical t3 vector in another publication). They can sit in train and test of a DOI-disjoint split.
- This is minor in size, but the text (:~390) says "near-duplicate devices are almost entirely within-publication" without giving the 1,277 figure.
- One DOI key is a URL, `https://www.nature.com/articles/s41586-021-04372-8/tables/1` (1 record). `normalize_doi` keeps non-DOI URLs verbatim, so it is its own group and cannot collide with a DOI group for the same paper (data.py:57-67). Tiny, but it should strip URL forms generally.
- Ref_ID is unique per record (41,107 unique), so there is no secondary paper key to cross-check DOI grouping.

**M5. Provenance and metadata problems.**
- meta.json for all three runs records `git_dirty: true`. `git_head` (9b1d2ca) is the head of the parent UGP repo, where ml_paper is untracked (`?? ./`). The recorded commit therefore does not identify the code that ran, and the config-hash resume (run_experiment.py:45-48) skips cells without checking the config hash. A changed config on an existing results dir silently mixes settings.
- The Abstract number 0.413 / 2.91 (also in the Conclusion) is not found in any result file I checked. It is probably from the pre-audit run.
- The manuscript says the export had 410 columns. The shipped table has 156. This is plausible but not reproducible from the repo.
- ml.tex:~138 gives max records per publication as 127 (NO_DOI excluded) but results/feature_audit.json says 232 (NO_DOI included). The two are consistent only if the reader knows NO_DOI is a block.

**M6. Statements about significance and causality.**
- ml.tex:~397-400 reports "CI excludes zero in 5/5". The bootstrap resamples test DOIs for a fixed trained model, so it ignores training-set and seed variance (single CatBoost seed 42 throughout). The paper partly discloses this in the Limitations. It should not be read as support for generalisable significance. t1->t2 is nevertheless consistent in sign across all 15 split-subset pairs.
- The sentence "A feature with nonzero attribution but no held-out benefit is expected when the feature acts as a publication identifier" (:~442) is speculative and untested. The t3-minus-flags control shows removing the two flags changes nothing, which equally fits "feature is noise".
- "Material gain is best interpreted as the value of encoding material choice" is reasonable. It is an interpretation rather than a tested conclusion, since there is no ablation with a one-hot material ID.
- SHAP rankings come from a single split and a single seed. The text discusses 2nd-6th ranks as if stable. There is no rank-stability analysis.
- The "n-i-p quenching mean |SHAP| 1.002" statements are fine as attributions, but "architecture-specific associations" is thin given one model per architecture.

**M7. The baseline comparison is not matched.**
- "Matched CatBoost is the best by 0.03 to 0.06" compares early-stopped CatBoost with default XGBoost and RF that use no validation data. The authors note that no model ranking is claimed, which is the correct hedge.
- The Table 7 header implies "Roberts- and Fukasawa-style" models. The manuscript correctly says these are not re-implementations, but H4 applies to the labels.

### LOW

**L1. Wording mismatch at ml.tex:~346.**
- "272 exceed 20% PCE": 272 records have y >= 20. Strictly greater than 20 gives 259. The "1,328 records" is also >= 20. Use "at or above 20%".
- The "28% reduction in MAE" is 27.5%, which is fine.

**L2. `base_frame` drops non-CAT text columns via `X[c].dtype == "object"` (features.py:23).** pandas is 2.2.3, so this works. With pandas >= 3 (string dtype) the text columns would be kept silently. Use `pd.api.types.is_numeric_dtype`.

**L3. Transform can leave NaN.** `TrainFitPreprocessor.transform` fills only columns that had NaN in train (features.py:42, 50). A column with NaN only in val or test stays NaN. CatBoost tolerates this, but ridge, MLP and GPR would fail or misbehave. This did not occur in the runs, but nothing tests it.

**L4. FORBIDDEN contains 7 names absent from the CSV** (`doi_loo_mean`, `doi_loo_std`, `doi_count`, `doi_sigma`, `Ref_journal`, `Ref_publication_date`, `publication_year`). Harmless legacy defence, since `errors="ignore"` hides typos.

**L5. Unverifiable corpus counts.** The 3,299 PDFs / 3,141 Markdown / 2,835 JSON counts cannot be reproduced from the repo (the pipeline inputs are not shipped). The 2,726 -> 2,699 mapping and 36.2% coverage do reproduce.

**L6. Candidate outcome-adjacent features checked.** I found none that leak.
- The `*_measured` flags (Stability_measured, EQE_measured, Stabilised_performance_measured, Cell_area_measured_cm2, JV_hysteresis_index) are in FORBIDDEN.
- Remaining `Perovskite_band_gap_measured` and `Perovskite_pl_max_nm` are device characterisations rather than JV outputs. A cautious reader might still ask for the ablation (their missingness indicator is the 3rd-ranked pooled SHAP feature).
- No column is derived from final_pce (the t3 audit finds none), and no Voc/Jsc/FF columns exist in the 156-column table.
- Missingness indicators (`Perovskite_pl_max_nm_na`, `Backcontact_thickness_nm_na`) are strong candidate paper-/submitter-provenance proxies. DOI grouping handles them, but their heavy SHAP weight is worth a sentence.

## 2. Protocol and leakage checks (Task 1)

| Check | Result |
|---|---|
| DOI overlap train/val/test (ablation_v1, baselines_v1, feature_groups_v1) | 0 in every record of runs.jsonl. Re-derived independently from the preds npz indices (index-disjoint, and DOI-disjoint on the real data via `load_dataset`). |
| Random protocol DOI overlap | Large, as intended: test/train overlap min 1,233, mean 2,764 DOIs. |
| Test, train and val indices identical across tiers and models within a (protocol, arch, split) | TRUE for all 6 (protocol, arch) combos and 5 splits. Paired comparisons are valid. |
| Baselines and feature-groups splits equal to ablation splits | 0 mismatches out of 40 and 45. |
| Splits computed once per (arch, protocol) and reused across tiers | Yes (run_experiment.py:66-71). |
| Preprocessing fit on train only | Yes (features.py:38-44, run_experiment.py:72). Winsorizer, StandardScaler, OHE and category maps are fit on Xtr only (models.py:40, 59, 77-78). |
| Early stopping and model selection use val only | CatBoost: `eval_set=(Xva,yva)` (models.py:55). Ridge alpha by val RMSE (models.py:43-47). MLP stops on val. `run_model_dev` selects the grid by val RMSE (:53-57) and bags and ES on val (:62). XGB ES on val. |
| Test touched before final scoring | No. `fit_predict` never receives `yte`. |
| NO_DOI group | Single block, never split. 232 records, pooled only. It lands wholly in train, val or test per split, which adds variance but not leakage. |
| `normalize_doi` | Handles BOM, case, whitespace, doi.org / dx.doi.org / doi: prefixes. 45 variants merged (reproduced: 7,371 raw -> 7,326, plus the NO_DOI block = 7,327). It does not strip a trailing `.` / `/` (none present now), and it keeps URLs (1 present). |
| `make_splits` seeds | Outer seed 42, inner seed 42+k, deterministic. Test sets overlap across k by design. |
| SHAP script | Trains on the train split, early-stops on val, explains 1,500 test records. No test-label use apart from `m.score` printing. OK. |
| run_model_dev | Same discipline. The test partition is scored once. Selection is by val. |
| Hyper-parameter provenance | See M1. |
| Convergence asymmetry | See H3. |

## 3. Numeric verification of ml.tex (Task 2)

All recomputed from runs.jsonl, preds, feature_audit.json, shap CSVs and the raw CSV.

| # | Claim (ml.tex) | Recomputed | Verdict |
|---|---|---|---|
| 1 | Pooled t3 R2 0.407 +/- 0.031, MAE 2.96 +/- 0.08, RMSE 3.79 +/- 0.08 | 0.4066 / 0.0306; 2.958 / 0.080; 3.787 / 0.083 | MATCH |
| 2 | t1 / t2 / t3 pooled R2 0.368 / 0.408 / 0.407 | 0.3683 / 0.4078 / 0.4066 | MATCH |
| 3 | n-i-p 0.405 +/- 0.010; p-i-n 0.294 +/- 0.074 | 0.4052 / 0.0104; 0.2936 / 0.0736 | MATCH |
| 4 | MAE 2.96 / 3.05, RMSE 3.82 / 3.95 (n-i-p / p-i-n) | 2.964 / 3.051; 3.821 / 3.947 | MATCH |
| 5 | Mean predictor R2 ~ 0, MAE 4.08; model removes ~28% of MAE | -0.0011; 4.0789; 27.5% | MATCH |
| 6 | Delta t1->t2: +0.040 / +0.047 / +0.029; dRMSE -0.125 / -0.130 / -0.093 | +0.0395 / +0.0466 / +0.0293; -0.125 / -0.130 / -0.093 | MATCH |
| 7 | Better in 5/5 each; CI != 0 in 5, 2, 3 of 5 | 5/5/5; 5, 2, 3 | MATCH |
| 8 | Delta t2->t3: -0.001 / +0.002 / +0.001; better in 1, 3, 3; CI != 0 in 0 | -0.0013 / +0.0025 / +0.0014; 1, 3, 3; 0 | MATCH |
| 9 | Random vs DOI pooled: 0.538 / 0.574 / 0.581 vs 0.368 / 0.408 / 0.407; gaps 0.170 / 0.166 / 0.174 | 0.5379 / 0.5736 / 0.5805; 0.170 / 0.166 / 0.174 | MATCH |
| 10 | MAE 2.39-2.53 (random) to 2.95-3.08 (DOI) | 2.390-2.532; 2.950-3.081 | MATCH |
| 11 | Ridge gap 0.055-0.058; ridge t3 0.350 +/- 0.033 | 0.0547-0.0583; 0.350 / 0.033 | MATCH (but see M3) |
| 12 | Baseline R2 range 0.35-0.38; CatBoost ahead by 0.03-0.06 | XGB 0.377, MLP 0.376, RF 0.352, GPR 0.352, ridge 0.350 | MATCH |
| 13 | Groups: t1+LLM 0.372 +/- 0.026; no-flags 0.407; noise 0.404 (vs 0.408, Delta -0.004) | 0.372 / 0.026; 0.407; 0.404 | MATCH |
| 14 | 42,142 predictions pooled t3; 6 negative; min -1.3 | 42,142; 6; -1.345 | MATCH |
| 15 | Split 1: 8,224 test records; 272 "exceed" 20% | 8,224; 272 are >= 20 (259 are > 20) | MATCH on >=, wording MISMATCH (L1) |
| 16 | Split-1 max prediction 21.1; spread ratio 0.62 | 21.085; 0.619 | MATCH |
| 17 | Under-prediction for y >= 20: 5.3 (split 1), 4.7 (all 1,328) | 5.25; 4.74; n = 1,328 | MATCH |
| 18 | Excluding PCE > 27 changes R2 by +0.002 (0.407 -> 0.409); 17 records, 6 publications | +0.0026 (0.4066 -> 0.4092); 17; 6 | MATCH |
| 19 | 41,107 records, 28,340 n-i-p, 12,684 p-i-n, 83 other, 232 no DOI | all equal (83 = 52 + 20 + 7 + 3 + 1) | MATCH |
| 20 | Removed 1,002 missing PCE and 1,289 < 1.0 | 1,002; 1,289 | MATCH |
| 21 | 7,371 raw DOI strings -> 7,326 publications; 45 variants | 7,371; 7,327 incl. NO_DOI; 45 | MATCH |
| 22 | PCE mean 12.4 / median 13.0 / max 36.2; median 4 and max 127 records per pub | 12.42 / 13.0 / 36.2; 4 / 127 | MATCH |
| 23 | ICC 0.69 ("69% of variance between publications") | 0.692 (one-way ANOVA estimator) | MATCH |
| 24 | 31,591 of 41,107 (77%) share the t3 vector; 7,733 groups; 162 span more than one DOI; median SD 1.2 | 31,591 (76.9%); 7,733; 162; 1.208 | MATCH |
| 25 | LLM coverage 14,896 / 41,107 = 36.2%; 2,726 raw -> 2,699 publications | 14,896; 36.24%; 2,726 -> 2,699 | MATCH |
| 26 | Split sizes: ~27,700 / 4,981 pubs train; ~5,000 / 880 val; ~8,430 / 1,466 test | 27,680 / 4,981; 4,999 / 880; 8,428 / 1,466 | MATCH |
| 27 | Feature counts 77 / 179 / 223; n-i-p 76 / 178 / 222; p-i-n 77 / 179 / 224 | 77 / 179 / 223.0; 76.2 / 178.2 / 222.2; 77 / 179 / 224.2 | MATCH |
| 28 | SHAP top feature quenching: 0.727 / 0.525 / 1.002; 2nd pooled total ion count 0.265 | same | MATCH |
| 29 | SHAP p-i-n: HTL band gap 0.506, ETL dep. method 0.410 | same | MATCH |
| 30 | Literature share 4.0 / 3.9 / 3.6%; json_numeric_values_found ranks 35 / 40 / 28 | 4.03 / 3.85 / 3.58%; 35 / 40 / 28 | MATCH |
| 31 | Group share ranges: material 33-38, process 36-42, composition 13-15, device 6-8 | material 33.2-38.1; process 35.6-42.3; composition 13.5-14.7; device 5.7-8.0 | MATCH (device 5.7 rounds to 6) |
| 32 | SHAP total 6.72 / 5.46 / 6.51 | 6.719 / 5.457 / 6.510 | MATCH |
| 33 | Quenching field counts 20,657 (1), 22,731 (0), 10 NaN | same | MATCH |
| 34 | Reference table: XGB t1 0.334 / t3 0.377; RF 0.295 / 0.352; MLP 0.343 / 0.376; GPR 0.330 / 0.352 | same | MATCH |
| 35 | Abstract: pooled R2 0.41 +/- 0.01; MAE 2.91; p-i-n 0.37; n-i-p 0.45 | 0.407 +/- 0.031; 2.958; 0.294; 0.405 | MISMATCH (stale, H1) |
| 36 | Abstract/Conclusion: Delta R2 +0.005 to +0.007 for LLM features, "every subset and metric" | -0.0013 / +0.0025 / +0.0014; pooled RMSE worse | MISMATCH (H1) |
| 37 | Abstract/Conclusion: random overestimates R2 by 0.17-0.24 | 0.166-0.174 pooled CatBoost; 0.20-0.29 by architecture | MISMATCH (H1) |
| 38 | Conclusion: 0.413 pooled; 0.365 / 0.452; LLM gain 0.074 random / 0.004 DOI | not in any result file; 0.0069 / -0.0013 | MISMATCH (H1) |
| 39 | Abstract/Conclusion: 2,726 articles | 2,699 publications (2,726 raw strings) | MISMATCH/inconsistent |

Not independently reproducible: PDF / Markdown / JSON counts (3,299 / 3,141 / 2,835), the "410 columns" of the raw export, the Qwen extraction details, and all external-literature numbers (H4).

## 4. Over-reach and logic (Task 3)

- Roberts / Fukasawa, vs lit_audit.md: see H4. lit_audit lists the Fukasawa numbers (36,937 / 248 / 0.7177 / 294 / 0.45 / 3.20 / 6,446 / 20,695) as UNVERIFIED and warns against stating Roberts used random splits (only the thesis did). The manuscript still states both. It correctly says Roberts evaluated Jsc and "repeated device configurations arose during feature reduction", which fixes lit_audit item 6.
- Novelty: "To our knowledge, no earlier study compares record-level and DOI-grouped evaluation on the same PDP export" is hedged ("search was not exhaustive"). This is acceptable. The Abstract's "typically" is not.
- "Directionally consistent with Fukasawa" (:~391) compares a temporal-new-records drop with a grouped-vs-random drop. The manuscript notes the numbers are not comparable, but it still uses the comparison as support. It is also built on unverified numbers.
- "Leakage" language: the Results use "overestimates", and the discussion carefully says it does not prove identical-device leakage. Good. The Abstract and Conclusion say "overestimated" without that caveat. The gap also mixes (a) paper-level leakage, (b) different test-set composition and variance, and (c) unconverged random models (H3, which biases toward understating).
- Causal or significance wording: "gave the largest gain" (Conclusion) compares tiers under a single model and single seed. t1->t2 is solid across splits, and t2->t3 is correctly null in the Results.
- "Mean +/- s.d. over five splits" is described correctly as non-independent. Good. No pooled p-values are claimed, which is the correct call.
- SHAP: caveats are well stated. The "publication identifier" explanation is speculation (M6).
- "Properly trained MLP and GPR competitive with tree ensemble": GPR is a 3,000-record subsample, and the manuscript says no ranking is claimed. Fine.

## 5. Test coverage assessment (Task 4)

Current tests (tests/test_psc.py, tests/test_pipeline_smoke.py) cover: DOI normalisation variants, synthetic split partition and reproducibility, synthetic DOI-disjointness with the NO_DOI block, preprocessor median fit on train (single column), tier nesting, one real-data DOI-disjointness check (pooled, seed 42), CatBoost eval_set identity, and an end-to-end smoke run.

| Required check | Status | Problem |
|---|---|---|
| DOI overlap | Partially tested | Real-data test covers pooled data and seed 42 only. n-i-p / p-i-n subsets and the random-split non-overlap contrast are untested. `test_run_records_have_zero_doi_overlap` reads the overlap counts from runs.jsonl, which the same script wrote (circular). |
| Train / val / test leakage | Weak | `test_prediction_invariant_to_test_targets` is tautological: `fit_predict` takes no test targets, so it passes by construction. No test checks that preds npz `test_idx` / `train_idx` / `val_idx` are mutually disjoint, or that they are identical across tiers (paired validity). |
| Duplicates | Vacuous | `df.drop(columns=["doi_group"]).duplicated().sum() == 0` includes the unique `Ref_ID` column, so it can never fail. Cross-DOI duplicate-vector straddling is untested. |
| Target leakage | Weak | Only the name `final_pce` is checked, and FORBIDDEN is checked only for 2 names. The 7 FORBIDDEN names absent from the CSV, the object-dtype drop and any near-target column are not tested. No outcome-column denylist test (Voc / Jsc / FF / JV_*). |
| Preprocessing fit on train only | Partial | Tests median and keep list on a single column. The `_na` set, the winsorizer, the scaler / OHE, the XGB / RF category maps and the MLP / GPR path are not tested. NaN-only-in-test behaviour (L3) is untested. |
| Split reproducibility | Partial | Synthetic data only. Real-data determinism, seed+k inner seeds, and the property that different k give different splits are not tested. |
| Sample counts | Missing | No test pins 41,107 records / 7,327 groups / 232 no-DOI / 28,340 + 12,684 architecture counts / 77-179-223 features, which the paper reports. |
| Models other than CatBoost / ridge | Missing | XGB / RF / MLP / GPR and `run_model_dev` are not tested for val-only selection. |

Proposed tests (code for pasting into tests/test_protocol_audit.py, not added):

```python
import json
from pathlib import Path
import numpy as np, pandas as pd, pytest
from psc.data import (ARCHS, DEFAULT_DATA, FORBIDDEN, NO_DOI_GROUP, TARGET, DOI_COL,
                      load_dataset, normalize_doi)
from psc.features import TrainFitPreprocessor, base_frame
from psc.splits import make_splits

RES = Path(__file__).resolve().parents[1] / "results"
needs_data = pytest.mark.skipif(not DEFAULT_DATA.exists(), reason="dataset not present")


@needs_data
@pytest.mark.parametrize("arch", ["overall", "pin", "nip"])
def test_doi_disjoint_every_arch_and_seed(arch):
    df = load_dataset(arch=arch); g = df["doi_group"].values
    for seed in (42, 7):
        for tr, va, te in make_splits(g, "doi", 5, 0.2, 0.15, seed):
            S = [set(g[i]) for i in (tr, va, te)]
            assert not (S[0] & S[1] or S[0] & S[2] or S[1] & S[2])
            assert len(np.concatenate([tr, va, te])) == len(g)


@needs_data
def test_sample_counts_match_paper():
    df = load_dataset()
    assert len(df) == 41107 and df["doi_group"].nunique() == 7327
    assert (df["doi_group"] == NO_DOI_GROUP).sum() == 232
    assert load_dataset(arch="nip").shape[0] == 28340 and load_dataset(arch="pin").shape[0] == 12684
    assert [base_frame(df, t).shape[1] for t in ("t1", "t2", "t3")] == [76, 178, 222] or True  # pre-imputation raw widths; pin the post-fit widths instead
    pre = TrainFitPreprocessor()
    widths = [pre.fit_transform(base_frame(df, t)).shape[1] for t in ("t1", "t2", "t3")]
    assert widths == [77, 179, 223]   # full-data fit; per-split widths vary by +/-2


@needs_data
def test_no_outcome_or_identifier_columns_in_any_tier():
    df = load_dataset()
    banned = set(FORBIDDEN) | {"doi_group"}
    for t in ("t1", "t2", "t3", "t1_lit", "t3_nometa", "t2_noise"):
        X = base_frame(df, t)
        assert not (banned & set(X.columns)), t
        assert not [c for c in X.columns if c.startswith(("JV_", "Ref_", "Stability_", "EQE_"))]
        assert not [c for c in X.columns if X[c].dtype == object and c not in
                    ("Cell_architecture", "ETL_deposition_method",
                     "HTL_deposition_method", "Perovskite_deposition_method")]


@needs_data
def test_no_single_feature_nearly_determines_target():
    """Cheap target-leak screen: |Spearman| of any numeric predictor with PCE must be < 0.6."""
    df = load_dataset(); X = base_frame(df, "t3").select_dtypes("number")
    corr = X.apply(lambda c: c.corr(df[TARGET], method="spearman")).abs()
    assert corr.max() < 0.6, corr.sort_values().tail(3)


@needs_data
def test_duplicate_check_is_not_vacuous_and_cross_doi_dups_reported():
    df = load_dataset(); X = base_frame(df, "t3")
    h = X.astype(str).agg("|".join, axis=1)
    n_doi = df.assign(h=h).groupby("h")["doi_group"].nunique()
    # paper reports 162 cross-publication duplicate groups; the test pins it so changes are noticed
    assert int((n_doi > 1).sum()) == 162


def test_preprocessor_all_statistics_train_only():
    rng = np.random.default_rng(0)
    cat = ["Cell_architecture", "ETL_deposition_method", "HTL_deposition_method", "Perovskite_deposition_method"]
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "c": np.nan})
    for k in cat: X[k] = "x"
    X.loc[::4, "a"] = np.nan
    tr, te = X.iloc[:100], X.iloc[100:].copy()
    te["b"] = np.where(np.arange(100) % 2, np.nan, 1e9)       # NaN pattern and scale only in test
    p1 = TrainFitPreprocessor().fit(tr)
    te2 = te.copy(); te2["a"] = -1e9; te2["b"] = 7.0
    p2 = TrainFitPreprocessor().fit(tr)
    assert p1.median_ == p2.median_ and p1.keep_ == p2.keep_ and p1.na_cols_ == p2.na_cols_
    assert "c" not in p1.keep_                                  # >80% missing in train
    Z = p1.transform(te)
    assert not Z.drop(columns=cat).isnull().any().any()         # L3: no NaN may survive transform


def test_ridge_scaler_winsorizer_fit_on_train_only():
    from psc.models import Winsorizer
    tr = pd.DataFrame({"a": np.arange(100.0)}); w = Winsorizer().fit(tr)
    assert w.hi_["a"] == pytest.approx(np.quantile(np.arange(100.0), 0.995))
    out = w.transform(pd.DataFrame({"a": [1e9]}))
    assert out["a"].iloc[0] == pytest.approx(w.hi_["a"])


@pytest.mark.skipif(not (RES / "ablation_v1" / "preds").exists(), reason="results absent")
def test_saved_preds_independent_partition_and_pairing():
    """Recompute disjointness from saved indices (not from self-reported counts) and
    check identical test/train/val indices across tiers and models."""
    P = RES / "ablation_v1" / "preds"
    for proto in ("doi", "random"):
        for arch in ("overall", "pin", "nip"):
            for k in range(1, 6):
                ref = None
                for t in ("t1", "t2", "t3"):
                    for m in ("mean", "ridge", "catboost"):
                        z = np.load(P / f"{proto}_{arch}_{t}_{m}_{k}.npz")
                        idx = (z["train_idx"], z["val_idx"], z["test_idx"])
                        assert not (set(idx[0]) & set(idx[1]) or set(idx[0]) & set(idx[2])
                                    or set(idx[1]) & set(idx[2]))
                        if ref is None: ref = idx
                        assert all(np.array_equal(a, b) for a, b in zip(ref, idx))


@needs_data
@pytest.mark.skipif(not (RES / "ablation_v1" / "preds").exists(), reason="results absent")
def test_saved_split_doi_overlap_recomputed_from_data():
    df = load_dataset(); g = df["doi_group"].values
    z = np.load(RES / "ablation_v1" / "preds" / "doi_overall_t3_catboost_1.npz")
    assert not (set(g[z["test_idx"]]) & (set(g[z["train_idx"]]) | set(g[z["val_idx"]])))
    assert np.allclose(df[TARGET].values[z["test_idx"]], z["y"])   # y alignment with data


@pytest.mark.skipif(not (RES / "ablation_v1" / "runs.jsonl").exists(), reason="results absent")
def test_catboost_not_capped_in_doi_protocol_and_cap_reported_for_random():
    rows = [json.loads(l) for l in (RES / "ablation_v1" / "runs.jsonl").read_text().splitlines()]
    cb = [r for r in rows if r["model"] == "catboost"]
    assert all(r["info"]["best_iteration"] < 1499 for r in cb if r["protocol"] == "doi")
    capped = sum(r["info"]["best_iteration"] >= 1499 for r in cb if r["protocol"] == "random")
    assert capped == 0, f"{capped} random-protocol fits hit the iteration cap (unconverged)"  # currently FAILS (31) -> H3


def test_xgb_rf_category_maps_from_train_only(monkeypatch):
    from psc import models
    from psc.data import CAT
    n = 40
    X = pd.DataFrame({c: ["a"] * n for c in CAT}); X["v"] = np.arange(n, dtype=float)
    Xte = X.copy(); Xte[CAT[0]] = "test_only_level"
    y = np.arange(n, dtype=float)
    p, _ = models.fit_predict("rf", X, y, X, y, Xte)     # must not raise; unseen level -> -1
    assert np.isfinite(p).all()


def test_split_seed_changes_and_repeats_differ():
    g = np.repeat(np.arange(200), 5).astype(object)
    a = list(make_splits(g, "doi", 3, 0.2, 0.15, 1)); b = list(make_splits(g, "doi", 3, 0.2, 0.15, 2))
    assert not np.array_equal(a[0][2], b[0][2]) and not np.array_equal(a[0][2], a[1][2])
```

Note: the stray line `assert [...] == [76,178,222] or True` in the second test is a placeholder to be deleted. Use only the post-fit width assertion.

## 6. Recommended fix order

1. Rewrite the Abstract and Conclusion from `results/manuscript_numbers.json`, and generate those numbers with a script rather than by hand. Remove the LLM-gain claim. Fix "+0.038", the 0.17-0.24 range and 2,726 versus 2,699. Resolve `%%DEVPARA%%` and define "final model" (H1, H2).
2. Rerun the random-protocol arm with a larger iteration cap, or report the capped fraction and say the gap is a lower bound (H3).
3. Verify or hedge the Roberts / Fukasawa statements, and soften "typically" to "often" (H4).
4. Correct the p-i-n "not harder" statement (M2). Add a low-capacity control for the ridge argument or soften it (M3).
5. Record the provenance of the CatBoost hyper-parameters (M1). Fix `git_head` / dirty metadata, for example by archiving ml_paper as its own repo (M5).
6. Add the tests above and remove the vacuous ones.
7. Fix the "exceed 20%" wording and the pandas-3 dtype brittleness (L1, L2).
