# Scientific audit report

## Defects found in the original pipeline (all fixed; legacy code kept in `legacy/`)
1. **Test-set early stopping** in every training script (`eval_set=(X_test, y_test)`), including CatBoost, XGBoost and the ensemble.
2. **Un-normalised DOIs**: 7,371 raw strings = 7,326 publications; 45 publications could straddle partitions.
3. **Preprocessing fitted on all data** (medians, >80 % missing filter, `_na` set).
4. **Estimator confound**: the headline t3 number came from a tuned, bagged, XGBoost-averaged ensemble while t1/t2 used a single CatBoost.
5. **Weak baselines**: the earlier ANN/GPR comparators were far below properly trained models (MLP 0.376, GPR 0.352 vs CatBoost 0.407 on pooled t3).
6. **Inconsistent counts and stale figure panels** (223/221/225 features, 44–46 LLM features, XGBoost depth-12 panel, 42K/15K/27K).
7. A divergent ridge baseline (R² ≈ −33,000) was caught and fixed during this audit; a truncated (1,500-iteration) record-level arm was caught by the independent review and rerun.

## Evidence per contribution (pooled data, five DOI-grouped holdouts, mean ± s.d.)
**A. Prediction.** Matched CatBoost: t1 0.368 ± 0.024, t2 0.408 ± 0.032, t3 0.407 ± 0.031 (MAE 2.96). Validation-tuned CatBoost + XGBoost ensemble: t3 0.418 ± 0.026 (MAE 2.92). Ridge 0.350, MLP 0.376, XGBoost-default 0.377, RF 0.352, GPR(3k) 0.352. Material properties: ΔR² = +0.040, cluster-bootstrap CI excludes 0 in 5/5 pooled splits. **LLM features: ΔR² = −0.001, 0/15 intervals exclude zero; t1+LLM 0.372 vs t1 0.368; t2+noise 0.404 vs t2 0.408.** The original claim of a consistent LLM gain is not supported and was removed.
**B. DOI-grouped evaluation.** All 265 DOI-grouped fits (135 ablation + 45 feature-group + 40 reference-model + 45 model-development) have zero DOI overlap between train, validation and test (recorded, and recomputed from saved indices by the independent reviewer for the ablation); the record-level arm overlaps by design. Record-level vs DOI-grouped (CatBoost, pooled): 0.587/0.618/0.627 vs 0.368/0.408/0.407, gap 0.21–0.22; the record-level fits were rerun with a 20,000-iteration cap and converged (best iteration 7.2k–11.7k; 6k-cap values were 0.005–0.007 lower); ridge gap 0.056.
**Stricter protocols (new).** Lead-author-grouped (laboratory proxy): CatBoost t3 0.395 ± 0.013; forward-in-time (3 windows): 0.335 ± 0.058 (mean predictor −0.155); material step +0.030/+0.052, literature step +0.001/−0.010. All with zero DOI/author overlap.
**Extraction proxy validation (new).** Against the PDP's independent curation: thickness/annealing Spearman 0.44–0.90, 59–85 % within 10 %; DOI-level flag precision/recall 0.94/0.82 (TiO2), 0.95/0.80 (Spiro), 0.85/0.83 (NiOx); LLM numerics equal the PDP value exactly in 59–82 % of records and add a value where PDP is empty for only 0.8–5.3 % of records — redundancy with t1 is a plausible reason for the null t2→t3 effect. ICC by DOI = 0.69; 77 % of records belong to duplicated predictor configurations, almost all within one publication.

## Claims removed or revised
* "LLM-derived features give a small but consistent improvement" → no detectable benefit.
* "Gap grows to 0.24 for t3" → gap is similar across tiers (0.21–0.22).
* Single-split Roberts/Fukasawa pipeline comparison → withdrawn; replaced by in-framework XGBoost-default / RF baselines (not re-implementations).
* "Publication-level evaluation is required" → recommended alongside record-level reporting.
* "p-i-n is not intrinsically harder" → p-i-n is somewhat harder (MAE reduction 21 % vs 28 %).
* Ridge "capacity" explanation of the gap → stated as consistent for pooled data only; not a clean capacity test.

## Unresolved risks
* LLM extraction was validated only against a second curated source (no gold annotation); the Qwen checkpoint is not recorded; the retrieval stage of the released pipeline is now legal-only with a per-DOI provenance log (the corpus used for the reported features should be regenerated or checked against that policy before release).
* No external validation on independently collected devices (temporal and lead-author-surname grouping were run); 1,277 records in 162 cross-publication duplicate groups can straddle a DOI split.
* Fixed CatBoost hyperparameters were inherited from earlier work, not tuned here; tier comparison under tuned models is covered only by the separate model-development experiment (same conclusion).
* Fukasawa 2024 / Roberts 2024 claims verified from the PDFs in `Lit/`; overlapping 2026 arXiv LLM–perovskite papers (arXiv:2602.04914, 2602.13312) not read; Digital Discovery AI-disclosure policy not retrieved.
