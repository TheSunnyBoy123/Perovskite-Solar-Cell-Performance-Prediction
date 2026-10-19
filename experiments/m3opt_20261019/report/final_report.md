# M3 optimisation for the PSC manuscript: final report (experiment m3opt_20261019, 2026-10-19)

All work is in `ml_paper/experiments/m3opt_20261019/`. Nothing outside that directory was modified: the manuscript,
existing results, tier definitions, DOI assignments and git state are untouched. No external services or LLMs were used.
Audit findings are in `audit_report.md`.

## 0. Headline

1. **No M3 variant beats M2 with statistical support under the pre-specified primary test.**
   * The frozen candidate N_svd16_clean (M2 + a 16-dimensional TF-IDF/SVD embedding of the template-cleaned JSON extraction)
     scored +0.0072 R² on covered reserved records, 95% CI [−0.0052, +0.0199], with 5/5 seeds positive.
   * That is the same size as production M3 (+0.0071 [−0.0007, +0.0157]). Neither excludes zero.
2. **Corrected flags (secondary, unplanned):** M3_strict, the outcome-key-filtered flags with the join fixed, gave +0.0106 [+0.0026, +0.0191] on covered
   reserved records and +0.0039 [+0.0003, +0.0077] on all of them. Three things stop this from being claimed:
   * it was one of about 30 secondary comparisons;
   * on dev, the same arm was null (E1 −0.0012, E2 +0.0006);
   * the reserved split is not pristine.
   It is a lead for future data, not a result.
3. **The manuscript's M3 repeated-holdout numbers are not reproducible from the repository.**
   * The manuscript states M3 0.409 ± 0.028 (pooled), 0.293 (p-i-n) and 0.405 (n-i-p).
   * The stored predictions (`results/t3final_matched_v1`), the committed `results/t3final_summary/numbers.json`, `psc.results.load_final`
     and an independent rerun here (bit-identical predictions) all give **0.4066 ± 0.028, 0.2879 ± 0.073 and 0.4034 ± 0.011**.
     M3 is therefore *below* M2 (0.4078 / 0.2911 / 0.4038) on all three; see §6.
4. Recommendation: no performance claim should change in M3's favour. The M3 holdout numbers need correcting, which requires your approval (§9).

## 1. Baseline vs best (identical records, paired)

dR2 = R²(arm) − R²(M2) on the same records. CI = publication-cluster bootstrap, 2000 resamples.

**Reserved v4final (one-shot diagnostic, configuration frozen beforehand; `runs/F_final_R`).** Train on v4dev with 5 seeds; mirrors `scripts/v4_final.py`.

| Arm | Full R² (n=14,125) | dR2 full [95% CI] | Covered R² (n=5,460) | dR2 covered [95% CI] | seeds > 0 (covered) |
|---|---|---|---|---|---|
| M1 | 0.3712 | −0.0323 [−0.0442, −0.0207] | 0.3177 | −0.0416 [−0.0631, −0.0208] | 0/5 |
| **M2 (baseline)** | **0.4035** | — | **0.3593** | — | — |
| M3_v4 (production, reproduces stored) | 0.4060 | +0.0025 [−0.0011, +0.0062] | 0.3664 | +0.0071 [−0.0007, +0.0157] | 4/5 |
| M3_strict | 0.4074 | +0.0039 [+0.0003, +0.0077] | 0.3699 | +0.0106 [+0.0026, +0.0191] | 5/5 |
| M3_clean | 0.4059 | +0.0024 [−0.0013, +0.0059] | 0.3668 | +0.0075 [−0.0000, +0.0150] | 3/5 |
| **N_svd16_clean (primary candidate)** | 0.4058 | +0.0023 [−0.0030, +0.0079] | 0.3664 | **+0.0072 [−0.0052, +0.0199]** | 5/5 |
| N_svd16_clean_perm (control) | 0.4003 | −0.0032 [−0.0075, +0.0009] | 0.3513 | −0.0080 [−0.0167, +0.0011] | 0/5 |

"Covered" uses the production definition. With the corrected join (n=5,489) every number moves by ≤ 0.0008 (`final_metrics.csv`).

**Manuscript repeated DOI holdouts (`make_splits(...,'doi',5,0.2,0.15,42)`; `runs/F_final_H`).** Test indices were asserted equal to `ablation_v1`.
M2 and M3_v4 predictions are bit-identical to the stored `ablation_v1` t2 and `t3final_matched_v1` t3f predictions.

| Subset | M2 R² | M3_v4 R² | dR2 [95% CI], splits > 0 | N_svd16_clean R² | dR2 [95% CI], splits > 0 |
|---|---|---|---|---|---|
| Pooled | 0.4078 ± 0.032 | 0.4066 ± 0.028 | −0.0012 [−0.0059, +0.0034], 2/5 | 0.4078 ± 0.033 | +0.0000 [−0.0046, +0.0049], 3/5 |
| p-i-n | 0.2911 ± 0.066 | 0.2879 ± 0.073 | −0.0032 [−0.0153, +0.0098], 3/5 | 0.2887 ± 0.070 | −0.0024 [−0.0199, +0.0166], 3/5 |
| n-i-p | 0.4038 ± 0.013 | 0.4034 ± 0.011 | −0.0004 [−0.0052, +0.0045], 2/5 | 0.4042 ± 0.012 | +0.0003 [−0.0068, +0.0074], 2/5 |

These holdout test sets overlap the v4dev publications used for selection, so any candidate is favoured here by selection. Even so, nothing gains.

## 2. What was tried (dev only, v4dev 26,982 records / 4,708 DOIs, DOI-grouped 5-fold)

Full fold-level ledger: `experiment_ledger.csv` (1,015 fits). Per-run paired tables: `runs/<run>/paired_vs_M2.csv`.
Noise floor: M2 refit with another seed gives |dR2| ≤ 0.002.

| Family | Variants | Covered dR2 vs M2 | Verdict |
|---|---|---|---|
| Join fix (IOP DOIs, empty JSON → NaN) | M3_fixed | +0.0005 | Failed (within noise) |
| Outcome-key filter | M3_strict | E1 −0.0012, E2 +0.0006 | Failed on dev |
| Template removal | M3_clean | E3 +0.0010 | Failed on dev |
| Missingness representation | NaN / zero / cat3 / coverage flag | −0.0020 to +0.0006 | Failed |
| Penalty | pen 0 vs 0.5 | +0.0015 vs −0.0012 | Failed |
| Flag aggregation | 6 group counts, groups+flags, prevalence-weighted | +0.0003 to +0.0006 | Failed |
| JSON numeric properties / design descriptors | jprop, design | +0.0006 / −0.0007 | Failed (band gap already in M1) |
| Fold-internal token target encoding | tokte | −0.0005 | Failed |
| kNN target encoding over TF-IDF | knn (raw / chem) | E1 +0.0031, E2 +0.0045, E3 chem −0.0001 | Inconsistent |
| **TF-IDF→SVD text embedding** | k=8/16/32; raw / clean / chem text | E1 +0.0067; E2 +0.0040 (k16), +0.0072 (k32); E3 clean +0.0054, chem +0.0036 | Consistent sign, CIs include 0 → frozen as candidate |
| Combinations | flags+svd, svd+knn | +0.0052 to +0.0066 | No gain over svd alone |
| DOI-permuted controls (every block) | — | −0.0030 to −0.0080 | Real text beats permuted by about 0.01 |

E4 robustness ran *after* the freeze, on fresh fold seeds 50–52, and did not change the frozen choice. Each comparison is against M2 under the same model and configuration.

| Setting | Covered dR2 [95% CI] | Pooled dR2 |
|---|---|---|
| CatBoost default | +0.0103 [+0.0024, +0.0185] | −0.0000 |
| CatBoost depth 6 / l2 10 (both arms) | +0.0070 [−0.0007, +0.0152] | −0.0003 |
| CatBoost 3-model bag (both arms) | +0.0109 [+0.0040, +0.0184] | +0.0008 |
| LightGBM (both arms) | +0.0151 [+0.0060, +0.0249] | +0.0057 [+0.0014, +0.0100] |
| LightGBM, production M3_v4 flags | +0.0051 [−0.0020, +0.0123] | +0.0022 |
| k=32 clean text (CatBoost) | +0.0138 [+0.0051, +0.0230] | +0.0034 |

E4 effects are larger than E2/E3 effects on other fold seeds. This illustrates how much the dev estimate moves with the fold partition (±0.005).
On the reserved split the frozen CatBoost candidate shrank back to +0.007. The LightGBM and k=32 variants were not frozen, and the reserved split has now been used,
so their dev advantage cannot be confirmed with the data available.

Feature diagnostics (`feature_statistics.csv`, `feature_group_permutation.csv`):
* Publication-level permutation of all 36 flags in a trained M3 raises covered held-out MSE by about 1% (4/5 folds). The model uses the flags, but M2 recovers the same information without them.
* The process and HTL-additive groups carry most of the importance.
* `htl_tbp`/`htl_litfsi` are near-duplicates (φ 0.86), and `hot_cast` appears in a single paper.

## 3. M3 − M2 on identical records

* Reserved, covered: +0.0071 [−0.0007, +0.0157].
* Reserved, full: +0.0025 [−0.0011, +0.0062].
* Manuscript holdouts, pooled: −0.0012 [−0.0059, +0.0034]; p-i-n −0.0032; n-i-p −0.0004.
* Dev (E2, 25 fold-fits): covered +0.0002 [−0.0053, +0.0054]; pooled −0.0021 [−0.0046, +0.0004].

Under all protocols the incremental value of the recipe flags is indistinguishable from zero. Its sign depends on the split.

## 4. Uncertainty

* CIs come from a publication-cluster bootstrap of seed-averaged per-record squared-error differences.
* Seed-level and fold-level sign counts are reported alongside.
* Dev R² variability across fold partitions is SD 0.002–0.005, comparable to every effect studied.
* No multiplicity correction was applied to the reserved secondary comparisons (7 arms × 5 subsets). They are labelled exploratory.

## 5. Generalisation

* **Architecture:** text and flag gains are confined to covered n-i-p records. On p-i-n they are null or negative everywhere: reserved −0.004 to +0.001, holdouts −0.002 to −0.003, dev E2 −0.003 to −0.005.
* **Covered vs pooled:** gains on covered records (36% of records) are diluted to about +0.002 pooled.
* **Coverage:** nothing applies to post-2020 papers, where JSON coverage is 0%.
* **Model family:** the sign is the same for CatBoost and LightGBM, but the magnitude depends on the model (§2).

## 6. Data-quality and reproducibility problems (details in `audit_report.md`)

1. **Manuscript M3 holdout numbers are not reproducible.**
   * `ml.tex` (uncommitted working copy) and the uncommitted working copy of `results/t3final_summary/numbers.json`
     report M3 R² 0.409 / 0.293 / 0.405 with MAE 2.947 / 3.043 / 2.964.
   * The committed `numbers.json`, the stored predictions and this rerun give 0.4066 / 0.2879 / 0.4034 with MAE 2.956 / 3.065 / 2.969.
   * Only these M3 means (R², MAE, RMSE) differ between committed and working copy; every other key differs only by rounding, and the SDs were kept.
   * The working n-i-p values match the **superseded 44-predictor t3 block** (`ablation_v1`, 223 features), which `psc.results.load_final` deliberately drops.
     The pooled and p-i-n values match no stored run.
   * The sentence "adding the LLM-derived features produced a small improvement" (ml.tex l.388, 494) is not supported. The correct reading is no improvement (−0.001 pooled).
2. **Join bug:** 53 IOP-style JSON files never join, and 17 empty-JSON publications get fake all-zero flags.
3. **Leaky key filter:** the outcome-key filter misses 2,455 keys, so stability-test conditions trigger the ambient and N₂ flags (112 papers).
4. **Template artefacts:** placeholder records appear in about 492 papers; 140 papers are template-only and counted as covered.
5. 3 JSON files carry an extraction whose `source` field names a different paper.
6. **Coverage confounded with era:** 0% coverage for 2021+.
7. **Flag quality:** pilot precision 0.74, recall 0.44 (model-assisted, not human-adjudicated).
8. **Missing scripts:** the README references 6 scripts that do not exist.
9. **Reserved split reused:** v4final was already scored by `v4_final.py` and by an earlier `sub.py` feature-drop experiment.

## 7. Rerun commands (from `ml_paper/experiments/m3opt_20261019/scripts`)

```bash
python3 build_canonical.py                                   # JSON -> canonical long CSV + integrity audit
python3 run_dev.py --config ../configs/E1_screen.json        # likewise E2_confirm, E3_clean, E4_robust (resumable)
python3 analyze.py E2_confirm --subsets all,covered,pin,nip  # paired tables -> runs/<run>/paired_vs_M2.csv
python3 analyze.py E4_robust --ref M2_lgbm --arms N_svd16_clean_lgbm,M3_v4_lgbm   # matched-reference comparisons
python3 diag_features.py                                     # feature_* files, configs/tiers.json
python3 final_eval.py --config ../configs/F_final.json --protocol R   # reserved diagnostic (already spent)
python3 final_eval.py --config ../configs/F_final.json --protocol H   # manuscript holdouts
python3 analyze_final.py                                     # report/final_metrics.csv, final_reproduction.json
python3 make_ledger.py                                       # experiment_ledger.csv
```
Runtime on 11 cores: about 8–15 s per CatBoost fit. E1–E4 took 2.7 h of summed fit time; F_final about 20 min.

## 8. Paths

* **Audit and snapshot:** `audit_report.md`, `audit/integrity_audit.json`, `snapshot/` (git status, HEAD b2efd1f, versions).
* **Feature files:**
  * canonical extraction: `features/json_long.csv.gz`, `features/paper_status.csv`, `features/record_json_state.csv`;
  * `feature_schema.json`, `feature_provenance.csv`, `feature_statistics.csv`, `feature_correlations.csv`, `feature_group_permutation.csv`.
* **Configs:** `configs/tiers.json`, `configs/E1_screen.json`…`E4_robust.json`, `configs/F_final.json` (frozen choice and rationale).
* **Scripts:** `scripts/build_canonical.py`, `m3feat.py`, `run_dev.py`, `analyze.py`, `diag_features.py`, `final_eval.py`, `analyze_final.py`, `make_ledger.py`.
* **Runs:** `runs/<run>/preds/*.npz` (out-of-fold predictions), `ledger.jsonl`, `meta.json` (config and data SHA-256), `paired_vs_*.csv`.
* **Final:** `runs/F_final_R/pred_<arm>.npy`, `runs/F_final_H/H_<arch>_<arm>_<k>.npz`.
* **Reports:** `experiment_ledger.csv`, `report/final_metrics.csv`, `report/final_reproduction.json`, `report/final_report.md`.
* **Logs and job IDs:** `logs/` (PIDs in `logs/pids.txt`).

## 9. Recommendation

* **Do not change the manuscript to claim an M3 gain.** No optimisation produced a pre-specified, statistically supported improvement over M2.
  The best frozen candidate equals production M3 on the reserved split, and the manuscript holdouts show zero or negative increments.
  Framing the LLM features as giving "no measurable incremental value beyond material descriptors under publication-grouped evaluation" is better supported than "a modest incremental contribution".
* **A correction is needed (awaiting your approval; I have not edited anything).** Replace the M3 repeated-holdout values with the reproducible ones:
  * pooled R² 0.407 ± 0.028, MAE 2.956, RMSE 3.787;
  * p-i-n 0.288 ± 0.073;
  * n-i-p 0.403 ± 0.011;
  and reword the "small improvement" sentences. Alternatively, identify the run behind 0.409 / 0.293 and commit it.
* **Optional further disclosure:** the join bug and the outcome-key leakage in production flags. Fixing them does not change conclusions on dev.
* **If M3 is to be pursued:** the only signal with consistent direction is a cleaned text embedding of the extraction on covered n-i-p papers, about +0.01 covered R²,
  larger under LightGBM. Confirming it needs data not yet used for selection, for example post-2020 papers once extracted, or a new extraction batch.
  It should not be claimed from this dataset.
