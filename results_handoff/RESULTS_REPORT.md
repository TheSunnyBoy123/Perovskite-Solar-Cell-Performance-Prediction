# RESULTS REPORT — perovskite solar-cell PCE prediction (handoff for manual manuscript writing)

Generated from existing artifacts only (no training, no LLM calls, no new experiments). Project root: `/Users/sunraj/Desktop/College/S6/UGP/ml_paper`. Every table below is produced by
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
* **Records vs publications.** Median 4 records per DOI group; the maximum of 232 reported in `results/feature_audit.json` is the no-DOI block — the largest real publication has 127 records (the manuscript's "maximum 127" is the publication-level figure; both are correct, see section I). ICC of PCE by DOI = 0.692 (69% of variance between publications). 31,591 records belong to duplicated original-t3 predictor configurations (7,733 groups; only 162 groups span >1 DOI; median within-group sd of PCE 1.21).
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

| flag | group | prevalence among 2,835 extracted papers |
|---|---|---|
| rc_cl_source | additive/precursor | 0.205 |
| rc_acetate | additive/precursor | 0.047 |
| rc_scn | additive/precursor | 0.023 |
| rc_snf2 | additive/precursor | 0.007 |
| rc_ava | additive/precursor | 0.008 |
| rc_hx_acid | additive/precursor | 0.070 |
| rc_alkali | additive/precursor | 0.024 |
| rc_cs | additive/precursor | 0.120 |
| rc_pea_family | additive/precursor | 0.608 |
| rc_lewis_base | additive/precursor | 0.347 |
| rc_water_add | additive/precursor | 0.041 |
| rc_polymer | additive/precursor | 0.177 |
| rc_carbon | additive/precursor | 0.284 |
| rc_dio | additive/precursor | 0.007 |
| rc_htl_tbp | HTL dopant | 0.245 |
| rc_htl_litfsi | HTL dopant | 0.235 |
| rc_htl_fk | HTL dopant | 0.049 |
| rc_htl_f4tcnq | HTL dopant | 0.008 |
| rc_htl_cu | HTL dopant | 0.045 |
| rc_htl_borate | HTL dopant | 0.003 |
| rc_etl_ticl4 | ETL dopant | 0.056 |
| rc_etl_dopant | ETL dopant | 0.062 |
| rc_undoped | undoped / interface passivation | 0.091 |
| rc_interface_pass | undoped / interface passivation | 0.157 |
| rc_antisolvent | process route | 0.064 |
| rc_vacuum | process route | 0.176 |
| rc_two_step | process route | 0.039 |
| rc_hot_cast | process route | 0.000 |
| rc_solvent_anneal | process route | 0.010 |
| rc_flexible | device context | 0.043 |
| rc_tandem | device context | 0.034 |
| rc_lead_free | device context | 0.040 |
| rc_inorganic | device context | 0.075 |
| rc_mesoscopic | device context | 0.223 |
| rc_n2_box | process route | 0.040 |
| rc_ambient | process route | 0.104 |

**Coverage of the V4 block.** Flags exist for papers with an extraction record: **15,027 of 41,107 records (36.6%), 2,731 publications** (`lx_covered`); NaN elsewhere. The 32 publications / 131 records by which this mask exceeds the earlier block's mask all have an extraction record with zero extracted materials (their flags are all 0, not missing); see `docs/COVERAGE_MASK_AUDIT.md`, `results/coverage_mask_audit.csv`. Reserved evaluation partition: 5,460 covered records / 1,005 DOIs of 14,125 / 2,619.
**Do not mix representations.** Results labelled "t2+E" in sections D, E and G (five-split ablation, model development, protocol comparisons, baselines, SHAP) were produced with the earlier 44-predictor representation E. A V4 t1 result on the reserved split was added after the first handoff (section C.0, `results/v4/final_t1/`); no V4 five-split ablation, no V4 Bag/XGB/Ensemble and no V4 SHAP exists.

## C. Main V4 results (reserved evaluation, `results/v4/final/`, `results/v4/final_t1/`)

### C.0 Common-split t1 / t2 / t3 (added after the first handoff; `scripts/v4_t1_baseline.py`)
Same reserved `v4final` partition, test records, matched default CatBoost configuration, preprocessing and 5-seed procedure for all three tiers. t1 was trained for this comparison (it was missing from `results/v4/final/`); t2 and t3 are the existing `t2_default` / `t3_default` predictions (not retrained; their t3−t2 deltas reproduce C.1/C.3). Feature columns after preprocessing: t1 77, t2 179, t3 215. t1 was scored after t2/t3 with identical code path, seeds and validation splits; no choice depended on it.
Source: `results/v4/final_t1/{metrics_3tier.csv, comparisons_3tier.json}`; table `tables/v4_three_tier_common_split.csv`.

| population | tier | test records | test DOIs | R² [95% publication-cluster CI] | MAE | RMSE |
|---|---|---|---|---|---|---|
| LLM-covered | t1 | 5460 | 1005 | 0.3177 [0.2645, 0.3652] | 3.021 | 3.866 |
| LLM-covered | t2 | 5460 | 1005 | 0.3593 [0.3042, 0.4077] | 2.911 | 3.747 |
| LLM-covered | t3 | 5460 | 1005 | 0.3664 [0.3159, 0.4124] | 2.892 | 3.726 |
| all reserved | t1 | 14125 | 2619 | 0.3712 [0.3370, 0.4050] | 3.054 | 3.896 |
| all reserved | t2 | 14125 | 2619 | 0.4035 [0.3696, 0.4353] | 2.955 | 3.794 |
| all reserved | t3 | 14125 | 2619 | 0.4060 [0.3717, 0.4384] | 2.949 | 3.787 |

Paired differences (bootstrap over publications, 2,000 resamples; ΔMAE/ΔRMSE are differences of the point metrics, not bootstrapped):

| population | comparison | ΔR² | 95% paired CI | seeds ΔR²>0 | ΔMAE (derived) | ΔRMSE (derived) |
|---|---|---|---|---|---|---|
| LLM-covered | t2 - t1 | +0.0416 | [+0.0208, +0.0631] | 5/5 | -0.110 | -0.120 |
| LLM-covered | t3 - t2 | +0.0071 | [-0.0007, +0.0157] | 4/5 | -0.019 | -0.021 |
| LLM-covered | t3 - t1 | +0.0487 | [+0.0293, +0.0691] | 5/5 | -0.129 | -0.140 |
| all reserved | t2 - t1 | +0.0323 | [+0.0207, +0.0442] | 5/5 | -0.099 | -0.101 |
| all reserved | t3 - t2 | +0.0025 | [-0.0011, +0.0062] | 4/5 | -0.005 | -0.008 |
| all reserved | t3 - t1 | +0.0348 | [+0.0229, +0.0468] | 5/5 | -0.105 | -0.109 |

Reading: R², MAE and RMSE improve monotonically t1 → t2 → t3 in both populations. The t1 → t2 step has a CI excluding zero; the t2 → t3 step (+0.0071 covered, +0.0025 all) has a CI that **includes zero**, so it is not statistically established. These are separate from the separately tuned comparisons below (t1 was not tuned).

**Protocol.** (`scripts/v4_final.py`, `docs/V4_REPORT.md`) Publications assigned by SHA-256 of the normalised DOI with salt `psc-v4` (`psc.v4.v4_partition`): `v4dev` 65% (26,982 records / 4,708 DOIs; all design, tuning, ablations) and `v4final` 35% (**14,125 records / 2,619 DOIs; of which 5,460 records / 1,005 DOIs are covered by V4 flags**), scored once.
Final models: trained on all `v4dev` records, DOI-disjoint 15% validation split per seed for early stopping (100 rounds), **5 seed models averaged**, predictions scored on `v4final`. t2 = `psc.features.base_frame(df,'t2')`; t3 = t2 + 36 `rc_*` columns (NaN for uncovered papers, first-feature-use penalty 0.5). "Default" = the matched CatBoost configuration (depth 8, lr 0.05, L2 3, subsample 0.8, colsample 0.7, min_data_in_leaf 5, ≤1,500 iterations, `psc.v4model.DEFAULT_PARAMS`);
"tuned" = best of 24 random-search configurations per tier (default included) by DOI-grouped 5-fold mean squared error on `v4dev` (`scripts/v4_tune.py`, `results/v4/tune.json`; t3 has lower tuning MSE than t2 in only 6/24 shared configurations; best CV MSE t2 14.348, t3 14.387).
CI: publication-cluster bootstrap, 2,000 resamples (seed 0), of the paired squared-error difference of the seed-averaged predictions, divided by the test-set target variance; 95% percentile interval. "seeds ΔR²>0" counts the five per-seed paired differences.
**Held-out status:** the `v4final` DOIs were seen in earlier experiments with other feature representations (earlier representation E, lx2/lx3 blocks); they are fresh for the V4 representation but **not untouched data**; the flag block was selected after development comparisons on `v4dev`.

Tuned configurations (`results/v4/final/tuned_cfg.json`):

| CatBoost parameter | t2 tuned | t3 tuned |
|---|---|---|
| depth | 10 | 9 |
| learning_rate | 0.05 | 0.05 |
| l2_leaf_reg | 1.0 | 30.0 |
| colsample_bylevel | 0.7 | 0.55 |
| min_data_in_leaf | 20 | 20 |
| subsample | 0.6 | 1.0 |
| random_strength | 2.0 | 2.0 |

### C.1–C.2 Covered records (papers with V4 flags; 5,460 records / 1,005 DOIs): same-settings (A) and separately tuned (B)
| comparison | arms (reference -> comparison) | R² t2 | R² t3 | ΔR² | 95% CI | seeds ΔR²>0 | MAE t2 | MAE t3 | ΔMAE* | RMSE t2 | RMSE t3 | ΔRMSE* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: same default settings | t2_default -> t3_default | 0.359 | 0.366 | +0.0071 | [-0.0007, +0.0157] | 4/5 | 2.911 | 2.892 | -0.019 | 3.747 | 3.726 | -0.021 |
| A: both under t3-tuned config | t3cfg_on_t2 -> t3_tuned | 0.364 | 0.371 | +0.0066 | [-0.0021, +0.0162] | 4/5 | 2.898 | 2.883 | -0.015 | 3.732 | 3.713 | -0.019 |
| A: both under t2-tuned config | t2_tuned -> t2cfg_on_t3 | 0.361 | 0.364 | +0.0024 | [-0.0077, +0.0131] | 3/5 | 2.906 | 2.900 | -0.006 | 3.741 | 3.733 | -0.007 |
| B: each tier with its own tuned config | t2_tuned -> t3_tuned | 0.361 | 0.371 | +0.0094 | [-0.0016, +0.0219] | 5/5 | 2.906 | 2.883 | -0.023 | 3.741 | 3.713 | -0.028 |

### C.3–C.4 All reserved records (14,125 records / 2,619 DOIs)
| comparison | arms (reference -> comparison) | R² t2 | R² t3 | ΔR² | 95% CI | seeds ΔR²>0 | MAE t2 | MAE t3 | ΔMAE* | RMSE t2 | RMSE t3 | ΔRMSE* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: same default settings | t2_default -> t3_default | 0.403 | 0.406 | +0.0025 | [-0.0011, +0.0062] | 4/5 | 2.955 | 2.949 | -0.005 | 3.794 | 3.787 | -0.008 |
| A: both under t3-tuned config | t3cfg_on_t2 -> t3_tuned | 0.405 | 0.408 | +0.0026 | [-0.0015, +0.0067] | 2/5 | 2.949 | 2.943 | -0.006 | 3.788 | 3.780 | -0.008 |
| A: both under t2-tuned config | t2_tuned -> t2cfg_on_t3 | 0.406 | 0.405 | -0.0016 | [-0.0064, +0.0030] | 2/5 | 2.942 | 2.951 | +0.009 | 3.785 | 3.790 | +0.005 |
| B: each tier with its own tuned config | t2_tuned -> t3_tuned | 0.406 | 0.408 | +0.0015 | [-0.0039, +0.0067] | 3/5 | 2.942 | 2.943 | +0.001 | 3.785 | 3.780 | -0.005 |

\* ΔMAE and ΔRMSE are **derived** here as (comparison − reference) from `metrics.csv`; the artifacts store no CI for MAE/RMSE. Negative = lower error for t3. "A both under t3-tuned/t2-tuned config" applies one tuned configuration to both tiers.
Matches `docs/V4_REPORT.md`: A default covered +0.0071 [−0.0007, +0.0157] (4/5), all +0.0025 [−0.0011, +0.0062] (4/5); B covered +0.0094 [−0.0016, +0.0219] (5/5), all +0.0015 [−0.0039, +0.0067] (3/5); absolute covered R² t2-tuned 0.361, t3-tuned 0.371. **All intervals include zero.**

### C.5 Tuning effects (not LLM comparisons) and negative controls
| scope | comparison | ΔR² | 95% CI | seeds>0 |
|---|---|---|---|---|
| covered (papers with V4 flags) | tuning effect on t2 (not an LLM comparison) | +0.0020 | [-0.0048, +0.0088] | 2/5 |
| covered (papers with V4 flags) | tuning effect on t3 (not an LLM comparison) | +0.0043 | [-0.0006, +0.0093] | 5/5 |
| all records | tuning effect on t2 (not an LLM comparison) | +0.0030 | [-0.0009, +0.0069] | 3/5 |
| all records | tuning effect on t3 (not an LLM comparison) | +0.0020 | [-0.0013, +0.0053] | 4/5 |

| scope | control arm (vs t2 default) | ΔR² | 95% CI | seeds>0 |
|---|---|---|---|---|
| covered (papers with V4 flags) | ctl_perm0_default | -0.0008 | [-0.0091, +0.0089] | 2/5 |
| covered (papers with V4 flags) | ctl_perm1_default | -0.0035 | [-0.0100, +0.0031] | 0/5 |
| covered (papers with V4 flags) | ctl_perm2_default | +0.0015 | [-0.0050, +0.0077] | 4/5 |
| covered (papers with V4 flags) | ctl_perm3_default | -0.0013 | [-0.0092, +0.0066] | 1/5 |
| covered (papers with V4 flags) | ctl_perm4_default | -0.0049 | [-0.0124, +0.0027] | 1/5 |
| covered (papers with V4 flags) | ctl_coverflag_default | +0.0015 | [-0.0024, +0.0053] | 2/5 |
| all records | ctl_perm0_default | +0.0007 | [-0.0029, +0.0044] | 2/5 |
| all records | ctl_perm1_default | -0.0016 | [-0.0050, +0.0020] | 1/5 |
| all records | ctl_perm2_default | -0.0020 | [-0.0056, +0.0017] | 2/5 |
| all records | ctl_perm3_default | -0.0008 | [-0.0043, +0.0027] | 2/5 |
| all records | ctl_perm4_default | -0.0028 | [-0.0064, +0.0008] | 1/5 |
| all records | ctl_coverflag_default | +0.0011 | [-0.0016, +0.0038] | 2/5 |

Controls: five copies of the 36-column block with rows permuted across publications (DOI-level permutation, `np.random.default_rng(100+j)`) and a coverage-indicator-only block (`rc_covered`), all with the matched default config. Covered permuted range −0.0049…+0.0015, all-records −0.0028…+0.0007; coverage-only +0.0015 / +0.0011.
Interpretation (team, `docs/V4_REPORT.md`): tuning alone moves t2 by about +0.002…+0.003, comparable to the t3 effect, and the t3 gain shrinks under t2's tuned configuration (+0.0024 covered; −0.0016 all), so the tuned comparison B should not be read as an LLM gain by itself.

### C.6 Metrics for every arm (`results/v4/final/metrics.csv`)
| scope | arm | tier | estimator config | R² | MAE | RMSE | n test records | n test DOIs |
|---|---|---|---|---|---|---|---|---|
| all records | t2_default | t2 | matched default CatBoost | 0.4035 | 2.955 | 3.794 | 14125 | 2619 |
| all records | t3_default | t3 | matched default CatBoost | 0.4060 | 2.949 | 3.787 | 14125 | 2619 |
| all records | t2_tuned | t2 | t2-tuned CatBoost config | 0.4065 | 2.942 | 3.785 | 14125 | 2619 |
| all records | t3_tuned | t3 | t3-tuned CatBoost config | 0.4080 | 2.943 | 3.780 | 14125 | 2619 |
| all records | t3cfg_on_t2 | t2 | t3-tuned config applied to t2 | 0.4054 | 2.949 | 3.788 | 14125 | 2619 |
| all records | t2cfg_on_t3 | t3 | t2-tuned config applied to t3 | 0.4049 | 2.951 | 3.790 | 14125 | 2619 |
| all records | ctl_perm0_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.4042 | 2.958 | 3.792 | 14125 | 2619 |
| all records | ctl_perm1_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.4019 | 2.961 | 3.799 | 14125 | 2619 |
| all records | ctl_perm2_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.4015 | 2.958 | 3.801 | 14125 | 2619 |
| all records | ctl_perm3_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.4027 | 2.958 | 3.797 | 14125 | 2619 |
| all records | ctl_perm4_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.4007 | 2.964 | 3.803 | 14125 | 2619 |
| all records | ctl_coverflag_default | t2 + coverage indicator only | matched default CatBoost | 0.4046 | 2.947 | 3.791 | 14125 | 2619 |
| covered (papers with V4 flags) | t2_default | t2 | matched default CatBoost | 0.3593 | 2.911 | 3.747 | 5460 | 1005 |
| covered (papers with V4 flags) | t3_default | t3 | matched default CatBoost | 0.3664 | 2.892 | 3.726 | 5460 | 1005 |
| covered (papers with V4 flags) | t2_tuned | t2 | t2-tuned CatBoost config | 0.3613 | 2.906 | 3.741 | 5460 | 1005 |
| covered (papers with V4 flags) | t3_tuned | t3 | t3-tuned CatBoost config | 0.3707 | 2.883 | 3.713 | 5460 | 1005 |
| covered (papers with V4 flags) | t3cfg_on_t2 | t2 | t3-tuned config applied to t2 | 0.3641 | 2.898 | 3.732 | 5460 | 1005 |
| covered (papers with V4 flags) | t2cfg_on_t3 | t3 | t2-tuned config applied to t3 | 0.3637 | 2.900 | 3.733 | 5460 | 1005 |
| covered (papers with V4 flags) | ctl_perm0_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.3585 | 2.920 | 3.749 | 5460 | 1005 |
| covered (papers with V4 flags) | ctl_perm1_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.3558 | 2.922 | 3.757 | 5460 | 1005 |
| covered (papers with V4 flags) | ctl_perm2_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.3608 | 2.902 | 3.742 | 5460 | 1005 |
| covered (papers with V4 flags) | ctl_perm3_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.3580 | 2.916 | 3.750 | 5460 | 1005 |
| covered (papers with V4 flags) | ctl_perm4_default | t2 + DOI-permuted recipe block | matched default CatBoost | 0.3544 | 2.919 | 3.761 | 5460 | 1005 |
| covered (papers with V4 flags) | ctl_coverflag_default | t2 + coverage indicator only | matched default CatBoost | 0.3607 | 2.904 | 3.742 | 5460 | 1005 |

### C.7 Development-stage evidence (v4dev only; DOI-grouped 5-fold × 3 seeds, 15 folds; recomputed from `results/v4/*.csv`)
* Block with first-feature-use penalty 0.5 (`pen_05.csv`, the final setting): covered-dev ΔR² = **+0.0070, 12/15 folds positive**; without penalty (`rec1.csv`): +0.0052, 12/15. (`docs/V4_REPORT.md` quotes +0.007, which corresponds to the penalised run; it is not a discrepancy.)
* Five DOI-permuted copies (`recctl.csv`, unpenalised): −0.0086, −0.0056, −0.0029, +0.0022, −0.0069 (matches the reported range −0.009…+0.002).
* Existing JSON content otherwise exhausted: token bags +0.004 (`blocks1.csv`: +0.0039), numeric quantities −0.0023, device-matched layer values −0.0012, PDP gap-filling −0.0030 (all from `blocks1.csv`); full-data all-record 5-fold with penalty-free block: +0.0027, 11/15 folds (`full_rec.csv`).
* Dropping any one flag group lowers the gain and no group alone reproduces it (`recabl.csv`: group-only −0.0053…+0.0005; drop-one +0.0010…+0.0030 vs +0.0052 for all) — a distributed signal near the noise floor. Role-scoped variant +0.001 (`rec2.csv`).
* These dev numbers are **development results** (the flag set and penalty were chosen on them) and must not be presented as held-out evidence.

## D. Earlier fixed-configuration ablation (earlier representation E; `results/ablation_v1`)

**Protocol.** Five repeated DOI-grouped holdouts (`GroupShuffleSplit`, test 20% of publications, seed 42; per-split inner DOI-disjoint validation = 15% of training publications, used only for early stopping). Config `configs/ablation.json`, data SHA-256 `600efce587a1bd81…`. One fixed CatBoost configuration for all tiers (depth 8, lr 0.05, L2 3, subsample 0.8, colsample 0.7, min leaf 5, seed 42, ≤1,500 iterations, 100-round early stopping on validation) — **hyperparameters inherited from earlier work, not tuned in this study**. Mean train/val/test sizes per split (pooled): ≈27680 / 4999 / 8428 records; ≈4981 / 880 / 1466 DOIs. Zero DOI overlap between train/validation/test in every split (recorded in `runs.jsonl`, `tests/test_protocol_audit.py`).
Test sets of the five splits overlap, so sd values describe split-to-split variability, not CIs. **t2+E in this section = t2 plus the earlier 44-predictor literature representation E (run-file code `t3`).** These values are unchanged by V4.

| subset | tier | mean features | mean test records/split | mean test DOIs/split | R² (mean ± sd, 5 splits) | MAE | RMSE |
|---|---|---|---|---|---|---|---|
| pooled | t1 | 77.0 | 8428 | 1466 | 0.368 ± 0.024 | 3.081 ± 0.040 | 3.908 ± 0.046 |
| pooled | t2 | 179.0 | 8428 | 1466 | 0.408 ± 0.032 | 2.950 ± 0.080 | 3.783 ± 0.085 |
| pooled | t2+E = t2 + earlier literature representation E | 223.0 | 8428 | 1466 | 0.407 ± 0.031 | 2.958 ± 0.080 | 3.787 ± 0.083 |
| p-i-n | t1 | 77.0 | 2599 | 451 | 0.244 ± 0.058 | 3.168 ± 0.049 | 4.085 ± 0.114 |
| p-i-n | t2 | 179.0 | 2599 | 451 | 0.291 ± 0.065 | 3.047 ± 0.045 | 3.955 ± 0.131 |
| p-i-n | t2+E = t2 + earlier literature representation E | 224.2 | 2599 | 451 | 0.294 ± 0.074 | 3.051 ± 0.067 | 3.947 ± 0.142 |
| n-i-p | t1 | 76.2 | 5612 | 1037 | 0.374 ± 0.016 | 3.062 ± 0.047 | 3.918 ± 0.061 |
| n-i-p | t2 | 178.2 | 5612 | 1037 | 0.404 ± 0.013 | 2.966 ± 0.036 | 3.825 ± 0.060 |
| n-i-p | t2+E = t2 + earlier literature representation E | 222.2 | 5612 | 1037 | 0.405 ± 0.010 | 2.964 ± 0.017 | 3.821 ± 0.045 |

Paired differences (per-split cluster bootstrap over test publications, 1,000 resamples; `paired_tier_diffs.csv`, `paired_bootstrap_summary_doi_catboost.csv`):

| subset | contrast | ΔR² mean | ΔR² sd | splits R² better | splits CI excludes 0 | ΔRMSE mean |
|---|---|---|---|---|---|---|
| n-i-p | t2-t1 | +0.029 | 0.016 | 5/5 | 3/5 | -0.093 |
| n-i-p | (t2+E)-t2 | +0.001 | 0.006 | 3/5 | 0/5 | -0.005 |
| n-i-p | (t2+E)-t1 | +0.031 | 0.012 | 5/5 | 4/5 | -0.097 |
| pooled | t2-t1 | +0.040 | 0.014 | 5/5 | 5/5 | -0.125 |
| pooled | (t2+E)-t2 | -0.001 | 0.003 | 1/5 | 0/5 | +0.004 |
| pooled | (t2+E)-t1 | +0.038 | 0.013 | 5/5 | 4/5 | -0.121 |
| p-i-n | t2-t1 | +0.047 | 0.014 | 5/5 | 2/5 | -0.130 |
| p-i-n | (t2+E)-t2 | +0.002 | 0.017 | 3/5 | 0/5 | -0.008 |
| p-i-n | (t2+E)-t1 | +0.049 | 0.020 | 5/5 | 4/5 | -0.138 |

Feature-group controls (earlier representation E; same protocol; `results/feature_groups_v1`):

| subset | variant | mean features | R² mean ± sd | MAE | RMSE |
|---|---|---|---|---|---|
| ablation holdouts n-i-p | t1 + earlier literature representation E (no material properties) | 120.2 | 0.378 ± 0.014 | 3.050 | 3.907 |
| ablation holdouts n-i-p | t2 + 32 N(0,1) noise columns | 210.2 | 0.402 ± 0.009 | 2.976 | 3.830 |
| ablation holdouts n-i-p | t2+E minus 2 reporting-completeness flags | 218.2 | 0.407 ± 0.012 | 2.960 | 3.816 |
| ablation holdouts pooled | t1 + earlier literature representation E (no material properties) | 121.0 | 0.372 ± 0.026 | 3.060 | 3.895 |
| ablation holdouts pooled | t2 + 32 N(0,1) noise columns | 211.0 | 0.404 ± 0.029 | 2.970 | 3.796 |
| ablation holdouts pooled | t2+E minus 2 reporting-completeness flags | 219.0 | 0.407 ± 0.028 | 2.953 | 3.787 |
| ablation holdouts p-i-n | t1 + earlier literature representation E (no material properties) | 122.2 | 0.251 ± 0.071 | 3.150 | 4.065 |
| ablation holdouts p-i-n | t2 + 32 N(0,1) noise columns | 211.0 | 0.288 ± 0.072 | 3.067 | 3.962 |
| ablation holdouts p-i-n | t2+E minus 2 reporting-completeness flags | 220.2 | 0.292 ± 0.070 | 3.052 | 3.951 |

Material-identity control (`docs/IDENTITY_CONTROL.md`, `results/identity_vs_t1id.csv`; full data, DOI-grouped 5-fold × seeds 42–44, matched config; 41,107 records / 7,327 DOI groups; **different protocol from the five holdouts**):

| tier | reference | R² ref | R² tier | ΔR² (tier − ref) | 95% CI | seeds>0 | folds>0 |
|---|---|---|---|---|---|---|---|
| t1 | t1_id | 0.4064 | 0.3698 | -0.0366 | [-1.075, -0.687] (ΔMSE) | 0 | 0/15 |
| t2 | t1_id | 0.4064 | 0.4073 | +0.0008 | [-0.102, 0.145] (ΔMSE) | 2 | 9/15 |
| t2_id | t1_id | 0.4064 | 0.4101 | +0.0036 | [-0.007, 0.184] (ΔMSE) | 3 | 12/15 |

Additional single-stage full-data results with other literature blocks (E2–E6, `results/ablE_vs_t2_*.csv`, matched config, DOI-grouped 5-fold × 3 seeds): earlier representation E −0.0024 (pooled), −0.0007 (n-i-p), −0.0002 (p-i-n) relative to t2 (t2 R² 0.4073 / 0.4246 / 0.3218); this is a **different protocol** from the five holdouts above (pooled t2+E − t2 = −0.0013) — both are null.
Per-split values, split sizes and feature counts: `tables/original_tier_ablation_per_split.csv`.

## E. Model-development comparison (`results/model_dev_v1`, **earlier representation E only**)

**Procedure** (`scripts/run_model_dev.py`, `configs/model_dev.json`): per (subset, tier, DOI-grouped split) — same 5 splits as section D — select among a predeclared CatBoost grid of six configurations
(depth/L2 ∈ {(6,3),(8,3),(10,3),(8,10),(10,10),(8,1)}; lr 0.05, ≤3,000 iterations, subsample 0.8, colsample 0.7, min leaf 5; 150-round early stopping) by **validation RMSE**; then (i) tuned CatBoost, (ii) 3-seed bag (seeds 42, 7, 123) of the selected configuration, (iii) XGBoost with early stopping on the validation set, (iv) mean of bag and XGBoost. The test partition is used only for final scores; identical grid and budget for every tier. 45 cells (3 subsets × 3 tiers × 5 splits).
**t2+E here = t2 plus the earlier representation E. No V4 entries exist; none were invented.**

| subset | tier | model | R² mean ± sd (5 splits) | MAE | RMSE |
|---|---|---|---|---|---|
| n-i-p | t1 | CatBoost tuned | 0.374 ± 0.016 | 3.061 | 3.919 |
| n-i-p | t1 | CatBoost 3-seed bag of selected config | 0.377 ± 0.015 | 3.056 | 3.911 |
| n-i-p | t1 | XGBoost | 0.368 ± 0.017 | 3.075 | 3.939 |
| n-i-p | t1 | Ensemble (mean of Bag and XGB) | 0.380 ± 0.015 | 3.046 | 3.900 |
| n-i-p | t2 | CatBoost tuned | 0.406 ± 0.018 | 2.950 | 3.818 |
| n-i-p | t2 | CatBoost 3-seed bag of selected config | 0.409 ± 0.017 | 2.946 | 3.808 |
| n-i-p | t2 | XGBoost | 0.402 ± 0.011 | 2.975 | 3.832 |
| n-i-p | t2 | Ensemble (mean of Bag and XGB) | 0.413 ± 0.013 | 2.941 | 3.795 |
| n-i-p | t2+E (t2 + earlier representation E) | CatBoost tuned | 0.407 ± 0.015 | 2.955 | 3.815 |
| n-i-p | t2+E (t2 + earlier representation E) | CatBoost 3-seed bag of selected config | 0.409 ± 0.015 | 2.952 | 3.807 |
| n-i-p | t2+E (t2 + earlier representation E) | XGBoost | 0.403 ± 0.015 | 2.971 | 3.829 |
| n-i-p | t2+E (t2 + earlier representation E) | Ensemble (mean of Bag and XGB) | 0.413 ± 0.015 | 2.944 | 3.796 |
| pooled | t1 | CatBoost tuned | 0.371 ± 0.023 | 3.066 | 3.901 |
| pooled | t1 | CatBoost 3-seed bag of selected config | 0.372 ± 0.023 | 3.066 | 3.897 |
| pooled | t1 | XGBoost | 0.368 ± 0.019 | 3.076 | 3.910 |
| pooled | t1 | Ensemble (mean of Bag and XGB) | 0.377 ± 0.021 | 3.055 | 3.881 |
| pooled | t2 | CatBoost tuned | 0.408 ± 0.029 | 2.938 | 3.781 |
| pooled | t2 | CatBoost 3-seed bag of selected config | 0.410 ± 0.029 | 2.938 | 3.776 |
| pooled | t2 | XGBoost | 0.409 ± 0.022 | 2.941 | 3.781 |
| pooled | t2 | Ensemble (mean of Bag and XGB) | 0.416 ± 0.026 | 2.924 | 3.758 |
| pooled | t2+E (t2 + earlier representation E) | CatBoost tuned | 0.408 ± 0.029 | 2.948 | 3.784 |
| pooled | t2+E (t2 + earlier representation E) | CatBoost 3-seed bag of selected config | 0.412 ± 0.029 | 2.938 | 3.771 |
| pooled | t2+E (t2 + earlier representation E) | XGBoost | 0.411 ± 0.024 | 2.935 | 3.773 |
| pooled | t2+E (t2 + earlier representation E) | Ensemble (mean of Bag and XGB) | 0.418 ± 0.026 | 2.921 | 3.752 |
| p-i-n | t1 | CatBoost tuned | 0.235 ± 0.061 | 3.184 | 4.110 |
| p-i-n | t1 | CatBoost 3-seed bag of selected config | 0.241 ± 0.060 | 3.180 | 4.095 |
| p-i-n | t1 | XGBoost | 0.235 ± 0.058 | 3.201 | 4.110 |
| p-i-n | t1 | Ensemble (mean of Bag and XGB) | 0.249 ± 0.059 | 3.168 | 4.073 |
| p-i-n | t2 | CatBoost tuned | 0.293 ± 0.066 | 3.037 | 3.950 |
| p-i-n | t2 | CatBoost 3-seed bag of selected config | 0.297 ± 0.069 | 3.032 | 3.938 |
| p-i-n | t2 | XGBoost | 0.285 ± 0.074 | 3.069 | 3.972 |
| p-i-n | t2 | Ensemble (mean of Bag and XGB) | 0.300 ± 0.072 | 3.033 | 3.928 |
| p-i-n | t2+E (t2 + earlier representation E) | CatBoost tuned | 0.295 ± 0.071 | 3.041 | 3.943 |
| p-i-n | t2+E (t2 + earlier representation E) | CatBoost 3-seed bag of selected config | 0.299 ± 0.069 | 3.033 | 3.933 |
| p-i-n | t2+E (t2 + earlier representation E) | XGBoost | 0.284 ± 0.077 | 3.093 | 3.974 |
| p-i-n | t2+E (t2 + earlier representation E) | Ensemble (mean of Bag and XGB) | 0.300 ± 0.074 | 3.046 | 3.928 |

Selected configurations change between splits (pooled):

| tier (pooled) | selected {depth, l2_leaf_reg} per split 1-5 |
|---|---|
| t1 | [{"depth": 10, "l2_leaf_reg": 3}, {"depth": 6, "l2_leaf_reg": 3}, {"depth": 10, "l2_leaf_reg": 10}, {"depth": 8, "l2_leaf_reg": 3}, {"depth": 8, "l2_leaf_reg": 10}] |
| t2 | [{"depth": 10, "l2_leaf_reg": 3}, {"depth": 8, "l2_leaf_reg": 10}, {"depth": 10, "l2_leaf_reg": 10}, {"depth": 6, "l2_leaf_reg": 3}, {"depth": 8, "l2_leaf_reg": 1}] |
| t2+E | [{"depth": 8, "l2_leaf_reg": 3}, {"depth": 10, "l2_leaf_reg": 10}, {"depth": 10, "l2_leaf_reg": 10}, {"depth": 8, "l2_leaf_reg": 1}, {"depth": 8, "l2_leaf_reg": 1}] |

Headline (team reading in `docs/AUDIT_REPORT.md`): pooled ensemble 0.418 ± 0.026, MAE 2.92 — 0.011 above matched CatBoost, produced by tuning/bagging/ensembling, not by features (t1 0.377 → t2 0.416 → t2+E 0.418).

## F. Feature engineering and extraction

* **Full-text retrieval and extraction coverage.** 7,499 raw DOI strings in the PDP; 3,299 PDFs retrieved → 3,141 Markdown conversions (OpenDataLoader PDF) → 2,835 structured JSON paper records (`json/`, not packaged) → 2,726 raw DOI strings = 2,699 normalised DOIs matched to the filtered data (14,896 records, 36.2%) for the earlier representation E; the V4 mask (`lx_covered`) is 15,027 records / 2,731 DOIs (36.6%). (Attrition counts: `docs/TODO_REGISTER.md` #6, `docs/AUDIT_REPORT.md`; the 3,299/3,141/2,835 counts come from the project logs/docs, not from a file in `results/`.)
* **LLM extraction.** Locally served Qwen (alias `qwen35`, vLLM, temperature 0.1, ≤16,384 output tokens, one request per document, zero-shot system prompt recovered verbatim in `docs/EXTRACTION.md`). **The exact checkpoint/quantisation is not recorded anywhere in the repository** (manuscript text says "Qwen3.5-9B" — unverifiable from artifacts). The number of few-shot examples cannot be verified; the recovered script is zero-shot.
* **Original literature block** (`src/psc/litfeat.py`, `docs/FEATURES.md`): 32 paper-level base columns (material-presence `has_*`, process `used_*`, thicknesses/anneal/spin/concentration, reporting-completeness `json_numeric_values_found`, `has_stability_measurement`) → 44 predictors with indicators; coverage 36.2% of records, constant within a paper. Per-feature coverage (`results/feature_audit.json`): flags 0.362; thicknesses 0.081–0.176; anneal 0.037–0.070; concentration 0.051; spin speed 0.100.
* **Quantity-level / re-parsed exploration (lx2, lx3; development-era, not final).** lx2.0: 61 publication-level columns (process, chemistry, context) used in a two-stage residual design on covered papers (dev +0.0022 [−0.0003, +0.0051], 5/5 seeds; confirm +0.0011 [−0.0015, +0.0036], 4/5; pre-registered CI-excludes-zero rule **not met**; gain concentrated in a minority of publications — removing the 25 most-helped publications turns it negative; `docs/LX_RESULTS.md`, `docs/PREREG_LX.md`). lx3.0: 39 columns (30 quantity `lx_qty_*`, 9 chemistry `lx_cat_*`); most quantity columns cover 2–15% of papers; only 2 of 30 survived the 80%-missingness filter in earlier runs. A quantity-covered experiment on the 9,441-record / 1,680-DOI eligible population (15 predeclared columns with ≥10% coverage; partition `qdev` 6,677 rec / 1,180 DOIs, `qconf` 2,764 / 500) gave dev ΔR² +0.0104 (t2 + quantities) and held-out +0.0026 [−0.010, +0.014], with the earlier representation E −0.0141 [−0.038, +0.006] on `qconf`; controls fluctuate −0.01…+0.005 (`docs/LX3_RESULTS.md`, `results/qty_holdout/summary.json`). Exploratory; population selected by availability. Table: `tables/control_experiments.csv`.
* **Why the V4 groups were retained** (evidence recorded in `docs/V4_REPORT.md` and section C.7): token bags, numeric quantities, device-matched layer values and PDP gap-filling gave ~0 or negative on development; recipe identities gave +0.007 on covered dev (12/15 folds) versus permuted controls −0.009…+0.002; no single group reproduces the gain; interpretation that the additive/recipe identities are the targeted missing information is the team's (see ceiling analysis, +0.022 for PDP's own curated columns).
* **Extraction validation (proxy only; no gold annotation exists; hallucination rate not measured).** Agreement with PDP's independent curation (`results/extraction_validation.json`, `scripts/validate_extraction.py`), earlier representation E, 2,699 DOIs:

| field | DOIs | Spearman | within 10% | exact equal where both exist | LLM value where PDP missing (fraction of records) |
|---|---|---|---|---|---|
| perovskite_thickness_nm | 507 | 0.79 | 0.78 | 0.73 | 0.053 |
| etl_thickness_nm | 433 | 0.63 | 0.59 | 0.59 | 0.046 |
| htl_thickness_nm | 338 | 0.90 | 0.78 | 0.78 | 0.038 |
| contact_thickness_nm | 1263 | 0.67 | 0.82 | 0.82 | 0.014 |
| perov_anneal_temp_C | 455 | 0.71 | 0.85 | 0.78 | 0.008 |
| perov_anneal_time_min | 406 | 0.44 | 0.73 | 0.74 | 0.009 |

| flag (earlier representation E) | DOIs | PDP prevalence | precision | recall | kappa |
|---|---|---|---|---|---|
| has_SnO2 | 2699 | 0.094 | 0.54 | 0.82 | 0.60 |
| has_TiO2 | 2699 | 0.560 | 0.94 | 0.82 | 0.74 |
| has_ZnO | 2699 | 0.048 | 0.56 | 0.91 | 0.67 |
| has_C60 | 2699 | 0.064 | 0.50 | 0.87 | 0.60 |
| has_PCBM | 2699 | 0.253 | 0.75 | 0.54 | 0.53 |
| has_BCP | 2699 | 0.002 | 0.02 | 1.00 | 0.03 |
| has_LiF | 2699 | 0.001 | 0.05 | 0.50 | 0.09 |
| has_Spiro | 2699 | 0.551 | 0.94 | 0.80 | 0.73 |
| has_PEDOT | 2699 | 0.196 | 0.81 | 0.87 | 0.80 |
| has_NiOx | 2699 | 0.072 | 0.85 | 0.83 | 0.83 |
| has_PTAA | 2699 | 0.056 | 0.83 | 0.83 | 0.82 |

  `has_BCP`/`has_LiF` cannot be assessed (PDP stack strings rarely contain those layers). Numerical values equal the PDP value exactly in 59–82% of records where both exist and add a value where PDP is empty in only 0.8–5.3% of records — the earlier representation E is largely redundant with t1. **No validation of the V4 flags against any reference was run**; they are regexes over the same extraction JSON.
* **Original-block coverage vs V4 coverage** are different masks (see section I).

## G. Other scientific results (all on the earlier representation E where a literature tier appears)

### G.1–G.3 Evaluation protocols (pooled; matched CatBoost, ridge, training-mean; `tables/evaluation_protocols.csv`)
| protocol | model | splits | mean n test | R² t1 | R² t2 | R² t2+E (earlier representation) | MAE t2+E |
|---|---|---|---|---|---|---|---|
| DOI-grouped | catboost | 5 | 8428 | 0.368 ± 0.024 | 0.408 ± 0.032 | 0.407 ± 0.031 | 2.96 |
| DOI-grouped | ridge | 5 | 8428 | 0.307 ± 0.030 | 0.354 ± 0.032 | 0.350 ± 0.033 | 3.14 |
| DOI-grouped | mean | 5 | 8428 | -0.001 ± 0.001 | -0.001 ± 0.001 | -0.001 ± 0.001 | 4.08 |
| record-level random | catboost | 5 | 8222 | 0.587 ± 0.009 | 0.618 ± 0.006 | 0.627 ± 0.006 | 2.20 |
| record-level random | ridge | 5 | 8222 | 0.363 ± 0.008 | 0.408 ± 0.008 | 0.408 ± 0.008 | 2.95 |
| record-level random | mean | 5 | 8222 | -0.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 | 4.06 |
| lead-author-grouped | catboost | 5 | 8280 | 0.363 ± 0.019 | 0.394 ± 0.018 | 0.395 ± 0.013 | 2.92 |
| lead-author-grouped | ridge | 5 | 8280 | 0.313 ± 0.011 | 0.353 ± 0.017 | 0.350 ± 0.018 | 3.05 |
| lead-author-grouped | mean | 5 | 8280 | -0.002 ± 0.002 | -0.002 ± 0.002 | -0.002 ± 0.002 | 3.98 |
| forward-in-time (3 windows) | catboost | 3 | 6436 | 0.293 ± 0.059 | 0.345 ± 0.049 | 0.335 ± 0.058 | 3.21 |
| forward-in-time (3 windows) | ridge | 3 | 6436 | 0.227 ± 0.075 | 0.286 ± 0.087 | 0.283 ± 0.091 | 3.34 |
| forward-in-time (3 windows) | mean | 3 | 6436 | -0.155 ± 0.027 | -0.155 ± 0.027 | -0.155 ± 0.027 | 4.57 |

* Record-level vs DOI-grouped gap (R² difference, CatBoost): t1 0.219, t2 0.210, t2+E 0.221; ridge: 0.056, 0.055, 0.058. Record-level CatBoost for the pooled data = converged 20,000-iteration rerun (best iterations 7,207–11,702; the 6,000-cap run was 0.005–0.007 lower; the original 1,500-cap run truncated 31/45 fits). Record-level p-i-n/n-i-p values come from the 6,000-cap rerun. `tables/evaluation_protocols_gap_ablation_v1.csv` holds the gap computed inside `ablation_v1` (uses the truncated record-level fits — different from the converged values above; see section I).
* Lead-author grouping uses the modal PDP lead-author surname per publication (coarse laboratory proxy; large surnames over-group). Forward-in-time: publications dated by earliest PDP record; three windows with quantiles (0.55,0.70], (0.70,0.85], (0.85,1.00]; train strictly earlier, most recent 15% of train as validation; zero DOI/author overlap.

Forward windows (pooled CatBoost, `strict_validation_v1`):

| window | tier | train until | test from | n train | n test | R² | MAE |
|---|---|---|---|---|---|---|---|
| 1 | t1 | 2017-12-26 | 2018-05-02 | 18529 | 6502 | 0.284 | 3.31 |
| 2 | t1 | 2018-07-20 | 2018-12-12 | 24074 | 6322 | 0.356 | 3.23 |
| 3 | t1 | 2019-01-27 | 2019-07-19 | 29429 | 6484 | 0.240 | 3.48 |
| 1 | t2 | 2017-12-26 | 2018-05-02 | 18529 | 6502 | 0.335 | 3.17 |
| 2 | t2 | 2018-07-20 | 2018-12-12 | 24074 | 6322 | 0.398 | 3.08 |
| 3 | t2 | 2019-01-27 | 2019-07-19 | 29429 | 6484 | 0.302 | 3.30 |
| 1 | t2+E | 2017-12-26 | 2018-05-02 | 18529 | 6502 | 0.329 | 3.19 |
| 2 | t2+E | 2018-07-20 | 2018-12-12 | 24074 | 6322 | 0.396 | 3.08 |
| 3 | t2+E | 2019-01-27 | 2019-07-19 | 29429 | 6484 | 0.279 | 3.34 |

### G.4 Reference models (pooled, 5 DOI-grouped holdouts; `results/baselines_v1`, `ablation_v1`)
| model | tier (t2+E = t2 + earlier representation E) | R² | MAE | RMSE |
|---|---|---|---|---|
| gpr_sub | t1 | 0.330 ± 0.024 | 3.197 ± 0.039 | 4.025 ± 0.053 |
| gpr_sub | t2+E | 0.352 ± 0.029 | 3.129 ± 0.051 | 3.957 ± 0.065 |
| mlp | t1 | 0.343 ± 0.020 | 3.144 ± 0.029 | 3.987 ± 0.025 |
| mlp | t2+E | 0.376 ± 0.023 | 3.042 ± 0.052 | 3.883 ± 0.056 |
| rf | t1 | 0.295 ± 0.018 | 3.219 ± 0.041 | 4.128 ± 0.058 |
| rf | t2+E | 0.352 ± 0.021 | 3.073 ± 0.049 | 3.958 ± 0.070 |
| xgb_default | t1 | 0.334 ± 0.031 | 3.148 ± 0.052 | 4.011 ± 0.070 |
| xgb_default | t2+E | 0.377 ± 0.025 | 3.011 ± 0.057 | 3.880 ± 0.056 |
| catboost (matched) | t1 | 0.368 ± 0.024 | 3.081 ± 0.040 | 3.908 ± 0.046 |
| catboost (matched) | t2+E | 0.407 ± 0.031 | 2.958 ± 0.080 | 3.787 ± 0.083 |
| mean | t1 | -0.001 ± 0.001 | 4.079 ± 0.036 | 4.920 ± 0.047 |
| mean | t2+E | -0.001 ± 0.001 | 4.079 ± 0.036 | 4.920 ± 0.047 |
| ridge | t1 | 0.307 ± 0.030 | 3.271 ± 0.039 | 4.093 ± 0.064 |
| ridge | t2+E | 0.350 ± 0.033 | 3.141 ± 0.054 | 3.963 ± 0.078 |

XGBoost defaults: `tree_method=hist`, no tuning/early stopping; RF 300 trees; MLP two hidden layers with validation early stopping; GPR exact on a 3,000-record random training subsample. These are **not** re-implementations of the Roberts/Fukasawa pipelines.

### G.5 Prediction distribution and high-PCE error (recomputed from `results/ablation_v1/preds/doi_overall_t3_catboost_*.npz`, matched CatBoost, earlier representation E)
split 1: n=8224, records with PCE ≥ 20%: 272, mean(pred − exp) for those = -5.25 pp, max prediction 21.1%, sd(pred)/sd(exp) = 0.62; five splits stacked: n=42142, ≥20%: 1328, mean(pred − exp) = -4.74 pp; negative predictions: 6 of 42142 (min -1.34%).
Squared-error training regresses to the mean; predictions saturate near 21%.

### G.6 Material-property vs material-identity control
See section D table (identity-only t1_id R² 0.4062 vs t2 0.4073 vs t1 0.3698; t2 − t1_id = +0.0008 CI includes zero) — team interpretation: the t2 gain over t1 reflects knowing *which* materials are present; transferable physical information is not demonstrated (leave-material-out test not run).

### G.7 SHAP (`scripts/shap_analysis.py`, matched CatBoost trained on t2+E (earlier representation E, not t3) on DOI-grouped split 1; TreeExplainer on 1,500 random test records; attribution, not benefit)
| model | group | features | Σ mean|SHAP| | share |
|---|---|---|---|---|
| overall | composition | 47 | 0.988 | 14.7% |
| overall | device | 9 | 0.535 | 8.0% |
| overall | literature | 44 | 0.271 | 4.0% |
| overall | material | 102 | 2.532 | 37.7% |
| overall | process | 21 | 2.393 | 35.6% |
| pin | composition | 47 | 0.802 | 14.7% |
| pin | device | 9 | 0.310 | 5.7% |
| pin | literature | 44 | 0.210 | 3.9% |
| pin | material | 102 | 2.078 | 38.1% |
| pin | process | 21 | 2.056 | 37.7% |
| nip | composition | 47 | 0.877 | 13.5% |
| nip | device | 9 | 0.484 | 7.4% |
| nip | literature | 44 | 0.233 | 3.6% |
| nip | material | 102 | 2.161 | 33.2% |
| nip | process | 21 | 2.756 | 42.3% |

| model | top 8 features by mean |SHAP| (PCE points) |
|---|---|
| overall | Perov_quenching_crystallisation 0.727; total_ion_count 0.265; Perovskite_pl_max_nm_na 0.229; HTL_deposition_method 0.212; Perov_anneal_time_min 0.202; Perovskite_avg_atomic_mass 0.197; Perovskite_avg_radius 0.192; HTL_band_gap 0.186 |
| pin | Perov_quenching_crystallisation 0.525; HTL_band_gap 0.506; ETL_deposition_method 0.410; Perovskite_thickness_nm 0.234; HTL_LUMO 0.229; b_Pb_fraction 0.168; Backcontact_thickness_nm_na 0.144; HTL_deposition_method 0.142 |
| nip | Perov_quenching_crystallisation 1.002; HTL_deposition_method 0.303; Perovskite_avg_radius 0.285; Perovskite_avg_atomic_mass 0.220; ETL_deposition_method 0.202; Perovskite_deposition_method 0.195; Perovskite_pl_max_nm_na 0.158; Backcontact_thickness_nm_na 0.149 |

### G.8 Prior PDP-based studies (as verified against the PDFs in `../Lit/`, `docs/LIT_VERIFICATION.md`)
| study | target | model | split | R² | RMSE | MAE | units / note |
|---|---|---|---|---|---|---|---|
| Roberts et al., Adv. Phys. Res. 3, 2400060 (2024) | Jsc (mA cm-2) | XGBoost | 80/10/10 train/validation/test; duplicates averaged (pseudo-target) | 0.3500 | 3.58 | NA | mA cm-2 (RMSE); NOT comparable with PCE metrics |
| Fukasawa et al., Energy Adv. 3, 812 (2024) | PCE | random forest (best model) | 8:2 train/test (no grouping by publication described); 25% of training used for validation | 0.7177 | NA | NA | R2 as reported (Table 2, zero-complement) |
| Fukasawa et al., Energy Adv. 3, 812 (2024) | PCE | random forest (best model) | 294 newly registered records (as of 24 Aug 2023) | 0.4500 | NA | 3.20 | PCE percent (MAE as reported) |
| Hussain et al., RSC Adv. (2023) | PCE | nan | not stated in the article | NA | NA | NA |  |

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
| V4 development diagnostics | `results/v4/{rec1,pen_,pen_05,pen_20,recctl,recabl,rec2,blocks1,full_rec}.csv`, `docs/V4_REPORT.md` | v4dev; DOI-grouped 5-fold × 3 seeds |
| Original ablation (section D) | `results/ablation_v1/{runs.jsonl,summary.csv,paired_*}`, `configs/ablation.json` | 5 DOI-grouped holdouts |
| Feature-group controls | `results/feature_groups_v1/runs.jsonl`, `configs/ablation_groups.json` | same 5 holdouts |
| Model development (section E) | `results/model_dev_v1/runs.jsonl`, `scripts/run_model_dev.py`, `configs/model_dev.json` | same 5 holdouts |
| Protocol comparisons (G.1–G.3) | `ablation_v1`, `ablation_random_cap6000` (6k iters), `random_convergence_cap20000` (pooled converged), `strict_validation_v1`; `psc/results.py: load_ablation` | see section G |
| Reference models | `results/baselines_v1/runs.jsonl`, `configs/baselines.json` | same 5 holdouts |
| SHAP | `results/shap/*.csv`, `scripts/shap_analysis.py` | split 1, 1,500 test records |
| Identity / E2–E6 / lx2 / lx3 / qty controls | `results/identity_vs_t1id.csv`, `ablE_vs_t2_*.csv`, `lx_*_summary.csv`, `lx_dev_vs_t2.csv`, `qty_dev_summary.csv`, `qty_holdout/summary.json`, docs `IDENTITY_CONTROL.md`, `LX_RESULTS.md`, `LX3_RESULTS.md` | DOI-grouped 5-fold or qdev/qconf; see CSV `partition` column |
| Extraction validation, coverage, ICC | `results/extraction_validation.json`, `results/feature_audit.json`, `results/coverage_mask_audit.json` | earlier block (validation); both masks (audit) |
| V4 common-split t1/t2/t3 (C.0) | `results/v4/final_t1/{metrics_3tier.csv,comparisons_3tier.json}`, `scripts/v4_t1_baseline.py` | v4final reserved partition; trained on v4dev; matched default CatBoost; 5 seed models |
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
