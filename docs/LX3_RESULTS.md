# Quantity-level literature features (schema lx3.0): results
Code: src/psc/litfeat3.py, scripts/build_lit_v3.py -> data/lit_v3.csv (39 cols, 2,835 papers; most quantity cols cover 2-15% of papers). Tiers: t3q (t3x+lx3), t3q_new, t3q_qty, t3q_cat (9 chemistry descriptors), t3q_nochem, t1_t3q.
Runs: results/lx3_dev_kfold_s42-44 (covered dev), lx3_full_kfold_s42-44 (full data), lx3c_{dev,confirm}_kfold_s42-46 (cat subset + matched controls). Analysis: scripts/lx_compare.py <prefix> --ref t2 [--partition dev|confirm].
Same CatBoost config/splits for all tiers; disjoint DOI-grouped 5-fold.

Covered dev (3 seeds), dR2 vs t2 (all CIs include 0): t1 -0.049, t3 -0.008, t3x -0.002, t3q -0.003, t3q_qty -0.003, t3q_new +0.006, t3q_cat +0.0075, t3q_nochem +0.003, t1+t3q -0.049 (vs t1 +0.0005).
Full data: t3 -0.0024, t3x/t3q -0.0019, t1+t3q vs t1 +0.0007 (material props +0.0375); Gaussian controls -0.005..-0.007.
t3q_cat: dev (5 seeds) +0.0066, CI dR2 [-0.0048,+0.0183], 5/5 seeds, 17/25 folds; matched controls -0.008..+0.003. Confirm (second look, NOT independent) +0.0018, [-0.0101,+0.0137], 3/5, 14/25; controls -0.012..+0.006.
t3q_cat was selected post hoc among ~9 dev tiers. Conclusion: small, same-signed, suggestive, not established; full-data null preserved.

---
# Quantity-covered experiment (exploratory; manuscript NOT revised yet)

## 1. What the earlier covered-subset runs tested
The 39 lx3.0 columns = 30 quantity (`lx_qty_*`) + 9 chemistry-descriptor (`lx_cat_*`). Under the 80%-missingness filter, on the covered dev records only 2 of the 30 quantity columns survived (`n_add_species`, `contact_thick_nm`). The earlier t3q / t3q_qty runs therefore did not test the numerical features; only `t3q_cat` (no missingness issue) was a real test of the chemistry block.

## 2. Population (defined from availability only, before any model result)
Eligible = covered publication with >=1 non-missing non-back-contact `lx_qty_*` value: 9,441 records / 1,680 DOIs (of 15,027 / 2,731 covered; median 4 records per DOI). Predeclared quantity features = eligible-paper coverage >=10%: 15 columns (list in `data/lx3_qty_selected.json`). Reproduce: `scripts/qty_population.py`.
Fresh partition for this representation: SHA-256 DOI hash (salt `psc-qty-v1`), 70/30 -> `qdev` 6,677 records / 1,180 DOIs, `qconf` 2,764 / 500, reserved and scored once. Label: the DOIs were seen in earlier experiments (other feature sets), so `qconf` is fresh for this representation, not an untouched dataset; results are exploratory.
Preprocessing: the 15 predeclared columns bypass the 80% filter (train-median imputation + `_na` flag, fit on training data only), all other columns follow the unchanged rule; they genuinely enter the matrix (`n_qty_used` = 15 in every t3qq/t3_qq/control run). Tiers: t3qq = t2 + 15 qty; t3qq_cat = + 9 chemistry descriptors; t3_qq = t3 (original 32) + 15 qty; controls: DOI-level permuted / Gaussian versions of the same 15 columns (3 each). Same CatBoost config, DOI-disjoint validation early stopping, all tiers.

## 3. Protocols
A. Development: disjoint DOI-grouped 5-fold within `qdev`, 5 seeds (results/qty_dev_kfold_s42-46). B. Held-out: train on all `qdev` (DOI-disjoint 15% validation for early stopping, per seed), score on `qconf` (results/qty_holdout; `scripts/run_qty_holdout.py`, `scripts/qty_holdout_compare.py`). CI: publication-cluster bootstrap (2000) of the seed-averaged paired squared-error difference, converted to R2.

## 4. Results (dR2 vs t2, same records)
| tier | dev R2 | dev MAE | dev RMSE | n feat (matrix) | dev dR2 [95% CI] | seeds>0 / folds>0 | qconf R2 | qconf dR2 [95% CI] | seeds>0 |
|---|---|---|---|---|---|---|---|---|---|
| t1 | 0.2757 | 3.190 | 4.019 | 69 | -0.041 [-0.069,-0.011] | 0/5, 2/25 | 0.334 | -0.039 [-0.068,-0.009] | 0/5 |
| t2 | 0.3171 | 3.065 | 3.902 | 171 | ref | | 0.372 | ref | |
| t3 (orig 32) | 0.3282 | 3.042 | 3.872 | 211 | +0.0107 [-0.007,+0.029] | 5/5, 19/25 | 0.358 | -0.0141 [-0.038,+0.006] | 0/5 |
| t3qq | 0.3277 | 3.043 | 3.872 | 201 | +0.0104 [-0.008,+0.031] | 4/5, 18/25 | 0.375 | +0.0026 [-0.010,+0.014] | 3/5 |
| t3qq_cat | 0.3359 | 3.016 | 3.848 | 219 | +0.0185 [-0.002,+0.041] | 5/5, 20/25 | 0.376 | +0.0035 [-0.013,+0.020] | 3/5 |
| t3_qq | 0.3339 | 3.030 | 3.855 | 241 | +0.0165 [-0.005,+0.042] | 5/5, 19/25 | 0.360 | -0.0123 [-0.042,+0.013] | 0/5 |
| controls (6, perm/noise of the 15 qty) | | | | | -0.002 .. +0.005 | | | -0.010 .. +0.003 | |
Dev CIs: publication-cluster bootstrap of the seed-averaged dMSE (scripts/lx_compare.py), divided by the qdev target variance (22.52) to give dR2; dev R2/MAE/RMSE columns are means over all 25 seed x fold cells.
qconf: 2,764 records / 500 publications; dev test sets 1,258-1,335 records / ~236 publications per fold.

## 5. Reading
* Numerical features do enter the model now; t3qq vs t2 is +0.010 on dev (4/5 seeds) but only +0.003 on `qconf` (3/5 seeds); all intervals include zero. Controls with the same availability mask fluctuate by -0.01..+0.005, so these gains are within control noise.
* The original t3 looks helpful on dev (+0.011, 5/5 seeds) but is -0.014 on `qconf` (0/5): the dev advantage does not replicate. This population is small (500 held-out publications), so the held-out estimate is itself noisy.
* t3_qq over t3 (does the quantity block add beyond the original features): dev +0.006, `qconf` +0.002 (R2 0.360 vs 0.358); intervals include zero.
* The chemistry descriptors (t3qq_cat) are again the best tier (dev +0.0185, `qconf` +0.0035) but remain within control noise on `qconf`, and that tier was chosen after earlier dev results.
Decision: suggestive at most; no established improvement of t3 over t2, even on papers where the numerical features are present. Population limits: availability-selected papers only; conclusions do not transfer to the full PDP dataset. Not done: manuscript revision (held per instruction).
