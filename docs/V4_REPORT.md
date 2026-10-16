# lx4 / t3 model development: report

## Bottom line
A richer LLM-derived block (36 recipe-identity flags, `data/lit_v4.json`) gives a **small, consistently positive but not statistically established** gain of t3 over t2 on unseen publications. It does **not** meet the "reproducible improvement over a fair optimized t2" bar with a CI excluding zero. No gain is claimed beyond what is below.

## Protocol
* New DOI-hash partition (salt `psc-v4`): `v4dev` 65% of publications (all development, tuning, ablations), `v4final` 35% scored once (`scripts/v4_final.py`). v4final DOIs were seen in earlier experiments with other feature sets, so it is fresh for this representation, not untouched data.
* Dev: DOI-grouped 5-fold, 3 seeds. Final: train on all v4dev (DOI-disjoint 15% validation for early stopping), 5 seed models averaged, publication-cluster bootstrap (2000) of paired squared error.
* t2 fixed (`psc.features.base_frame(df,'t2')`). t3 = t2 + `rc_*` recipe flags (NaN for papers without extraction, CatBoost first-feature-use penalty 0.5 on those columns).

## Diagnostics that shaped the design (dev only)
* Existing JSON content is nearly exhausted: token bags +0.004 R2; numeric quantities, device-matched layer values, PDP gap-filling: ~0 or negative. 51 numeric field families screened vs t2 residual: none survive multiplicity.
* Ceiling: giving t2 the PDP's own curated process columns for the same papers adds only +0.022 R2 (additives +0.012, solvents/quench +0.003). The additive/recipe identities are therefore what the LLM block targets.
* Recipe block: +0.007 (covered dev, 12/15 folds); five DOI-permuted copies -0.009..+0.002. Dropping any one flag group cuts the gain, and no single group works alone (distributed, near noise floor). A role-scoped variant gave +0.001, so variant differences are within dev noise.
* Full dev data under equal tuning (24 shared CatBoost configs): t3 beats t2 in only 6/24 configs.

## Final results on v4final (dR2 = t3 minus t2, positive = t3 better; CI = publication-cluster bootstrap)
| comparison | covered (5,460 rec / 1,005 DOI) | full (14,125 rec / 2,619 DOI) |
|---|---|---|
| A: same default config | +0.0071 [-0.0007,+0.0157], 4/5 seeds | +0.0025 [-0.0011,+0.0062], 4/5 |
| A: both under t3's tuned config | +0.0066 [-0.0021,+0.0162] | +0.0026 [-0.0015,+0.0067] |
| A: both under t2's tuned config | +0.0024 [-0.0077,+0.0131] | -0.0016 [-0.0064,+0.0030] |
| B: each with its own tuned config | +0.0094 [-0.0016,+0.0219], 5/5 | +0.0015 [-0.0039,+0.0067], 3/5 |
| 5 permuted-block controls vs t2 | -0.0049..+0.0015 | -0.0028..+0.0007 |
| coverage-flag-only control | +0.0015 | +0.0011 |
Absolute covered R2: t2 tuned 0.361, t3 tuned 0.371. Full: both about 0.40. Tuning alone moves t2 by about +0.002 to +0.003, similar to the t3 effect, so tuned results should not be read as an LLM gain by themselves. Metrics for every arm: `results/v4/final/metrics.csv`, comparisons: `comparisons.json`.

## Reading
Direction replicates dev (+0.007 covered both times), beats all permuted and coverage controls, but the intervals include zero and the gain depends on the estimator config (it shrinks under t2's tuned config). Full-data benefit is diluted by 37% coverage. Larger signal would need full-text re-extraction of the ~4,500 uncovered publications (not done: no local full texts, and sending texts to an external LLM was not approved).

## Limitations
Recipe flags are paper-level regex over LLM JSON (noisy, not device-specific); the PDP itself already holds curated additive columns outside the restricted t1/t2 definition; one reserved split of about 1,000 covered publications gives roughly +/-0.01 R2 resolution.

## Artifacts
* Features: `data/lit_v4.json` (schema lx4.0; `scripts/build_lit_v4.py`, `src/psc/litfeat4.py`, `src/psc/v4.py`)
* Pipeline: `scripts/v4_dev.py` (dev comparisons), `v4_tune.py` (-> `results/v4/tune.json`), `v4_final.py` (-> `results/v4/final/`), `v4_save_bundle.py`
* Models (fit on all data, 5 seeds): `models/v4_t3_bundle.pkl`, `models/v4_t2_bundle.pkl`; tuned configs `results/v4/final/tuned_cfg.json`
* Inference (no LLM/network): `python scripts/predict.py --input devices.csv --out preds.csv [--model models/v4_t2_bundle.pkl]`
