# Schema-lx2.0 literature block: results so far (covered-DOI subset, stage-2 residual design)

Setup: records of publications that have extracted JSON (36.6% of records, 2,731 DOIs), split into `dev` / `confirm` by DOI hash. Stage 1 = matched t2 CatBoost; stage 2 = regularised CatBoost on the block `x` (process + chemistry + context, 61 columns) fitted to DOI-grouped inner-OOF residuals; lambda chosen on a DOI-disjoint validation set. Reference = mean of 6 matched DOI-level controls (3 permuted-real, 3 Gaussian) passed through the identical stage 2 (the stage-2 model also absorbs an intercept recalibration, so t2 itself is not the right reference). Disjoint grouped 5-fold, 5 seeds, DOI-cluster bootstrap CI.

| Partition | role | records / DOIs | dR2 vs all controls | seeds > 0 | folds > 0 | 95% CI (dMSE, avg over seeds) |
|---|---|---|---|---|---|---|
| dev | design (3 stage-2 settings x 2 blocks tried) | 7,640 / 1,368 | +0.0022 | 5/5 | 19/25 | [-0.036, 0.141] |
| confirm | locked, single pre-specified config (PREREG_LX.md) | 7,387 / 1,363 | +0.0011 | 4/5 | 19/25 | [-0.079, 0.137] |

Against permuted-real controls only, confirm is +0.0006 (2/5 seeds); against Gaussian controls +0.0016 (4/5).

Reading: the sign is the same on both partitions (9/10 seeds, 38/50 folds positive), the magnitude is ~0.001-0.002 R2, and the pre-registered rule (CI excludes 0) is NOT met. The defensible statement is "small, same-signed, not statistically established". Block `chem` alone showed no effect on dev. Single-stage CatBoost with the block appended does not help (dev: dR2 -0.002 to +0.004 vs t2, noise block -0.03).
Not done: leave-top-publications-out check (only meaningful if the primary rule were met), pin/nip subsets, device-level features (needs full text; legal OA retrieval reaches ~25% of DOIs).

## Robustness (scripts/lx_robust.py; seed-averaged paired difference, DOI-cluster bootstrap)
| | dR2 (seed-avg) | 95% CI dR2 | per-seed dR2 | nip dR2 | pin dR2 |
|---|---|---|---|---|---|
| dev | +0.0022 | [-0.0003, +0.0051] | +.0029 +.0027 +.0012 +.0035 +.0004 | +0.0023 | +0.0019 |
| confirm | +0.0011 | [-0.0015, +0.0036] | +.0041 +.0025 +.0008 -.0019 +.0002 | +0.0010 | +0.0017 |

* Same sign in both architectures and both partitions; CIs include 0 in all cases (pre-registered criterion not met).
* The gain is concentrated: the top 5% of publications account for >100% of the total gain on both partitions (245% dev, 523% confirm); removing the 10 most-helped publications leaves ~0 (dev +0.0006, confirm -0.018 dMSE) and removing 25 turns it negative (confirm CI excludes 0 on the negative side). Descriptive, because removing the best publications is a selection, but it shows the effect is not a broad, stable one.
* Conclusion: a small positive, same-signed average effect exists in this design, but it is driven by a minority of publications and is not statistically distinguishable from zero. It must be reported as such; it cannot be a principal claim of the paper.

## Additional dev-only variant (not adopted)
Paper-level stage 2 (fit on per-publication mean residuals, weights capped at 20 records; `stage2_mode: paper`, `configs/lx_stackP_dev_s*.json`): dev dR2 +0.0013 (CI [-0.0011, +0.0039]), per-seed +.0030 +.0034 +.0005 -.0002 -.0003, leave-top-25 negative. Weaker and less consistent than the record-level stage 2 (+0.0022, 5/5 seeds), so it was not adopted and was not run on `confirm`. Total dev design evaluations: 4 stage-2 settings x blocks {chem, x}; the confirm partition was scored once.

**Decision (user, 2026-10-14):** the literature-block result is accepted as "small, same-signed, not statistically established"; it is reported with the concentration caveat and is not a principal claim.

## Full-data single-stage ablation E2-E6 (results/ablE_kfold_s42-44; cluster job 788203)
Matched CatBoost, disjoint grouped 5-fold x 3 seeds, 41,107 records / 7,327 DOIs. dR2 = tier - t2 (t2 R2: overall .4073, n-i-p .4246, p-i-n .3218). All 95% CIs (DOI-cluster bootstrap on dMSE) include 0 except the Gaussian-noise controls overall/n-i-p, which are significantly worse than t2.

| tier | overall | n-i-p | p-i-n |
|---|---|---|---|
| E2 t3 (original 44 lit cols) | -0.0024 | -0.0007 | -0.0002 |
| E3 t3x (new block, proc+chem+ctx) | -0.0019 | +0.0013 | -0.0051 |
| E4 t3b (original + new) | -0.0039 | -0.0001 | -0.0050 |
| E5 t3x_meta (completeness only) | -0.0059 | -0.0016 | -0.0049 |
| E6 DOI-permuted real block (3 seeds, range) | -0.0023 .. -0.0032 | -0.0021 .. +0.0002 | -0.0028 .. -0.0039 |
| E6 DOI-level Gaussian, same mask (range) | -0.0059 .. -0.0074 | -0.0071 .. -0.0104 | -0.0083 .. -0.0103 |

Reading: appended to a single-stage model, no literature block improves on t2 (this repeats the original finding for t3). The new block (E3) is the least harmful real block overall, better than the original block (E2) overall and in n-i-p, worse in p-i-n, and better than its DOI-permuted control by only 0.0004-0.0013 R2 overall. Completeness-only features (E5) are the worst real block, so the new block's small advantage is not a reporting-style artefact. The two-stage design above is where the small same-signed gain appears (covered subset only); the single-stage full-data result does not support any claim of an LLM-feature improvement over t2.
