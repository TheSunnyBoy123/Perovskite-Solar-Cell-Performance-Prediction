# Pre-specified confirmation analysis for the schema-lx2.0 literature block (written before any `confirm`-partition run)

* Partitions: publications are assigned to `dev` / `confirm` by SHA-256 of the normalised DOI (`psc.data.doi_partition`), independent of data and outcome. All design choices (blocks, stage-2 hyper-parameters, lambda grid) were made on `dev` only.
* Dev log (3 stage-2 settings tried, 2 blocks): default stage 2 (depth 4, lr .03, l2 10, 600 it) with block `x` (= proc+chem+ctx) was selected: dR2 vs matched controls +0.0022 (5/5 seeds, 19/25 folds, 95% CI includes 0). Block `chem` alone: no effect. Variants A (depth 3, l2 30) and B (depth 6, l2 3): weaker (+0.0016, +0.0013).
* Confirmation: the single pre-specified configuration (`configs/lx_stack_confirm_s*.json`, block `x`, default stage 2, subset `lx_covered`, 5 disjoint grouped folds x seeds 42-46, same code as dev) is run once on the `confirm` partition.
* Primary statistic: Delta MSE = mean MSE of matched DOI-level controls (3 permuted-real + 3 Gaussian) minus MSE of the real block, pooled out-of-fold, DOI-cluster bootstrap 95% CI, per seed and averaged.
* Decision rule: "reproducible incremental gain" only if the mean Delta is > 0 with the CI excluding 0 AND >= 4/5 seeds positive AND not driven by a few publications (leave-top-k-publications-out check). Otherwise the result is reported as not confirmed, however the point estimate looks.
* No further tuning after the confirm run. Any later extra analysis is labelled exploratory.
