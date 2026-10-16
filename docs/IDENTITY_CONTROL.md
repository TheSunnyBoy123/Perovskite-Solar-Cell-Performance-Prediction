# Material-identity control (task 6.4) — full data, disjoint grouped 5-fold x seeds 42-44, matched CatBoost

| tier | pooled R2 | MAE | RMSE |
|---|---|---|---|
| t1 | 0.3696 | 3.051 | 3.892 |
| t1_id (t1 + ETL, HTL, Perovskite as native categorical) | 0.4062 | 2.940 | 3.778 |
| t2 (t1 + 102 material-property columns) | 0.4070 | 2.934 | 3.775 |
| t2_id | 0.4098 | 2.929 | 3.766 |

Paired, DOI-cluster bootstrap (41,107 records, 7,327 DOIs): t1_id - t1 = +0.0366 R2 (15/15 folds, CI excludes 0); t2 - t1_id = +0.0008 (CI [-0.102, +0.145] dMSE, 9/15 folds); t2_id - t1_id = +0.0036 (12/15 folds, CI [-0.007, +0.184] dMSE).

Interpretation (required by the task): the material-property block improves over t1 only because it tells the model WHICH transport/absorber materials are present. Identity alone reproduces essentially the full t2 gain; the physical descriptors add at most ~0.001-0.004 R2 and not distinguishably from zero. The manuscript's "material-property contribution" claim must be reworded to "knowing the transport materials helps"; transferable physical information is not demonstrated by this control (a leave-material-out test would be needed to show it).
Runs: results/identity_kfold_s42..44 (cluster job 788194), `scripts/lx_compare.py`.
