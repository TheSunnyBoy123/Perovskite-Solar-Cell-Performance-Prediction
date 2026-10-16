#!/bin/bash
# Full reviewer reproduction (run from ml_paper/). Produces, in order:
#  results/results_folds_{overall,pin,nip}.csv  (t1/t2/t3 x 5 DOI folds)
#  results/results_random_overall.csv            (matched record-level splits)
#  results/results_boost_{overall,pin,nip}.csv   (boosted t3: tuned+bagged CB+XGB)
#  results/results_nona_{overall,pin,nip}.csv    (Tier-3 minus _na indicators)
#  results/summary_tiers.csv                     (means/SD/folds, built by auditor)
#  models/, predictions/ (incl. oof_{arch}.csv pooled held-out tables)
#  figures/ (SHAP, OOF diagnostics, reviewer fig_*.pdf/png)
set -e
pip install -r pipeline/requirements.txt
for arch in overall pin nip; do
  python3 pipeline/train_tiers.py $arch all
done
python3 pipeline/train_random.py
for arch in overall pin nip; do
  python3 pipeline/train_boost.py $arch
done
for arch in overall pin nip; do
  python3 pipeline/oof_train.py $arch
  python3 pipeline/plot_oof.py $arch
  python3 pipeline/nna_train.py $arch
done
for arch in overall pin nip; do
  python3 pipeline/shap_final.py $arch
done
python3 pipeline/shap_composite.py
for fig in tiers diag shapgroups split; do
  python3 pipeline/make_figures.py $fig
done
mv fig_feature_tiers.pdf fig_feature_tiers.png \
   fig_oof_diagnostics.pdf fig_oof_diagnostics.png \
   fig_shap_groups.pdf fig_shap_groups.png \
   fig_split_effect.pdf fig_split_effect.png figures/
rm -rf catboost_info  # CatBoost telemetry (keeps dir free of regenerable logs)
