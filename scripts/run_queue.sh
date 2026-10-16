#!/bin/bash
# Runs remaining experiments sequentially (resumable). Run from ml_paper/.
set -e
python3 scripts/run_experiment.py --config configs/ablation_groups.json
python3 scripts/run_experiment.py --config configs/baselines.json
python3 scripts/shap_analysis.py overall pin nip
