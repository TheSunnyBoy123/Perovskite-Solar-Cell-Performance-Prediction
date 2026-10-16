#!/bin/bash
# second queue: convergence check, then full clean-environment reproduction (venv path given as $1)
cd "$(dirname "$0")/.."
while pgrep -f "run_experiment.py --config configs/strict_validation.json" >/dev/null; do sleep 20; done
python3 scripts/run_experiment.py --config configs/random_convergence.json
${1:-python3} scripts/run_experiment.py --config configs/repro_cleanenv.json
