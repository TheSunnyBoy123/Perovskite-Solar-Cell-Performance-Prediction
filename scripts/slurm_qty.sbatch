#!/bin/bash
#SBATCH -J psc_qty
#SBATCH -p small
#SBATCH -c 48
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm_%x_%A.out
cd "${SLURM_SUBMIT_DIR}" || exit 1
module load hpc/python/3.13 2>/dev/null
~/venvs/psc/bin/python scripts/run_qty_holdout.py
