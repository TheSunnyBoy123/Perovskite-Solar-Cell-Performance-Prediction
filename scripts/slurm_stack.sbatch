#!/bin/bash
#SBATCH -J lxstack
#SBATCH -p small
#SBATCH -c 48
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm_%x_%A_%a.out
# usage: sbatch --array=42-46 --export=PREFIX=lx_stack_confirm_s scripts/slurm_stack.sbatch
cd "${SLURM_SUBMIT_DIR}" || exit 1
module load hpc/python/3.13 2>/dev/null
~/venvs/psc/bin/python scripts/run_stack.py --config configs/${PREFIX}${SLURM_ARRAY_TASK_ID}.json
