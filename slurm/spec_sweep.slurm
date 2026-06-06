#!/usr/bin/env bash
# Phase 2 — full speculative decoding sweep (14B + 32B × γ × NGL).
# This is the long-running phase; expect several hours.
#
# Submit:
#   sbatch slurm/spec_sweep.slurm
# Or chained after baseline:
#   sbatch --dependency=afterok:<baseline_jobid> slurm/spec_sweep.slurm

#SBATCH --job-name=spec-sweep
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=data/logs/slurm-%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/_env.sh"

bash scripts/run_spec_sweep.sh
