#!/usr/bin/env bash
# Phase 1 — non-speculative throughput baseline.
# Establishes the denominator for every speedup ratio.
#
# Submit:
#   sbatch slurm/baseline.slurm

#SBATCH --job-name=spec-baseline
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/slurm-%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/_env.sh"

bash scripts/run_baseline.sh
