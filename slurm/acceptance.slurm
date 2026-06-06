#!/usr/bin/env bash
# Phase 2b — acceptance rates by task type × thinking mode.
# Runs llama-speculative against the curated prompt sets.
#
# Submit:
#   sbatch slurm/acceptance.slurm

#SBATCH --job-name=spec-accept
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=data/logs/slurm-%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/_env.sh"

bash scripts/run_acceptance.sh
