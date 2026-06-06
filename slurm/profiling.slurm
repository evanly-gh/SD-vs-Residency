#!/usr/bin/env bash
# Phase 3 — hardware profiling session for one (model, ngl, gamma, mode) tuple.
# Parameterized via --export; submit once per condition you want to profile.
#
# Submit:
#   sbatch --export=ALL,MODEL=32b,NGL=32,GAMMA=4,MODE=spec slurm/profiling.slurm
#   sbatch --export=ALL,MODEL=32b,NGL=32,GAMMA=4,MODE=base slurm/profiling.slurm
#
# Pick MODEL/NGL/GAMMA based on the crossover identified in the spec sweep —
# the script is not auto-submitted by submit_all.sh because that choice depends
# on inspecting Phase 2 results first.

#SBATCH --job-name=spec-prof
#SBATCH --account=intelligentsystems
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/slurm-%x-%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/_env.sh"

: "${MODEL:?set --export=MODEL=14b|32b}"
: "${NGL:?set --export=NGL=<int>}"
: "${GAMMA:?set --export=GAMMA=<int>}"
: "${MODE:=spec}"   # spec or base

bash scripts/run_profiling.sh "${MODEL}" "${NGL}" "${GAMMA}" "${MODE}"
