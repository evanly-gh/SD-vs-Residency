#!/usr/bin/env bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/sd-qwen35/logs/slurm-%x-%j.out
#SBATCH --error=/mmfs1/gscratch/intelligentsystems/evanly/sd-qwen35/logs/slurm-%x-%j.err
#
# Required environment variables (set via --export in submit_all.sh):
#   SD_CONDITION      e.g. "A3"
#   SD_TARGET_MODEL   local path e.g. "/mmfs1/.../models/Qwen3.5-27B"
#   SD_DRAFT_MODEL    local path, empty string for baseline
#   SD_MODE           "standard" or "thinking"
#   SD_PORT           unique port per condition (8100-8107)
#   SD_OUTPUT_FILE    e.g. "/mmfs1/.../results/A3.json"
#   SD_NUM_PROMPTS    number of prompts to run (default: all 150; set to 5 for smoke test)
#
# Usage (direct sbatch, for smoke testing):
#   sbatch --job-name=sd-A3 \
#     --export=ALL,SD_CONDITION=A3,...  \
#     slurm/job_template.sh

set -euo pipefail

GSCRATCH="/mmfs1/gscratch/intelligentsystems/evanly"
ENV_PREFIX="${GSCRATCH}/envs/sd-qwen35"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_NUM_PROMPTS="${SD_NUM_PROMPTS:-150}"

echo "=== SD Experiment: condition=${SD_CONDITION} mode=${SD_MODE} port=${SD_PORT} ==="
echo "Target:  ${SD_TARGET_MODEL}"
echo "Draft:   ${SD_DRAFT_MODEL:-<none, baseline>}"
echo "Output:  ${SD_OUTPUT_FILE}"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "GPUs:    ${CUDA_VISIBLE_DEVICES:-<not set>}"
date -u

# Load CUDA
module load cuda/12.4

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

# Suppress NCCL noise and use spawn for CUDA multiprocessing
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# Check port availability — exit early if another job on the same node holds it
if ss -tlnp 2>/dev/null | grep -q ":${SD_PORT} "; then
    echo "ERROR: Port ${SD_PORT} is already in use on $(hostname). Exiting."
    echo "Resubmit this job or pick a different port."
    exit 1
fi

mkdir -p "$(dirname "${SD_OUTPUT_FILE}")"

cd "${REPO_DIR}"

python run_experiment.py \
    --condition          "${SD_CONDITION}" \
    --target-model       "${SD_TARGET_MODEL}" \
    --draft-model        "${SD_DRAFT_MODEL}" \
    --mode               "${SD_MODE}" \
    --port               "${SD_PORT}" \
    --output             "${SD_OUTPUT_FILE}" \
    --prompts            "data/prompts.json" \
    --batch-sizes        1 4 8 16 \
    --num-spec-tokens    4 \
    --max-tokens         512 \
    --gpu-memory-util    0.90 \
    --num-prompts        "${SD_NUM_PROMPTS}" \
    --seed               42 \
    --logs-dir           "${GSCRATCH}/sd-qwen35/logs"

echo "=== Job complete: ${SD_CONDITION} ==="
date -u
