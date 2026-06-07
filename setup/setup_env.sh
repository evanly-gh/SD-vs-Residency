#!/usr/bin/env bash
# Sets up the conda environment for the SD-vs-Residency experiment on Hyak Klone.
# Run once from a login node BEFORE submitting any SLURM jobs.
#
# Usage:
#   bash setup/setup_env.sh
#
# After this completes, download model weights (see bottom of script).

set -euo pipefail

GSCRATCH="/mmfs1/gscratch/intelligentsystems/evanly"
ENV_PREFIX="${GSCRATCH}/envs/sd-qwen35"
MODELS_DIR="${GSCRATCH}/models"
RESULTS_DIR="${GSCRATCH}/sd-qwen35/results"
LOGS_DIR="${GSCRATCH}/sd-qwen35/logs"
VLLM_VERSION="0.9.2"
PYTHON_VERSION="3.11"

echo "=== SD-vs-Residency Environment Setup ==="
echo "GSCRATCH:    $GSCRATCH"
echo "ENV_PREFIX:  $ENV_PREFIX"
echo "vLLM:        $VLLM_VERSION"
echo ""

# Create persistent directories
mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$LOGS_DIR"
echo "Created storage directories under $GSCRATCH/sd-qwen35/"

# Symlink results/ into the repo for convenience
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -L "${REPO_DIR}/results" ]; then
    ln -s "$RESULTS_DIR" "${REPO_DIR}/results"
    echo "Symlinked repo/results -> $RESULTS_DIR"
fi

# Load modules
module load cuda/12.4 2>/dev/null || echo "Warning: could not load cuda/12.4 — run on a compute node or check module name"

# Create conda environment in gscratch (not home dir, which has a 10GB quota)
if [ -d "$ENV_PREFIX" ]; then
    echo "Environment already exists at $ENV_PREFIX — skipping creation."
    echo "To reinstall, remove it first: rm -rf $ENV_PREFIX"
else
    echo "Creating conda environment at $ENV_PREFIX ..."
    conda create -y --prefix "$ENV_PREFIX" python="$PYTHON_VERSION"
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "Installing PyTorch (CUDA 12.4)..."
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

echo "Installing vLLM $VLLM_VERSION..."
pip install "vllm==${VLLM_VERSION}" --quiet

echo "Installing experiment dependencies..."
pip install \
    openai \
    requests \
    datasets \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    huggingface_hub \
    --quiet

echo ""
echo "=== Installed versions ==="
python -c "
import torch, vllm, openai, datasets, matplotlib, seaborn, pandas, numpy
print(f'torch:      {torch.__version__}')
print(f'cuda:       {torch.version.cuda}')
print(f'vllm:       {vllm.__version__}')
print(f'openai:     {openai.__version__}')
print(f'datasets:   {datasets.__version__}')
print(f'matplotlib: {matplotlib.__version__}')
print(f'seaborn:    {seaborn.__version__}')
print(f'pandas:     {pandas.__version__}')
print(f'numpy:      {numpy.__version__}')
"

echo ""
echo "=== Environment ready ==="
echo "Activate with: conda activate $ENV_PREFIX"
echo ""

# -----------------------------------------------------------------------
# Model weight downloads — uncomment and run separately (slow, run once).
# Use a tmux session on the login node; downloads can take 30-60 min each.
# -----------------------------------------------------------------------
echo "=== Model download commands (run manually in tmux) ==="
echo ""
echo "huggingface-cli download Qwen/Qwen3.5-0.8B  --local-dir ${MODELS_DIR}/Qwen3.5-0.8B"
echo "huggingface-cli download Qwen/Qwen3.5-2B    --local-dir ${MODELS_DIR}/Qwen3.5-2B"
echo "huggingface-cli download Qwen/Qwen3.5-4B    --local-dir ${MODELS_DIR}/Qwen3.5-4B"
echo "huggingface-cli download Qwen/Qwen3.5-9B    --local-dir ${MODELS_DIR}/Qwen3.5-9B"
echo "huggingface-cli download Qwen/Qwen3.5-27B   --local-dir ${MODELS_DIR}/Qwen3.5-27B"
echo "huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir ${MODELS_DIR}/Qwen3.5-35B-A3B"
echo ""
echo "After downloading, generate prompts:"
echo "  python data/generate_prompts.py"
echo ""
echo "Then run a smoke test:"
echo "  sbatch --export=ALL,SD_CONDITION=A3,SD_TARGET_MODEL=${MODELS_DIR}/Qwen3.5-27B,SD_DRAFT_MODEL=${MODELS_DIR}/Qwen3.5-4B,SD_MODE=standard,SD_PORT=8103,SD_OUTPUT_FILE=${RESULTS_DIR}/smoke_A3.json slurm/job_template.sh"
