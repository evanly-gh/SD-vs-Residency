#!/usr/bin/env bash
# Downloads all Qwen3.5 weights needed for the 9 conditions, smoke-test pair first.
# Idempotent: huggingface-cli resumes partial/complete downloads.
#
# Usage:  bash setup/download_models.sh
set -uo pipefail

GSCRATCH="/mmfs1/gscratch/intelligentsystems/evanly"
MODELS_DIR="${GSCRATCH}/models"
mkdir -p "$MODELS_DIR"

# Smoke-test pair (4B draft + 27B target) FIRST, then the rest by ascending size.
MODELS=(
    Qwen3.5-4B          # smoke draft  (~8 GB)
    Qwen3.5-27B         # smoke target / Axis 1 target (~54 GB)
    Qwen3.5-0.8B        # A1 draft (~1.6 GB)
    Qwen3.5-2B          # A2 draft (~4 GB)
    Qwen3.5-9B          # A4 draft (~18 GB)
    Qwen3.5-35B-A3B     # B2/B3 + baseline_moe target (~70 GB)
)

for m in "${MODELS[@]}"; do
    dest="${MODELS_DIR}/${m}"
    echo "=== $(date -u +%H:%M:%S) Downloading Qwen/${m} -> ${dest} ==="
    huggingface-cli download "Qwen/${m}" --local-dir "${dest}" \
        --exclude "*.pth" "original/*" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "WARNING: download of Qwen/${m} exited with code $rc (continuing)"
    else
        echo "DONE: Qwen/${m}"
        touch "${dest}/.download_complete"
    fi
done
echo "=== All model downloads attempted: $(date -u) ==="
