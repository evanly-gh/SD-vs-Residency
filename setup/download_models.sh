#!/usr/bin/env bash
# Download Qwen3 GGUF model files from Hugging Face.
#
# WHAT THIS DOES
# ──────────────
# Uses the `hf` CLI (from the `huggingface_hub` Python package) to pull
# specific GGUF files from bartowski's quantized repos on Hugging Face.
# NOTE: `huggingface-cli` is deprecated; the current command is `hf`.
# Files land in the repo's models/ directory, which is gitignored.
#
# STORAGE REQUIREMENTS
# ────────────────────
#   Qwen3-0.6B  Q4_K_M  ~  400 MB  (draft model — always needed)
#   Qwen3-14B   Q4_K_M  ~  8.5 GB  (VRAM-resident target)
#   Qwen3-32B   Q4_K_M  ~ 19.0 GB  (CPU-offload target — optional, download last)
#   ─────────────────────────────────
#   Total                ~ 28 GB
#
# RECOMMENDATION FOR 12 GB VRAM (YOUR SETUP)
# ────────────────────────────────────────────
# Start by downloading only the 0.6B and 14B (~9 GB total). Run the full
# pipeline on those first to validate your build and confirm the 14B fits
# in VRAM alongside the draft model. Only then download the 32B for the
# CPU-offload sweep.
#
# USAGE
# ─────
#   bash setup/download_models.sh            # downloads 0.6B + 14B
#   bash setup/download_models.sh --all      # also downloads 32B (19 GB extra)
#   bash setup/download_models.sh --32b      # only downloads 32B

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
mkdir -p "${MODELS_DIR}"

if ! command -v hf &>/dev/null; then
    echo "==> hf CLI not found. Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

download_model() {
    local repo="$1"
    local filename="$2"
    local dest="${MODELS_DIR}/${filename}"

    if [[ -f "${dest}" ]]; then
        echo "  Already exists: ${filename} — skipping."
        return
    fi

    echo "==> Downloading ${filename} from ${repo}..."
    echo "    (This may take a while on slow connections.)"
    hf download "${repo}" "${filename}" --local-dir "${MODELS_DIR}"
    echo "    Done: ${dest}"
}

DO_32B=false
DO_ONLY_32B=false

for arg in "$@"; do
    case "${arg}" in
        --all) DO_32B=true ;;
        --32b) DO_32B=true; DO_ONLY_32B=true ;;
    esac
done

if [[ "${DO_ONLY_32B}" == false ]]; then
    echo ""
    echo "── Draft model (Qwen3-0.6B, ~480 MB) ───────────────────────────────────"
    download_model "bartowski/Qwen_Qwen3-0.6B-GGUF" "Qwen_Qwen3-0.6B-Q4_K_M.gguf"

    echo ""
    echo "── Target model: VRAM-resident (Qwen3-14B, ~9 GB) ──────────────────────"
    download_model "bartowski/Qwen_Qwen3-14B-GGUF" "Qwen_Qwen3-14B-Q4_K_M.gguf"
fi

if [[ "${DO_32B}" == true ]]; then
    echo ""
    echo "── Target model: CPU-offload condition (Qwen3-32B, ~20 GB) ─────────────"
    echo "    NOTE: This is a 20 GB download. Make sure you have ~21 GB free disk."
    download_model "bartowski/Qwen_Qwen3-32B-GGUF" "Qwen_Qwen3-32B-Q4_K_M.gguf"
fi

echo ""
echo "==> Downloaded models:"
ls -lh "${MODELS_DIR}"/*.gguf 2>/dev/null || echo "  (no .gguf files found)"

echo ""
echo "==> Next: run setup/verify_setup.sh to confirm vocab alignment and GPU loading."
