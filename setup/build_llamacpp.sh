#!/usr/bin/env bash
# Build llama.cpp with CUDA support for NVIDIA RTX 6000.
# Run this script once from anywhere; it clones into a scratch directory by default.
#
# Prerequisites:
#   - NVIDIA driver + CUDA toolkit (nvcc in PATH)
#   - cmake >= 3.21, git, make, g++
#
# After building, llama-bench and llama-cli will be at:
#   ${LLAMA_CPP_DIR}/build/bin/llama-bench
#   ${LLAMA_CPP_DIR}/build/bin/llama-cli

set -euo pipefail

# Prefer scratch storage to avoid $HOME quota. Override with LLAMA_CPP_DIR.
DEFAULT_LLAMA_DIR="/gscratch/scrubbed/${USER}/llama.cpp"
LLAMA_DIR="${LLAMA_CPP_DIR:-${DEFAULT_LLAMA_DIR}}"
# Optional: set CUDA_ARCH to force a specific compute capability (e.g., 89 for Ada).
CUDA_ARCH="${CUDA_ARCH:-}"

echo "==> Checking CUDA availability..."
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Load a CUDA module or install the CUDA toolkit."
    echo "  Check: which nvcc"
    exit 1
fi
echo "  nvcc: $(which nvcc)"
echo "  CUDA: $(nvcc --version | grep -i release | head -1 || echo 'unknown')"

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
    exit 1
fi
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo 'unknown')"

echo "==> Cloning llama.cpp into ${LLAMA_DIR}..."
if [[ -d "${LLAMA_DIR}" ]]; then
    echo "  Directory exists. Pulling latest..."
    git -C "${LLAMA_DIR}" pull --ff-only
else
    git clone https://github.com/ggerganov/llama.cpp "${LLAMA_DIR}"
fi

if [[ -n "${LLAMA_CPP_COMMIT:-}" && "${LLAMA_CPP_COMMIT}" != "latest" ]]; then
    echo "  Checking out pinned commit: ${LLAMA_CPP_COMMIT:0:9}..."
    git -C "${LLAMA_DIR}" checkout "${LLAMA_CPP_COMMIT}"
fi

echo "==> Configuring CMake (CUDA backend)..."
CMAKE_ARGS=(
    -DGGML_CUDA=ON
    -DCMAKE_BUILD_TYPE=Release
    -DLLAMA_CURL=OFF
)
if [[ -n "${CUDA_ARCH}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
    echo "  Using CUDA arch: ${CUDA_ARCH}"
fi

cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" "${CMAKE_ARGS[@]}"

echo "==> Building (using $(nproc) cores)..."
cmake --build "${LLAMA_DIR}/build" --config Release -j"$(nproc)"

echo ""
echo "==> Build complete. Verifying binaries..."
for bin in llama-bench llama-cli; do
    BIN_PATH="${LLAMA_DIR}/build/bin/${bin}"
    if [[ -x "${BIN_PATH}" ]]; then
        echo "  ✓ ${BIN_PATH}"
    else
        echo "  ✗ MISSING: ${BIN_PATH}"
        exit 1
    fi
done

echo ""
echo "==> Done. Run setup/verify_setup.sh next to confirm GPU detection."
