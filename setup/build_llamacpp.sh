#!/usr/bin/env bash
# Build llama.cpp with ROCm/HIP support for Navi 22 (gfx1031).
# Run this script once from anywhere; it clones into $HOME/llama.cpp.
#
# Prerequisites:
#   - ROCm installed and in PATH (check with: rocminfo | grep gfx)
#   - cmake >= 3.21, git, make, g++
#
# After building, llama-bench and llama-cli will be at:
#   $HOME/llama.cpp/build/bin/llama-bench
#   $HOME/llama.cpp/build/bin/llama-cli

set -euo pipefail

LLAMA_DIR="${HOME}/llama.cpp"
# Navi 22 = gfx1031. Confirm with: rocminfo | grep 'Name:' | grep gfx
AMDGPU_TARGET="gfx1031"

echo "==> Checking ROCm availability..."
if ! command -v hipcc &>/dev/null; then
    echo "ERROR: hipcc not found. Is ROCm installed and in PATH?"
    echo "  Check: which hipcc  or  ls /opt/rocm/bin/hipcc"
    exit 1
fi
echo "  hipcc: $(which hipcc)"
ROCM_VERSION="$(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"
echo "  ROCm version: ${ROCM_VERSION}"

# ROCm 6.x ships clang 17, which looks for GCC 12 C++ standard headers.
# If only GCC 11 is installed (Ubuntu 22.04 default), the HIP compile test
# fails with 'cmath' file not found.
if ! dpkg -l gcc-12 2>/dev/null | grep -q '^ii'; then
    echo ""
    echo "WARNING: gcc-12 not found. ROCm 6.x clang 17 needs GCC 12 headers."
    echo "  Fix: sudo apt install -y gcc-12 g++-12"
    echo "  (Attempting build anyway — it may fail with 'cmath' not found)"
    echo ""
fi

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

echo "==> Configuring CMake (HIP backend, target=${AMDGPU_TARGET})..."
# NOTE: The flag was renamed from GGML_HIPBLAS to GGML_HIP in llama.cpp.
# Passing GGML_HIPBLAS=ON is silently ignored in current builds.
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGET}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF

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
