#!/usr/bin/env bash
# Smoke-tests the full environment before any benchmarking begins.
# Checks: GPU detection, vocab alignment, short inference pass, profiling tools.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"  # "ok" or error message
    if [[ "${result}" == "ok" ]]; then
        echo "  ✓ ${label}"
        ((PASS++)) || true
    else
        echo "  ✗ ${label}: ${result}"
        ((FAIL++)) || true
    fi
}

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Speculative Decoding Setup Verification"
echo "══════════════════════════════════════════════════════"

# ── Binaries ────────────────────────────────────────────────────────────────
echo ""
echo "── llama.cpp binaries ──────────────────────────────────────────────────"
[[ -x "${LLAMA_BENCH}" ]]        && check "llama-bench"       "ok" || check "llama-bench"       "not found at ${LLAMA_BENCH}"
[[ -x "${LLAMA_CLI}" ]]          && check "llama-cli"         "ok" || check "llama-cli"         "not found at ${LLAMA_CLI}"
[[ -x "${LLAMA_SPECULATIVE}" ]]  && check "llama-speculative" "ok" || check "llama-speculative" "not found at ${LLAMA_SPECULATIVE}"

# ── GPU ─────────────────────────────────────────────────────────────────────
echo ""
echo "── GPU / ROCm ──────────────────────────────────────────────────────────"
if command -v rocm-smi &>/dev/null; then
    check "rocm-smi" "ok"
    echo "    $(rocm-smi --showproductname 2>/dev/null | grep -i 'card\|rx\|radeon' | head -1 || echo '(product name unavailable)')"
else
    check "rocm-smi" "not found — is ROCm in PATH?"
fi

if command -v rocminfo &>/dev/null; then
    GFX=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "unknown")
    check "rocminfo GPU arch" "ok"
    echo "    Detected arch: ${GFX} (expected gfx1031 for Navi 22)"
    if [[ "${GFX}" != "gfx1031" ]]; then
        echo "    WARNING: arch mismatch — update AMDGPU_TARGET in build_llamacpp.sh"
    fi
else
    check "rocminfo" "not found"
fi

# ── Models ──────────────────────────────────────────────────────────────────
echo ""
echo "── Model files ─────────────────────────────────────────────────────────"
for model_var in MODEL_DRAFT MODEL_14B MODEL_32B; do
    model_path="${!model_var}"
    model_name="$(basename "${model_path}")"
    if [[ -f "${model_path}" ]]; then
        size=$(du -sh "${model_path}" | cut -f1)
        check "${model_name} (${size})" "ok"
    else
        [[ "${model_var}" == "MODEL_32B" ]] \
            && echo "  - ${model_name}: not downloaded (optional; needed for 32B CPU-offload sweep)" \
            || check "${model_name}" "not found — run setup/download_models.sh"
    fi
done

# ── Vocabulary alignment ─────────────────────────────────────────────────────
# NOTE: llama-bench does NOT support speculative decoding in this build.
# Vocab alignment is checked via a 4-token llama-speculative run instead.
echo ""
echo "── Vocabulary alignment (draft ↔ targets) ──────────────────────────────"
LLAMA_SPECULATIVE="${LLAMA_CPP_DIR}/build/bin/llama-speculative"
if [[ -x "${LLAMA_SPECULATIVE}" && -f "${MODEL_DRAFT}" && -f "${MODEL_14B}" ]]; then
    echo "    Running 4-token spec run to check vocab compatibility..."
    SPEC_OUT=$("${LLAMA_SPECULATIVE}" \
        -m "${MODEL_14B}" \
        --spec-draft-model "${MODEL_DRAFT}" \
        -ngl "${GPU_LAYERS_14B}" \
        --spec-draft-ngl "${GPU_LAYERS_DRAFT}" \
        -c 512 -n 4 \
        --spec-draft-n-max 2 \
        --spec-draft-n-min 2 \
        --prompt "Hello" \
        2>&1) || SPEC_EXIT=$?

    if echo "${SPEC_OUT}" | grep -qi "vocab.*mismatch\|tokenizer.*mismatch\|incompatible"; then
        check "Vocabulary alignment" "MISMATCH DETECTED — check model files"
        echo "${SPEC_OUT}" | grep -i "mismatch\|vocab\|token" | head -5
    elif [[ "${SPEC_EXIT:-0}" -ne 0 ]]; then
        check "Vocabulary alignment" "run failed (exit ${SPEC_EXIT:-?}) — check stderr below"
        echo "${SPEC_OUT}" | tail -10
    else
        check "Vocabulary alignment" "ok"
    fi
else
    echo "  - Skipped (requires llama-speculative + both model files)"
fi

# ── VRAM headroom estimate ────────────────────────────────────────────────────
echo ""
echo "── VRAM headroom estimate (Navi 22, 12 GB) ─────────────────────────────"
echo "    Qwen3-14B Q4_K_M  : ~8.5 GB"
echo "    Qwen3-0.6B Q4_K_M : ~0.4 GB"
echo "    KV cache @ 8192ctx : ~1.3 GB"
echo "    ─────────────────────────────"
echo "    Estimated total    : ~10.2 GB  (1.8 GB headroom)"
echo "    Status: Should fit. Watch for OOM during actual runs."

# ── Profiling tools ──────────────────────────────────────────────────────────
echo ""
echo "── Profiling tools ─────────────────────────────────────────────────────"
command -v rocm-smi   &>/dev/null && check "rocm-smi"   "ok" || check "rocm-smi"   "missing"
command -v radeontop  &>/dev/null && check "radeontop"  "ok" || echo "  - radeontop: not installed (optional; install with: sudo apt install radeontop)"
command -v perf       &>/dev/null && check "perf stat"  "ok" || echo "  - perf: not installed (optional; for PCIe bandwidth monitoring)"

# ── Python analysis stack ─────────────────────────────────────────────────────
echo ""
echo "── Python analysis stack ───────────────────────────────────────────────"
python3 -c "import pandas; print('  ✓ pandas', pandas.__version__)" 2>/dev/null || echo "  ✗ pandas — run: conda env create -f environment.yml"
python3 -c "import scipy; print('  ✓ scipy', scipy.__version__)" 2>/dev/null   || echo "  ✗ scipy"
python3 -c "import matplotlib; print('  ✓ matplotlib', matplotlib.__version__)" 2>/dev/null || echo "  ✗ matplotlib"
python3 -c "import seaborn; print('  ✓ seaborn', seaborn.__version__)" 2>/dev/null || echo "  ✗ seaborn"
python3 -c "import pingouin; print('  ✓ pingouin', pingouin.__version__)" 2>/dev/null || echo "  ✗ pingouin (for mixed-effects ANOVA)"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ${PASS} passed, ${FAIL} failed"
echo "══════════════════════════════════════════════════════"
[[ "${FAIL}" -eq 0 ]] && echo "  Ready to benchmark." || echo "  Fix the failures above before running sweeps."
echo ""
