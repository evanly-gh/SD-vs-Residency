#!/usr/bin/env bash
# Run non-speculative baseline benchmarks across all model/NGL configurations.
# This establishes the denominator for every speedup ratio calculation.
#
# Outputs JSON files to data/raw/ named:
#   baseline_{model}_{ngl}layers.json
#
# Run this (Week 2 in the proposal timeline) before the speculative sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

echo "==> Baseline sweep starting at $(date)"
echo "    Repetitions: ${REPETITIONS} (first 2 discarded in analysis)"
echo "    Tokens to generate: ${N_GEN}"
echo "    Context size: ${CTX_SIZE}"
echo ""

run_baseline() {
    local model_path="$1"
    local model_label="$2"
    local ngl="$3"
    local out_file="${RAW_DIR}/baseline_${model_label}_ngl${ngl}.json"

    if [[ -f "${out_file}" ]]; then
        echo "  [SKIP] Already exists: $(basename "${out_file}")"
        return
    fi

    echo "  Running: ${model_label} ngl=${ngl}..."
    "${LLAMA_BENCH}" \
        -m "${model_path}" \
        -ngl "${ngl}" \
        -c "${CTX_SIZE}" \
        -n "${N_GEN}" \
        -p "${N_PROMPT}" \
        -r "${REPETITIONS}" \
        -o json \
        --output-file "${out_file}" \
        2>>"${LOGS_DIR}/baseline_${model_label}_ngl${ngl}.stderr.log"

    echo "    → $(basename "${out_file}")"
}

# ── 14B: fully resident (ngl=99 pushes all layers to GPU) ────────────────────
echo "── Qwen3-14B baseline ──────────────────────────────────────────────────"
if [[ -f "${MODEL_14B}" ]]; then
    run_baseline "${MODEL_14B}" "14b" "${GPU_LAYERS_14B}"
else
    echo "  SKIP: ${MODEL_14B} not found"
fi

# ── 32B: sweep across NGL values ─────────────────────────────────────────────
echo ""
echo "── Qwen3-32B baseline (NGL sweep) ─────────────────────────────────────"
if [[ -f "${MODEL_32B}" ]]; then
    for ngl in "${NGL_SWEEP[@]}"; do
        run_baseline "${MODEL_32B}" "32b" "${ngl}"
    done
else
    echo "  SKIP: ${MODEL_32B} not found (run: bash setup/download_models.sh --32b)"
fi

echo ""
echo "==> Baseline sweep complete at $(date)"
echo "    Results in: ${RAW_DIR}/"
ls -lh "${RAW_DIR}"/baseline_*.json 2>/dev/null | awk '{print "    " $5, $9}'
