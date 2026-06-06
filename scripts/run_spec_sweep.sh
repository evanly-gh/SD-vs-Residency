#!/usr/bin/env bash
# Run the full speculative decoding sweep using llama-speculative.
#
# NOTE: In this build of llama.cpp, llama-bench does NOT support speculative
# decoding. The dedicated llama-speculative binary is used instead.
# Baseline (no spec decoding) throughput is still measured by run_baseline.sh
# via llama-bench for clean, comparable JSON output.
#
# This script runs llama-speculative N times per condition, logs timing output,
# and saves per-run tps to data/raw/spec_*.jsonl. The analysis script computes
# speedup ratios against the llama-bench baseline.
#
# Key llama-speculative flags used here:
#   --spec-draft-model / -md       : draft model path
#   --spec-draft-ngl / -ngld       : draft GPU layers
#   --spec-draft-n-max N           : max draft tokens per step (enforces fixed γ)
#   --spec-draft-n-min N           : min draft tokens (set = n-max for fixed γ)
#
# Outputs JSONL to data/raw/spec_{model}_{ngl}layers_gamma{g}.jsonl
# (one JSON object per run, containing tps and accept_pct)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

# Fixed prompt for throughput measurement — same content across all conditions.
# Kept short so the measurement is dominated by generation, not prefill.
BENCH_PROMPT="Write a detailed explanation of how transformers work in machine learning."

echo "==> Speculative decoding sweep starting at $(date)"
echo "    Tool: llama-speculative (llama-bench has no spec decoding in this build)"
echo "    Gamma values: ${GAMMA_VALUES[*]}"
echo "    Runs per condition: ${REPETITIONS} (first 2 discarded in analysis)"
echo ""

parse_tps() {
    # Current llama-speculative prints:  "decoded NNN tokens in M.MMM seconds, speed:   TT.TTT t/s"
    # Older builds used:                  "eval time = ... (TT.T tokens per second)"
    # Try the new format first, fall back to the old.
    local v
    v=$(grep -oP 'decoded\s+\d+\s+tokens\s+in\s+[\d.]+\s+seconds,\s+speed:\s+\K[\d.]+' "${1}" | tail -1 || true)
    if [[ -z "${v}" ]]; then
        v=$(grep -oP 'eval time.*?\K[\d.]+(?= tokens per second)' "${1}" | grep -v '^inf' | tail -1 || true)
    fi
    echo "${v}"
}

parse_accept_pct() {
    # Current format: "accept    = 38.806%"
    # Older format:   "draft accepted: 312 / 804 (38.8%)"
    local v
    v=$(grep -oP 'accept\s*=\s*\K[\d.]+(?=%)' "${1}" | tail -1 || true)
    if [[ -z "${v}" ]]; then
        v=$(grep -oP 'draft accepted:.*?\K[\d.]+(?=%)' "${1}" | tail -1 || true)
    fi
    echo "${v}"
}

run_spec() {
    local model_path="$1"
    local model_label="$2"
    local ngl="$3"
    local gamma="$4"
    local out_jsonl="${RAW_DIR}/spec_${model_label}_ngl${ngl}_gamma${gamma}.jsonl"

    # Skip if we already have enough runs
    if [[ -f "${out_jsonl}" ]]; then
        local existing
        existing=$(wc -l < "${out_jsonl}")
        if [[ "${existing}" -ge "${REPETITIONS}" ]]; then
            echo "  [SKIP] Already have ${existing} runs: $(basename "${out_jsonl}")"
            return
        fi
    fi

    echo "  Running: ${model_label} ngl=${ngl} γ=${gamma} ..."

    for i in $(seq 1 "${REPETITIONS}"); do
        local run_log="${LOGS_DIR}/spec_${model_label}_ngl${ngl}_gamma${gamma}_run${i}.log"

        "${LLAMA_SPECULATIVE}" \
            -m "${model_path}" \
            --spec-draft-model "${MODEL_DRAFT}" \
            -ngl "${ngl}" \
            --spec-draft-ngl "${GPU_LAYERS_DRAFT}" \
            -c "${CTX_SIZE}" \
            -n "${N_GEN}" \
            --spec-draft-n-max "${gamma}" \
            --spec-draft-n-min "${gamma}" \
            --prompt "${BENCH_PROMPT}" \
            -e \
            2>&1 | tee "${run_log}" > /dev/null

        local tps accept_pct
        tps=$(parse_tps "${run_log}")
        accept_pct=$(parse_accept_pct "${run_log}")

        if [[ -z "${tps}" ]]; then
            echo "    WARNING: run ${i} — could not parse tps from log"
            tps="null"
        fi

        # Append one JSON record per run to the JSONL file
        printf '{"run":%d,"model":"%s","ngl":%d,"gamma":%d,"tps":%s,"accept_pct":%s}\n' \
            "${i}" "${model_label}" "${ngl}" "${gamma}" \
            "${tps}" "${accept_pct:-null}" \
            >> "${out_jsonl}"

        echo "    run ${i}: ${tps:-?} tok/s  accept=${accept_pct:-?}%"
    done

    echo "    → $(basename "${out_jsonl}")"
}

# ── 14B: fully resident, all gamma values ─────────────────────────────────────
echo "── Qwen3-14B speculative sweep ─────────────────────────────────────────"
if [[ -f "${MODEL_14B}" && -f "${MODEL_DRAFT}" ]]; then
    for gamma in "${GAMMA_VALUES[@]}"; do
        run_spec "${MODEL_14B}" "14b" "${GPU_LAYERS_14B}" "${gamma}"
    done
else
    echo "  SKIP: 14B or draft model not found"
fi

# ── 32B: NGL sweep × all gamma values ────────────────────────────────────────
echo ""
echo "── Qwen3-32B speculative sweep (NGL × γ) ──────────────────────────────"
if [[ -f "${MODEL_32B}" && -f "${MODEL_DRAFT}" ]]; then
    for ngl in "${NGL_SWEEP[@]}"; do
        for gamma in "${GAMMA_VALUES[@]}"; do
            run_spec "${MODEL_32B}" "32b" "${ngl}" "${gamma}"
        done
    done
else
    echo "  SKIP: ${MODEL_32B} not found (run: bash setup/download_models.sh --32b)"
fi

echo ""
echo "==> Speculative sweep complete at $(date)"
echo "    Results in: ${RAW_DIR}/"
ls -lh "${RAW_DIR}"/spec_*.jsonl 2>/dev/null | awk '{print "    " $5, $9}'
