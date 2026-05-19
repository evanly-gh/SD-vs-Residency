#!/usr/bin/env bash
# Hardware profiling session for a single target configuration.
# Runs rocm-smi in the background while llama-bench executes, capturing:
#   - GPU compute utilization (%)
#   - VRAM usage (MB)
#   - GPU clock frequencies (MHz)
#   - PCIe transfer rates (where rocm-smi exposes them)
#
# Usage:
#   bash scripts/run_profiling.sh <model_label> <ngl> <gamma> [spec|base]
#
# Example (profile crossover condition for 32B at ngl=32):
#   bash scripts/run_profiling.sh 32b 32 4 spec
#   bash scripts/run_profiling.sh 32b 32 4 base
#
# Per the proposal: discard first 2 min (thermal stabilization), then collect
# 8 min of steady-state data. This script handles the discard automatically
# by splitting the profiling session into warm-up and collection phases.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

MODEL_LABEL="${1:?Usage: $0 <model_label> <ngl> <gamma> [spec|base]}"
NGL="${2:?}"
GAMMA="${3:?}"
MODE="${4:-spec}"   # "spec" or "base"

# Select model
case "${MODEL_LABEL}" in
    14b) MODEL_PATH="${MODEL_14B}" ;;
    32b) MODEL_PATH="${MODEL_32B}" ;;
    *) echo "ERROR: model_label must be 14b or 32b"; exit 1 ;;
esac

SESSION_ID="${MODEL_LABEL}_ngl${NGL}_gamma${GAMMA}_${MODE}"
PROFILE_CSV="${PROFILING_DIR}/profile_${SESSION_ID}.csv"
PROFILE_LOG="${PROFILING_DIR}/profile_${SESSION_ID}.bench.json"

echo "==> Profiling session: ${SESSION_ID}"
echo "    Thermal discard: ${THERMAL_DISCARD_SECS}s"
echo "    Collection window: ${PROFILING_SECS}s"
echo ""

# ── rocm-smi polling function ────────────────────────────────────────────────
start_rocm_monitor() {
    local csv_file="$1"
    echo "timestamp_s,gpu_use_pct,mem_use_mb,sclk_mhz,mclk_mhz" > "${csv_file}"
    local start_time
    start_time=$(date +%s)
    while true; do
        local now
        now=$(date +%s)
        local elapsed=$(( now - start_time ))

        # rocm-smi --showuse --showmemuse outputs GPU%/VRAM in parseable form
        local gpu_use mem_use sclk mclk
        gpu_use=$(rocm-smi -d "${GPU_ID}" --showuse 2>/dev/null \
            | grep -oP 'GPU use \(%\)\s*:\s*\K\d+' | head -1 || echo "NA")
        mem_use=$(rocm-smi -d "${GPU_ID}" --showmemuse 2>/dev/null \
            | grep -oP 'GPU Memory Allocated.*?:\s*\K[\d.]+' | head -1 || echo "NA")
        sclk=$(rocm-smi -d "${GPU_ID}" --showclkfrq 2>/dev/null \
            | awk '/sclk/{getline; print $2; exit}' || echo "NA")
        mclk=$(rocm-smi -d "${GPU_ID}" --showclkfrq 2>/dev/null \
            | awk '/mclk/{getline; print $2; exit}' || echo "NA")

        echo "${elapsed},${gpu_use},${mem_use},${sclk},${mclk}" >> "${csv_file}"
        sleep "$(echo "${ROCM_SMI_INTERVAL_MS} / 1000" | bc -l)"
    done
}

# ── Build llama-bench command ─────────────────────────────────────────────────
build_bench_cmd() {
    local cmd=(
        "${LLAMA_BENCH}"
        -m "${MODEL_PATH}"
        -ngl "${NGL}"
        -c "${CTX_SIZE}"
        -n "${N_GEN}"
        -p "${N_PROMPT}"
        -r 20          # many repetitions to fill the profiling window
        -o json
        --output-file "${PROFILE_LOG}"
    )
    if [[ "${MODE}" == "spec" ]]; then
        cmd+=( -md "${MODEL_DRAFT}" -ngld "${GPU_LAYERS_DRAFT}" -d "${GAMMA}" )
    fi
    echo "${cmd[@]}"
}

# ── Phase 1: Thermal warm-up (discard) ───────────────────────────────────────
echo "── Phase 1: thermal warm-up (${THERMAL_DISCARD_SECS}s, data discarded) ──"
WARMUP_CSV="${PROFILING_DIR}/profile_${SESSION_ID}.warmup.csv"
start_rocm_monitor "${WARMUP_CSV}" &
MONITOR_PID=$!

# Run bench in a loop until warm-up time has elapsed
WARMUP_END=$(( $(date +%s) + THERMAL_DISCARD_SECS ))
while [[ $(date +%s) -lt "${WARMUP_END}" ]]; do
    eval "$(build_bench_cmd)" > /dev/null 2>&1 || true
done

kill "${MONITOR_PID}" 2>/dev/null || true
rm -f "${WARMUP_CSV}"
echo "    Warm-up complete."

# ── Phase 2: Steady-state collection ─────────────────────────────────────────
echo "── Phase 2: steady-state collection (${PROFILING_SECS}s) ───────────────"
start_rocm_monitor "${PROFILE_CSV}" &
MONITOR_PID=$!

COLLECT_END=$(( $(date +%s) + PROFILING_SECS ))
while [[ $(date +%s) -lt "${COLLECT_END}" ]]; do
    eval "$(build_bench_cmd)" >> /dev/null 2>&1 || true
done

kill "${MONITOR_PID}" 2>/dev/null || true

echo "    Collection complete."
echo ""
echo "==> Profiling output:"
echo "    Resource utilization CSV : ${PROFILE_CSV}"
echo "    llama-bench JSON         : ${PROFILE_LOG}"
echo ""
echo "    Rows collected: $(wc -l < "${PROFILE_CSV}")"
echo "    GPU utilization summary:"
awk -F',' 'NR>1 && $2!="NA" {sum+=$2; n++} END {printf "      mean=%.1f%%, samples=%d\n", sum/n, n}' "${PROFILE_CSV}" || true
