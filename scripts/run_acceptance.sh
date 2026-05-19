#!/usr/bin/env bash
# Measure draft token acceptance rates using llama-cli with real task prompts.
# This is the task-type and thinking-mode characterization component.
#
# For each (task, thinking_mode, gamma) combination, runs llama-cli once and
# parses its native acceptance rate output (n_drafted / n_accept / accept%).
#
# Outputs plain-text logs to data/logs/ named:
#   accept_{model}_{ngl}layers_gamma{g}_{task}_{think|nothink}.log
#
# The analysis script (analysis/parse_results.py) extracts the accept% values.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

# Task categories and their prompt files
declare -A TASK_PROMPTS=(
    ["code"]="${PROMPTS_CODE}"
    ["reasoning"]="${PROMPTS_REASONING}"
    ["chat"]="${PROMPTS_CHAT}"
)

THINKING_MODES=("think" "nothink")

echo "==> Acceptance rate sweep starting at $(date)"
echo ""

run_acceptance() {
    local model_path="$1"
    local model_label="$2"
    local ngl="$3"
    local gamma="$4"
    local task="$5"
    local think_mode="$6"    # "think" or "nothink"
    local prompt_idx="$7"
    local prompt_text="$8"

    local label="${model_label}_ngl${ngl}_gamma${gamma}_${task}_${think_mode}_p${prompt_idx}"
    local log_file="${LOGS_DIR}/accept_${label}.log"

    if [[ -f "${log_file}" ]]; then
        echo "  [SKIP] ${label}"
        return
    fi

    # Prepend thinking mode control token to the prompt
    local full_prompt
    if [[ "${think_mode}" == "think" ]]; then
        full_prompt="/think ${prompt_text}"
    else
        full_prompt="/no_think ${prompt_text}"
    fi

    echo "  ${label}..."
    "${LLAMA_SPECULATIVE}" \
        -m "${model_path}" \
        --spec-draft-model "${MODEL_DRAFT}" \
        -ngl "${ngl}" \
        --spec-draft-ngl "${GPU_LAYERS_DRAFT}" \
        -c "${CTX_SIZE}" \
        -n "${N_GEN}" \
        --spec-draft-n-max "${gamma}" \
        --spec-draft-n-min "${gamma}" \
        --prompt "${full_prompt}" \
        -e \
        2>&1 | tee "${log_file}" > /dev/null

    # Quick sanity: check accept% appeared in log
    if grep -q "accept" "${log_file}"; then
        local rate
        rate=$(grep -oP 'accept\s*=\s*\K[\d.]+' "${log_file}" | tail -1)
        echo "    → accept%: ${rate:-?}"
    else
        echo "    WARNING: acceptance stats not found in log — check ${log_file}"
    fi
}

# Load prompts from JSON files (requires python3 + json module — stdlib)
get_prompts() {
    local json_file="$1"
    python3 -c "
import json, sys
with open('${json_file}') as f:
    prompts = json.load(f)
for i, p in enumerate(prompts):
    print(f'{i}|||{p[\"prompt\"]}')
"
}

# ── 14B fully-resident acceptance sweep ──────────────────────────────────────
echo "── Qwen3-14B acceptance rates ──────────────────────────────────────────"
if [[ ! -f "${MODEL_14B}" || ! -f "${MODEL_DRAFT}" ]]; then
    echo "  SKIP: models not found"
else
    for task in "${!TASK_PROMPTS[@]}"; do
        prompt_file="${TASK_PROMPTS[${task}]}"
        while IFS='|||' read -r idx prompt; do
            for think_mode in "${THINKING_MODES[@]}"; do
                for gamma in "${GAMMA_VALUES[@]}"; do
                    run_acceptance "${MODEL_14B}" "14b" "${GPU_LAYERS_14B}" \
                        "${gamma}" "${task}" "${think_mode}" "${idx}" "${prompt}"
                done
            done
        done < <(get_prompts "${prompt_file}")
    done
fi

# ── 32B NGL sweep acceptance (spot-check at crossover NGL values) ─────────────
# Running the full NGL × task × thinking × gamma sweep would be very long.
# We spot-check the NGL values that produced crossover in run_spec_sweep results.
# Edit NGL_ACCEPTANCE below after reviewing initial speedup results.
NGL_ACCEPTANCE=(16 32 48)   # edit after seeing crossover from spec sweep

echo ""
echo "── Qwen3-32B acceptance rates (spot-check NGL values) ──────────────────"
if [[ ! -f "${MODEL_32B}" || ! -f "${MODEL_DRAFT}" ]]; then
    echo "  SKIP: 32B not downloaded"
else
    for ngl in "${NGL_ACCEPTANCE[@]}"; do
        for task in "${!TASK_PROMPTS[@]}"; do
            prompt_file="${TASK_PROMPTS[${task}]}"
            while IFS='|||' read -r idx prompt; do
                for think_mode in "${THINKING_MODES[@]}"; do
                    # Only sweep γ=4 and γ=8 for the 32B acceptance spot-check
                    for gamma in 4 8; do
                        run_acceptance "${MODEL_32B}" "32b" "${ngl}" \
                            "${gamma}" "${task}" "${think_mode}" "${idx}" "${prompt}"
                    done
                done
            done < <(get_prompts "${prompt_file}")
        done
    done
fi

echo ""
echo "==> Acceptance sweep complete at $(date)"
echo "    Logs in: ${LOGS_DIR}/"
