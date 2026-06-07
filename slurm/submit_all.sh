#!/usr/bin/env bash
# Submits all speculative decoding experiment conditions as parallel SLURM jobs.
#
# Usage:
#   bash slurm/submit_all.sh                        # submit all 9 conditions
#   bash slurm/submit_all.sh --only K1,K2,baseline  # submit subset
#
# The 9 conditions (MTP speculative decoding on the Qwen3.5 family):
#   Axis 1 — speculative-depth sweep on dense 27B: baseline (k=0 AR), K1..K4, K6
#   Axis 2 — baseline_moe (35B-A3B AR), M3 (MoE k=3), M3_think (MoE k=3 thinking)
# baseline / baseline_moe are the autoregressive denominators for the dense /
# MoE speedups respectively.
#   bash slurm/submit_all.sh --dry-run              # print sbatch commands, don't submit
#   bash slurm/submit_all.sh --smoke-test           # 5 prompts only, condition A3

set -euo pipefail

GSCRATCH="/mmfs1/gscratch/intelligentsystems/evanly"
MODELS_DIR="${GSCRATCH}/models"
RESULTS_DIR="${GSCRATCH}/sd-qwen35/results"
LOGS_DIR="${GSCRATCH}/sd-qwen35/logs"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${REPO_DIR}/slurm/job_template.sh"

DRY_RUN=0
SMOKE_TEST=0
ONLY_CONDITIONS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1 ;;
        --smoke-test)  SMOKE_TEST=1 ;;
        --only)        ONLY_CONDITIONS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Symlink results into repo if not already done
if [ ! -L "${REPO_DIR}/results" ] && [ ! -d "${REPO_DIR}/results" ]; then
    ln -s "$RESULTS_DIR" "${REPO_DIR}/results"
    echo "Symlinked repo/results -> $RESULTS_DIR"
fi

# condition -> "target_model_name  spec_method  k  mode  port"
# model names are resolved to full paths below.
#
# Axis 1 (dense 27B target): speculative-depth (k) sweep via MTP self-speculation.
#   vLLM forces any Qwen3.5 "draft" onto its native MTP head, so cross-size
#   draft-model SD is impossible on this family; the comparable axis is k.
# Axis 2: dense-vs-MoE at fixed k (K3 vs M3) and thinking effect (M3 vs M3_think),
#   with baseline_moe as the MoE autoregressive denominator.
declare -A CONFIGS
CONFIGS[baseline]="Qwen3.5-27B      none  0  standard  8100"
CONFIGS[K1]="Qwen3.5-27B      mtp   1  standard  8101"
CONFIGS[K2]="Qwen3.5-27B      mtp   2  standard  8102"
CONFIGS[K3]="Qwen3.5-27B      mtp   3  standard  8103"
CONFIGS[K4]="Qwen3.5-27B      mtp   4  standard  8104"
CONFIGS[K6]="Qwen3.5-27B      mtp   6  standard  8105"
CONFIGS[baseline_moe]="Qwen3.5-35B-A3B  none  0  standard  8106"
CONFIGS[M3]="Qwen3.5-35B-A3B  mtp   3  standard  8107"
CONFIGS[M3_think]="Qwen3.5-35B-A3B  mtp   3  thinking  8108"

# Determine which conditions to submit
if [ -n "$ONLY_CONDITIONS" ]; then
    IFS=',' read -ra SUBMIT_LIST <<< "$ONLY_CONDITIONS"
elif [ "$SMOKE_TEST" -eq 1 ]; then
    SUBMIT_LIST=(K2)
else
    SUBMIT_LIST=(baseline K1 K2 K3 K4 K6 baseline_moe M3 M3_think)
fi

NUM_PROMPTS=150
[ "$SMOKE_TEST" -eq 1 ] && NUM_PROMPTS=5

echo "=== Submitting conditions: ${SUBMIT_LIST[*]} ==="
[ "$DRY_RUN" -eq 1 ] && echo "(DRY RUN — not actually submitting)"
[ "$SMOKE_TEST" -eq 1 ] && echo "(SMOKE TEST — $NUM_PROMPTS prompts only)"
echo ""

declare -A SUBMITTED_IDS

for CONDITION in "${SUBMIT_LIST[@]}"; do
    if [ -z "${CONFIGS[$CONDITION]+x}" ]; then
        echo "ERROR: Unknown condition '$CONDITION'. Valid: ${!CONFIGS[*]}"
        exit 1
    fi

    read -r TARGET_NAME SPEC_METHOD K MODE PORT <<< "${CONFIGS[$CONDITION]}"

    TARGET_MODEL="${MODELS_DIR}/${TARGET_NAME}"

    OUTPUT_FILE="${RESULTS_DIR}/${CONDITION}.json"
    [ "$SMOKE_TEST" -eq 1 ] && OUTPUT_FILE="${RESULTS_DIR}/smoke_${CONDITION}.json"

    # Verify model directory exists
    if [ ! -d "$TARGET_MODEL" ]; then
        echo "WARNING: Target model not found: $TARGET_MODEL"
        echo "  Run: huggingface-cli download Qwen/${TARGET_NAME} --local-dir ${TARGET_MODEL}"
    fi

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",SD_REPO_DIR=${REPO_DIR}"
    EXPORT_VARS+=",SD_CONDITION=${CONDITION}"
    EXPORT_VARS+=",SD_TARGET_MODEL=${TARGET_MODEL}"
    EXPORT_VARS+=",SD_SPEC_METHOD=${SPEC_METHOD}"
    EXPORT_VARS+=",SD_K=${K}"
    EXPORT_VARS+=",SD_MODE=${MODE}"
    EXPORT_VARS+=",SD_PORT=${PORT}"
    EXPORT_VARS+=",SD_OUTPUT_FILE=${OUTPUT_FILE}"
    EXPORT_VARS+=",SD_NUM_PROMPTS=${NUM_PROMPTS}"

    CMD=(
        sbatch
        --job-name="sd-${CONDITION}"
        --export="${EXPORT_VARS}"
        "$SCRIPT"
    )

    printf "%-12s  target=%-18s  spec=%-5s k=%-2s  mode=%-9s  port=%s\n" \
        "$CONDITION" "$TARGET_NAME" "$SPEC_METHOD" "$K" "$MODE" "$PORT"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY RUN] Would run: ${CMD[*]}"
    else
        JOB_ID=$("${CMD[@]}" | grep -oP '\d+')
        SUBMITTED_IDS[$CONDITION]="$JOB_ID"
        echo "  Submitted job $JOB_ID -> $OUTPUT_FILE"
    fi
done

echo ""
if [ "$DRY_RUN" -eq 0 ] && [ ${#SUBMITTED_IDS[@]} -gt 0 ]; then
    echo "=== All jobs submitted ==="
    echo "Monitor: squeue -u $USER --format='%.10i %.12j %.8T %.10M %.6D %R'"
    echo ""
    echo "Logs: $LOGS_DIR"
    echo "Results: $RESULTS_DIR"
    echo ""
    echo "When wave 1 (A1-A4, baseline) is done, run analysis:"
    echo "  python analysis/plot_all.py --results-dir $RESULTS_DIR"
fi
