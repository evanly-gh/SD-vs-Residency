#!/usr/bin/env bash
# Submits all speculative decoding experiment conditions as parallel SLURM jobs.
#
# Usage:
#   bash slurm/submit_all.sh                        # submit all 9 conditions
#   bash slurm/submit_all.sh --only A1,A2,baseline  # submit subset
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

# condition -> "target_model_name  draft_model_name  mode  port"
# model names are resolved to full paths below
declare -A CONFIGS
CONFIGS[baseline]="Qwen3.5-27B    ''              standard  8100"
CONFIGS[A1]="Qwen3.5-27B    Qwen3.5-0.8B   standard  8101"
CONFIGS[A2]="Qwen3.5-27B    Qwen3.5-2B     standard  8102"
CONFIGS[A3]="Qwen3.5-27B    Qwen3.5-4B     standard  8103"
CONFIGS[A4]="Qwen3.5-27B    Qwen3.5-9B     standard  8104"
CONFIGS[B1]="Qwen3.5-27B    Qwen3.5-4B     standard  8105"
CONFIGS[B2]="Qwen3.5-35B-A3B  Qwen3.5-4B   standard  8106"
CONFIGS[B3]="Qwen3.5-35B-A3B  Qwen3.5-4B   thinking  8107"

# Determine which conditions to submit
if [ -n "$ONLY_CONDITIONS" ]; then
    IFS=',' read -ra SUBMIT_LIST <<< "$ONLY_CONDITIONS"
elif [ "$SMOKE_TEST" -eq 1 ]; then
    SUBMIT_LIST=(A3)
else
    SUBMIT_LIST=(baseline A1 A2 A3 A4 B1 B2 B3)
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

    read -r TARGET_NAME DRAFT_NAME MODE PORT <<< "${CONFIGS[$CONDITION]}"

    TARGET_MODEL="${MODELS_DIR}/${TARGET_NAME}"
    if [ "$DRAFT_NAME" = "''" ] || [ -z "$DRAFT_NAME" ]; then
        DRAFT_MODEL=""
    else
        DRAFT_MODEL="${MODELS_DIR}/${DRAFT_NAME}"
    fi

    OUTPUT_FILE="${RESULTS_DIR}/${CONDITION}.json"
    [ "$SMOKE_TEST" -eq 1 ] && OUTPUT_FILE="${RESULTS_DIR}/smoke_${CONDITION}.json"

    # Verify model directories exist
    if [ ! -d "$TARGET_MODEL" ]; then
        echo "WARNING: Target model not found: $TARGET_MODEL"
        echo "  Run: huggingface-cli download Qwen/${TARGET_NAME} --local-dir ${TARGET_MODEL}"
    fi
    if [ -n "$DRAFT_MODEL" ] && [ ! -d "$DRAFT_MODEL" ]; then
        echo "WARNING: Draft model not found: $DRAFT_MODEL"
        echo "  Run: huggingface-cli download Qwen/${DRAFT_NAME} --local-dir ${DRAFT_MODEL}"
    fi

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",SD_CONDITION=${CONDITION}"
    EXPORT_VARS+=",SD_TARGET_MODEL=${TARGET_MODEL}"
    EXPORT_VARS+=",SD_DRAFT_MODEL=${DRAFT_MODEL}"
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

    printf "%-10s  target=%-20s  draft=%-20s  mode=%-10s  port=%s\n" \
        "$CONDITION" "$TARGET_NAME" "${DRAFT_NAME:-<baseline>}" "$MODE" "$PORT"

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
