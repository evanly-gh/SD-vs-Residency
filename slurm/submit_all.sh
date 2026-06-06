#!/usr/bin/env bash
# Submit the full experiment pipeline as a dependency chain:
#   baseline → spec_sweep → acceptance → analysis
#
# Profiling is NOT auto-chained: the NGL value worth profiling depends on
# inspecting the speedup sweep first. Submit it manually afterwards, e.g.:
#   sbatch --export=ALL,MODEL=32b,NGL=32,GAMMA=4,MODE=spec slurm/profiling.slurm
#
# Usage:
#   bash slurm/submit_all.sh           # full chain
#   bash slurm/submit_all.sh --skip-accept   # skip the long acceptance phase
#   bash slurm/submit_all.sh --only baseline,analysis   # specific phases only

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_DIR}/.."

SKIP_ACCEPT=0
ONLY=""
for arg in "$@"; do
    case "$arg" in
        --skip-accept) SKIP_ACCEPT=1 ;;
        --only) ;;  # handled below
        --only=*) ONLY="${arg#--only=}" ;;
    esac
done
# Support `--only baseline,spec_sweep` (two-arg form)
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--only" ]]; then
        j=$((i+1)); ONLY="${!j}"
    fi
done

should_run() {
    local name="$1"
    [[ -z "${ONLY}" ]] && return 0
    [[ ",${ONLY}," == *",${name},"* ]]
}

submit() {
    local script="$1"; shift
    local jid
    jid=$(sbatch --parsable "$@" "${script}")
    echo "${jid}"
}

DEP=""
chain_dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || echo ""; }

echo "==> Submitting experiment chain"
LAST_JID=""

if should_run baseline; then
    JID=$(submit slurm/baseline.slurm $(chain_dep "${LAST_JID}"))
    echo "  baseline    : ${JID}"
    LAST_JID="${JID}"
fi

if should_run spec_sweep; then
    JID=$(submit slurm/spec_sweep.slurm $(chain_dep "${LAST_JID}"))
    echo "  spec_sweep  : ${JID}"
    LAST_JID="${JID}"
fi

if [[ "${SKIP_ACCEPT}" -eq 0 ]] && should_run acceptance; then
    JID=$(submit slurm/acceptance.slurm $(chain_dep "${LAST_JID}"))
    echo "  acceptance  : ${JID}"
    LAST_JID="${JID}"
fi

if should_run analysis; then
    # afterany so analysis runs even if a sweep partial-fails (parses what's there)
    DEP_FLAG=""
    [[ -n "${LAST_JID}" ]] && DEP_FLAG="--dependency=afterany:${LAST_JID}"
    JID=$(submit slurm/analysis.slurm ${DEP_FLAG})
    echo "  analysis    : ${JID}"
fi

echo ""
echo "Monitor with: watch -n 10 squeue -u \$USER"
