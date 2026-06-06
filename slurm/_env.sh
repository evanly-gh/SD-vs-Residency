#!/usr/bin/env bash
# Sourced by every *.slurm script.
# Centralizes module loads and env setup so a single edit propagates everywhere.

set -euo pipefail

# Slurm copies the batch script to /var/spool/slurmd/jobNNN/slurm_script so
# BASH_SOURCE[0] inside the batch script does NOT point back to this file.
# Slurm sets SLURM_SUBMIT_DIR to the dir where sbatch was invoked — we submit
# from the repo root, so use that. Fallback to BASH_SOURCE for direct sourcing.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"

# Required for llama.cpp binaries built on the cluster:
#   gcc/13.2.0  — supplies libstdc++ with GLIBCXX_3.4.26+ / CXXABI_1.3.13
#   cuda/12.4.1 — supplies libcudart.so.12 (driver libcuda.so.1 is on the GPU node)
# Don't `source /etc/profile.d/modules.sh` blindly — module is already in env on klone.
module purge 2>/dev/null || true
module load gcc/13.2.0
module load cuda/12.4.1

# Belt-and-suspenders: ensure llama.cpp's own lib dir is also discoverable.
LLAMACPP_LIB="${HOME}/llama.cpp/build/bin"
export LD_LIBRARY_PATH="${LLAMACPP_LIB}:${LD_LIBRARY_PATH:-}"

# Pin the GPU to the one Slurm assigned us; the visible GPU is index 0 in-job.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Thread pool for llama.cpp CPU work.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "── Slurm job environment ────────────────────────────────────────────────"
echo "  Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "  Node         : $(hostname)"
echo "  Partition    : ${SLURM_JOB_PARTITION:-N/A}"
echo "  CPUs         : ${SLURM_CPUS_PER_TASK:-?}"
echo "  Mem          : ${SLURM_MEM_PER_NODE:-?} MB"
echo "  GPU(s)       : ${CUDA_VISIBLE_DEVICES}"
echo "  Repo         : ${REPO_ROOT}"
echo "  Modules      : $(module list 2>&1 | tail -n +2 | tr '\n' ' ')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

# Fast-fail self-check: prove all shared libs resolve (libcuda.so.1 needs the
# GPU node's driver; running --help would init CUDA and fail on a CPU node).
if [[ -x "${LLAMACPP_LIB}/llama-speculative" ]]; then
    # libcuda.so.1 (NVIDIA driver) only exists on GPU nodes — allow that one.
    missing=$(ldd "${LLAMACPP_LIB}/llama-speculative" 2>&1 | grep 'not found' | grep -v libcuda.so.1 || true)
    if [[ -n "${missing}" ]]; then
        echo "  ✗ llama-speculative has unresolved shared libs:"
        echo "${missing}" | sed 's/^/    /'
        exit 1
    fi
    echo "  ✓ llama-speculative shared libs resolve"
fi
echo "─────────────────────────────────────────────────────────────────────────"
