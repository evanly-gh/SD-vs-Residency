#!/usr/bin/env bash
# Central configuration — sourced by all other scripts.
# Edit the paths in this file to match your environment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── llama.cpp binaries ────────────────────────────────────────────────────────
# NOTE: In this build, speculative decoding is in llama-speculative, NOT llama-bench.
# llama-bench is used for baseline (no spec decoding) throughput only.
LLAMA_CPP_DIR="${HOME}/llama.cpp"
LLAMA_BENCH="${LLAMA_CPP_DIR}/build/bin/llama-bench"
LLAMA_CLI="${LLAMA_CPP_DIR}/build/bin/llama-cli"
LLAMA_SPECULATIVE="${LLAMA_CPP_DIR}/build/bin/llama-speculative"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR="${REPO_ROOT}/models"
MODEL_DRAFT="${MODELS_DIR}/Qwen_Qwen3-0.6B-Q4_K_M.gguf"
MODEL_14B="${MODELS_DIR}/Qwen_Qwen3-14B-Q4_K_M.gguf"
MODEL_32B="${MODELS_DIR}/Qwen_Qwen3-32B-Q4_K_M.gguf"

# ── Hardware ──────────────────────────────────────────────────────────────────
# NVIDIA RTX 6000 (cluster partition: gpu-rtx6k). Use CUDA_VISIBLE_DEVICES to
# select the GPU in your job; inside the job, GPU_ID=0 targets the visible GPU.
GPU_ID=0

# 14B fully resident: 8.5 GB model + ~0.4 GB draft + ~1.3 GB KV cache ≈ 10.2 GB
# Set to 99 to push all layers to GPU; llama.cpp caps at actual layer count.
GPU_LAYERS_14B=99
GPU_LAYERS_DRAFT=99

# 32B CPU-offload sweep: layer counts for -ngl flag (0 = all CPU, 64 = all GPU)
# Qwen3-32B has 64 transformer layers.
NGL_SWEEP=(0 16 32 48 64)

# ── Benchmark parameters ──────────────────────────────────────────────────────
CTX_SIZE=8192          # Fixed for all conditions; large enough for thinking-mode chains
N_GEN=512              # Tokens to generate per run
N_PROMPT=128           # Synthetic prompt tokens for llama-bench throughput runs
REPETITIONS=7          # 1 cold + 1 warm + 5 measured; first 2 dropped in analysis
GAMMA_VALUES=(4 6 8 10) # Speculation lengths (γ) to sweep

# ── Prompt files ──────────────────────────────────────────────────────────────
PROMPTS_DIR="${REPO_ROOT}/prompts"
PROMPTS_CODE="${PROMPTS_DIR}/code_gen.json"
PROMPTS_REASONING="${PROMPTS_DIR}/reasoning.json"
PROMPTS_CHAT="${PROMPTS_DIR}/chat.json"

# ── Output directories ────────────────────────────────────────────────────────
DATA_DIR="${REPO_ROOT}/data"
RAW_DIR="${DATA_DIR}/raw"
LOGS_DIR="${DATA_DIR}/logs"
PROFILING_DIR="${DATA_DIR}/profiling"
RESULTS_DIR="${REPO_ROOT}/results"
FIGURES_DIR="${RESULTS_DIR}/figures"

# ── Profiling ─────────────────────────────────────────────────────────────────
THERMAL_DISCARD_SECS=120   # Discard first 2 min of profiling session
PROFILING_SECS=480         # Collect for 8 min steady-state
GPU_SMI_INTERVAL_MS=500    # nvidia-smi polling interval

# ── Derived ───────────────────────────────────────────────────────────────────
mkdir -p "${RAW_DIR}" "${LOGS_DIR}" "${PROFILING_DIR}" "${FIGURES_DIR}"
