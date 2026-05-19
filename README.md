# When Does Speculative Decoding Actually Help?

**Characterizing VRAM Residency and Task Draftability as Primary Determinants of Speedup for Dense GGUF Models on Consumer Hardware**

Ali Karim · George Lee · Evan Li — University of Washington

---

## What This Project Is

Speculative decoding is widely recommended for accelerating LLM inference, but practitioners report wildly contradictory results: one person gets a 3× speedup, another running an identical-sounding setup gets a slowdown. This project runs a controlled, hardware-level study to find out exactly when and why speculative decoding helps or hurts — on the consumer hardware (commodity GPUs, GGUF models, llama.cpp) that most people actually use.

The technique works by running a small "draft" model to speculatively generate several tokens at once, then verifying them in a single target model forward pass. Because LLM inference is memory-bandwidth-bound, verifying k tokens costs roughly the same as generating one, so you get k−1 accepted tokens nearly for free. But this breaks down if:
- The acceptance rate is too low (draft tokens are wrong too often)
- The model is offloaded to CPU over PCIe (bandwidth profile changes completely)
- The dequantization overhead of verifying multiple tokens exceeds the savings

**Primary question:** Is VRAM residency — whether the model fits in GPU memory or must offload layers to CPU — a primary determinant of whether speculative decoding produces a net throughput improvement?

---

## Hardware Context: Your 12 GB AMD GPU (Navi 22)

Your GPU is an **AMD Radeon RX 6700 / 6700 XT** (Navi 22, `gfx1031`), with **12 GB GDDR6 VRAM**. This is a tier not studied in the existing literature, which has only examined:

- Enterprise: A100, H100 (80 GB HBM)
- Community benchmarks: RTX 3090 / 4090 (24 GB GDDR6X)

You sit below the commonly assumed 16 GB floor for consumer LLM work. Here is what that means for this study:

| Component | Size | Notes |
|---|---|---|
| Qwen3-14B Q4_K_M | ~8.5 GB | Target model — fits with headroom |
| Qwen3-0.6B Q4_K_M | ~0.4 GB | Draft model — tiny |
| KV cache @ 8192 ctx | ~1.3 GB | Estimated; Qwen3-14B has 8 KV heads |
| **Total estimated** | **~10.2 GB** | **~1.8 GB headroom in 12 GB** |

The 14B model **should fit** fully resident. This is tighter than the 16 GB case in the proposal, which makes your hardware an especially interesting data point — KV cache pressure may force additional layer spilling that wouldn't occur on 16 GB hardware.

The 32B model (~19 GB) **will not fit** — it will be used for the CPU-offload sweep with the `-ngl` flag controlling how many layers stay in VRAM.

---

## Step 0: Install the Conda Environment

```bash
conda env create -f environment.yml
conda activate speculative-decoding
```

This installs pandas, scipy, matplotlib, seaborn, pingouin (for ANOVA), and huggingface_hub (for model downloads).

---

## Step 1: Build llama.cpp with ROCm/HIP

You have ROCm installed but llama.cpp not yet built. Run:

```bash
bash setup/build_llamacpp.sh
```

**What it does:**
1. Clones the latest llama.cpp into `~/llama.cpp`
2. Configures CMake with `-DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1031`
3. Builds with all CPU cores
4. Verifies that `llama-bench` and `llama-cli` binaries exist

**Prerequisites:** `cmake`, `git`, `g++`, and ROCm in your PATH. If `hipcc` isn't found, check that `/opt/rocm/bin` is in your `$PATH`.

**Expected build time:** 5–15 minutes depending on your CPU.

After building, binaries will be at:
```
~/llama.cpp/build/bin/llama-bench
~/llama.cpp/build/bin/llama-cli
```

---

## Step 2: Download the Models

### What models you need and why

| Model | Size | Role | When to download |
|---|---|---|---|
| `Qwen3-0.6B-Q4_K_M.gguf` | ~400 MB | Draft model for all spec decoding runs | First |
| `Qwen3-14B-Q4_K_M.gguf` | ~8.5 GB | VRAM-resident target (14B fits in 12 GB) | First |
| `Qwen3-32B-Q4_K_M.gguf` | ~19 GB | CPU-offload target (won't fit in 12 GB) | After pipeline validated |

**Why Qwen3?** It has a purpose-built 0.6B draft model in the same family with a matching vocabulary (151,936 tokens). Vocabulary mismatch between draft and target causes silent failures in llama.cpp — using the same family eliminates this confound. The 14B and 32B sizes put you cleanly in the VRAM-resident and CPU-offloaded conditions respectively.

**Why Q4_K_M?** It's the default quantization in the llama.cpp docs, the most-downloaded format on Hugging Face, and the point where the speculative decoding tension is most acute: memory bandwidth savings from 4-bit weights are large enough to matter, but verification overhead may offset them.

### How to download (do NOT go to HuggingFace manually)

Use `huggingface-cli`, which downloads directly into the `models/` directory:

```bash
# Download draft + 14B first (~9 GB total, start here)
bash setup/download_models.sh

# After validating the pipeline, download 32B (~19 GB more)
bash setup/download_models.sh --32b
```

The download script installs `huggingface_hub` if needed. Files land in `models/` (gitignored, so they don't get committed). Downloads are resumable — if interrupted, just re-run the same command.

**Storage you need:**
- For the Phase 1 pipeline (14B + 0.6B): ~9 GB
- For the full study including 32B sweep: ~28 GB

**Which bartowski repos?** The script pulls from `bartowski/Qwen3-{0.6B,14B,32B}-GGUF`. Bartowski's quantizations are the community standard. If a repo name has changed, verify on [huggingface.co/bartowski](https://huggingface.co/bartowski) and update `setup/download_models.sh`.

---

## Step 3: Verify Setup

```bash
bash setup/verify_setup.sh
```

This checks: llama.cpp binaries exist, GPU arch is detected correctly, model files are present, vocabulary alignment between draft and target is valid, and your Python analysis stack is installed. Fix any failures before proceeding.

---

## Step 4: Run the Experiments

The experiment runs in three phases matching the proposal timeline. Run each phase in order — later phases depend on earlier output.

### Phase 1 (Week 2): Baseline throughput

Establishes the denominator for every speedup ratio. No speculative decoding.

```bash
bash scripts/run_baseline.sh
```

This runs `llama-bench` on:
- Qwen3-14B at `ngl=99` (fully resident)
- Qwen3-32B at `ngl = 0, 16, 32, 48, 64` (if downloaded)

Each configuration runs 7 times; the first 2 (cold + warm) are discarded in analysis, giving 5 stable measurements per condition. Results land in `data/raw/baseline_*.json`.

### Phase 2 (Weeks 3–4): Speculative decoding sweep

```bash
bash scripts/run_spec_sweep.sh
```

Crosses every combination of:
- Model: 14B (ngl=99) and 32B (ngl sweep)
- Speculation length γ: 4, 6, 8, 10 tokens

Results land in `data/raw/spec_*.json`. **This is the most time-consuming phase.** For the 32B sweep (5 NGL × 4 γ × 7 runs), budget several hours.

### Phase 2b: Acceptance rate by task and thinking mode

```bash
bash scripts/run_acceptance.sh
```

Runs `llama-cli` (not `llama-bench`) with real prompts from the three task categories (code generation, structured reasoning, open-ended chat), under both thinking mode (`/think`) and non-thinking mode (`/no_think`). Logs acceptance rate statistics — `n_drafted / n_accepted / accept%` — to `data/logs/accept_*.log`.

**Thinking mode:** Qwen3 supports two response modes:
- `/think` prepended to a prompt: generates a long structured chain-of-thought inside `<think>` tags before answering. We hypothesize this will be far more draftable than non-thinking output because the `<think>` token stream is highly structured and repetitive.
- `/no_think`: direct response with no chain-of-thought.

This is an open empirical question — if the 0.6B draft model cannot reliably predict the 14B's thought process, the hypothesis will show up as low acceptance rates in thinking mode.

### Phase 3 (Week 5): Hardware profiling

For the conditions closest to the speedup crossover point (one just above 1.0, one just below), run a sustained profiling session:

```bash
# Profile the spec condition at crossover
bash scripts/run_profiling.sh 32b 32 4 spec

# Profile the baseline at the same NGL for comparison
bash scripts/run_profiling.sh 32b 32 4 base
```

The script discards the first 2 minutes (thermal stabilization) and collects 8 minutes of steady-state data. rocm-smi monitors GPU utilization and VRAM usage at 500 ms intervals. Data lands in `data/profiling/profile_*.csv`.

**What we're looking for:** The resource that hits saturation first in net-negative conditions. Candidates:
- PCIe bandwidth saturation (most likely cause in the CPU-offload regime)
- KV cache memory pressure (spills extra layers to CPU when holding two caches)
- Dequantization overhead (verification pass dequantizes more weights per step)
- GPU compute underutilization (draft model too small to keep GPU busy)

---

## Step 5: Analyze Results

```bash
# Parse all raw data into tidy CSVs
python analysis/parse_results.py

# Statistical analysis (ANOVA, threshold estimation, α*)
python analysis/run_anova.py

# Generate figures
python analysis/plot_speedup.py
python analysis/plot_acceptance.py
python analysis/plot_profiling.py
```

### Key outputs

| File | What it shows |
|---|---|
| `results/speedup_by_gamma_14b.png` | Speedup ratio at each γ for 14B resident condition |
| `results/speedup_by_ngl_32b.png` | Speedup vs. NGL for 32B CPU-offload sweep — shows the residency threshold |
| `results/throughput_heatmap.png` | NGL × γ heatmap — green = beneficial, red = harmful |
| `results/acceptance_by_task.png` | Draft acceptance rate by task × thinking mode |
| `results/acceptance_vs_speedup.png` | Scatter plot identifying α* |
| `results/bottleneck_comparison_*.png` | Resource utilization: spec vs. baseline at crossover |

### What the analysis produces

**Residency threshold:** For the 32B sweep, the analysis fits a piecewise linear model to speedup ratio as a function of NGL (0, 16, 32, 48, 64) and estimates the crossover layer count — the exact `-ngl` value at which speculative decoding transitions from net-negative to net-positive.

**Minimum viable acceptance rate (α\*):** The acceptance rate below which no tested γ yields speedup > 1.0. Identified by plotting speedup ratio against acceptance rate across all conditions.

**Effect sizes:** Partial η² for task type, thinking mode, and residency as predictors of speedup ratio. Tells you which factor matters most.

**Thinking mode hypothesis test:** If thinking-mode conditions show substantially higher α than non-thinking on the same prompts, the hypothesis that `<think>` output is more draftable is confirmed. If not, thinking mode is not a useful predictor.

---

## Repository Structure

```
.
├── config.sh                   # Central config — paths, GPU, parameters
├── environment.yml             # Conda environment
│
├── setup/
│   ├── build_llamacpp.sh       # Build llama.cpp with ROCm/HIP for gfx1031
│   ├── download_models.sh      # Download Qwen3 GGUFs via huggingface-cli
│   └── verify_setup.sh         # Smoke test before benchmarking
│
├── scripts/
│   ├── run_baseline.sh         # Phase 1: non-speculative throughput
│   ├── run_spec_sweep.sh       # Phase 2: speculative decoding sweep
│   ├── run_acceptance.sh       # Phase 2b: acceptance rates by task/thinking mode
│   └── run_profiling.sh        # Phase 3: rocm-smi hardware profiling
│
├── prompts/
│   ├── code_gen.json           # 10 MT-Bench coding prompts
│   ├── reasoning.json          # 10 MT-Bench structured reasoning prompts
│   └── chat.json               # 10 ShareGPT open-ended chat prompts
│
├── analysis/
│   ├── parse_results.py        # Raw JSON + logs → tidy CSV
│   ├── run_anova.py            # ANOVA, threshold estimation, α* identification
│   ├── plot_speedup.py         # Throughput and speedup ratio figures
│   ├── plot_acceptance.py      # Acceptance rate figures
│   └── plot_profiling.py       # Resource utilization figures
│
├── data/                       # gitignored — generated data
│   ├── raw/                    # llama-bench JSON output
│   ├── logs/                   # llama-cli acceptance rate logs
│   └── profiling/              # rocm-smi CSV profiling sessions
│
├── models/                     # gitignored — GGUF model files
└── results/
    └── figures/                # Generated plots
```

---

## Experimental Design

### Independent variables

| Variable | Values | Role |
|---|---|---|
| VRAM residency / `-ngl` | 0, 16, 32, 48, 64 layers (32B model) + 99 (14B, fully resident) | Primary IV — determines PCIe vs. VRAM bottleneck |
| Task type | Code generation, structured reasoning, open-ended chat | Secondary IV — controls acceptance rate |
| Thinking mode | `/think` vs. `/no_think` | Secondary IV — hypothesized to change draftability |
| Speculation length γ | 4, 6, 8, 10 tokens | Optimization variable — swept per condition |
| Spec decoding on/off | Baseline vs. `--model-draft` | Treatment variable |

### What is fixed

| Variable | Fixed value | Why |
|---|---|---|
| Quantization | Q4_K_M | Most common consumer format; most contested point for spec decoding |
| Draft model quantization | Q4_K_M | Matches target; SpecMQuant showed mismatched draft quant degrades α |
| Context length | 8,192 tokens | Large enough for thinking-mode chains (1,000–4,000 tokens) |
| Batch size | 1 | Single-user local inference — the consumer case |
| Model family | Qwen3 | Controls vocabulary mismatch (a known confound) |

### Measurement protocol

Each llama-bench configuration runs 7 times: 1 cold-start discard, 1 warm-start discard, then 5 measured. The median of the 5 measured runs is the reported throughput. The median is preferred over mean to reduce sensitivity to thermal throttling and OS jitter.

**Speedup ratio** = `tokens/sec (spec)` / `tokens/sec (baseline)` at matching model + NGL configuration.
- Above 1.0 → speculative decoding is beneficial
- Below 1.0 → speculative decoding is making inference slower

---

## Why llama-bench, Not llama-server

The proposal explicitly excludes `llama-server` as the primary benchmarking tool. Its continuous batching loop reduces draft token generation by 10–15% relative to `llama-bench` and `llama-cli`, which would systematically understate speculative decoding performance and make results non-comparable to community benchmarks. `llama-bench` and `llama-cli` run in single-request mode, which is correct for single-user local inference.

---

## Expected Run Count

| Condition | Count |
|---|---|
| 14B baseline | 1 NGL × 7 runs = 7 |
| 14B speculative | 4 γ × 7 runs = 28 |
| 32B baseline | 5 NGL × 7 runs = 35 |
| 32B speculative | 5 NGL × 4 γ × 7 runs = 140 |
| Acceptance rate (14B) | 3 tasks × 2 modes × 10 prompts × 4 γ = 240 llama-cli runs |
| Profiling sessions | ~4 targeted sessions (~40 min each including warm-up) |
| **Total timed measurements** | **~450 + 240 acceptance runs** |

---

## Timeline

| Week | Phase | Goal |
|---|---|---|
| 1 | Environment setup | Build llama.cpp; download models; verify GPU detection and vocab alignment |
| 2 | Baseline characterization | Non-speculative throughput sweep across all NGL values; confirm residency effect is observable without spec decoding |
| 3–4 | Main speculative sweep | Full IV sweep: spec on/off × NGL × γ (+ acceptance rate sweep in parallel) |
| 5 | Profiling and analysis | Targeted profiling at crossover conditions; ANOVA; threshold estimation; α* identification |
| 6 | Write-up | Final report; GitHub reproducibility package |

---

## Troubleshooting

### `hipcc not found` during build
ROCm is installed but not in PATH. Add to your shell config:
```bash
export PATH="/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
```

### llama.cpp reports wrong GPU arch or falls back to CPU
Check your GPU arch: `rocminfo | grep gfx`. If it's not `gfx1031`, update `AMDGPU_TARGET` in `setup/build_llamacpp.sh` and rebuild.

### Out of memory (OOM) during 14B run
The 14B model is estimated at ~10.2 GB with the draft model and KV cache at 8192 context. If you OOM:
1. Reduce context: change `CTX_SIZE=8192` to `CTX_SIZE=4096` in `config.sh` (this may truncate thinking-mode outputs)
2. Try reducing the number of GPU layers slightly (`GPU_LAYERS_14B=36`) to offload the last few layers

### Vocabulary mismatch error
`verify_setup.sh` checks this. If it fires, confirm you downloaded Qwen3-0.6B (not Qwen3.5-0.8B, which has a different tokenizer with 248,320 vocab size and will silently fail).

### `huggingface-cli` download fails or is slow
Downloads are resumable — just re-run the same command. If you're on a slow connection, consider using `--quiet` flag or running overnight. The 32B model is 19 GB.

### rocm-smi shows wrong values or crashes
Some versions of ROCm have changed rocm-smi's output format. `run_profiling.sh` parses specific field names — if they've changed, check `rocm-smi --help` and update the `grep` patterns in the script.

---

## Adapting This for Other Hardware

If teammates have different GPUs, update `config.sh` before running on their machine:

```bash
# For NVIDIA (RTX 3090, 4090, etc.)
# Build llama.cpp with: cmake -DGGML_CUDA=ON
# Replace rocm-smi with nvidia-smi in run_profiling.sh
# GPU_ID=0 stays the same

# For 16 GB GPU (RTX 3080, 4070 Ti, RX 7900 XT)
# 14B model still fits; 32B still offloads
# GPU_LAYERS_14B=99 still correct
```

The sweep scripts and analysis pipeline are hardware-agnostic — only the build step and the profiling monitoring commands differ.

---

## Related Work

| Paper | Finding | Gap this project addresses |
|---|---|---|
| Leviathan et al. 2023; Chen et al. 2023 | Foundational spec decoding: 2–3× on T4/TPU, FP16 | FP16 only; no quantization; no consumer hardware; no residency variable |
| EAGLE / EAGLE-2 (2024) | 2.5–3.5× on A100 with autoregressive draft heads | A100 only; no GGUF; no consumer GPU |
| SpecMQuant (2025) | Tree-style verification undermines W4 gains on A100 | A100 + Marlin W4A16 kernels only; not GGUF Q4_K_M; no residency variable |
| SpecKV (2025) | Optimal γ varies with compression level | BitsAndBytes only; no GGUF; no residency variable |
| TaskSpec (2025) | α varies 20–73% by task type on FP16 | Enterprise hardware; no GGUF; no thinking-mode variable |
| MoESD (2025) | Spec decoding net-negative for all MoE on consumer hardware | MoE only; dense transformers not tested |
| SpecExec (2024) | Spec decoding with full-model CPU offload (multi-GPU) | Full offload only; partial -ngl regime not studied |

---

## License

Code in this repository is MIT licensed. Model weights (Qwen3 family) are Apache 2.0. Evaluation prompts: MT-Bench is open-source; ShareGPT is CC BY 4.0.
