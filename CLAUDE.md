# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: When Does Speculative Decoding Actually Help?

A controlled UW research study (Karim, Lee, Li) measuring whether speculative decoding produces net throughput gains on consumer-class hardware (GGUF / llama.cpp). The **primary hypothesis** is that VRAM residency — whether the target model fits in GPU VRAM vs. partially offloads to CPU over PCIe — is the dominant determinant of speedup. Secondary factors: task type (code / reasoning / chat), Qwen3 thinking mode (`/think` vs `/no_think`), and speculation length γ.

Target hardware: **NVIDIA RTX 6000** (UW cluster partition `gpu-rtx6k`). Models are Qwen3 GGUFs at Q4_K_M:
- Draft: `Qwen3-0.6B` (~0.4 GB)
- Target small (VRAM-resident): `Qwen3-14B` (~8.5 GB)
- Target large (CPU-offload sweep): `Qwen3-32B` (~19 GB) — won't fit; swept via `-ngl ∈ {0,16,32,48,64}` out of 64 layers.

## Architecture / Pipeline

The repo is a thin orchestration layer around `llama.cpp` binaries plus a Python analysis stack. Everything flows through one config file:

- [config.sh](config.sh) is sourced by every shell script. It defines binary paths, model paths, the NGL sweep, γ values `(4,6,8,10)`, fixed `CTX_SIZE=8192`, `N_GEN=512`, `N_PROMPT=128`, and `REPETITIONS=7` (first 2 runs are cold/warm and dropped in analysis).
- **Important quirk**: in this build, `llama-bench` does NOT support speculative decoding. Baselines use `llama-bench` (clean JSON output); spec runs use the separate `llama-speculative` binary. The parser handles both formats.
- `LLAMA_CPP_DIR` defaults to `/gscratch/scrubbed/$USER/llama.cpp` (cluster scratch). Override via env var when working locally.
- The top line of [config.sh](config.sh) currently contains a stray `${LLAMA_BENCH} ...` invocation glued before the shebang — likely accidental; leave alone unless asked.

### Phase scripts (run in order)

1. [scripts/run_baseline.sh](scripts/run_baseline.sh) — `llama-bench` non-spec runs → `data/raw/baseline_{model}_ngl{N}.json`. Skips if output exists.
2. [scripts/run_spec_sweep.sh](scripts/run_spec_sweep.sh) — `llama-speculative` across `{14b@ngl99, 32b@NGL_SWEEP} × γ`. Forces fixed γ by setting `--spec-draft-n-min == --spec-draft-n-max`. Per-run JSONL is appended to `data/raw/spec_*.jsonl`; tps and accept% are grep'd out of the binary's stderr.
3. [scripts/run_acceptance.sh](scripts/run_acceptance.sh) — uses `llama-speculative` (despite the comment saying `llama-cli`) on real prompts from [prompts/](prompts/), prepending `/think` or `/no_think` to control Qwen3 thinking mode. 32B side spot-checks `NGL_ACCEPTANCE=(16 32 48)` with only γ∈{4,8} to limit runtime.
4. [scripts/run_profiling.sh](scripts/run_profiling.sh) — two-phase nvidia-smi sampling (2 min thermal discard + 8 min steady-state) around looped `llama-bench` runs at crossover conditions. Note: the profiling phase uses `llama-bench` even in "spec" mode (passing `-md/-ngld/-d`), relying on those flags being accepted — verify against the installed llama.cpp version before trusting "spec" profile sessions.

### Analysis

- [analysis/parse_results.py](analysis/parse_results.py) consolidates `baseline_*.json`, `spec_*.jsonl`, and `accept_*.log` into `results/all_results.csv` + `results/acceptance_rates.csv`. Drops the first 2 runs to match the cold/warm discard policy, then computes `speedup_ratio = spec.median_ts / baseline.median_ts` joined on `(model, ngl)`.
- [analysis/run_anova.py](analysis/run_anova.py) does descriptive speedup summary, piecewise-linear residency-threshold fit on the 32B NGL sweep, and α* (minimum viable acceptance) identification. Uses `pingouin` for ANOVA when present, else falls back to descriptives.
- Plotting modules are independent and read the CSVs.

### Output layout (all gitignored)

```
data/raw/         baseline JSON + spec JSONL
data/logs/        stderr + per-run llama-speculative logs + accept_*.log
data/profiling/   nvidia-smi CSV + per-session bench JSON
results/          all_results.csv, acceptance_rates.csv
results/figures/  generated PNGs
```

## Common Commands

```bash
# One-time setup
conda env create -f environment.yml && conda activate speculative-decoding
bash setup/build_llamacpp.sh          # builds with -DGGML_CUDA=ON
bash setup/download_models.sh         # 0.6B + 14B
bash setup/download_models.sh --32b   # add 32B (~19 GB)
bash setup/verify_setup.sh            # smoke test before benchmarking

# Experiments (each phase respects existing outputs and skips)
bash scripts/run_baseline.sh
bash scripts/run_spec_sweep.sh
bash scripts/run_acceptance.sh
bash scripts/run_profiling.sh 32b 32 4 spec   # <model> <ngl> <gamma> <spec|base>

# Analysis
python analysis/parse_results.py
python analysis/run_anova.py
python analysis/plot_speedup.py
```

To rerun a single condition: delete the matching output file (the scripts skip-if-exists by filename) and re-run the phase script.

## Measurement Protocol (don't change without discussion)

- 7 reps per condition, first 2 always dropped, median (not mean) of the remaining 5 is the reported tps.
- Single-request (`batch=1`) inference — this is the consumer-local case. **Do not switch to `llama-server`**: its continuous batching depresses draft-token throughput 10–15% vs. `llama-bench`/`llama-cli` and was explicitly excluded by the proposal.
- Same Qwen3 family for draft and target (vocab 151,936) — vocabulary mismatch causes silent failures in llama.cpp. `setup/verify_setup.sh` checks this.
- Q4_K_M for both target and draft — SpecMQuant showed mismatched draft quant degrades acceptance.

## Adapting Paths

When teammates run this on different machines, the only thing they should need to edit is [config.sh](config.sh) (binary dir, model paths, GPU id). The sweep and analysis code is hardware-agnostic. For non-NVIDIA backends only `setup/build_llamacpp.sh` and the `nvidia-smi` queries in `run_profiling.sh` change.
