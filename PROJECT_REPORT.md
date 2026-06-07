# When Does Speculative Decoding Actually Help?

**Characterizing VRAM Residency, Speculation Length, and Task Draftability as Determinants of Speedup for Dense GGUF Models on Consumer Hardware**

Ali Karim · George Lee · Evan Li — University of Washington

---

## Abstract

Speculative decoding is widely recommended for accelerating LLM inference, but practitioners report inconsistent results — one setup gets a 3× speedup, another gets a slowdown. We run a controlled study on two dense Qwen3 models (14B and 32B) under GGUF Q4\_K\_M quantization on an NVIDIA RTX 6000 Ada GPU, sweeping GPU layer count (VRAM residency), speculation length γ, task type, and thinking mode. Our results show that (1) γ=4 is consistently the best speculation length across all conditions, (2) the 32B model benefits from speculative decoding even under heavy CPU offload (1.3–1.5× speedup at γ=4), (3) the fully VRAM-resident 14B model gains only marginally (1.07× at γ=4) because it is already bandwidth-saturated, and (4) contrary to our hypothesis, thinking mode (`/think`) slightly *reduces* draft acceptance rate rather than improving it — task type is the dominant draftability factor, with reasoning prompts achieving ~50% acceptance vs. ~32% for chat.

---

## 1. Introduction

Speculative decoding accelerates autoregressive LLM inference by running a small, fast "draft" model to propose k tokens at once, then verifying them in a single forward pass of the larger target model. Because verification is parallel, it costs roughly the same as generating one token — so if most drafted tokens are accepted, you get k−1 tokens nearly for free.

In practice, reported speedups vary wildly. The academic literature shows 2–3× on A100s with FP16 models, but practitioners on consumer hardware with quantized GGUF models report everything from 1.5× to outright slowdowns. We hypothesize three confounds:

1. **VRAM residency:** Partially CPU-offloaded models have a different bandwidth bottleneck than fully GPU-resident models. Speculative decoding's benefit depends on which resource is saturated.
2. **Speculation length γ:** Most practitioners use a default of 4–8. The optimal value is task-dependent.
3. **Task draftability:** Some output distributions (structured reasoning) are more predictable than others (open-ended chat), yielding higher acceptance rates.

We further hypothesize that Qwen3's thinking mode (`/think`), which generates chain-of-thought before answering, produces more repetitive and predictable token sequences — making it more draftable.

---

## 2. Setup

### Hardware and Software

- **GPU:** NVIDIA RTX 6000 Ada Generation (48 GB GDDR6 ECC) on UW Hyak Klone cluster, partition `gpu-rtx6k`
- **Inference engine:** llama.cpp, CUDA backend (`-DGGML_CUDA=ON`)
- **Benchmarking tools:** `llama-bench` for throughput; `llama-speculative` for spec decoding; `llama-cli` for acceptance rate logging

### Models

| Role | Model | Quantization | VRAM footprint |
|---|---|---|---|
| Target (small) | Qwen3-14B | Q4\_K\_M | ~8.5 GB |
| Target (large) | Qwen3-32B | Q4\_K\_M | ~19 GB |
| Draft (all runs) | Qwen3-0.6B | Q4\_K\_M | ~0.4 GB |

All three models share a 151,936-token vocabulary, eliminating tokenizer mismatch as a confound. Q4\_K\_M is the most widely used consumer quantization format.

### Why These Models

The 14B model fits entirely in VRAM (8.5 GB of 48 GB available), establishing a fully GPU-resident baseline. The 32B model (19 GB) also fits fully at ngl=64, but we sweep ngl across {0, 16, 32, 48, 64} to artificially vary how many layers are GPU-resident vs. CPU-offloaded, creating a controlled residency gradient.

### Experimental Variables

| Variable | Values | Role |
|---|---|---|
| Model | Qwen3-14B, Qwen3-32B | Size / residency condition |
| GPU layers (`-ngl`) | 0, 16, 32, 48, 64 (32B only) | Controls CPU-offload degree |
| Speculation length γ | 4, 6, 8, 10 | Primary optimization variable |
| Task type | Code generation, Reasoning, Chat | Controls acceptance rate baseline |
| Thinking mode | `/think` vs. `/no_think` | Tests draftability hypothesis |

**Fixed:** context length = 8,192 tokens; batch size = 1 (single-user local inference).

### Measurement Protocol

Each throughput condition runs **7 times**: discard the first (cold-start) and second (warm-start) run, report the **median of the remaining 5**. This reduces sensitivity to thermal throttling and OS jitter.

**Speedup ratio** = `tokens/sec (spec)` / `tokens/sec (baseline)` at the same model and ngl. Values above 1.0 indicate a net benefit; below 1.0 indicate speculative decoding is making inference slower.

Acceptance rate is measured via `llama-cli` with 10 prompts per task × thinking mode combination.

---

## 3. Results

### 3.1 Throughput and Speedup

#### Qwen3-14B (fully VRAM-resident, ngl=99)

| γ | Median tok/s | Speedup vs. baseline (53.1 tok/s) |
|---|---|---|
| Baseline | 53.1 | 1.000× |
| 4 | 56.8 | **1.069×** |
| 6 | 48.8 | 0.918× |
| 8 | 52.4 | 0.988× |
| 10 | 51.0 | 0.960× |

For the fully resident 14B model, only γ=4 yields a net improvement (+6.9%). Larger speculation windows are net-negative. This is consistent with the model being already bandwidth-saturated on the GPU — the verification overhead of longer drafts erodes the savings.

#### Qwen3-32B: Speedup by NGL and γ

| ngl (GPU layers) | γ=4 | γ=6 | γ=8 | γ=10 |
|---|---|---|---|---|
| 0 (full CPU) | **1.318×** | 1.175× | 0.997× | 0.987× |
| 16 | **1.507×** | 1.342× | 1.139× | 0.890× |
| 32 | **1.487×** | 1.229× | 1.017× | 0.978× |
| 48 | **1.361×** | 1.276× | 1.102× | 0.944× |
| 64 (mostly GPU) | **1.332×** | 1.065× | 1.258× | 1.191× |

**Key findings:**
- γ=4 wins at every NGL value without exception.
- γ=10 is net-negative at every NGL value except 64 (where it barely helps).
- The 32B model benefits from spec decoding even at ngl=0 (full CPU offload) — 1.32× speedup at γ=4. This is counter to the intuition that CPU offload should make spec decoding useless.
- Peak speedup occurs at ngl=16: **1.51×** at γ=4. This may reflect a bandwidth sweet spot where partial GPU residency creates idle GPU cycles that spec decoding's verification pass can fill.

### 3.2 Acceptance Rate by Task and Thinking Mode

Mean draft token acceptance rate (%) across 10 prompts per cell, all at γ=4–10 combined:

| Task | No-think | Think | Difference |
|---|---|---|---|
| Chat | 33.2% | 30.5% | −2.7pp |
| Code | 39.3% | 35.9% | −3.4pp |
| Reasoning | 48.9% | 48.0% | −0.9pp |
| **Overall** | **40.5%** | **38.1%** | **−2.4pp** |

**Thinking mode does not improve draftability.** In all three task categories, no-think mode achieves equal or higher acceptance rate. The differences are small (−0.9pp to −3.4pp) but directionally consistent: thinking mode slightly hurts.

**Task type is the dominant factor.** Reasoning prompts (48.5% overall) are dramatically more draftable than chat prompts (31.8%) — a 16.7 percentage point gap. Code sits in between at 37.6%. This makes intuitive sense: reasoning outputs are internally consistent and step-to-step predictable; chat responses are more variable.

---

## 4. Discussion

### Why γ=4 always wins

Speculative decoding's expected gain is governed by acceptance rate α and the draft/verify cost ratio. When α is ~30–50% (our range), longer drafts waste verification budget on tokens that will be rejected. The 0.6B draft model simply isn't accurate enough at positions 5–10 to justify the overhead. γ=4 hits the acceptance rate "sweet spot" where accepted tokens outweigh the overhead on every configuration tested.

### Why 32B benefits more than 14B

The 14B model is fast (~53 tok/s). Its GPU is already busy generating tokens, and the marginal cost of verification is relatively high. The 32B model at any NGL is slower (1.3–25 tok/s depending on offload), meaning each token costs more wall-clock time. When a spec window succeeds, the savings are proportionally larger because the verification pass is cheaper relative to the alternative of generating each token one at a time over PCIe or CPU.

### Why thinking mode hurts acceptance rate

Our hypothesis was that `/think` output — structured chain-of-thought — would be more predictable because reasoning commits to logical paths. This appears to be wrong, at least for the Qwen3-0.6B draft model. One explanation: the 0.6B model wasn't specifically trained to predict Qwen3-14B's thinking-mode reasoning traces. The `<think>` distribution involves longer, more complex sentence constructions than the 0.6B model is calibrated on. The near-zero effect on reasoning tasks (−0.9pp) suggests thinking mode and no-think mode are nearly identical in draftability for structured reasoning, but thinking mode imposes a mild penalty on chat and code tasks where the CoT preamble adds unpredictable tokens before the actual answer.

### Practical recommendation

For single-user local inference with Qwen3 Q4\_K\_M GGUF models in llama.cpp:
1. **Use γ=4.** Larger windows reliably hurt.
2. **Spec decoding is worth enabling for the 32B model** regardless of how many layers are offloaded to CPU — you get 1.3–1.5× improvement.
3. **For the 14B model, the gain is marginal** (6.9%). Only use spec decoding if you have other prompts suggesting high draftability.
4. **Don't rely on thinking mode to improve spec decoding.** It doesn't. If your task involves reasoning-heavy prompts, you'll get higher acceptance rate than chat regardless of thinking mode.

---

## 5. Related Work

| Paper | Finding | How we differ |
|---|---|---|
| Leviathan et al. 2023; Chen et al. 2023 | 2–3× speedup on T4/TPU, FP16 | FP16 only; no quantization; no consumer hardware |
| EAGLE / EAGLE-2 (2024) | 2.5–3.5× on A100 with autoregressive draft heads | A100 only; custom draft architecture; no GGUF |
| Dovetail (EMNLP 2025) | CPU/GPU heterogeneous spec decoding; 1.79–10.1× | PyTorch int8 quantization; custom model; not GGUF |
| SpecOffload (2025) | Spec decoding in CPU offload pipeline; 2.54× | Mixtral MoE models; not GGUF/llama.cpp |
| SpecMemo (2025) | VRAM budget lower bound for mobile spec decoding | Mobile GPU focus; not llama.cpp Q4\_K\_M |
| TaskSpec (2025) | α varies 20–73% by task on FP16 | Enterprise hardware; no quantization; no thinking mode |

The GGUF/llama.cpp inference stack — which represents the majority of practitioner local inference deployments — remains underrepresented in the academic literature. Our study provides directly actionable benchmarks for this setting.

---

## 6. Limitations

- **Single hardware configuration:** Results are specific to RTX 6000 Ada (48 GB). Different PCIe bandwidth, VRAM sizes, or GPU generations will shift the crossover points.
- **Single model family:** Qwen3 vocab-matched draft/target pairs may have higher acceptance rates than cross-family pairs. Results may not generalize to e.g. Llama + Qwen draft.
- **No quality validation:** We measure throughput only. Speculative decoding is mathematically lossless (identical output distribution), but we did not empirically verify output quality is preserved.
- **γ granularity:** We sweep γ ∈ {4, 6, 8, 10}. The true optimum might lie between these values.
- **10 prompts per cell for acceptance rate:** Small sample; estimates have wide confidence intervals, particularly for high-variance tasks like chat.

---

## 7. Conclusion

We conducted a controlled study of speculative decoding across 24 (model × ngl × γ) throughput conditions and 600 acceptance rate measurements on dense Qwen3 GGUF models. The key findings are:

1. **γ=4 is optimal across all tested conditions.** Longer speculation windows (γ=8, 10) are net-negative at realistic acceptance rates.
2. **The 32B model benefits substantially from spec decoding even under full CPU offload** (1.3× at γ=4, ngl=0), with peak speedup 1.51× at ngl=16.
3. **The 14B fully VRAM-resident model gains only 6.9%** at the best γ, suggesting bandwidth saturation limits the benefit.
4. **Thinking mode does not improve draftability** — it slightly reduces acceptance rate. Task type is the dominant factor: reasoning tasks are ~17 percentage points more draftable than chat.

These results suggest that speculative decoding is most valuable for large models under partial CPU offload, that practitioners should default to γ=4, and that task selection matters more than thinking mode for acceptance rate optimization.

---

## Appendix: Raw Data Summary

**Baseline throughput:**

| Model | ngl | Baseline tok/s |
|---|---|---|
| Qwen3-14B | 99 (full GPU) | 53.1 |
| Qwen3-32B | 0 (full CPU) | 1.32 |
| Qwen3-32B | 16 | 1.77 |
| Qwen3-32B | 32 | 2.58 |
| Qwen3-32B | 48 | 4.47 |
| Qwen3-32B | 64 | 18.93 |

**Acceptance rate by model and task (γ=4, all thinking modes combined):**

| Task | Mean α% | Std |
|---|---|---|
| Reasoning | 48.5% | 11.2% |
| Code | 37.6% | 13.3% |
| Chat | 31.8% | 11.4% |
