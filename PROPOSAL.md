# Project Proposal: Speculative Decoding Bottlenecks on the Qwen3.5 Family

**CSE 493G1 / 599G1 — Deep Learning for Computer Vision, Spring 2026**
**Evan Li — Solo**

---

## Research Question

> *How does draft model size affect speculative decoding efficiency within the Qwen3.5
> family, and does crossing the dense-to-MoE architecture boundary in the target model
> change this relationship? Does Qwen3.5's thinking mode systematically alter draft
> acceptance rate?*

Speculative decoding (SD) accelerates autoregressive LLM inference by using a cheap draft
model to propose k tokens, then verifying all k in a single parallel target forward pass.
The speedup rests on one critical assumption: in the memory-bound regime, verifying k tokens
costs roughly the same as verifying 1, because weight loading dominates and weights are
loaded once regardless of sequence length. This assumption holds cleanly for dense targets.

For Mixture-of-Experts (MoE) targets it may not. Each draft token routes to a different
expert subset; verifying k tokens forces the target to load the union of all activated
experts across the draft tree. Recent work (MoE-Spec, arXiv:2602.16052; Utility-Driven SD,
arXiv:2506.20675; MoESD, arXiv:2505.19645) has begun studying SD on MoE targets but no
paper has established empirical scaling laws for **dense-draft-to-MoE-target** pairings
using the Qwen3.5 family specifically — this is explicitly identified as an open research
direction in the SD scaling law literature (arXiv:2505.07858).

---

## Motivation

Three specific gaps motivate this project:

1. **Draft size sweep within Qwen3.5 is unstudied.** Prior within-family scaling work
   covers OPT and LLaMA (arXiv:2402.01528, NAACL 2025). The finding — larger draft raises
   acceptance rate but reduces throughput due to latency growth — has not been reproduced
   for Qwen3.5, which has a different architecture, training recipe, and tokenizer. The
   optimal draft size is not known for this family.

2. **MoE verification cost is non-constant.** On OLMoE-1B-7B, a 127-token draft tree
   activates 54 of 64 experts per layer (MoE-Spec, 2026), approaching full-model cost.
   Whether this problem is severe or mild for Qwen3.5-35B-A3B (only 3B of 35B active per
   token) is unknown. Crucially, the draft size sweep interacts with MoE verification cost
   in a way no paper has characterized: larger drafts produce higher-quality tokens, which
   may route to more concentrated expert subsets, reducing the expert-overactivation problem.

3. **Thinking mode.** All Qwen3.5 instruct models support a `<think>` token that triggers
   extended chain-of-thought generation before the final answer. These tokens may be easier
   for a dense draft to predict (formulaic reasoning steps) or harder (highly context-
   dependent). No paper has measured this. The effect could be large given that thinking
   responses can be 10x longer than standard responses.

---

## Planned Experiments

### Models

All models are fully resident on GPU in bf16 simultaneously (draft + target co-resident,
no CPU offload):

| Role | Model | bf16 VRAM | HuggingFace |
|---|---|---|---|
| Draft | Qwen3.5-0.8B | 1.6 GB | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| Draft | Qwen3.5-2B | 4.0 GB | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Draft | Qwen3.5-4B | 8.0 GB | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Draft | Qwen3.5-9B | 18.0 GB | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Target (dense) | Qwen3.5-27B | 54.0 GB | Qwen/Qwen3.5-27B |
| Target (MoE) | Qwen3.5-35B-A3B | 70.0 GB | Qwen/Qwen3.5-35B-A3B |

### Experiment Matrix

**Axis 1 — Draft size sweep (target fixed at 27B dense):**

| Condition | Draft | Target | Mode | GPU layout |
|---|---|---|---|---|
| A1 | 0.8B | 27B | standard | 2× L40S (55.6 GB) |
| A2 | 2B | 27B | standard | 2× L40S (58 GB) |
| A3 | 4B | 27B | standard | 2× L40S (62 GB) |
| A4 | 9B | 27B | standard | 2× L40S (72 GB) |
| Baseline | — | 27B | standard | 2× L40S (autoregressive) |

This axis answers: *what is the optimal draft size for Qwen3.5-27B, and does the
acceptance-rate / throughput tradeoff from OPT/LLaMA transfer to Qwen3.5?*

**Axis 2 — Architecture crossing (draft fixed at 4B, best from Axis 1 expected):**

| Condition | Draft | Target | Mode | GPU layout |
|---|---|---|---|---|
| B1 | 4B | 27B (dense) | standard | 2× L40S (62 GB) |
| B2 | 4B | 35B-A3B (MoE) | standard | 2× L40S (78 GB) |
| B3 | 4B | 35B-A3B (MoE) | thinking | 2× L40S (78 GB) |

This axis answers: *does the dense→MoE architecture boundary change the SD tradeoff, and
does thinking mode help or hurt draft acceptance?*

Note: 9B + 35B-A3B = 88 GB, exceeding 2× L40S (96 GB usable after KV cache). That pairing
uses 2× A100 80GB if needed, but is not in the primary matrix.

### What I Will Measure

For each condition, across batch sizes {1, 4, 8, 16}:
- **Token acceptance rate (TAR):** accepted tokens / draft tokens proposed
- **Throughput:** tokens/second (wall-clock)
- **Speedup ratio:** throughput with SD / throughput of autoregressive baseline
- **Mean accepted length:** average consecutive accepted tokens per SD step

### Target Plots

**Plot 1 — TAR vs draft model size (line chart):**
Conditions A1–A4, batch size 1. X-axis: draft parameter count (log scale). Y-axis:
acceptance rate. Expected shape: monotonically increasing, diminishing returns.
This directly measures whether Qwen3.5's within-family alignment scales with draft size.

**Plot 2 — Throughput speedup vs draft model size (line chart):**
Same conditions. X-axis: draft size. Y-axis: tokens/sec speedup over baseline. Expected
shape: peaks somewhere in the 2B–4B range, then declines as draft latency dominates.
This is the key result — the optimal draft size for Qwen3.5-27B.

**Plot 3 — Dense vs MoE target (grouped bar chart):**
Conditions B1 vs B2 at batch sizes {1, 4, 8, 16}. Two metrics side by side: TAR and
speedup ratio. Tests whether the MoE boundary changes the draft size→speedup relationship,
and whether the non-monotonic batch size pattern from MoESD (arXiv:2505.19645) appears.

**Plot 4 — Thinking mode effect (paired bar chart):**
Conditions B2 vs B3: TAR and speedup for standard vs thinking mode on the MoE target.
If measurable: broken down by token phase (thinking tokens vs answer tokens), since
thinking tokens should exhibit different draft-predictability than answer tokens.

---

## Dataset / Prompts

Three prompt categories, 50 prompts each (150 total):

- **Standard QA:** short factual questions (Wikipedia-style). Minimal thinking in thinking
  mode.
- **Reasoning/math:** GSM8K-style multi-step problems. Long thinking-phase responses.
- **Code generation:** function completion tasks. Mixed thinking behavior.

Task category is a covariate in all plots — this allows checking whether optimal draft size
is task-dependent, which the instructor rubric rewards as deeper analysis.

---

## Implementation

### Framework

**vLLM** (production SD support) with MTP speculative decoding:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --tensor-parallel-size 2 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --reasoning-parser qwen3
```

Acceptance rate from Prometheus metrics:
```
rate(vllm:spec_decode_num_accepted_tokens_total[1m])
/ rate(vllm:spec_decode_num_draft_tokens_total[1m])
```

Per-position acceptance rate vector (`vllm:spec_decode_num_accepted_tokens_per_pos`)
enables the phase breakdown in Plot 4.

For expert activation coverage, I will instrument the MoE routing layer in HuggingFace
Transformers via a forward hook — vLLM alone does not expose per-expert activation counts.
This is a bonus measurement on top of the core vLLM runs.

### Hardware (Hyak Klone)

| Condition set | GPU layout | Partition |
|---|---|---|
| A1–A4 (draft sweep, 27B target) | 2× L40S | cse/gpu-l40s |
| B1–B2 (MoE crossing, 35B-A3B target) | 2× L40S | cse/gpu-l40s |
| B3 (thinking mode) | 2× L40S | cse/gpu-l40s |
| Autoregressive baselines | 2× L40S | cse/gpu-l40s |

All model weights stored in `/gscratch/<group>/` — not `/gscratch/scrubbed/` (auto-deletes
after 21 days of inactivity).

**SLURM job template:**

```bash
#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --job-name=sd-sweep

module load cuda/12.4
conda activate vllm-env

python run_sd_experiment.py \
  --draft Qwen/Qwen3.5-${DRAFT_SIZE} \
  --target Qwen/Qwen3.5-27B \
  --batch-sizes 1 4 8 16 \
  --num-speculative-tokens 4 \
  --prompts data/prompts.json \
  --output results/${CONDITION}.json
```

Conditions A1–A4 and B1–B3 are independent SLURM jobs and run in parallel. The 8-hour
checkpoint requeue limit is not a concern — each condition runs in under 3 hours.

---

## Related Work

**Decoding Speculative Decoding (arXiv:2402.01528, NAACL 2025):** Empirically shows that
within-family draft size scaling (OPT, LLaMA) raises acceptance rate but reduces throughput
due to latency growth. My project replicates this design on Qwen3.5, a more recent and
architecturally distinct family, and extends it to an MoE target.

**MoESD (arXiv:2505.19645, 2025):** First systematic study of SD on MoE targets. Shows
non-monotonic speedup vs batch size on Qwen2-57B-14A-Instruct (2.29x peak). My project
extends this to Qwen3.5-35B-A3B and adds the draft size sweep as a controlled variable
that MoESD does not study.

**SD Scaling Laws (arXiv:2505.07858, 2025):** Establishes log-linear scaling laws for
acceptance rate on dense architectures (Llama2/3, Qwen2.5). Explicitly identifies MoE
draft optimization as an open research direction — the empirical gap my project addresses.

**Utility-Driven Speculative Decoding (arXiv:2506.20675, 2026):** Identifies that optimal
k is non-static for MoE targets (varying by task and request), motivating my use of
multiple prompt categories to measure task-dependence.

---

## Potential Concerns and Mitigations

**Concern: This is a systems paper, not a vision paper.**
Mitigation: The instructor rubric rewards narrow, testable research questions, good
experimental design, and honest analysis. I frame this as a model behavior question: how
does draft model size and target architecture shape what a neural network accepts or rejects
from a smaller model predicting its outputs? That is a learning question, not a networking
question.

**Concern: vLLM's MTP method may have bugs on these specific models.**
Mitigation: A forum report noted `num_speculative_tokens=2` failing on Qwen3.5-27B-FP8. I
run a smoke test on the smallest pair (0.8B draft → 27B target, batch size 1) before
committing to the full matrix. HuggingFace native SD is the fallback.

**Concern: Nine conditions × four batch sizes = 36 runs is a lot for two days.**
Mitigation: All conditions are independent and submit to SLURM in parallel. With 8 free
L40S GPUs, all Axis 1 conditions (A1–A4 + baseline) run simultaneously. Axis 2 conditions
(B1–B3) run in a second wave. Wall-clock time is ~6 hours total across both waves, plus
queue wait time.

**Concern: Optimal draft from Axis 1 may not be 4B.**
Mitigation: Axis 2 uses 4B as a placeholder pending Axis 1 results. If Axis 1 shows the
optimal is clearly 2B or 9B, I update Axis 2 conditions B1–B3 before submitting that wave.
The two-wave structure allows this.

---

## Expected Findings

Based on prior literature, I expect:

- **Axis 1:** TAR increases monotonically with draft size (0.8B → 9B). Throughput speedup
  peaks at 2B or 4B and declines for 9B — consistent with OPT/LLaMA results. The optimal
  draft is well below the target size (27B), likely under 15% of target parameter count.

- **Axis 2, dense vs MoE:** Speedup for the MoE target (B2) will be lower than dense (B1)
  at batch size 1 due to expert activation overhead, but will close or exceed B1 at batch
  sizes 4–8, consistent with MoESD's non-monotonic pattern.

- **Axis 2, thinking mode:** Thinking tokens will show higher TAR than answer tokens (more
  predictable phrasing), but the longer thinking sequences may not translate to proportional
  wall-clock gain if TTFT dominates over TPOT in that regime.

Negative results are valid. If draft size has no effect above 2B, or if MoE target behaves
identically to dense, those are clean findings with direct practical implications for
Qwen3.5 deployment.
