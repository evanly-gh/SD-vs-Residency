# Speculative Decoding on the Qwen3.5 Family — Project Report & Poster Handoff

**Evan Li** · CSE 493G1 / 599G1 — Deep Learning,  Spring 2026, University of Washington CSE · solo project
**Status:** ✅ Complete — 9 conditions × 4 batch sizes on Hyak Klone (2× L40S).
**Artifacts:** figures in `plots/`; raw per-condition JSON in
`/mmfs1/gscratch/intelligentsystems/evanly/sd-qwen35/results/`; this document is
the single source for building the poster.

---

## Abstract

Speculative decoding (SD) accelerates autoregressive LLM inference by proposing
several tokens cheaply and verifying them in one parallel target forward pass. We
study SD on the **Qwen3.5** family, which ships a native **Multi-Token Prediction
(MTP)** head for self-speculation. Across 150 prompts (math / QA / code) and batch
sizes 1–16 we measure how (1) **speculative depth k** trades acceptance against
throughput on a dense 27B target, (2) the **dense→MoE** architecture boundary
changes the payoff, and (3) **thinking (chain-of-thought) mode** affects draft
acceptance. We find a clear **optimal depth of k≈4 (2.88× speedup)** on the dense
27B with sharp degradation by k=6; that the **MoE 35B-A3B target benefits *more*
from speculation than the dense model** (up to 2.95× at batch 16, gap widening with
concurrency); and that **thinking tokens are marginally harder to draft** (≈1.5–2 pt
lower acceptance) than standard tokens.

---

## 1. Research questions

1. **Depth / acceptance tradeoff (Axis 1).** As speculative depth *k* grows, how do
   token acceptance rate (TAR), mean accepted length, and throughput speedup trade
   off, and where is the optimum for the dense Qwen3.5-27B?
2. **Architecture crossing (Axis 2).** Does crossing the **dense → Mixture-of-Experts
   (MoE)** boundary in the target (27B dense vs 35B-A3B MoE, fixed k=3) change the SD
   payoff?
3. **Thinking mode.** Does Qwen3.5's `<think>` chain-of-thought generation
   systematically alter draft acceptance / speedup vs standard mode?

---

## 2. Background & motivation

SD's speedup rests on a memory-bound-regime assumption: verifying *k* proposed
tokens costs roughly the same as verifying one, because weight loading dominates and
weights are read once regardless of sequence length. This holds cleanly for **dense**
targets. For **MoE** targets it may not: each speculated token can route to a
different expert subset, so verifying a *k*-token draft can force the target to load
the union of all activated experts — inflating verification cost. Three gaps
motivate this study:

- **No within-family SD scaling characterization for Qwen3.5.** Prior within-family
  work (OPT, LLaMA) showed a tradeoff between acceptance and latency; it has not been
  reproduced for Qwen3.5, which has a different architecture, training recipe, and an
  explicit MTP speculation path.
- **MoE verification cost is non-constant.** On other MoE models a wide draft tree can
  activate most experts per layer, approaching full-model cost. Whether this is severe
  or mild for Qwen3.5-35B-A3B (only 3B of 35B active per token) was unknown.
- **Thinking mode is unmeasured.** Extended chain-of-thought may be *easier* to draft
  (formulaic reasoning) or *harder* (highly context-dependent); the effect on
  acceptance had not been quantified.

---

## 3. Related work

- **Decoding Speculative Decoding** (arXiv:2402.01528, NAACL 2025): within-family
  draft scaling on OPT/LLaMA — larger drafts raise acceptance but can reduce
  throughput as draft latency grows. We test the analogous tradeoff on Qwen3.5 (along
  the speculative-depth axis) and extend it to an MoE target.
- **MoESD** (arXiv:2505.19645, 2025): first systematic SD study on MoE targets;
  reports non-monotonic speedup vs batch size. We extend to Qwen3.5-35B-A3B.
- **SD Scaling Laws** (arXiv:2505.07858, 2025): log-linear acceptance scaling on dense
  models; explicitly flags MoE draft optimization as open — the gap we address.
- **Utility-Driven SD** (arXiv:2506.20675, 2026): optimal speculation length is
  non-static for MoE targets and varies by task — motivating our multi-category prompt
  set and the per-category breakdown.

---

## 4. Method

### 4.1 Models

| Role | Model | Architecture | bf16 VRAM |
|---|---|---|---|
| Dense target | **Qwen3.5-27B** | dense | ~52 GB |
| MoE target | **Qwen3.5-35B-A3B** | Mixture-of-Experts (3B active / 35B total) | ~67 GB |
| Speculator | each target's **native MTP head** | multi-token prediction | shared with target |

All weights are bf16, fully GPU-resident (no CPU offload), tensor-parallel across
2× L40S.

### 4.2 Speculation mechanism

Qwen3.5 ships a native **MTP** head, and SD is performed via **self-speculation**:
the target's own MTP head proposes *k* tokens, which the target then verifies. We
sweep *k* (the `num_speculative_tokens` knob) as the Axis-1 variable. This is the
SD configuration Qwen3.5 is designed for and the one exposed by the serving stack;
it probes the same acceptance-vs-speedup tradeoff as a draft-model sweep, with
speculative depth as the control variable. (See §8 for why depth — rather than a
separate draft-model size — is the realized Axis-1 knob.)

```bash
# SD condition: target's MTP head, depth k
--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'
# baseline condition: plain autoregressive (no speculative-config)
```

### 4.3 Serving, hardware, decoding

| Item | Value |
|---|---|
| Engine | vLLM 0.22.1 (V1), torch 2.11+cu130, transformers 4.57.6 |
| Hardware | 2× NVIDIA L40S (45 GB ea.), Hyak `cse/gpu-l40s`, tensor-parallel = 2 |
| Decoding | greedy (temperature 0), `max_tokens` = 512 (8192 ctx for thinking) |
| Execution | eager mode (see §8) — uniform across all conditions |

### 4.4 Dataset

150 prompts, fixed seed 42, sha256-checksummed (`data/prompts.json`), three
categories of 50 each — a covariate in all analyses:

- **Math** — GSM8K (multi-step reasoning)
- **QA** — SQuAD v2 (short factual questions, answerable subset)
- **Code** — HumanEval (Python function completion)

### 4.5 Metrics (vLLM Prometheus `spec_decode_*` counters)

- **TAR** (token acceptance rate) = accepted draft tokens / proposed draft tokens
- **Mean accepted length** = 1 + accepted/drafts (verified tokens per decode step)
- **Throughput** = output tokens / wall-clock second
- **Speedup** = SD throughput / matching autoregressive-baseline throughput
  (same target, same batch size)

### 4.6 Conditions (9)

| Condition | Target | Method | k | Mode | Purpose |
|---|---|---|---|---|---|
| `baseline` | 27B dense | none | — | standard | Axis-1 AR denominator |
| `K1`…`K4`, `K6` | 27B dense | MTP | 1,2,3,4,6 | standard | **Axis 1** depth sweep |
| `baseline_moe` | 35B-A3B MoE | none | — | standard | Axis-2 AR denominator |
| `M3` | 35B-A3B MoE | MTP | 3 | standard | **Axis 2** dense-vs-MoE (vs `K3`) |
| `M3_think` | 35B-A3B MoE | MTP | 3 | thinking | **Axis 3** thinking (vs `M3`) |

Batch sizes {1, 4, 8, 16} for every condition.

---

## 5. Results

### 5.1 Axis 1 — speculative depth on dense Qwen3.5-27B

Per-token acceptance, mean accepted length, and speedup (batch size = 1; full
per-batch data in `plots/summary_table.csv`):

| k | TAR | mean accepted len | throughput (tok/s) | speedup |
|---|-----|-------------------|--------------------|---------|
| 0 (baseline) | — | — | 21.0 | 1.00× |
| 1 | 0.943 | 1.94 | 31.7 | 1.51× |
| 2 | 0.887 | 2.77 | 43.0 | 2.05× |
| 3 | 0.824 | 3.47 | 54.6 | 2.60× |
| **4** | **0.758** | **4.03** | **60.5** | **2.88× ← optimum** |
| 6 | 0.641 | 4.85 | 36.2 | 1.73× |

**Mechanism (the headline story).** As depth grows, per-token **acceptance falls
monotonically** (0.94 → 0.64) because the MTP head's deeper predictions are
progressively less reliable. **Mean accepted length keeps rising but sub-linearly**
(1.94 → 4.85, well below the ideal *k*+1), so each step commits more tokens — until,
at k=6, so many proposed tokens are rejected that the wasted draft-and-verify work
**drops throughput below the k=2 level**. The net throughput speedup therefore
**peaks at k≈4 (2.88×) and then collapses (1.73× at k=6)** — a concrete optimal
speculative depth for Qwen3.5-27B.

**Batch-size stability.** Speedup is remarkably flat across batch sizes 1→16 (k=4:
2.88 / 2.99 / 2.94 / 2.79×). MTP self-speculation keeps helping under concurrency —
notable because classic draft-model SD usually decays as the batch fills the GPU.
→ **Figures `plot1_tar_vs_k`, `plot2_speedup_vs_k`** (plot 2 also breaks speedup down
by task category at batch 1).

### 5.2 Axis 2 — dense vs MoE target (k = 3)

Dense Qwen3.5-27B (`K3`) vs MoE Qwen3.5-35B-A3B (`M3`), each vs its own AR baseline:

| batch | dense TAR | dense speedup | MoE TAR | MoE speedup |
|---|---|---|---|---|
| 1 | 0.824 | 2.60× | 0.801 | 2.70× |
| 4 | 0.825 | 2.79× | 0.799 | 2.83× |
| 8 | 0.829 | 2.70× | 0.796 | 2.92× |
| 16 | 0.827 | 2.55× | 0.794 | **2.95×** |

**Finding.** Crossing dense→MoE *slightly lowers* acceptance (0.80 vs 0.82) but
yields a **higher speedup that widens with batch size** (MoE 2.70→2.95× vs dense
2.60→2.55×). The MoE's autoregressive baseline is far more memory-bound (only 3B of
35B active per token ⇒ very low arithmetic intensity, just 10.3 tok/s at batch 1 vs
21.0 for the dense 27B), so batched parallel verification recovers proportionally
more of that lost throughput. For Qwen3.5 this is the **opposite of a "MoE penalty"**
— MoE speculation pays off *more* than dense, monotonically in batch size.
→ **Figure `plot3_dense_vs_moe`.**

### 5.3 Axis 3 — thinking vs standard mode (MoE, k = 3)

`M3` (standard) vs `M3_think` (thinking), both Qwen3.5-35B-A3B at k=3:

| batch | std TAR | think TAR | std tok/s | think tok/s |
|---|---|---|---|---|
| 1 | 0.801 | 0.784 | 27.9 | 27.7 |
| 4 | 0.799 | 0.787 | 63.2 | 96.6 |
| 8 | 0.796 | 0.780 | 102.9 | 184.2 |
| 16 | 0.794 | 0.778 | 170.8 | 341.6 |

**Finding (lead with acceptance).** Thinking-mode tokens are **marginally harder for
the MTP head to predict** — TAR is consistently ~1.5–2 points lower under thinking
(0.78 vs 0.80) at every batch size. Extended chain-of-thought does **not** make
drafting easier; if anything it slightly reduces acceptance.

**Caveat on thinking throughput.** Absolute decode throughput is much higher under
thinking (342 vs 171 tok/s at batch 16), but this is a **batch-utilization effect,
not an SD effect**: long CoT responses keep concurrent requests alive together and
the GPU batch full, whereas the standard-mode baseline drains as short answers
finish. We therefore report the **TAR comparison as the clean result** and do *not*
claim a thinking-induced speculative speedup. → **Figure `plot4_thinking_vs_std`.**

### 5.4 Headline numbers (for the poster)

- **Optimal speculative depth for Qwen3.5-27B: k = 4 → 2.88× speedup.**
  Over-speculation (k=6) collapses to 1.73×.
- **MoE ≥ dense:** Qwen3.5-35B-A3B reaches **2.95× at batch 16**, beating the dense
  27B at every batch and widening with concurrency.
- **Acceptance falls smoothly with depth** (0.94→0.64 for k=1→6) and is **slightly
  lower for the MoE target and for thinking mode** than for dense / standard.

---

## 6. Discussion — hypotheses vs results

| Proposal expectation | Result |
|---|---|
| Acceptance/speedup tradeoff with diminishing returns; a clear interior optimum | **Confirmed** — speedup peaks at k=4, degrades by k=6 |
| MoE *penalty* at small batch, recovering at larger batch (à la MoESD non-monotonicity) | **Refuted** — MoE *out-performs* dense at all batches, monotonically; no penalty observed for Qwen3.5-35B-A3B |
| Thinking tokens easier to draft (higher acceptance) | **Refuted** — thinking acceptance is slightly *lower* |

The MoE result is the most interesting: because Qwen3.5-35B-A3B activates so few
parameters per token, its autoregressive decode is severely memory-bound, and MTP
verification — which amortizes weight loading across *k* tokens — reclaims more of
that headroom than on the denser, more compute-balanced 27B.

---

## 7. Figures (`plots/`, PNG + PDF each)

| File | Content | Suggested poster role |
|---|---|---|
| `plot2_speedup_vs_k` | Speedup vs k, per batch + per task category | **Hero** — the optimal-depth result |
| `plot1_tar_vs_k` | TAR & mean accepted length vs k | The "why" panel beside the hero |
| `plot3_dense_vs_moe` | K3 vs M3 — TAR & speedup by batch | Architecture-crossing panel |
| `plot4_thinking_vs_std` | M3 vs M3_think — TAR & speedup by batch | Thinking-mode panel |
| `summary_table.csv` | Every (condition × batch) row | Backing data / appendix |

---

## 8. Notes on method realization (brief)

- **Axis-1 knob = speculative depth, via native MTP self-speculation.** The released
  Qwen3.5 models perform SD through their built-in MTP head, which is bound to the
  model's own hidden size; the serving stack routes any Qwen3.5 speculator through
  that head. The realized, supported way to vary "how much we speculate" is therefore
  the number of speculative tokens *k* (this study's Axis 1) rather than a separately
  sized draft checkpoint. Both probe the same acceptance-vs-speedup tradeoff.
- **Eager execution.** All conditions run with CUDA-graph/compile disabled for stable,
  unattended startup. This lowers *absolute* tok/s uniformly across every condition;
  acceptance rates and the **SD-vs-baseline speedup ratios** — the reported metrics —
  are unaffected.
- **Serving stack.** Current vLLM (0.22.1) is required to load the Qwen3.5
  architecture; torch/transformers track that requirement.
- The optional MoE per-expert activation instrumentation (a stretch goal) is not
  included in this run.

---

## 9. Reproduce

```bash
# Hyak login node, from the repo root
module load conda/Miniforge3-25.9.1-0
conda activate /mmfs1/gscratch/intelligentsystems/evanly/envs/sd-qwen35

python data/generate_prompts.py            # -> data/prompts.json (seeded, checksummed)
bash slurm/submit_all.sh                    # submit all 9 conditions to SLURM
bash slurm/status.sh                        # monitor queue + per-condition completeness
python analysis/plot_all.py \
  --results-dir /mmfs1/gscratch/intelligentsystems/evanly/sd-qwen35/results \
  --out-dir plots                           # regenerate all figures + summary_table.csv
```

The conda env's `activate.d` redirects all caches to gscratch (HOME has a 10 GB
quota) and sets the CUDA toolkit + sampler flags the engine needs, so every SLURM
job inherits a working environment automatically.

## 10. Repository layout

| Path | What |
|---|---|
| `data/generate_prompts.py` | builds the 150-prompt seeded, checksummed set |
| `run_experiment.py` | launches vLLM (MTP/baseline), runs prompts, collects Prometheus deltas, writes per-batch-size checkpoint JSON |
| `slurm/submit_all.sh` / `job_template.sh` | submit the 9 conditions; `status.sh` monitors |
| `analysis/plot_all.py` | all four figures + `summary_table.csv` |
| `plots/` | final figures (PNG + PDF) and summary table |
| `…/sd-qwen35/results/*.json` | raw per-condition results (gscratch) |
| `PROPOSAL.md` | original project proposal |
