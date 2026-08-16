# Speculative Decoding on the Qwen3.5 Family

**When does speculative decoding actually help?** A controlled measurement of how
**speculative depth**, the **dense→MoE** target boundary, and **thinking (chain-of-thought)
mode** affect token acceptance rate and throughput speedup on the Qwen3.5 family — using
the models' **native Multi-Token-Prediction (MTP) head** for self-speculation, served with
vLLM (V1) on 2× NVIDIA L40S.

> CSE 493G1 / 599G1 — Deep Learning, Spring 2026, University of Washington · Evan Li (solo).
> Full write-up and poster source: [HANDOFF.md](HANDOFF.md). Original design: [PROPOSAL.md](PROPOSAL.md).

---

## Overview & motivation

Speculative decoding (SD) accelerates autoregressive LLM inference by proposing several tokens
cheaply and verifying them in one parallel target forward pass. The speedup rests on a
memory-bound-regime assumption: verifying *k* proposed tokens costs roughly the same as
verifying one, because weight loading dominates and weights are read once regardless of
sequence length. This holds cleanly for **dense** targets; for **MoE** targets each speculated
token can route to a different expert subset, so a *k*-token draft can force the target to load
the union of activated experts and inflate verification cost.

Qwen3.5 ships a native **MTP head** for self-speculation, but its SD scaling had not been
characterized: (1) no within-family speculative-depth sweep for Qwen3.5, (2) unknown whether the
MoE-verification penalty is severe or mild for Qwen3.5-35B-A3B (only 3B of 35B active/token),
and (3) the effect of thinking-mode tokens on draft acceptance was unmeasured. This project
closes those three gaps with a controlled, seeded, reproducible experiment.

## Research questions

1. **Depth / acceptance tradeoff (Axis 1).** As speculative depth *k* grows, how do token
   acceptance rate (TAR), mean accepted length, and throughput speedup trade off, and where is
   the optimum for the dense Qwen3.5-27B?
2. **Architecture crossing (Axis 2).** Does crossing the dense→MoE boundary in the target
   (27B dense vs 35B-A3B MoE, fixed *k*=3) change the SD payoff?
3. **Thinking mode (Axis 3).** Does Qwen3.5's `<think>` chain-of-thought generation
   systematically alter draft acceptance / speedup versus standard mode?

## Method

### Speculation mechanism — native MTP self-speculation

Qwen3.5 performs SD through its built-in MTP head: the target's own head proposes *k* tokens,
which the target then verifies. vLLM routes **any** Qwen3.5 speculator through that MTP head,
which is bound to the target's hidden size — so a separately-sized draft **checkpoint** is not
usable on this family. The realized, supported Axis-1 knob is therefore the **number of
speculative tokens *k*** (speculative depth), not a draft-model size. Both probe the same
acceptance-vs-speedup tradeoff. This pivot (draft-size sweep → depth sweep) is documented in
[HANDOFF.md §8](HANDOFF.md) and enforced in [`run_experiment.py`](run_experiment.py).

```bash
# SD condition: target's MTP head, depth k
--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'
# baseline condition: plain autoregressive (no --speculative-config)
```

### Serving, hardware, decoding

| Item | Value |
|---|---|
| Engine | vLLM 0.22.1 (V1), torch 2.11 + CUDA 13, transformers 4.57.6 |
| Hardware | 2× NVIDIA L40S (≈45 GB ea.), Hyak Klone `cse/gpu-l40s`, tensor-parallel = 2 |
| Precision | bf16, fully GPU-resident (no CPU offload) |
| Decoding | greedy (temperature 0), `max_tokens` = 512 (8192-token ctx for thinking, else 4096) |
| Execution | `--enforce-eager` (CUDA-graph/compile off) — uniform across all conditions for reliable unattended startup; lowers *absolute* tok/s uniformly, leaving TAR and the SD/baseline **speedup ratio** unaffected |

### Models

| Role | Model | Architecture | bf16 VRAM |
|---|---|---|---|
| Dense target | Qwen3.5-27B | dense | ~52 GB |
| MoE target | Qwen3.5-35B-A3B | Mixture-of-Experts (3B active / 35B total) | ~67 GB |
| Speculator | each target's **native MTP head** | multi-token prediction | shared with target |

### Dataset

150 prompts, fixed seed 42, sha256-checksummed ([`data/prompts.json`](data/prompts.json)),
three categories of 50 each (a covariate in all analyses), built by
[`data/generate_prompts.py`](data/generate_prompts.py):

- **Math** — GSM8K (multi-step reasoning)
- **QA** — SQuAD v2 (short factual questions, answerable subset)
- **Code** — HumanEval (Python function completion)

### Metrics (vLLM Prometheus `spec_decode_*` counters)

- **TAR** (token acceptance rate) = accepted draft tokens / proposed draft tokens
- **Mean accepted length** = 1 + accepted/drafts (verified tokens per decode step)
- **Throughput** = output tokens / wall-clock second
- **Speedup** = SD throughput / matching autoregressive baseline throughput (same target,
  same batch size)

`run_experiment.py` snapshots the counters before/after each batch-size run and reports deltas.
The parser strips the Prometheus `_total` suffix so it is robust across vLLM 0.9.x↔0.22.x
counter renames.

### The 9 conditions (× batch sizes {1, 4, 8, 16})

| Condition | Target | Method | k | Mode | Role |
|---|---|---|---|---|---|
| `baseline` | 27B dense | none | — | standard | Axis-1 autoregressive denominator |
| `K1`…`K4`, `K6` | 27B dense | MTP | 1,2,3,4,6 | standard | **Axis 1** depth sweep |
| `baseline_moe` | 35B-A3B MoE | none | — | standard | Axis-2/3 MoE autoregressive denominator |
| `M3` | 35B-A3B MoE | MTP | 3 | standard | **Axis 2** dense-vs-MoE (vs `K3`) |
| `M3_think` | 35B-A3B MoE | MTP | 3 | thinking | **Axis 3** thinking (vs `M3`) |

Each condition is an independent SLURM job on its own port (8100–8108), submitted in parallel.

## Key results

Full per-(condition × batch) data: [`plots/summary_table.csv`](plots/summary_table.csv).

### Axis 1 — speculative depth on dense Qwen3.5-27B (batch size 1)

| k | TAR | mean accepted len | throughput (tok/s) | speedup |
|---|-----|-------------------|--------------------|---------|
| 0 (baseline) | — | — | 20.99 | 1.00× |
| 1 | 0.943 | 1.94 | 31.74 | 1.51× |
| 2 | 0.887 | 2.77 | 43.00 | 2.05× |
| 3 | 0.824 | 3.47 | 54.62 | 2.60× |
| **4** | **0.758** | **4.03** | **60.52** | **2.88× ← optimum** |
| 6 | 0.641 | 4.85 | 36.24 | 1.73× |

As depth grows, per-token acceptance falls monotonically (0.94→0.64) and mean accepted length
rises **sub-linearly** (below the ideal *k*+1); net speedup **peaks at k≈4 (2.88×) then
collapses to 1.73× at k=6** as rejected-draft work dominates. Speedup is remarkably flat across
batch sizes (k=4: 2.88 / 2.99 / 2.94 / 2.79× at bs 1/4/8/16) — MTP self-speculation keeps
helping under concurrency, unlike classic draft-model SD.

### Axis 2 — dense vs MoE target (k=3)

| batch | dense TAR | dense speedup | MoE TAR | MoE speedup |
|---|---|---|---|---|
| 1 | 0.824 | 2.60× | 0.801 | 2.70× |
| 4 | 0.825 | 2.79× | 0.799 | 2.83× |
| 8 | 0.829 | 2.70× | 0.796 | 2.92× |
| 16 | 0.827 | 2.55× | 0.794 | **2.95×** |

Crossing dense→MoE slightly lowers acceptance (0.80 vs 0.82) but yields a **higher speedup that
widens with batch size** (MoE 2.70→2.95× vs dense 2.60→2.55×). The MoE's autoregressive
baseline is far more memory-bound (10.33 tok/s at bs 1 vs 20.99 for the dense 27B), so batched
parallel verification recovers proportionally more throughput. For Qwen3.5 this is the
**opposite of a "MoE penalty"** — MoE speculation pays off *more* than dense.

### Axis 3 — thinking vs standard mode (MoE, k=3)

Thinking-mode tokens are **marginally harder for the MTP head to predict** — TAR is consistently
~1.5–2 points lower under thinking (≈0.78 vs ≈0.80) at every batch size. This is the clean SD
result. Absolute decode throughput is much higher under thinking (341.6 vs 170.8 tok/s at bs 16;
apparent speedup up to 5.90×), but this is a **batch-utilization effect, not an SD effect**: long
CoT responses keep concurrent requests alive together and the GPU batch full, whereas the
standard-mode baseline drains as short answers finish — so the TAR comparison, not the headline
throughput, is reported as the finding.

### Headline numbers

- **Optimal speculative depth for Qwen3.5-27B: k=4 → 2.88× speedup**; over-speculation (k=6)
  collapses to 1.73×.
- **MoE ≥ dense:** Qwen3.5-35B-A3B reaches **2.95× at batch 16**, beating the dense 27B at
  every batch and widening with concurrency (refutes the expected MoE penalty).
- **Acceptance falls smoothly with depth** (0.94→0.64 for k=1→6) and is slightly lower for the
  MoE target and for thinking mode.

## Reproducibility / setup

Runs on Hyak Klone. Weights, results, and logs live in `gscratch`
(`/mmfs1/gscratch/intelligentsystems/evanly/`); `results/` is symlinked into the repo and
gitignored.

```bash
# 1. Login node — one-time environment + weights
bash setup/setup_env.sh                     # conda env at gscratch/envs/sd-qwen35
bash setup/download_models.sh               # Qwen3.5-27B + 35B-A3B (+ others), resumable

# 2. Generate the seeded, checksummed prompt set
conda activate /mmfs1/gscratch/intelligentsystems/evanly/envs/sd-qwen35
python data/generate_prompts.py             # -> data/prompts.json

# 3. Smoke test (5 prompts, one condition)
bash slurm/submit_all.sh --smoke-test

# 4. Submit all 9 conditions in parallel
bash slurm/submit_all.sh                     # or: --only K1,K2,baseline
bash slurm/status.sh                         # monitor queue + per-condition completeness

# 5. Figures + summary table
python analysis/plot_all.py \
  --results-dir /mmfs1/gscratch/intelligentsystems/evanly/sd-qwen35/results \
  --out-dir plots
```

Reproducibility features: fixed seed 42 everywhere; sha256 checksum on the prompt set;
`config.json` hash recorded per model in each result's `meta`; vLLM version, SLURM job id,
hostname, and timestamp captured per run; **atomic per-batch-size writes + `--resume`** so
preempted jobs restart from the last completed `(prompt_id, batch_size)` pair.

> Note: `setup/download_models.sh` still lists the draft checkpoints (0.8B/2B/4B/9B) from the
> original draft-size design; only the 27B and 35B-A3B targets are required for the realized
> MTP depth-sweep study.

## Usage

```bash
# Single condition, no SLURM (e.g. local smoke test)
python run_experiment.py \
  --condition K4 \
  --target-model /path/to/Qwen3.5-27B \
  --spec-method mtp --num-spec-tokens 4 \
  --mode standard --port 8104 \
  --output results/K4.json \
  --batch-sizes 1 4 8 16 --num-prompts 5
```

Baseline runs pass `--spec-method none` (no `--speculative-config`); thinking mode passes
`--mode thinking` (toggled per-request via `chat_template_kwargs.enable_thinking`).

## Project structure

| Path | Purpose |
|---|---|
| [`data/generate_prompts.py`](data/generate_prompts.py) | Builds the 150-prompt (GSM8K/SQuAD v2/HumanEval) seeded, sha256-checksummed set |
| [`data/prompts.json`](data/prompts.json) | The frozen prompt set + checksum |
| [`run_experiment.py`](run_experiment.py) | Launches vLLM (MTP or baseline), polls `/health`, fires prompts with `AsyncOpenAI`, collects Prometheus deltas, writes atomic checkpoint JSON per batch size |
| [`slurm/submit_all.sh`](slurm/submit_all.sh) | Submits the 9 conditions in parallel; `--dry-run`, `--smoke-test`, `--only` |
| [`slurm/job_template.sh`](slurm/job_template.sh) | SBATCH script (2× L40S, port-collision guard, conda activation) |
| [`slurm/status.sh`](slurm/status.sh) | Overnight queue + per-condition completeness monitor |
| [`analysis/plot_all.py`](analysis/plot_all.py) | Globs result JSONs, joins each SD run to its matching AR baseline, emits 4 figures (PNG+PDF) + `summary_table.csv` |
| [`setup/setup_env.sh`](setup/setup_env.sh), [`setup/download_models.sh`](setup/download_models.sh) | Conda env + weight downloads on Hyak |
| [`plots/`](plots/) | Final figures and summary table |
| [`HANDOFF.md`](HANDOFF.md) | Full project report / poster source (authoritative results) |
| [`PROPOSAL.md`](PROPOSAL.md) | Original research proposal |

An earlier prototype iteration (`config.sh`, `scripts/`, `setup/build_llamacpp.sh`, and the
`CLAUDE.md` guidance) explored the same question via **llama.cpp / GGUF** with a VRAM-residency
framing; it is retained for history but superseded by the vLLM MTP pipeline above.

## Artifacts

Figures in [`plots/`](plots/) (PNG + PDF each):

| File | Content |
|---|---|
| `plot1_tar_vs_k` | TAR & mean accepted length vs k (dense 27B) — the "why" panel |
| `plot2_speedup_vs_k` | Speedup vs k, per batch + per task category — the optimal-depth hero result |
| `plot3_dense_vs_moe` | K3 vs M3 — TAR & speedup by batch size |
| `plot4_thinking_vs_std` | M3 vs M3_think — TAR & speedup by batch size |
| `summary_table.csv` | Every (condition × batch) row — backing data |
