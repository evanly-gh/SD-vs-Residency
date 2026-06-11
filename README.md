# Speculative Decoding Bottlenecks on the Qwen3.5 Family

Measures how draft model size, the dense→MoE target boundary, and thinking mode affect
speculative decoding efficiency (token acceptance rate + throughput speedup) on the Qwen3.5
family, using vLLM on Hyak Klone. See [PROPOSAL.md](PROPOSAL.md) for the full research design.

## Repository layout

| File | Purpose |
|---|---|
| [data/generate_prompts.py](data/generate_prompts.py) | Pulls 50 math (GSM8K) + 50 QA (SQuAD v2) + 50 code (HumanEval), shuffles with a fixed seed, writes `prompts.json` with a sha256 checksum |
| [setup/setup_env.sh](setup/setup_env.sh) | Creates the conda env at `/mmfs1/.../envs/sd-qwen35`, pins vLLM 0.9.2 + torch 2.5.1+cu124, prints all model download commands |
| [slurm/job_template.sh](slurm/job_template.sh) | SBATCH script; reads config from env vars, checks port collision, runs `run_experiment.py` |
| [slurm/submit_all.sh](slurm/submit_all.sh) | Submits all 9 conditions in parallel; supports `--dry-run`, `--smoke-test`, `--only A1,A2` |
| [run_experiment.py](run_experiment.py) | Launches vLLM with the correct `--speculative-config`, polls `/health`, fires prompts with `AsyncOpenAI`, collects Prometheus deltas, writes atomic JSON after each batch size |
| [analysis/plot_all.py](analysis/plot_all.py) | Globs all result JSONs, joins speedup against the matching baseline, generates all 4 plots as PNG+PDF |

## The 9 conditions

| Condition | Draft | Target | Mode | Role |
|---|---|---|---|---|
| `baseline` | — | 27B (dense) | standard | Axis 1 autoregressive denominator |
| `A1` | 0.8B | 27B (dense) | standard | Draft-size sweep |
| `A2` | 2B | 27B (dense) | standard | Draft-size sweep |
| `A3` | 4B | 27B (dense) | standard | Draft-size sweep |
| `A4` | 9B | 27B (dense) | standard | Draft-size sweep |
| `baseline_moe` | — | 35B-A3B (MoE) | standard | Axis 2 autoregressive denominator for B2/B3 |
| `B1` | 4B | 27B (dense) | standard | Architecture crossing (dense) |
| `B2` | 4B | 35B-A3B (MoE) | standard | Architecture crossing (MoE) |
| `B3` | 4B | 35B-A3B (MoE) | thinking | Thinking-mode effect |

## Order of operations on Hyak

```bash
# 1. Login node — one time
bash setup/setup_env.sh
# (run the huggingface-cli download commands it prints, in tmux)

# 2. Generate prompts
conda activate /mmfs1/.../envs/sd-qwen35
python data/generate_prompts.py

# 3. Smoke test (5 prompts, condition A3)
bash slurm/submit_all.sh --smoke-test

# 4. Wave 1
bash slurm/submit_all.sh --only baseline,A1,A2,A3,A4

# 5. Wave 2 (after inspecting wave 1)
bash slurm/submit_all.sh --only B1,B2,B3

# 6. Plots
python analysis/plot_all.py --results-dir /mmfs1/.../sd-qwen35/results
```

> **MoE baseline (`baseline_moe`, the 9th condition):** `baseline_moe` is the autoregressive
> denominator for the MoE-target speedups (B2/B3). Submit it together with Wave 2 so the
> comparison uses a matching architecture:
>
> ```bash
> bash slurm/submit_all.sh --only baseline_moe,B1,B2,B3
> ```
>
> If `baseline_moe` is omitted, `plot_all.py` falls back to normalizing MoE speedup against
> the 27B dense baseline (flagged in the plot notes), which is an apples-to-oranges comparison.
