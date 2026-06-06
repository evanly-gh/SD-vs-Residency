#!/usr/bin/env bash
# Parse + analyze + plot. CPU-only, fast — but kept as a batch job so it can be
# chained --dependency=afterok onto the long sweeps and produce figures
# automatically when those finish.
#
# Submit:
#   sbatch slurm/analysis.slurm

#SBATCH --job-name=spec-analyze
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=data/logs/slurm-%x-%j.out

set -euo pipefail
# Analysis is CPU-only; only need the cd-to-repo + (optional) conda. Skip module loads.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# Use the existing rl env on /gscratch — has pandas, matplotlib, seaborn, scipy.
# Home is over quota so we can't pip install into ~/.local; the env on scrubbed
# storage is the only writable Python with these libs available.
PY=/gscratch/intelligentsystems/evanly/conda_envs/rl/bin/python

${PY} analysis/parse_results.py
${PY} analysis/run_anova.py || echo "  (run_anova.py skipped or failed — not fatal)"
${PY} analysis/plot_speedup.py
${PY} analysis/plot_acceptance.py || true
${PY} analysis/plot_profiling.py || true
${PY} analysis/plot_summary.py || true
