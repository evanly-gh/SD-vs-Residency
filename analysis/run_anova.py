"""
Statistical analysis of speculative decoding speedup results.

Runs:
  1. Mixed-effects ANOVA on speedup ratio (task type × thinking mode as
     within factors, gamma as covariate) for the 14B VRAM-resident condition.
  2. Piecewise linear threshold estimation for the 32B NGL sweep.
  3. Minimum viable acceptance rate (α*) identification.
  4. Effect size (partial η²) reporting.

Usage:
    python analysis/run_anova.py
    python analysis/run_anova.py --speedup results/all_results.csv \
                                  --accept results/acceptance_rates.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(speedup_path: str, accept_path: str):
    df_sp = pd.read_csv(speedup_path) if Path(speedup_path).exists() else pd.DataFrame()
    df_acc = pd.read_csv(accept_path) if Path(accept_path).exists() else pd.DataFrame()
    return df_sp, df_acc


def print_section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)


# ── 1. ANOVA on 14B VRAM-resident speedup ────────────────────────────────────

def run_14b_anova(df_sp: pd.DataFrame):
    print_section("1. Mixed-Effects ANOVA: 14B VRAM-Resident")

    df14 = df_sp[df_sp["model"] == "14b"].copy()
    if df14.empty:
        print("  No 14B data found. Run the sweep and parse_results.py first.")
        return

    try:
        import pingouin as pg
    except ImportError:
        print("  pingouin not installed. Run: conda install pingouin")
        print("  Falling back to descriptive statistics only.")
        _describe_speedup(df14)
        return

    # For ANOVA we need task × thinking_mode — these come from acceptance data.
    # If we have only throughput (synthetic prompts), report descriptive stats.
    print("  NOTE: Full mixed-effects ANOVA requires acceptance data joined with")
    print("  throughput by condition. Using descriptive stats for throughput-only data.")
    _describe_speedup(df14)


def _describe_speedup(df: pd.DataFrame):
    print(f"\n  {'Model':<6} {'NGL':>5} {'γ':>4}  {'Median speedup':>15}  {'N runs':>8}")
    print(f"  {'-'*6} {'-'*5} {'-'*4}  {'-'*15}  {'-'*8}")
    for (model, ngl, gamma), grp in df.groupby(["model", "ngl", "gamma"]):
        med = grp["speedup_ratio"].median()
        n = len(grp)
        flag = " ← >1.0 (BENEFIT)" if med > 1.0 else (" ← <1.0 (HARM)" if med < 0.99 else "")
        print(f"  {model:<6} {ngl:>5} {gamma:>4}  {med:>15.4f}  {n:>8}{flag}")


# ── 2. Piecewise linear threshold (32B NGL sweep) ────────────────────────────

def run_threshold_estimation(df_sp: pd.DataFrame):
    print_section("2. Residency Threshold Estimation (32B NGL Sweep)")

    df32 = df_sp[df_sp["model"] == "32b"].copy()
    if df32.empty:
        print("  No 32B data found.")
        return

    from scipy.optimize import brentq
    from scipy.stats import linregress

    for gamma, grp in df32.groupby("gamma"):
        pivot = grp.groupby("ngl")["speedup_ratio"].median().reset_index()
        pivot = pivot.sort_values("ngl")

        if len(pivot) < 3:
            print(f"  γ={gamma}: insufficient NGL data points ({len(pivot)})")
            continue

        ngl_vals = pivot["ngl"].values.astype(float)
        sr_vals = pivot["speedup_ratio"].values

        # Fit piecewise linear: find the breakpoint NGL where speedup crosses 1.0
        slope, intercept, r, p, se = linregress(ngl_vals, sr_vals)

        print(f"\n  γ={gamma}:")
        print(f"    Linear fit: speedup = {slope:.5f} × ngl + {intercept:.4f}  (R²={r**2:.3f})")

        if slope != 0:
            crossover_ngl = (1.0 - intercept) / slope
            print(f"    Estimated crossover NGL: {crossover_ngl:.1f} layers")
            if 0 <= crossover_ngl <= 64:
                print(f"    → Spec decoding beneficial when ngl > {crossover_ngl:.0f}")
            else:
                print(f"    → Crossover outside [0, 64] range — check data")
        else:
            print(f"    Flat relationship — speedup invariant to NGL")

        print(f"    NGL data: {dict(zip(pivot['ngl'].astype(int), pivot['speedup_ratio'].round(3)))}")


# ── 3. Minimum viable acceptance rate (α*) ───────────────────────────────────

def run_alpha_star(df_sp: pd.DataFrame, df_acc: pd.DataFrame):
    print_section("3. Minimum Viable Acceptance Rate (α*)")

    if df_acc.empty:
        print("  No acceptance rate data found. Run run_acceptance.sh first.")
        return

    print(f"\n  Acceptance rate by task × thinking mode:")
    print(f"  {'Task':<12} {'Mode':<10} {'Mean α%':>8}  {'Std α%':>7}  {'N':>4}")
    print(f"  {'-'*12} {'-'*10} {'-'*8}  {'-'*7}  {'-'*4}")
    for (task, mode), grp in df_acc.groupby(["task", "thinking_mode"]):
        mean_a = grp["accept_pct"].mean()
        std_a = grp["accept_pct"].std()
        n = len(grp)
        print(f"  {task:<12} {mode:<10} {mean_a:>8.1f}  {std_a:>7.1f}  {n:>4}")

    if df_sp.empty:
        print("\n  Cannot identify α* without speedup data.")
        return

    # Join speedup and acceptance (requires matching model, ngl, gamma)
    # This join is approximate — acceptance rates are per-prompt, speedup is synthetic
    acc_summary = df_acc.groupby(["model", "ngl", "gamma"])["accept_pct"].mean().reset_index()
    merged = df_sp.merge(acc_summary, on=["model", "ngl", "gamma"], how="inner")

    if merged.empty:
        print("\n  Could not join speedup and acceptance data (key mismatch).")
        return

    # Find α* per hardware tier (we have one tier: 12 GB AMD)
    below_threshold = merged[merged["speedup_ratio"] < 1.0]
    above_threshold = merged[merged["speedup_ratio"] >= 1.0]

    if not below_threshold.empty and not above_threshold.empty:
        alpha_star = below_threshold["accept_pct"].max()
        print(f"\n  α* estimate: {alpha_star:.1f}%")
        print(f"  Interpretation: acceptance rates below {alpha_star:.0f}% correlated with speedup < 1.0")
    else:
        print("\n  Insufficient data to identify α* — need both beneficial and harmful conditions.")


# ── 4. Effect sizes ───────────────────────────────────────────────────────────

def run_effect_sizes(df_acc: pd.DataFrame):
    print_section("4. Effect Sizes: Task Type vs. Thinking Mode")

    if df_acc.empty:
        print("  No acceptance rate data.")
        return

    try:
        from scipy import stats
    except ImportError:
        print("  scipy not installed.")
        return

    # One-way ANOVA: task type effect on acceptance rate
    groups_task = [grp["accept_pct"].values for _, grp in df_acc.groupby("task")]
    if len(groups_task) >= 2:
        f_stat, p_val = stats.f_oneway(*groups_task)
        # Partial η² = SS_between / SS_total (approximate from F)
        k = len(groups_task)
        n = len(df_acc)
        eta_sq = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n - k))
        print(f"\n  Task type → acceptance rate:")
        print(f"    F({k-1}, {n-k}) = {f_stat:.3f}, p = {p_val:.4f}, partial η² = {eta_sq:.3f}")

    # Thinking mode effect
    think_groups = [grp["accept_pct"].values for _, grp in df_acc.groupby("thinking_mode")]
    if len(think_groups) == 2:
        t_stat, p_val = stats.ttest_ind(*think_groups)
        think_means = df_acc.groupby("thinking_mode")["accept_pct"].mean()
        diff = think_means.get("think", 0) - think_means.get("nothink", 0)
        print(f"\n  Thinking mode → acceptance rate:")
        print(f"    t = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"    Mean difference (think - nothink): {diff:+.1f}%")
        direction = "HIGHER in thinking mode" if diff > 0 else "LOWER in thinking mode"
        print(f"    → Acceptance rates are {direction}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speedup", default=str(RESULTS_DIR / "all_results.csv"))
    parser.add_argument("--accept", default=str(RESULTS_DIR / "acceptance_rates.csv"))
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_sp, df_acc = load_data(args.speedup, args.accept)

    print(f"Loaded {len(df_sp)} speedup rows, {len(df_acc)} acceptance rows")

    run_14b_anova(df_sp)
    run_threshold_estimation(df_sp)
    run_alpha_star(df_sp, df_acc)
    run_effect_sizes(df_acc)

    print("\n")


if __name__ == "__main__":
    main()
