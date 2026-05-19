"""
Plots for draft token acceptance rate data.

Generates:
  1. acceptance_by_task.png       — mean α% by task type × thinking mode
  2. acceptance_vs_speedup.png    — scatter of α% vs speedup ratio (α* identification)
  3. acceptance_by_gamma.png      — α% vs γ for each task × thinking mode

Usage:
    python analysis/plot_acceptance.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

TASK_LABELS = {"code": "Code Gen", "reasoning": "Structured Reasoning", "chat": "Open-ended Chat"}
THINK_PALETTE = {"think": "#2196F3", "nothink": "#FF9800"}


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}\nRun run_acceptance.sh + parse_results.py first.")
    return pd.read_csv(p)


# ── Plot 1: acceptance by task × thinking mode ────────────────────────────────

def plot_acceptance_by_task(df_acc: pd.DataFrame, out_dir: Path):
    df = df_acc.copy()
    df["task_label"] = df["task"].map(TASK_LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))

    task_order = list(TASK_LABELS.values())
    x = np.arange(len(task_order))
    width = 0.35

    for i, mode in enumerate(["think", "nothink"]):
        means, cis = [], []
        for task_label in task_order:
            grp = df[(df["task_label"] == task_label) & (df["thinking_mode"] == mode)]["accept_pct"]
            means.append(grp.mean() if len(grp) else np.nan)
            cis.append(grp.sem() * 1.96 if len(grp) > 1 else 0)
        label = "Thinking mode (/think)" if mode == "think" else "Non-thinking (/no_think)"
        ax.bar(x + i * width, means, width, yerr=cis, capsize=4,
               color=THINK_PALETTE[mode], alpha=0.85, label=label, edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(task_order)
    ax.set_ylabel("Draft acceptance rate α (%)")
    ax.set_title("Draft Token Acceptance Rate by Task Type and Thinking Mode\nQwen3-14B target + Qwen3-0.6B draft, Q4_K_M, 12 GB AMD")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Reference lines from TaskSpec (2025) — enterprise FP16 baselines
    ax.axhline(20, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(2.8, 21, "TaskSpec reasoning floor (20%)", fontsize=8, color="gray")
    ax.axhline(73, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(2.8, 74, "TaskSpec text-gen ceiling (73%)", fontsize=8, color="gray")

    path = out_dir / "acceptance_by_task.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: α% vs speedup ratio (scatter) ─────────────────────────────────────

def plot_acceptance_vs_speedup(df_acc: pd.DataFrame, df_sp: pd.DataFrame, out_dir: Path):
    if df_sp.empty:
        print("  Skipping α* scatter — no speedup data.")
        return

    acc_summary = df_acc.groupby(["model", "ngl", "gamma"])["accept_pct"].mean().reset_index()
    merged = df_sp.merge(acc_summary, on=["model", "ngl", "gamma"], how="inner")
    if merged.empty:
        print("  Skipping α* scatter — could not join speedup and acceptance data.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = merged["gamma"].map({4: "#E91E63", 6: "#9C27B0", 8: "#03A9F4", 10: "#4CAF50"})
    scatter = ax.scatter(
        merged["accept_pct"], merged["speedup_ratio"],
        c=colors, alpha=0.7, s=60, edgecolors="white", linewidths=0.5,
    )
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(0, color="none")
    ax.set_xlabel("Draft acceptance rate α (%)")
    ax.set_ylabel("Speedup ratio (spec / baseline)")
    ax.set_title("Acceptance Rate vs. Speedup Ratio\n(α* = minimum α for beneficial speculation)")

    # Estimate α* visually
    below = merged[merged["speedup_ratio"] < 1.0]
    if not below.empty:
        alpha_star = below["accept_pct"].max()
        ax.axvline(alpha_star, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(alpha_star + 0.5, ax.get_ylim()[1] * 0.98,
                f"α* ≈ {alpha_star:.0f}%", color="red", fontsize=9, va="top")

    # Legend for gamma values
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E91E63", label="γ=4"),
        Patch(facecolor="#9C27B0", label="γ=6"),
        Patch(facecolor="#03A9F4", label="γ=8"),
        Patch(facecolor="#4CAF50", label="γ=10"),
    ]
    ax.legend(handles=legend_elements, title="γ", loc="lower right", fontsize=9)

    path = out_dir / "acceptance_vs_speedup.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: α% vs γ by task × mode ───────────────────────────────────────────

def plot_acceptance_by_gamma(df_acc: pd.DataFrame, out_dir: Path):
    tasks = df_acc["task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, sorted(tasks)):
        for mode in ["think", "nothink"]:
            grp = df_acc[(df_acc["task"] == task) & (df_acc["thinking_mode"] == mode)]
            if grp.empty:
                continue
            pivot = grp.groupby("gamma")["accept_pct"].mean().reset_index().sort_values("gamma")
            label = "/think" if mode == "think" else "/no_think"
            ax.plot(pivot["gamma"], pivot["accept_pct"],
                    marker="o", linewidth=2, color=THINK_PALETTE[mode], label=label)

        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlabel("Speculation length (γ)")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(mticker.FixedLocator([4, 6, 8, 10]))
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Acceptance rate α (%)")
    fig.suptitle("Acceptance Rate vs. Speculation Length by Task and Thinking Mode", y=1.02)

    path = out_dir / "acceptance_by_gamma.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", default=str(RESULTS_DIR / "acceptance_rates.csv"))
    parser.add_argument("--speedup", default=str(RESULTS_DIR / "all_results.csv"))
    args = parser.parse_args()

    df_acc = load(args.accept)
    df_sp = pd.read_csv(args.speedup) if Path(args.speedup).exists() else pd.DataFrame()
    print(f"Loaded {len(df_acc)} acceptance rows, {len(df_sp)} speedup rows")

    plot_acceptance_by_task(df_acc, FIGURES_DIR)
    plot_acceptance_vs_speedup(df_acc, df_sp, FIGURES_DIR)
    plot_acceptance_by_gamma(df_acc, FIGURES_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
