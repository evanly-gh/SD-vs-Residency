"""
Plots for throughput and speedup ratio results.

Generates:
  1. speedup_by_gamma_14b.png — speedup ratio vs γ for 14B resident model
  2. speedup_by_ngl_32b.png  — speedup ratio vs NGL for 32B offload sweep (one line per γ)
  3. throughput_heatmap.png  — tokens/sec heatmap across NGL × γ for 32B

Usage:
    python analysis/plot_speedup.py
    python analysis/plot_speedup.py --input results/all_results.csv
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

PALETTE = sns.color_palette("colorblind", 4)
GAMMA_COLORS = {4: PALETTE[0], 6: PALETTE[1], 8: PALETTE[2], 10: PALETTE[3]}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}\nRun parse_results.py first.")
    return pd.read_csv(p)


# ── Plot 1: 14B speedup by γ ──────────────────────────────────────────────────

def plot_14b_speedup(df: pd.DataFrame, out_dir: Path):
    df14 = df[df["model"] == "14b"].copy()
    if df14.empty:
        print("  No 14B data for speedup plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for gamma, grp in df14.groupby("gamma"):
        med_sr = grp["speedup_ratio"].median()
        color = GAMMA_COLORS.get(int(gamma), "gray")
        ax.bar(
            x=f"γ={int(gamma)}",
            height=med_sr,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
            label=f"γ={int(gamma)}",
        )
        ax.text(
            f"γ={int(gamma)}", med_sr + 0.005,
            f"{med_sr:.3f}×",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Break-even (1.0×)")
    ax.set_xlabel("Speculation length (γ)")
    ax.set_ylabel("Speedup ratio (spec / baseline)")
    ax.set_title("Speculative Decoding Speedup — Qwen3-14B, 12 GB AMD (Navi 22)\nFully VRAM-resident, Q4_K_M")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(df14["speedup_ratio"].max() * 1.15, 1.2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    path = out_dir / "speedup_by_gamma_14b.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: 32B speedup vs NGL (one line per γ) ───────────────────────────────

def plot_32b_ngl_sweep(df: pd.DataFrame, out_dir: Path):
    df32 = df[df["model"] == "32b"].copy()
    if df32.empty:
        print("  No 32B data for NGL sweep plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for gamma, grp in df32.groupby("gamma"):
        pivot = grp.groupby("ngl")["speedup_ratio"].median().reset_index().sort_values("ngl")
        color = GAMMA_COLORS.get(int(gamma), "gray")
        ax.plot(
            pivot["ngl"], pivot["speedup_ratio"],
            marker="o", linewidth=2, color=color, label=f"γ={int(gamma)}",
        )

    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Break-even (1.0×)")
    ax.set_xlabel("GPU layers loaded (-ngl)")
    ax.set_ylabel("Speedup ratio (spec / baseline)")
    ax.set_title("Speculative Decoding Speedup vs. VRAM Residency\nQwen3-32B on 12 GB AMD — CPU Offload Sweep")
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_xticklabels(["0\n(all CPU)", "16\n(25%)", "32\n(50%)", "48\n(75%)", "64\n(all GPU)"])
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Shade the region where spec is net-negative
    ylim = ax.get_ylim()
    ax.fill_between([0, 64], [1.0, 1.0], [ylim[0], ylim[0]],
                    alpha=0.08, color="red", label="_nolegend_")
    ax.set_ylim(ylim)

    path = out_dir / "speedup_by_ngl_32b.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: throughput heatmap ────────────────────────────────────────────────

def plot_throughput_heatmap(df: pd.DataFrame, out_dir: Path):
    df32 = df[df["model"] == "32b"].copy()
    if df32.empty:
        print("  No 32B data for heatmap.")
        return

    pivot = df32.groupby(["ngl", "gamma"])["speedup_ratio"].median().unstack("gamma")
    pivot.index = [f"ngl={i}" for i in pivot.index]
    pivot.columns = [f"γ={g}" for g in pivot.columns]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True, fmt=".3f",
        cmap="RdYlGn",
        center=1.0,
        vmin=max(0.5, pivot.values.min() - 0.05),
        vmax=min(2.0, pivot.values.max() + 0.05),
        linewidths=0.5,
        cbar_kws={"label": "Speedup ratio"},
    )
    ax.set_title("Speedup Ratio — Qwen3-32B CPU Offload Sweep\n(green > 1.0 = beneficial, red < 1.0 = harmful)")
    ax.set_xlabel("Speculation length (γ)")
    ax.set_ylabel("GPU layers loaded (-ngl)")

    path = out_dir / "throughput_heatmap.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS_DIR / "all_results.csv"))
    args = parser.parse_args()

    df = load(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    plot_14b_speedup(df, FIGURES_DIR)
    plot_32b_ngl_sweep(df, FIGURES_DIR)
    plot_throughput_heatmap(df, FIGURES_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
