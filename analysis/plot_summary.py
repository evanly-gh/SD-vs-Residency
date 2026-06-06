"""
Cross-cutting summary figures that span multiple result sources.

Adds graphs that the per-source plot_*.py scripts don't produce:
  1. throughput_bars_14b.png       — raw tok/s side-by-side (baseline vs each γ)
  2. throughput_bars_32b_ngl.png   — raw tok/s by NGL, grouped by condition
  3. accept_vs_gamma_grid.png      — α(γ) per task × thinking mode, faceted
  4. residency_curve_with_band.png — 32B speedup vs NGL with stddev band
  5. summary_table.png             — 1-page numeric summary rendered as an image

Usage:
    python analysis/plot_summary.py
"""

from __future__ import annotations

import argparse
import ast
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

GAMMA_PALETTE = dict(zip([4, 6, 8, 10], sns.color_palette("viridis", 4)))


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _parse_samples(s):
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


# ── 1. Raw throughput bars, 14B ───────────────────────────────────────────────

def plot_throughput_bars_14b(df: pd.DataFrame, out_dir: Path):
    df14 = df[df["model"] == "14b"].copy()
    if df14.empty:
        print("  [skip] throughput_bars_14b: no 14B data")
        return

    base_tps = df14[df14["condition"] == "baseline"]["median_ts"].median()
    spec = df14[df14["condition"] == "spec"].groupby("gamma")["median_ts"].median()

    labels = ["baseline"] + [f"γ={int(g)}" for g in spec.index]
    values = [base_tps] + list(spec.values)
    colors = ["#888"] + [GAMMA_PALETTE.get(int(g), "gray") for g in spec.index]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for b, v in zip(bars, values):
        if pd.notna(v):
            ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Raw Throughput — Qwen3-14B (VRAM-resident)\nBaseline vs. speculative at each γ")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "throughput_bars_14b.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── 2. Throughput by NGL for 32B ──────────────────────────────────────────────

def plot_throughput_bars_32b(df: pd.DataFrame, out_dir: Path):
    df32 = df[df["model"] == "32b"].copy()
    if df32.empty:
        print("  [skip] throughput_bars_32b_ngl: no 32B data")
        return

    base = df32[df32["condition"] == "baseline"].groupby("ngl")["median_ts"].median()
    spec = (df32[df32["condition"] == "spec"]
            .groupby(["ngl", "gamma"])["median_ts"].median().unstack("gamma"))

    ngls = sorted(set(base.index) | set(spec.index))
    width = 0.15
    x = np.arange(len(ngls))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 2 * width, [base.get(n, np.nan) for n in ngls], width,
           label="baseline", color="#888", edgecolor="white")
    for i, g in enumerate(sorted(spec.columns)):
        ax.bar(x + (i - 1) * width,
               [spec[g].get(n, np.nan) for n in ngls],
               width, label=f"γ={int(g)}",
               color=GAMMA_PALETTE.get(int(g), "gray"), edgecolor="white")

    ax.set_xticks(x); ax.set_xticklabels([f"ngl={n}" for n in ngls])
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Raw Throughput — Qwen3-32B, baseline vs. speculative, across NGL")
    ax.legend(ncol=5, fontsize=9, loc="upper left")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "throughput_bars_32b_ngl.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. α(γ) per task × thinking mode (facet grid) ─────────────────────────────

def plot_accept_vs_gamma_grid(df_acc: pd.DataFrame, out_dir: Path):
    if df_acc.empty:
        print("  [skip] accept_vs_gamma_grid: no acceptance data")
        return

    g = sns.FacetGrid(
        df_acc, col="task", row="thinking_mode",
        height=3, aspect=1.3, margin_titles=True, sharey=True,
    )
    g.map_dataframe(sns.lineplot, x="gamma", y="accept_pct",
                    estimator="mean", errorbar=("ci", 95), marker="o")
    g.set_axis_labels("γ (draft tokens per step)", "Acceptance % (α)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.figure.suptitle(
        "Draft acceptance rate as a function of γ\n"
        "Rows: thinking mode  •  Columns: task type",
        y=1.03,
    )
    for ax in g.axes.flat:
        ax.axhline(20, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0, 100)

    path = out_dir / "accept_vs_gamma_grid.png"
    g.figure.tight_layout()
    g.figure.savefig(path); plt.close(g.figure)
    print(f"  Saved: {path}")


# ── 4. Residency curve with stddev band ───────────────────────────────────────

def plot_residency_curve(df: pd.DataFrame, out_dir: Path):
    df32 = df[(df["model"] == "32b") & (df["condition"] == "spec")].copy()
    if df32.empty:
        print("  [skip] residency_curve_with_band: no 32B spec data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma, grp in df32.groupby("gamma"):
        agg = grp.groupby("ngl")["speedup_ratio"].agg(["mean", "std"]).reset_index()
        color = GAMMA_PALETTE.get(int(gamma), "gray")
        ax.plot(agg["ngl"], agg["mean"], marker="o", linewidth=2,
                color=color, label=f"γ={int(gamma)}")
        ax.fill_between(agg["ngl"],
                        agg["mean"] - agg["std"].fillna(0),
                        agg["mean"] + agg["std"].fillna(0),
                        alpha=0.15, color=color)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("GPU layers loaded (-ngl)")
    ax.set_ylabel("Speedup ratio")
    ax.set_title("Residency–Speedup Curve with ±1σ Bands — Qwen3-32B")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    path = out_dir / "residency_curve_with_band.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. One-page summary table rendered to PNG ─────────────────────────────────

def plot_summary_table(df: pd.DataFrame, df_acc: pd.DataFrame, out_dir: Path):
    rows = []
    if not df.empty:
        for (mdl, ngl, gamma), g in df.dropna(subset=["speedup_ratio"]) \
                .groupby(["model", "ngl", "gamma"], dropna=False):
            rows.append({
                "model": mdl, "ngl": int(ngl),
                "gamma": int(gamma) if pd.notna(gamma) else "—",
                "median tps": f"{g['median_ts'].median():.2f}",
                "median speedup": f"{g['speedup_ratio'].median():.3f}×",
                "n runs": int(g["n_samples"].sum()),
            })
    if not rows:
        print("  [skip] summary_table: no throughput data")
        return

    summary = pd.DataFrame(rows).sort_values(["model", "ngl", "gamma"])

    accept_rows = []
    if not df_acc.empty:
        for (task, mode), g in df_acc.groupby(["task", "thinking_mode"]):
            accept_rows.append({
                "task": task, "mode": mode,
                "mean α %": f"{g['accept_pct'].mean():.1f}",
                "n prompts": len(g),
            })

    fig, axes = plt.subplots(
        1, 2 if accept_rows else 1, figsize=(13, max(4, 0.35 * len(summary) + 2)),
    )
    if not accept_rows:
        axes = [axes]
    axes[0].axis("off")
    tbl = axes[0].table(cellText=summary.values, colLabels=summary.columns,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    axes[0].set_title("Throughput / speedup summary", pad=12)

    if accept_rows:
        acc_df = pd.DataFrame(accept_rows)
        axes[1].axis("off")
        tbl2 = axes[1].table(cellText=acc_df.values, colLabels=acc_df.columns,
                             loc="center", cellLoc="center")
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(9); tbl2.scale(1, 1.4)
        axes[1].set_title("Acceptance rate summary", pad=12)

    fig.tight_layout()
    path = out_dir / "summary_table.png"
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS_DIR / "all_results.csv"))
    parser.add_argument("--accept", default=str(RESULTS_DIR / "acceptance_rates.csv"))
    args = parser.parse_args()

    df = _load(Path(args.input))
    df_acc = _load(Path(args.accept))

    print(f"Loaded {len(df)} throughput rows, {len(df_acc)} acceptance rows.")
    plot_throughput_bars_14b(df, FIGURES_DIR)
    plot_throughput_bars_32b(df, FIGURES_DIR)
    plot_accept_vs_gamma_grid(df_acc, FIGURES_DIR)
    plot_residency_curve(df, FIGURES_DIR)
    plot_summary_table(df, df_acc, FIGURES_DIR)
    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
