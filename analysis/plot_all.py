"""
plot_all.py — Generate all 4 plots for the SD-vs-Residency experiment.

Usage:
  python analysis/plot_all.py
  python analysis/plot_all.py --results-dir /mmfs1/.../sd-qwen35/results --out-dir plots

Produces:
  plots/plot1_tar_vs_draft_size.{png,pdf}
  plots/plot2_speedup_vs_draft_size.{png,pdf}
  plots/plot3_dense_vs_moe.{png,pdf}
  plots/plot4_thinking_vs_standard.{png,pdf}
  plots/summary_table.csv
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")
matplotlib.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

# Draft model sizes for Axis 1 conditions (in billions)
DRAFT_SIZE_MAP = {
    "A1": 0.8,
    "A2": 2.0,
    "A3": 4.0,
    "A4": 9.0,
}

BATCH_SIZES = [1, 4, 8, 16]
COLORS = sns.color_palette("colorblind", n_colors=6)
BS_COLORS = {bs: COLORS[i] for i, bs in enumerate(BATCH_SIZES)}
BS_MARKERS = {1: "o", 4: "s", 8: "^", 16: "D"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> pd.DataFrame:
    rows = []
    json_files = glob.glob(str(Path(results_dir) / "*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")

    for path in json_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        meta = data["meta"]
        condition = meta["condition"]

        for bs_str, agg in data["aggregate"]["by_batch_size"].items():
            bs = int(bs_str)
            row = {
                "condition":       condition,
                "axis":            meta.get("axis", 0),
                "draft_model":     meta.get("draft_model") or "",
                "target_model":    meta["target_model"],
                "mode":            meta["mode"],
                "batch_size":      bs,
                "is_baseline":     meta.get("is_baseline", False),
                "is_sd":           meta.get("speculative_decoding", False),
                "tar":             agg.get("tar"),
                "mean_accepted_length": agg.get("mean_accepted_length"),
                "throughput_tps":  agg.get("throughput_tps"),
                "draft_size_b":    DRAFT_SIZE_MAP.get(condition),
                "total_output_tokens": agg.get("total_output_tokens"),
            }
            # Per-category throughput
            for cat, cat_agg in (agg.get("by_category") or {}).items():
                row[f"throughput_{cat}"] = cat_agg.get("throughput_tps")
            rows.append(row)

    df = pd.DataFrame(rows)

    # Join speedup_ratio: throughput_tps / baseline_throughput_tps
    # Match baseline by (target_model_basename, batch_size)
    df["target_name"] = df["target_model"].apply(lambda p: Path(p).name)
    baselines = df[df["is_baseline"]].copy()
    baselines = baselines.rename(columns={"throughput_tps": "baseline_tps"})
    baselines = baselines[["target_name", "batch_size", "baseline_tps"]]

    df = df.merge(baselines, on=["target_name", "batch_size"], how="left")
    df["speedup_ratio"] = df["throughput_tps"] / df["baseline_tps"]

    # For B2/B3 (MoE target, no separate baseline), use 27B baseline as denominator
    # This is flagged in plot subtitles
    df_27b_baseline = baselines[baselines["target_name"].str.contains("27B")]
    df_27b_baseline = df_27b_baseline.rename(columns={"baseline_tps": "baseline_27b_tps"})
    df = df.merge(df_27b_baseline[["batch_size", "baseline_27b_tps"]], on="batch_size", how="left")
    mask_no_baseline = df["baseline_tps"].isna() & df["is_sd"]
    df.loc[mask_no_baseline, "speedup_ratio"] = (
        df.loc[mask_no_baseline, "throughput_tps"] / df.loc[mask_no_baseline, "baseline_27b_tps"]
    )

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, out_dir: str, name: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = Path(out_dir) / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")


def add_note(ax, text: str) -> None:
    ax.annotate(
        text, xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, color="gray", va="bottom",
    )


# ---------------------------------------------------------------------------
# Plot 1: TAR vs draft model size
# ---------------------------------------------------------------------------

def plot1_tar_vs_draft_size(df: pd.DataFrame, out_dir: str) -> None:
    axis1 = df[df["condition"].isin(DRAFT_SIZE_MAP) & df["is_sd"]].copy()
    if axis1.empty:
        print("  [Plot 1] No Axis 1 data — skipping.")
        return

    categories = ["aggregate", "math", "qa", "code"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    fig.suptitle("Plot 1: Token Acceptance Rate vs Draft Model Size", fontweight="bold")

    for ax, cat in zip(axes, categories):
        for bs in BATCH_SIZES:
            sub = axis1[axis1["batch_size"] == bs].sort_values("draft_size_b")
            if sub.empty:
                continue

            if cat == "aggregate":
                y = sub["tar"].values
            else:
                col = f"throughput_{cat}"
                # For per-category TAR we don't have it stored separately;
                # use per-category throughput as a proxy and note this
                y = sub[col].values if col in sub.columns else np.full(len(sub), np.nan)

            x = sub["draft_size_b"].values
            valid = ~np.isnan(y.astype(float))
            if not any(valid):
                continue

            ax.plot(
                x[valid], y[valid],
                marker=BS_MARKERS[bs], color=BS_COLORS[bs],
                label=f"bs={bs}", linewidth=1.5, markersize=6,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Draft model size (B params)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}B"))
        ax.set_xticks([0.8, 2, 4, 9])
        ax.set_title(cat.capitalize())
        ax.grid(True, alpha=0.3)

        if cat == "aggregate":
            ax.set_ylabel("Token Acceptance Rate")
            ax.legend(loc="lower right", framealpha=0.7)

    add_note(axes[0], "Target: Qwen3.5-27B (dense), standard mode")
    plt.tight_layout()
    save_fig(fig, out_dir, "plot1_tar_vs_draft_size")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Speedup vs draft model size
# ---------------------------------------------------------------------------

def plot2_speedup_vs_draft_size(df: pd.DataFrame, out_dir: str) -> None:
    axis1 = df[df["condition"].isin(DRAFT_SIZE_MAP) & df["is_sd"]].copy()
    if axis1.empty:
        print("  [Plot 2] No Axis 1 data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Plot 2: Throughput Speedup vs Draft Model Size", fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="Baseline (autoregressive)", zorder=1)

    for bs in BATCH_SIZES:
        sub = axis1[axis1["batch_size"] == bs].sort_values("draft_size_b")
        if sub.empty:
            continue
        x = sub["draft_size_b"].values
        y = sub["speedup_ratio"].values
        valid = ~np.isnan(y.astype(float))
        if not any(valid):
            continue
        ax.plot(
            x[valid], y[valid],
            marker=BS_MARKERS[bs], color=BS_COLORS[bs],
            label=f"batch size = {bs}", linewidth=1.8, markersize=7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Draft model size (B params)")
    ax.set_ylabel("Throughput speedup (× over autoregressive baseline)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}B"))
    ax.set_xticks([0.8, 2, 4, 9])
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.3)
    add_note(ax, "Target: Qwen3.5-27B (dense), standard mode. Baseline = 27B autoregressive.")
    plt.tight_layout()
    save_fig(fig, out_dir, "plot2_speedup_vs_draft_size")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Dense vs MoE target
# ---------------------------------------------------------------------------

def plot3_dense_vs_moe(df: pd.DataFrame, out_dir: str) -> None:
    b1 = df[df["condition"] == "B1"].copy()
    b2 = df[df["condition"] == "B2"].copy()
    if b1.empty or b2.empty:
        print("  [Plot 3] B1 or B2 data missing — skipping.")
        return

    fig, (ax_tar, ax_spd) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plot 3: Dense vs MoE Target — TAR and Speedup by Batch Size",
                 fontweight="bold")

    x = np.arange(len(BATCH_SIZES))
    width = 0.35

    for ax, metric, ylabel in [
        (ax_tar, "tar", "Token Acceptance Rate"),
        (ax_spd, "speedup_ratio", "Throughput Speedup (×)"),
    ]:
        b1_vals = [b1[b1["batch_size"] == bs][metric].values[0]
                   if not b1[b1["batch_size"] == bs].empty else np.nan
                   for bs in BATCH_SIZES]
        b2_vals = [b2[b2["batch_size"] == bs][metric].values[0]
                   if not b2[b2["batch_size"] == bs].empty else np.nan
                   for bs in BATCH_SIZES]

        bars1 = ax.bar(x - width / 2, b1_vals, width, label="B1: Dense 27B target",
                       color=COLORS[0], alpha=0.85)
        bars2 = ax.bar(x + width / 2, b2_vals, width, label="B2: MoE 35B-A3B target",
                       color=COLORS[1], alpha=0.85)

        ax.set_xlabel("Concurrent batch size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in BATCH_SIZES])
        ax.legend(framealpha=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        if metric == "speedup_ratio":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=0)

        # Value labels
        for bar in [*bars1, *bars2]:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    add_note(ax_tar, "Draft: Qwen3.5-4B (dense), standard mode.\n"
                     "B2 speedup normalized to 27B autoregressive baseline.")
    plt.tight_layout()
    save_fig(fig, out_dir, "plot3_dense_vs_moe")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Thinking vs standard mode
# ---------------------------------------------------------------------------

def plot4_thinking_vs_standard(df: pd.DataFrame, out_dir: str) -> None:
    b2 = df[df["condition"] == "B2"].copy()
    b3 = df[df["condition"] == "B3"].copy()
    if b2.empty or b3.empty:
        print("  [Plot 4] B2 or B3 data missing — skipping.")
        return

    fig, (ax_tar, ax_spd) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plot 4: Standard vs Thinking Mode — TAR and Speedup (MoE Target)",
                 fontweight="bold")

    x = np.arange(len(BATCH_SIZES))
    width = 0.35

    for ax, metric, ylabel in [
        (ax_tar, "tar", "Token Acceptance Rate"),
        (ax_spd, "speedup_ratio", "Throughput Speedup (×)"),
    ]:
        b2_vals = [b2[b2["batch_size"] == bs][metric].values[0]
                   if not b2[b2["batch_size"] == bs].empty else np.nan
                   for bs in BATCH_SIZES]
        b3_vals = [b3[b3["batch_size"] == bs][metric].values[0]
                   if not b3[b3["batch_size"] == bs].empty else np.nan
                   for bs in BATCH_SIZES]

        bars2 = ax.bar(x - width / 2, b2_vals, width, label="B2: Standard mode",
                       color=COLORS[2], alpha=0.85)
        bars3 = ax.bar(x + width / 2, b3_vals, width, label="B3: Thinking mode",
                       color=COLORS[3], alpha=0.85, hatch="//")

        ax.set_xlabel("Concurrent batch size")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in BATCH_SIZES])
        ax.legend(framealpha=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        if metric == "speedup_ratio":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, zorder=0)

        for bar in [*bars2, *bars3]:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    add_note(ax_tar, "Target: Qwen3.5-35B-A3B (MoE). Draft: Qwen3.5-4B.\n"
                     "Speedup normalized to 27B autoregressive baseline.")
    plt.tight_layout()
    save_fig(fig, out_dir, "plot4_thinking_vs_standard")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame, out_dir: str) -> None:
    conditions = ["baseline", "A1", "A2", "A3", "A4", "B1", "B2", "B3"]
    cols = ["condition", "batch_size", "tar", "throughput_tps", "speedup_ratio"]
    sub = df[df["condition"].isin(conditions)][cols].copy()
    sub = sub.sort_values(["condition", "batch_size"])

    print("\n=== Summary Table ===")
    print(sub.to_string(index=False, float_format="{:.3f}".format))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sub.to_csv(Path(out_dir) / "summary_table.csv", index=False)
    print(f"\nSaved: {out_dir}/summary_table.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir",     default="plots")
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    df = load_all_results(args.results_dir)
    print(f"Loaded {len(df)} rows across {df['condition'].nunique()} conditions\n")

    print("Generating Plot 1 (TAR vs draft size)...")
    plot1_tar_vs_draft_size(df, args.out_dir)

    print("Generating Plot 2 (Speedup vs draft size)...")
    plot2_speedup_vs_draft_size(df, args.out_dir)

    print("Generating Plot 3 (Dense vs MoE)...")
    plot3_dense_vs_moe(df, args.out_dir)

    print("Generating Plot 4 (Thinking vs standard)...")
    plot4_thinking_vs_standard(df, args.out_dir)

    print_summary_table(df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
