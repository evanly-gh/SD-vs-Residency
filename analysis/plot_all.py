"""
plot_all.py — Generate all plots for the Qwen3.5 MTP speculative-decoding study.

Experiment (see README / handoff doc):
  Axis 1 — speculative-depth (k) sweep on the dense 27B target via MTP
           self-speculation: baseline (k=0, autoregressive), K1..K4, K6.
  Axis 2 — dense vs MoE at k=3 (K3 vs M3) and thinking effect (M3 vs M3_think),
           with baseline_moe as the MoE autoregressive denominator.

Usage:
  python analysis/plot_all.py --results-dir /mmfs1/.../sd-qwen35/results --out-dir plots

Produces (PNG + PDF each):
  plot1_tar_vs_k          TAR and mean accepted length vs k (dense)
  plot2_speedup_vs_k      Throughput speedup vs k (dense), per batch size
  plot3_dense_vs_moe      K3 vs M3: TAR and speedup by batch size
  plot4_thinking_vs_std   M3 vs M3_think: TAR and speedup by batch size
  summary_table.csv
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    pass
matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

BATCH_SIZES = [1, 4, 8, 16]
COLORS = sns.color_palette("colorblind", n_colors=6)
BS_COLORS = {bs: COLORS[i] for i, bs in enumerate(BATCH_SIZES)}
BS_MARKERS = {1: "o", 4: "s", 8: "^", 16: "D"}
DENSE_TARGET = "Qwen3.5-27B"
MOE_TARGET = "Qwen3.5-35B-A3B"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> pd.DataFrame:
    rows = []
    files = [f for f in glob.glob(str(Path(results_dir) / "*.json"))
             if not Path(f).name.startswith("smoke_")]
    if not files:
        raise FileNotFoundError(f"No result JSONs in {results_dir}")

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data["meta"]
        k = meta.get("num_speculative_tokens") or 0
        for bs_str, agg in data["aggregate"]["by_batch_size"].items():
            row = {
                "condition": meta["condition"],
                "target_name": Path(meta["target_model"]).name,
                "spec_method": meta.get("spec_method", "none"),
                "mode": meta.get("mode", "standard"),
                "k": int(k),
                "batch_size": int(bs_str),
                "is_baseline": meta.get("is_baseline", meta.get("spec_method", "none") == "none"),
                "tar": agg.get("tar"),
                "mean_accepted_length": agg.get("mean_accepted_length"),
                "throughput_tps": agg.get("throughput_tps"),
                "total_output_tokens": agg.get("total_output_tokens"),
            }
            for cat, cagg in (agg.get("by_category") or {}).items():
                row[f"throughput_{cat}"] = cagg.get("throughput_tps")
            rows.append(row)

    df = pd.DataFrame(rows)
    # Speedup = throughput / matching autoregressive baseline (same target, same batch).
    base = df[df["is_baseline"]][["target_name", "batch_size", "throughput_tps"]]
    base = base.rename(columns={"throughput_tps": "baseline_tps"})
    df = df.merge(base, on=["target_name", "batch_size"], how="left")
    df["speedup_ratio"] = df["throughput_tps"] / df["baseline_tps"]
    return df


def save_fig(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = Path(out_dir) / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"  saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — TAR and mean accepted length vs k (dense 27B)
# ---------------------------------------------------------------------------

def plot1_tar_vs_k(df, out_dir):
    d = df[(df["target_name"] == DENSE_TARGET) & (df["spec_method"] == "mtp")].copy()
    if d.empty:
        print("  [plot1] no dense MTP data — skipping")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plot 1 — Acceptance vs Speculative Depth (Qwen3.5-27B, MTP)",
                 fontweight="bold")
    for bs in BATCH_SIZES:
        s = d[d["batch_size"] == bs].sort_values("k")
        if s.empty:
            continue
        ax1.plot(s["k"], s["tar"], marker=BS_MARKERS[bs], color=BS_COLORS[bs],
                 label=f"bs={bs}", lw=1.6, ms=6)
        ax2.plot(s["k"], s["mean_accepted_length"], marker=BS_MARKERS[bs],
                 color=BS_COLORS[bs], label=f"bs={bs}", lw=1.6, ms=6)
    ax1.set_xlabel("Speculative tokens k"); ax1.set_ylabel("Token acceptance rate")
    ax1.set_title("Token Acceptance Rate"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_xlabel("Speculative tokens k"); ax2.set_ylabel("Mean accepted length (tokens/step)")
    ax2.set_title("Mean Accepted Length"); ax2.grid(alpha=0.3); ax2.legend()
    # Ideal-acceptance reference (accept all k): mean length = k+1
    ks = sorted(d["k"].unique())
    ax2.plot(ks, [k + 1 for k in ks], "k--", lw=1, alpha=0.6, label="perfect (k+1)")
    ax2.legend()
    plt.tight_layout()
    save_fig(fig, out_dir, "plot1_tar_vs_k")


# ---------------------------------------------------------------------------
# Plot 2 — Speedup vs k (dense 27B), per batch size + per category
# ---------------------------------------------------------------------------

def plot2_speedup_vs_k(df, out_dir):
    d = df[(df["target_name"] == DENSE_TARGET) & (df["spec_method"] == "mtp")].copy()
    if d.empty:
        print("  [plot2] no dense MTP data — skipping")
        return
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plot 2 — Throughput Speedup vs Speculative Depth (Qwen3.5-27B)",
                 fontweight="bold")
    axb.axhline(1.0, color="black", ls="--", lw=1.1, label="autoregressive baseline")
    for bs in BATCH_SIZES:
        s = d[d["batch_size"] == bs].sort_values("k")
        if s.empty:
            continue
        axb.plot(s["k"], s["speedup_ratio"], marker=BS_MARKERS[bs],
                 color=BS_COLORS[bs], label=f"bs={bs}", lw=1.8, ms=7)
    axb.set_xlabel("Speculative tokens k"); axb.set_ylabel("Speedup x over baseline")
    axb.set_title("By batch size"); axb.grid(alpha=0.3); axb.legend()

    # Per-category speedup at batch size 1 (needs per-category baseline)
    base_cat = df[(df["target_name"] == DENSE_TARGET) & (df["is_baseline"])]
    cats = ["math", "qa", "code"]
    b1 = d[d["batch_size"] == 1]
    if not b1.empty and not base_cat.empty:
        for i, cat in enumerate(cats):
            col = f"throughput_{cat}"
            bcol = base_cat[base_cat["batch_size"] == 1]
            if col not in d.columns or col not in base_cat.columns or bcol.empty:
                continue
            denom = bcol[col].values[0]
            s = b1.sort_values("k")
            if denom and denom > 0:
                axc.plot(s["k"], s[col] / denom, marker="o", color=COLORS[i],
                         label=cat, lw=1.6, ms=6)
        axc.axhline(1.0, color="black", ls="--", lw=1.0)
    axc.set_xlabel("Speculative tokens k"); axc.set_ylabel("Speedup x over baseline")
    axc.set_title("By task category (bs=1)"); axc.grid(alpha=0.3); axc.legend()
    plt.tight_layout()
    save_fig(fig, out_dir, "plot2_speedup_vs_k")


# ---------------------------------------------------------------------------
# Grouped-bar helper for two conditions
# ---------------------------------------------------------------------------

def _grouped_two(df, condA, labelA, condB, labelB, title, note, out_dir, fname):
    a = df[df["condition"] == condA]
    b = df[df["condition"] == condB]
    if a.empty or b.empty:
        print(f"  [{fname}] missing {condA} or {condB} — skipping")
        return
    fig, (axt, axs) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontweight="bold")
    x = np.arange(len(BATCH_SIZES)); w = 0.36
    for ax, metric, ylabel in [(axt, "tar", "Token acceptance rate"),
                               (axs, "speedup_ratio", "Throughput speedup x")]:
        av = [a[a["batch_size"] == bs][metric].values[0] if not a[a["batch_size"] == bs].empty else np.nan for bs in BATCH_SIZES]
        bv = [b[b["batch_size"] == bs][metric].values[0] if not b[b["batch_size"] == bs].empty else np.nan for bs in BATCH_SIZES]
        r1 = ax.bar(x - w / 2, av, w, label=labelA, color=COLORS[0], alpha=0.85)
        r2 = ax.bar(x + w / 2, bv, w, label=labelB, color=COLORS[1], alpha=0.85)
        ax.set_xlabel("Concurrent batch size"); ax.set_ylabel(ylabel)
        ax.set_xticks(x); ax.set_xticklabels([str(bs) for bs in BATCH_SIZES])
        ax.grid(axis="y", alpha=0.3); ax.legend()
        if metric == "speedup_ratio":
            ax.axhline(1.0, color="black", ls="--", lw=1.0)
        for bar in [*r1, *r2]:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
    axt.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction",
                 fontsize=8, color="gray")
    plt.tight_layout()
    save_fig(fig, out_dir, fname)


def plot3_dense_vs_moe(df, out_dir):
    _grouped_two(df, "K3", "Dense 27B (k=3)", "M3", "MoE 35B-A3B (k=3)",
                 "Plot 3 — Dense vs MoE Target (MTP, k=3)",
                 "Speedup vs each target's own autoregressive baseline.",
                 out_dir, "plot3_dense_vs_moe")


def plot4_thinking_vs_std(df, out_dir):
    _grouped_two(df, "M3", "Standard", "M3_think", "Thinking",
                 "Plot 4 — Standard vs Thinking Mode (MoE 35B-A3B, MTP k=3)",
                 "Speedup vs 35B-A3B autoregressive baseline (baseline_moe).",
                 out_dir, "plot4_thinking_vs_std")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(df, out_dir):
    cols = ["condition", "target_name", "spec_method", "k", "mode",
            "batch_size", "tar", "mean_accepted_length", "throughput_tps", "speedup_ratio"]
    sub = df[cols].sort_values(["target_name", "k", "condition", "batch_size"])
    print("\n=== Summary ===")
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sub.to_csv(Path(out_dir) / "summary_table.csv", index=False)
    print(f"\nsaved {out_dir}/summary_table.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()
    print(f"Loading results from {args.results_dir}")
    df = load_all_results(args.results_dir)
    print(f"Loaded {len(df)} rows, conditions: {sorted(df['condition'].unique())}\n")
    plot1_tar_vs_k(df, args.out_dir)
    plot2_speedup_vs_k(df, args.out_dir)
    plot3_dense_vs_moe(df, args.out_dir)
    plot4_thinking_vs_std(df, args.out_dir)
    summary_table(df, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
