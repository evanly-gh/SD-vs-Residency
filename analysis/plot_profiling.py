"""
Plots for hardware resource utilization profiling data.

Generates:
  1. gpu_utilization_{session}.png  — GPU % and VRAM usage over time
  2. bottleneck_comparison.png      — side-by-side resource profiles for
                                      spec vs baseline at crossover conditions

Usage:
    python analysis/plot_profiling.py
    python analysis/plot_profiling.py --session 32b_ngl32_gamma4_spec
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
PROFILING_DIR = REPO_ROOT / "data" / "profiling"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def load_profile(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["gpu_use_pct"] != "NA"].copy()
    df["gpu_use_pct"] = pd.to_numeric(df["gpu_use_pct"], errors="coerce")
    df["mem_use_mb"] = pd.to_numeric(df["mem_use_mb"], errors="coerce")
    df["sclk_mhz"] = pd.to_numeric(df["sclk_mhz"], errors="coerce")
    return df.dropna(subset=["gpu_use_pct"])


def plot_session(session_id: str, out_dir: Path):
    csv_path = PROFILING_DIR / f"profile_{session_id}.csv"
    df = load_profile(csv_path)
    if df is None:
        print(f"  Profile not found: {csv_path}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # GPU utilization
    ax1.plot(df["timestamp_s"], df["gpu_use_pct"], color="#1976D2", linewidth=1)
    ax1.fill_between(df["timestamp_s"], df["gpu_use_pct"], alpha=0.2, color="#1976D2")
    ax1.set_ylabel("GPU Utilization (%)")
    ax1.set_ylim(0, 105)
    ax1.axhline(90, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax1.text(df["timestamp_s"].max() * 0.01, 91, "90% saturation", fontsize=8, color="red")
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    mean_gpu = df["gpu_use_pct"].mean()
    ax1.set_title(f"Session: {session_id}  |  Mean GPU util: {mean_gpu:.1f}%")

    # VRAM usage
    if df["mem_use_mb"].notna().any():
        ax2.plot(df["timestamp_s"], df["mem_use_mb"], color="#388E3C", linewidth=1)
        ax2.fill_between(df["timestamp_s"], df["mem_use_mb"], alpha=0.2, color="#388E3C")
        ax2.set_ylabel("VRAM Used (MB)")
        ax2.axhline(12 * 1024, color="red", linestyle=":", linewidth=1, alpha=0.6)
        ax2.text(df["timestamp_s"].max() * 0.01, 12 * 1024 + 50, "12 GB limit", fontsize=8, color="red")
        ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        mean_vram = df["mem_use_mb"].mean()
        ax2.set_title(f"Mean VRAM: {mean_vram:.0f} MB  ({mean_vram/1024:.1f} GB of 12 GB)")

    ax2.set_xlabel("Time (s) — thermal discard already removed")

    path = out_dir / f"gpu_utilization_{session_id}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bottleneck_comparison(out_dir: Path):
    """Compare spec vs baseline at the crossover NGL condition."""
    # Find matching spec/base session pairs
    profiles = {}
    for csv_path in sorted(PROFILING_DIR.glob("profile_*.csv")):
        session = csv_path.stem.replace("profile_", "")
        df = load_profile(csv_path)
        if df is not None:
            profiles[session] = df

    if not profiles:
        print("  No profiling CSVs found — run scripts/run_profiling.sh first.")
        return

    # Find spec/base pairs with matching model+ngl+gamma
    pairs = {}
    for session in profiles:
        if session.endswith("_spec"):
            base_session = session[:-5] + "_base"
            if base_session in profiles:
                key = session[:-5]
                pairs[key] = (session, base_session)

    if not pairs:
        print("  No matching spec/base profiling pairs found.")
        print(f"  Available sessions: {list(profiles.keys())}")
        return

    for pair_key, (spec_id, base_id) in pairs.items():
        df_spec = profiles[spec_id]
        df_base = profiles[base_id]

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
        fig.suptitle(f"Bottleneck Attribution — {pair_key}", fontsize=12)

        metrics = [
            ("gpu_use_pct", "GPU Utilization (%)", 0, 100, "#1976D2"),
            ("mem_use_mb", "VRAM (MB)", None, None, "#388E3C"),
        ]

        for col, (metric, ylabel, ymin, ymax, color) in enumerate(metrics):
            for row, (df, label) in enumerate([(df_base, "Baseline"), (df_spec, "Speculative")]):
                ax = axes[row][col]
                if metric in df.columns and df[metric].notna().any():
                    ax.plot(df["timestamp_s"], df[metric], color=color, linewidth=1)
                    ax.fill_between(df["timestamp_s"], df[metric], alpha=0.2, color=color)
                    mean_val = df[metric].mean()
                    ax.set_title(f"{label} — mean {mean_val:.1f}")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_ylabel(ylabel)
                if ymin is not None:
                    ax.set_ylim(ymin, ymax)
                if row == 1:
                    ax.set_xlabel("Time (s)")

        path = out_dir / f"bottleneck_comparison_{pair_key}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None,
                        help="Single session ID to plot (e.g. 32b_ngl32_gamma4_spec). Default: all.")
    args = parser.parse_args()

    if args.session:
        plot_session(args.session, FIGURES_DIR)
    else:
        for csv_path in sorted(PROFILING_DIR.glob("profile_*.csv")):
            session = csv_path.stem.replace("profile_", "")
            if not session.endswith(".warmup"):
                plot_session(session, FIGURES_DIR)
        plot_bottleneck_comparison(FIGURES_DIR)

    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
