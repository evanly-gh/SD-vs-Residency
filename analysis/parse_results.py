"""
Parse raw benchmark data into a single tidy DataFrame ready for analysis.

Data sources:
  data/raw/baseline_*.json   — llama-bench JSON (no spec decoding)
  data/raw/spec_*.jsonl      — llama-speculative JSONL (one record per run)
  data/logs/accept_*.log     — llama-speculative acceptance rate logs

Note: In this build of llama.cpp, llama-bench does not support speculative
decoding. Baseline uses llama-bench; speculative runs use llama-speculative.

Usage:
    python analysis/parse_results.py
    python analysis/parse_results.py --output results/all_results.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
LOGS_DIR = REPO_ROOT / "data" / "logs"
RESULTS_DIR = REPO_ROOT / "results"


# ── Filename parsing ──────────────────────────────────────────────────────────

BASELINE_RE = re.compile(r"baseline_(?P<model>\w+)_ngl(?P<ngl>\d+)\.json$")
SPEC_RE = re.compile(r"spec_(?P<model>\w+)_ngl(?P<ngl>\d+)_gamma(?P<gamma>\d+)\.jsonl$")
ACCEPT_RE = re.compile(
    r"accept_(?P<model>\w+)_ngl(?P<ngl>\d+)_gamma(?P<gamma>\d+)"
    r"_(?P<task>code|reasoning|chat)_(?P<think>think|nothink)_p(?P<prompt_idx>\d+)\.log$"
)


def parse_bench_json(path: Path) -> list[dict]:
    """Extract per-run sample data from a llama-bench JSON file (baseline only)."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    # llama-bench JSON structure: top-level list of build+result blocks
    for block in data if isinstance(data, list) else [data]:
        for result in block.get("results", [block]):
            samples_ts = result.get("samples_ts", [])
            if not samples_ts:
                continue
            # Drop first 2 runs (cold + warm), keep remaining for median
            trimmed = samples_ts[2:] if len(samples_ts) > 2 else samples_ts
            rows.append({
                "n_prompt": result.get("n_prompt", 0),
                "n_gen": result.get("n_gen", 0),
                "n_gpu_layers": result.get("n_gpu_layers", -1),
                "avg_ts": result.get("avg_ts", None),
                "stddev_ts": result.get("stddev_ts", None),
                "median_ts": sorted(trimmed)[len(trimmed) // 2] if trimmed else None,
                "samples_ts": trimmed,
                "n_samples": len(trimmed),
            })
    return rows


def parse_spec_jsonl(path: Path) -> list[dict]:
    """
    Parse llama-speculative JSONL output (one JSON record per run).
    Written by run_spec_sweep.sh; each line has: run, model, ngl, gamma, tps, accept_pct.
    Drops first 2 runs (cold + warm) to match the baseline measurement protocol.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tps = obj.get("tps")
            if tps is None or tps == "null":
                continue
            rows.append({
                "run": obj.get("run", 0),
                "tps": float(tps),
                "accept_pct": obj.get("accept_pct"),
            })

    # Drop first 2 runs (cold + warm start)
    trimmed = rows[2:] if len(rows) > 2 else rows
    if not trimmed:
        return []

    tps_vals = [r["tps"] for r in trimmed]
    median_ts = sorted(tps_vals)[len(tps_vals) // 2]
    accept_vals = [r["accept_pct"] for r in trimmed if r.get("accept_pct") is not None]

    return [{
        "median_ts": median_ts,
        "avg_ts": sum(tps_vals) / len(tps_vals),
        "stddev_ts": (sum((x - sum(tps_vals)/len(tps_vals))**2 for x in tps_vals) / len(tps_vals)) ** 0.5,
        "samples_ts": tps_vals,
        "n_samples": len(tps_vals),
        "mean_accept_pct": sum(accept_vals) / len(accept_vals) if accept_vals else None,
    }]


def parse_accept_log(path: Path) -> dict | None:
    """
    Extract acceptance rate stats from a llama-cli log file.
    llama-cli prints something like:
      draft_accepted = 312/512, accept = 60.94%
    or:
      n_drafted = 512, n_accepted = 312, accept = 60.94%
    """
    text = path.read_text(errors="replace")

    # Try several patterns that have appeared across llama.cpp versions
    patterns = [
        r"accept\s*=\s*([\d.]+)\s*%",
        r"acceptance\s+rate[:\s]+([\d.]+)\s*%",
        r"n_accept\s*/\s*n_drafted\s*=\s*[\d]+\s*/\s*[\d]+\s*\(([\d.]+)%\)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            accept_pct = float(m.group(1))
            # Also try to get raw counts
            n_drafted = n_accepted = None
            dm = re.search(r"n_drafted\s*[=:]\s*(\d+)", text, re.IGNORECASE)
            am = re.search(r"n_accept(?:ed)?\s*[=:]\s*(\d+)", text, re.IGNORECASE)
            if dm:
                n_drafted = int(dm.group(1))
            if am:
                n_accepted = int(am.group(1))
            return {
                "accept_pct": accept_pct,
                "n_drafted": n_drafted,
                "n_accepted": n_accepted,
            }
    return None


# ── Main assembly ─────────────────────────────────────────────────────────────

def load_throughput_data() -> pd.DataFrame:
    rows = []

    for path in sorted(RAW_DIR.glob("baseline_*.json")):
        m = BASELINE_RE.match(path.name)
        if not m:
            continue
        for r in parse_bench_json(path):
            rows.append({
                "condition": "baseline",
                "model": m.group("model"),
                "ngl": int(m.group("ngl")),
                "gamma": None,
                **r,
            })

    for path in sorted(RAW_DIR.glob("spec_*.jsonl")):
        m = SPEC_RE.match(path.name)
        if not m:
            continue
        for r in parse_spec_jsonl(path):
            rows.append({
                "condition": "spec",
                "model": m.group("model"),
                "ngl": int(m.group("ngl")),
                "gamma": int(m.group("gamma")),
                **r,
            })

    if not rows:
        print("WARNING: No throughput JSON files found in data/raw/", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ngl"] = df["ngl"].astype(int)
    return df


def load_acceptance_data() -> pd.DataFrame:
    rows = []

    for path in sorted(LOGS_DIR.glob("accept_*.log")):
        m = ACCEPT_RE.match(path.name)
        if not m:
            continue
        stats = parse_accept_log(path)
        if stats is None:
            print(f"  WARNING: could not parse accept stats from {path.name}", file=sys.stderr)
            continue
        rows.append({
            "model": m.group("model"),
            "ngl": int(m.group("ngl")),
            "gamma": int(m.group("gamma")),
            "task": m.group("task"),
            "thinking_mode": m.group("think"),
            "prompt_idx": int(m.group("prompt_idx")),
            **stats,
        })

    if not rows:
        print("WARNING: No acceptance log files found in data/logs/", file=sys.stderr)
        return pd.DataFrame()

    return pd.DataFrame(rows)


def compute_speedup(df_tp: pd.DataFrame) -> pd.DataFrame:
    """Add speedup_ratio column = spec median_ts / baseline median_ts.

    llama-bench emits two rows per (model, ngl): a prompt-eval row (n_gen=0,
    ~1000s of tps) and a token-generation row (n_prompt=0, ~50 tps). Only the
    generation row is comparable to llama-speculative throughput, so filter
    baselines to n_gen > 0 before computing the ratio.
    """
    if df_tp.empty:
        return df_tp

    base_rows = df_tp[(df_tp["condition"] == "baseline") & (df_tp.get("n_gen", 0) > 0)]
    baseline = (
        base_rows.groupby(["model", "ngl"])["median_ts"]
        .median()
        .rename("baseline_ts")
    )
    spec = df_tp[df_tp["condition"] == "spec"].copy()
    spec = spec.join(baseline, on=["model", "ngl"])
    spec["speedup_ratio"] = spec["median_ts"] / spec["baseline_ts"]
    return spec


def main():
    parser = argparse.ArgumentParser(description="Parse benchmark results into tidy CSV")
    parser.add_argument("--output", default=str(RESULTS_DIR / "all_results.csv"))
    parser.add_argument("--accept-output", default=str(RESULTS_DIR / "acceptance_rates.csv"))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading throughput data...")
    df_tp = load_throughput_data()
    if not df_tp.empty:
        df_spec = compute_speedup(df_tp)
        df_spec.to_csv(args.output, index=False)
        print(f"  Saved {len(df_spec)} rows → {args.output}")
        print(f"\n  Speedup summary (median across runs):")
        summary = df_spec.groupby(["model", "ngl", "gamma"])["speedup_ratio"].median()
        print(summary.to_string())

    print("\nLoading acceptance rate data...")
    df_acc = load_acceptance_data()
    if not df_acc.empty:
        df_acc.to_csv(args.accept_output, index=False)
        print(f"  Saved {len(df_acc)} rows → {args.accept_output}")
        print(f"\n  Acceptance rate summary (mean by task × thinking mode):")
        summary = df_acc.groupby(["task", "thinking_mode"])["accept_pct"].mean()
        print(summary.to_string())


if __name__ == "__main__":
    main()
