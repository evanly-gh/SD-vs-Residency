"""
run_experiment.py — Speculative decoding experiment runner.

Launches a vLLM server with (optionally) a draft model, runs the prompt set,
collects token acceptance rate and throughput metrics via Prometheus deltas,
and writes a structured JSON result file.

Usage (via SLURM job_template.sh):
  python run_experiment.py \
    --condition A3 \
    --target-model /path/to/Qwen3.5-27B \
    --draft-model  /path/to/Qwen3.5-4B \
    --mode standard \
    --port 8103 \
    --output /path/to/results/A3.json \
    --batch-sizes 1 4 8 16

Direct (smoke test, no SLURM):
  python run_experiment.py --condition A3 --target-model ... --draft-model ... \
    --mode standard --port 8103 --output results/smoke_A3.json \
    --num-prompts 5 --batch-sizes 1
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--condition",       required=True)
    p.add_argument("--target-model",    required=True)
    p.add_argument("--draft-model",     default="")
    p.add_argument("--mode",            default="standard", choices=["standard", "thinking"])
    p.add_argument("--port",            type=int, default=8100)
    p.add_argument("--output",          required=True)
    p.add_argument("--prompts",         default="data/prompts.json")
    p.add_argument("--batch-sizes",     type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--num-spec-tokens", type=int, default=4)
    p.add_argument("--max-tokens",      type=int, default=512)
    p.add_argument("--gpu-memory-util", type=float, default=0.90)
    p.add_argument("--num-prompts",     type=int, default=None,
                   help="Limit to first N prompts (for smoke testing)")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--logs-dir",        default="results/logs")
    p.add_argument("--resume",          action="store_true",
                   help="Skip (prompt_id, batch_size) pairs already in output JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def build_vllm_command(args: argparse.Namespace) -> list[str]:
    is_sd = bool(args.draft_model)
    is_thinking = args.mode == "thinking"

    # B3 thinking mode needs more context for long CoT responses
    max_model_len = 8192 if is_thinking else 4096

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  args.target_model,
        "--port",                   str(args.port),
        "--tensor-parallel-size",   "2",
        "--dtype",                  "bfloat16",
        "--gpu-memory-utilization", str(args.gpu_memory_util),
        "--seed",                   str(args.seed),
        "--reasoning-parser",       "qwen3",
        "--max-model-len",          str(max_model_len),
        "--disable-log-requests",
        "--trust-remote-code",
    ]

    if is_sd:
        spec_config = json.dumps({
            "model":                  args.draft_model,
            "num_speculative_tokens": args.num_spec_tokens,
            "method":                 "draft_model",
        })
        cmd += ["--speculative-config", spec_config]

    # Standard mode explicitly disables thinking to prevent accidental CoT
    if not is_thinking:
        cmd += ["--default-chat-template-kwargs", '{"enable_thinking": false}']

    return cmd


def launch_vllm(
    cmd: list[str],
    port: int,
    logs_dir: str,
    condition: str,
    timeout_s: int = 600,
) -> subprocess.Popen:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log_path = Path(logs_dir) / f"vllm_{condition}_{job_id}.log"

    print(f"[vLLM] Starting server, log -> {log_path}")
    print(f"[vLLM] Command: {' '.join(cmd)}")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # new process group so killpg works
    )

    health_url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout_s
    print(f"[vLLM] Waiting for server on port {port} (timeout {timeout_s}s)...")

    import requests as req
    while time.monotonic() < deadline:
        try:
            r = req.get(health_url, timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        if proc.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"vLLM process exited with code {proc.returncode}. "
                f"Check log: {log_path}"
            )
        time.sleep(5)
    else:
        shutdown_vllm(proc)
        raise TimeoutError(f"vLLM did not become ready within {timeout_s}s. Check {log_path}")

    # Fire a dummy request to confirm inference is actually working
    print("[vLLM] Server healthy. Firing dummy request...")
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    for attempt in range(3):
        try:
            client.chat.completions.create(
                model=args_model_name_from_path(proc),  # vLLM uses the model path as name
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            break
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Dummy request failed after 3 attempts: {e}")
            time.sleep(5)

    print("[vLLM] Server ready for inference.")
    return proc


def args_model_name_from_path(proc) -> str:
    # vLLM registers the model under its path/ID as given on CLI
    # We retrieve it from the /v1/models endpoint
    import requests as req
    try:
        r = req.get(f"http://localhost:{_active_port}/v1/models", timeout=5)
        models = r.json()["data"]
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return _active_target_model  # fallback to global set in main()


def shutdown_vllm(proc: subprocess.Popen) -> None:
    print("[vLLM] Shutting down server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("[vLLM] Timeout waiting for clean exit, sending SIGKILL...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    print(f"[vLLM] Process exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Prometheus metric parsing
# ---------------------------------------------------------------------------

def get_prometheus_snapshot(port: int) -> dict:
    import requests as req
    r = req.get(f"http://localhost:{port}/metrics", timeout=10)
    r.raise_for_status()

    snapshot = {
        "draft_tokens_total":    0.0,
        "accepted_tokens_total": 0.0,
        "num_drafts":            0.0,
        "per_pos_draft":         {},
        "per_pos_accepted":      {},
    }

    for line in r.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue

        # Parse: metric_name{labels} value
        # or: metric_name value (no labels)
        if "{" in line:
            metric_part, val_str = line.rsplit("}", 1)
            metric_name_raw, labels_str = metric_part.split("{", 1)
            value = float(val_str.strip())
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            metric_name_raw, val_str = parts[0], parts[1]
            labels_str = ""
            value = float(val_str)

        name = metric_name_raw.strip()

        if name == "vllm:spec_decode_num_draft_tokens_total":
            snapshot["draft_tokens_total"] = value
        elif name == "vllm:spec_decode_num_accepted_tokens_total":
            snapshot["accepted_tokens_total"] = value
        elif name == "vllm:spec_decode_num_drafts":
            snapshot["num_drafts"] = value
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            # label: position="0"
            pos = None
            for part in labels_str.split(","):
                if "position" in part:
                    pos = int(part.split('"')[1])
            if pos is not None:
                snapshot["per_pos_accepted"][pos] = value
        elif name == "vllm:spec_decode_num_draft_tokens_per_pos_total":
            pos = None
            for part in labels_str.split(","):
                if "position" in part:
                    pos = int(part.split('"')[1])
            if pos is not None:
                snapshot["per_pos_draft"][pos] = value

    return snapshot


def delta_snapshots(before: dict, after: dict) -> dict:
    draft_delta    = after["draft_tokens_total"]    - before["draft_tokens_total"]
    accepted_delta = after["accepted_tokens_total"] - before["accepted_tokens_total"]
    drafts_delta   = after["num_drafts"]            - before["num_drafts"]

    tar = (accepted_delta / draft_delta) if draft_delta > 0 else None
    mean_accepted_len = (
        1.0 + accepted_delta / drafts_delta if drafts_delta > 0 else None
    )

    all_pos = sorted(
        set(before["per_pos_accepted"]) | set(after["per_pos_accepted"])
    )
    per_pos_accepted_delta = [
        after["per_pos_accepted"].get(p, 0) - before["per_pos_accepted"].get(p, 0)
        for p in all_pos
    ]
    per_pos_draft_delta = [
        after["per_pos_draft"].get(p, 0) - before["per_pos_draft"].get(p, 0)
        for p in all_pos
    ]

    return {
        "draft_tokens_delta":     draft_delta,
        "accepted_tokens_delta":  accepted_delta,
        "tar":                    tar,
        "mean_accepted_length":   mean_accepted_len,
        "per_pos_accepted_delta": per_pos_accepted_delta,
        "per_pos_draft_delta":    per_pos_draft_delta,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

async def run_single_prompt(
    client,
    model_id: str,
    prompt: dict,
    mode: str,
    max_tokens: int,
) -> dict:
    enable_thinking = mode == "thinking"
    start = time.monotonic()
    response = await client.chat.completions.create(
        model=model_id,
        messages=prompt["messages"],
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    wall_time = time.monotonic() - start

    output_tokens = response.usage.completion_tokens
    thinking_tokens = None
    # Extract thinking token count if present
    choice = response.choices[0]
    if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
        # Approximate thinking token count from character length (rough)
        # vLLM doesn't separate thinking/answer token counts in usage
        rc = choice.message.reasoning_content
        thinking_tokens = len(rc.split())  # word count as proxy

    return {
        "prompt_id":       prompt["id"],
        "category":        prompt["category"],
        "output_tokens":   output_tokens,
        "wall_time_s":     round(wall_time, 4),
        "tokens_per_sec":  round(output_tokens / wall_time, 2) if wall_time > 0 else 0,
        "thinking_token_count": thinking_tokens,
    }


async def run_batch_async(
    port: int,
    model_id: str,
    prompts: list[dict],
    batch_size: int,
    mode: str,
    max_tokens: int,
) -> tuple[list[dict], dict]:
    """
    Runs all prompts in chunks of batch_size using concurrent requests.
    Returns (per_prompt_results, prom_delta) where prom_delta covers the whole set.
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

    # Snapshot before the entire batch set
    snap_before = get_prometheus_snapshot(port)
    wall_start = time.monotonic()

    all_results = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        tasks = [
            run_single_prompt(client, model_id, p, mode, max_tokens)
            for p in chunk
        ]
        chunk_results = await asyncio.gather(*tasks)
        all_results.extend(chunk_results)

    wall_total = time.monotonic() - wall_start
    snap_after = get_prometheus_snapshot(port)

    prom = delta_snapshots(snap_before, snap_after)
    prom["total_wall_time_s"] = round(wall_total, 3)
    prom["total_output_tokens"] = sum(r["output_tokens"] for r in all_results)
    prom["throughput_tps"] = round(
        prom["total_output_tokens"] / wall_total if wall_total > 0 else 0, 2
    )

    return all_results, prom


def run_batch(port, model_id, prompts, batch_size, mode, max_tokens):
    return asyncio.run(
        run_batch_async(port, model_id, prompts, batch_size, mode, max_tokens)
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_aggregate(prompt_results: list[dict], prom: dict) -> dict:
    by_cat = {}
    categories = set(r["category"] for r in prompt_results)
    for cat in categories:
        cat_results = [r for r in prompt_results if r["category"] == cat]
        total_tok = sum(r["output_tokens"] for r in cat_results)
        total_time = sum(r["wall_time_s"] for r in cat_results)
        by_cat[cat] = {
            "throughput_tps": round(total_tok / total_time if total_time > 0 else 0, 2),
        }

    return {
        "tar":                   round(prom["tar"], 4) if prom["tar"] is not None else None,
        "mean_accepted_length":  round(prom["mean_accepted_length"], 3) if prom["mean_accepted_length"] is not None else None,
        "throughput_tps":        prom["throughput_tps"],
        "speedup_ratio":         None,  # filled in by plot_all.py
        "total_output_tokens":   prom["total_output_tokens"],
        "total_wall_time_s":     prom["total_wall_time_s"],
        "by_category":           by_cat,
    }


# ---------------------------------------------------------------------------
# Model hash
# ---------------------------------------------------------------------------

def compute_model_hash(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return "unknown"
    with open(config_path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def load_checkpoint(output_path: str) -> dict | None:
    p = Path(output_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def get_completed_pairs(checkpoint: dict | None) -> set[tuple[str, int]]:
    if checkpoint is None:
        return set()
    return {
        (r["prompt_id"], r["batch_size"])
        for r in checkpoint.get("per_prompt_results", [])
    }


def write_atomic(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Globals used by the dummy-request helper (set in main before launch)
_active_port: int = 8100
_active_target_model: str = ""


def main():
    global _active_port, _active_target_model

    args = parse_args()
    _active_port = args.port
    _active_target_model = args.target_model

    is_sd = bool(args.draft_model)
    is_baseline = not is_sd

    # Load prompts
    with open(args.prompts, encoding="utf-8") as f:
        prompt_data = json.load(f)
    all_prompts = prompt_data["prompts"]
    if args.num_prompts:
        all_prompts = all_prompts[: args.num_prompts]
    print(f"Loaded {len(all_prompts)} prompts from {args.prompts}")

    # Resume support
    checkpoint = load_checkpoint(args.output) if args.resume else None
    completed = get_completed_pairs(checkpoint)
    if completed:
        print(f"Resuming: {len(completed)} (prompt_id, batch_size) pairs already done")

    # Build result skeleton
    result = checkpoint or {
        "meta": {
            "condition":              args.condition,
            "axis":                   1 if args.condition.startswith("A") or args.condition == "baseline" else 2,
            "draft_model":            args.draft_model or None,
            "target_model":           args.target_model,
            "mode":                   args.mode,
            "num_speculative_tokens": args.num_spec_tokens if is_sd else None,
            "batch_sizes":            args.batch_sizes,
            "num_prompts":            len(all_prompts),
            "vllm_version":           importlib.metadata.version("vllm"),
            "target_model_hash":      compute_model_hash(args.target_model),
            "draft_model_hash":       compute_model_hash(args.draft_model) if is_sd else None,
            "slurm_job_id":           os.environ.get("SLURM_JOB_ID", "local"),
            "hostname":               socket.gethostname(),
            "timestamp_utc":          None,  # set at end
            "random_seed":            args.seed,
            "speculative_decoding":   is_sd,
            "is_baseline":            is_baseline,
        },
        "per_prompt_results": [],
        "aggregate": {"by_batch_size": {}},
    }

    # Launch vLLM
    cmd = build_vllm_command(args)
    proc = launch_vllm(cmd, args.port, args.logs_dir, args.condition)

    # Get model ID from vLLM's registered name
    import requests as req
    model_id = args.target_model
    try:
        r = req.get(f"http://localhost:{args.port}/v1/models", timeout=5)
        models = r.json()["data"]
        if models:
            model_id = models[0]["id"]
    except Exception:
        pass
    print(f"[vLLM] Model ID: {model_id}")

    try:
        for batch_size in args.batch_sizes:
            print(f"\n=== Batch size {batch_size} ({len(all_prompts)} prompts) ===")

            # Filter to prompts not yet completed for this batch_size
            remaining = [
                p for p in all_prompts
                if (p["id"], batch_size) not in completed
            ]
            if not remaining:
                print(f"  All prompts already done for batch_size={batch_size}, skipping.")
                continue

            prompt_results, prom = run_batch(
                port=args.port,
                model_id=model_id,
                prompts=remaining,
                batch_size=batch_size,
                mode=args.mode,
                max_tokens=args.max_tokens,
            )

            # Attach batch_size to each result and merge Prometheus metrics
            for i, pr in enumerate(prompt_results):
                pr["batch_size"] = batch_size
                if is_sd:
                    # Distribute aggregate prom metrics proportionally
                    # (per-prompt prom data is not available per-request)
                    pr["draft_tokens_delta"]    = None  # only aggregate available
                    pr["accepted_tokens_delta"] = None
                    pr["tar_this_prompt"]        = None
                    pr["mean_accepted_length"]   = None
                    pr["per_pos_draft_delta"]    = prom["per_pos_draft_delta"]
                    pr["per_pos_accepted_delta"] = prom["per_pos_accepted_delta"]
                else:
                    pr["draft_tokens_delta"]    = None
                    pr["accepted_tokens_delta"] = None
                    pr["tar_this_prompt"]        = None
                    pr["mean_accepted_length"]   = None
                    pr["per_pos_draft_delta"]    = []
                    pr["per_pos_accepted_delta"] = []

            result["per_prompt_results"].extend(prompt_results)
            completed.update((p["id"], batch_size) for p in remaining)

            # Aggregate for this batch size
            agg = compute_aggregate(prompt_results, prom)
            result["aggregate"]["by_batch_size"][str(batch_size)] = agg

            # Print summary
            tar_str = f"{agg['tar']:.3f}" if agg["tar"] is not None else "N/A (baseline)"
            print(f"  TAR: {tar_str}  |  "
                  f"Throughput: {agg['throughput_tps']:.1f} tok/s  |  "
                  f"Total tokens: {agg['total_output_tokens']}")

            # Atomic write after each batch_size (safe on preemption)
            write_atomic(args.output, result)
            print(f"  Written: {args.output}")

    finally:
        shutdown_vllm(proc)

    # Finalize timestamp
    from datetime import datetime, timezone
    result["meta"]["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    write_atomic(args.output, result)
    print(f"\n=== Done. Results: {args.output} ===")


if __name__ == "__main__":
    main()
