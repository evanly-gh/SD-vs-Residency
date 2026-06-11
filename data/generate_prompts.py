import argparse
import hashlib
import json
import random
from pathlib import Path


def load_math_prompts(n: int = 50, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)
    prompts = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        prompts.append({
            "id": f"math_{i:03d}",
            "category": "math",
            "messages": [
                {"role": "system", "content": "Solve the math problem step by step. Show your work."},
                {"role": "user", "content": row["question"]},
            ],
        })
    return prompts


def load_qa_prompts(n: int = 50, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(seed)
    # filter to rows that have at least one answer (not unanswerable)
    answerable = [i for i, row in enumerate(ds) if row["answers"]["text"]]
    indices = rng.sample(answerable, n)
    prompts = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        prompts.append({
            "id": f"qa_{i:03d}",
            "category": "qa",
            "messages": [
                {"role": "system", "content": "Answer the question concisely based on your knowledge."},
                {"role": "user", "content": row["question"]},
            ],
        })
    return prompts


def load_code_prompts(n: int = 50, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    prompts = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        prompts.append({
            "id": f"code_{i:03d}",
            "category": "code",
            "messages": [
                {"role": "system", "content": "Complete the Python function. Return only the implementation, no explanation."},
                {"role": "user", "content": row["prompt"]},
            ],
        })
    return prompts


def compute_checksum(prompts: list[dict]) -> str:
    content = json.dumps(prompts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/prompts.json")
    parser.add_argument("--n-each", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading math prompts (GSM8K)...")
    math = load_math_prompts(args.n_each, args.seed)
    print(f"  {len(math)} prompts loaded")

    print("Loading QA prompts (SQuAD v2)...")
    qa = load_qa_prompts(args.n_each, args.seed)
    print(f"  {len(qa)} prompts loaded")

    print("Loading code prompts (HumanEval)...")
    code = load_code_prompts(args.n_each, args.seed)
    print(f"  {len(code)} prompts loaded")

    all_prompts = math + qa + code
    rng = random.Random(args.seed)
    rng.shuffle(all_prompts)

    checksum = compute_checksum(all_prompts)
    output = {
        "checksum": checksum,
        "seed": args.seed,
        "n_math": len(math),
        "n_qa": len(qa),
        "n_code": len(code),
        "prompts": all_prompts,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(all_prompts)} prompts to {args.output}")
    print(f"Checksum: {checksum}")


if __name__ == "__main__":
    main()
