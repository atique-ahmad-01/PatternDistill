"""
Analyze baseline results.

Usage:
    # Summary only
    python scripts/analyze.py

    # Show wrong answers for a model
    python scripts/analyze.py --model 1.7b --wrong

    # Show all answers for a model
    python scripts/analyze.py --model 8b --all

    # Inspect a specific problem by id
    python scripts/analyze.py --model 8b --id 1
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "baseline"


def load(model: str) -> list:
    path = RESULTS_DIR / f"results_{model}.jsonl"
    if not path.exists():
        print(f"  Not found: {path}")
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def print_summary(label: str, results: list):
    if not results:
        return
    n       = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\n{'='*60}")
    print(f"{label} — {n} problems")
    print(f"  Accuracy : {correct/n:.3f}  ({correct}/{n})")
    print(f"  Avg lat  : {sum(r['latency_s'] for r in results)/n:.2f}s")
    print(f"  Avg tok/s: {sum(r['tokens_per_s'] for r in results)/n:.1f}")

    by_level = {}
    for r in results:
        lvl = r["level"]
        by_level.setdefault(lvl, {"correct": 0, "total": 0})
        by_level[lvl]["correct"] += r["correct"]
        by_level[lvl]["total"]   += 1
    print(f"\n  By level:")
    for lvl, v in sorted(by_level.items()):
        print(f"    L{lvl}: {v['correct']}/{v['total']}  acc={v['correct']/v['total']:.2f}")


def print_record(r: dict, show_solution: bool = True):
    status = "✓ CORRECT" if r["correct"] else "✗ WRONG"
    print(f"\n--- ID {r['id']} | {r['subject']} L{r['level']} | {status} ---")
    print(f"  Problem : {r['problem'][:120]}...")
    print(f"  Gold    : {r['gold_answer']}")
    print(f"  Pred    : {r['pred_answer']}")
    if show_solution and "solution" in r:
        sol = r["solution"].strip()
        # Show last 300 chars of solution (where the answer usually is)
        snippet = sol[-300:] if len(sol) > 300 else sol
        print(f"  Solution (tail):\n    {snippet.replace(chr(10), chr(10)+'    ')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["1.7b", "8b"], default=None)
    parser.add_argument("--wrong", action="store_true", help="Show wrong answers")
    parser.add_argument("--all",   action="store_true", help="Show all answers")
    parser.add_argument("--id",    type=int, default=None, help="Inspect one problem by id")
    args = parser.parse_args()

    models = [("1.7b", "Qwen 1.7B"), ("8b", "Qwen 8B")]
    if args.model:
        models = [(args.model, f"Qwen {args.model}")]

    for model_key, label in models:
        results = load(model_key)
        if not results:
            continue

        print_summary(label, results)

        if args.id is not None:
            matches = [r for r in results if r["id"] == args.id]
            if matches:
                print_record(matches[0])
            else:
                print(f"  ID {args.id} not found")

        elif args.wrong:
            wrong = [r for r in results if not r["correct"]]
            print(f"\n  Wrong answers ({len(wrong)}):")
            for r in wrong:
                print_record(r)

        elif args.all:
            for r in results:
                print_record(r)


if __name__ == "__main__":
    main()
