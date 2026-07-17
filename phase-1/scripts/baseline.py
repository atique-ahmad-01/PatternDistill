"""
Phase 1 — Vanilla Baseline

Runs both Qwen 1.7B and Qwen 8B on MATH500 with the exact same prompt.
No patterns, no fine-tuning — pure vanilla comparison.

Uses HuggingFace transformers directly (no Ollama needed).

Features:
  - Same prompt for both models (fair comparison)
  - Saves after every single problem (no data loss on crash)
  - Auto-resumes from where it left off if interrupted
  - Full GPU utilisation via device_map="cuda"

Usage:
    python scripts/baseline.py --max-problems 5   # smoke test
    python scripts/baseline.py                    # full run

Output:
    results/baseline/results_1.7b.jsonl
    results/baseline/results_8b.jsonl
    results/baseline/summary.json
    logs/baseline_<timestamp>.log
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use HF_TOKEN if set in environment for higher rate limits
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    ("Qwen/Qwen3-1.7B", "Qwen 1.7B", "results_1.7b.jsonl"),
    ("Qwen/Qwen3-8B",   "Qwen 8B",   "results_8b.jsonl"),
]
DATASET   = "HuggingFaceH4/MATH-500"
MAX_TOKENS = 8192
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR  = Path(__file__).parent.parent / "results" / "baseline"
LOGS_DIR = Path(__file__).parent.parent / "logs"

# ── Prompt (same for both models) ─────────────────────────────────────────────

SYSTEM = (
    "You are a math problem solver. "
    "Solve the problem step by step, then state the final answer clearly."
)
USER_TEMPLATE = (
    "Problem: {problem}\n\n"
    "End your response with: The answer is [your answer]."
)

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"baseline_{timestamp}.log"

    logger = logging.getLogger("baseline")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file : {log_file}")
    logger.info(f"Device   : {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU      : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return logger


# ── Answer extraction ─────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_boxed(text: str) -> str:
    """Extract content of the LAST \\boxed{} handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        if text[i:i+7] == r"\boxed{":
            depth = 1
            j = i + 7
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            results.append(text[i+7: j-1])
            i = j
        else:
            i += 1
    return results[-1].strip() if results else ""


def extract_answer(text: str) -> str:
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def normalize(ans: str) -> str:
    ans = ans.strip()
    # Strip enclosing dollar signs ($...$)
    ans = re.sub(r"^\$+(.+?)\$+$", r"\1", ans.strip())
    # Strip markdown bold/italic (**text** or *text*)
    ans = re.sub(r"\*+(.+?)\*+", r"\1", ans)
    # Unwrap \text{...} → just the inner text
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\,", "").replace("\\!", "")
    return ans.lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    seen, records = set(), []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"])
                records.append(r)
    return records


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Model runner ──────────────────────────────────────────────────────────────

def load_model(model_id: str, log: logging.Logger):
    log.info(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log.info(f"  Loading model: {model_id}")
    # Use 4-bit quantization for 8B to fit in 16GB VRAM alongside system processes
    free_vram = torch.cuda.mem_get_info()[0] / 1e9
    # GPU 0 only (RTX 5070 Ti, 16GB). GPU 1 has hardware issues — never use it.
    # 1.7B in bfloat16 ~ 3.4GB → fits fine on GPU 0.
    # 8B in bfloat16 ~ 16GB → use 4-bit quant (~4GB) to fit safely on GPU 0.
    is_large = any(x in model_id for x in ["8B", "8b", "7B", "7b", "13B", "13b", "70B", "70b"])
    use_4bit = is_large
    log.info(f"  Free VRAM: {free_vram:.1f}GB — {'4-bit quant (GPU 0)' if use_4bit else 'bfloat16 (GPU 0)'}")

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda:0",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
    model.eval()
    log.info(f"  Model loaded")
    return model, tokenizer


def run_model(model_id: str, label: str, problems: list,
              out_path: Path, log: logging.Logger) -> list:

    # Resume: skip already-done problems
    done    = {r["id"] for r in load_jsonl(out_path)}
    results = load_jsonl(out_path)
    correct = sum(r["correct"] for r in results)

    if done:
        log.info(f"  Resuming — {len(done)} already done, skipping")

    remaining = [item for i, item in enumerate(problems) if i not in done]
    if not remaining:
        log.info("  All problems already done.")
        return results

    model, tokenizer = load_model(model_id, log)

    for item in remaining:
        i       = problems.index(item)
        problem = item["problem"]
        subject = item.get("subject", "")
        level   = str(item.get("level", ""))

        # Build chat messages
        messages = [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": USER_TEMPLATE.format(problem=problem)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # Limit total sequence length to avoid OOM on long problems
        input_len = inputs["input_ids"].shape[1]
        max_new = min(MAX_TOKENS, 32768 - input_len)

        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,          # greedy — deterministic
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t_start

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw        = tokenizer.decode(new_tokens, skip_special_tokens=True)
        solution   = strip_thinking(raw)
        pred       = extract_answer(raw)
        ok         = is_correct(pred, item.get("answer", ""))
        out_tokens = len(new_tokens)
        tps        = out_tokens / latency if latency > 0 else 0

        correct += ok
        n        = len(results) + 1

        record = {
            "id":            i,
            "problem":       problem,
            "subject":       subject,
            "level":         level,
            "gold_answer":   item.get("answer", ""),
            "pred_answer":   pred,
            "correct":       ok,
            "solution":      solution,
            "raw_response":  raw,
            "latency_s":     round(latency, 3),
            "output_tokens": int(out_tokens),
            "tokens_per_s":  round(tps, 2),
        }

        append_jsonl(out_path, record)
        results.append(record)

        log.info(
            f"  [{label}] [{n}/{len(problems)}] {subject} L{level} | "
            f"{'✓' if ok else '✗'} | lat={latency:.2f}s tps={tps:.1f} | acc={correct/n:.3f}"
        )

    # Free VRAM before loading next model
    del model, tokenizer
    import gc
    gc.collect()
    gc.collect()  # second pass catches circular refs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log.info(f"  VRAM after cleanup: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free")

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(results: list) -> dict:
    n          = len(results)
    correct    = sum(r["correct"] for r in results)
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_level   = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        by_subject[r["subject"]]["correct"] += r["correct"]
        by_subject[r["subject"]]["total"]   += 1
        by_level[r["level"]]["correct"]     += r["correct"]
        by_level[r["level"]]["total"]       += 1

    return {
        "n":                 n,
        "accuracy":          round(correct / n, 4),
        "correct":           correct,
        "avg_latency_s":     round(sum(r["latency_s"] for r in results) / n, 3),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in results) / n, 1),
        "avg_tokens_per_s":  round(sum(r["tokens_per_s"] for r in results) / n, 2),
        "by_subject": {
            k: {"accuracy": round(v["correct"] / v["total"], 4),
                "correct": v["correct"], "n": v["total"]}
            for k, v in sorted(by_subject.items())
        },
        "by_level": {
            k: {"accuracy": round(v["correct"] / v["total"], 4),
                "correct": v["correct"], "n": v["total"]}
            for k, v in sorted(by_level.items())
        },
    }


def print_summary(label: str, s: dict, log: logging.Logger):
    log.info(f"\n{'='*60}")
    log.info(f"{label} — {s['n']} problems")
    log.info(f"  Accuracy:       {s['accuracy']:.3f}  ({s['correct']}/{s['n']})")
    log.info(f"  Avg latency:    {s['avg_latency_s']}s")
    log.info(f"  Avg tokens out: {s['avg_output_tokens']}")
    log.info(f"  Avg tokens/s:   {s['avg_tokens_per_s']}")
    log.info(f"\n  By subject:")
    for subj, v in s["by_subject"].items():
        log.info(f"    {subj:<35} acc={v['accuracy']:.3f}  (n={v['n']})")
    log.info(f"\n  By difficulty level:")
    for lvl, v in s["by_level"].items():
        log.info(f"    Level {lvl}: acc={v['accuracy']:.3f}  (n={v['n']})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems (smoke-test)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging()

    log.info(f"Loading dataset: {DATASET}")
    ds = load_dataset(DATASET, split="test")
    problems = list(ds)
    if args.max_problems:
        problems = problems[: args.max_problems]
        log.info(f"  Smoke-test mode: {len(problems)} problems")
    else:
        log.info(f"  {len(problems)} problems")

    all_summaries = {}

    for model_id, label, filename in MODELS:
        log.info(f"\n{'='*60}")
        log.info(f"Running {label} ({model_id}) ...")
        log.info(f"{'='*60}")

        results = run_model(model_id, label, problems, OUT_DIR / filename, log)
        s = summarize(results)
        all_summaries[label] = s
        print_summary(label, s, log)

    # Save combined summary
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    s1  = all_summaries["Qwen 1.7B"]
    s8  = all_summaries["Qwen 8B"]
    gap = round(s8["accuracy"] - s1["accuracy"], 4)
    log.info(f"\n{'='*60}")
    log.info(f"  Qwen 1.7B accuracy: {s1['accuracy']:.3f}")
    log.info(f"  Qwen 8B   accuracy: {s8['accuracy']:.3f}")
    log.info(f"  Gap (8B - 1.7B):    {gap:+.3f}  ← this is what we aim to close")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
