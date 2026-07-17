"""
Phase 2A — Teacher Pattern Generation

Runs the 8B teacher (Qwen3-8B) on MATH-500[0:400] and extracts three things
per problem:
  <cot>      step-by-step reasoning
  <pattern>  1-2 sentence abstract rule (the thing we'll inject / distill)
  <answer>   final answer

The pattern is the key output — it must be action-oriented, not descriptive.
Good: "When solving systems of equations, first isolate one variable by
      substitution, then back-substitute to find the remaining unknowns."
Bad:  "This problem involves systems of equations."

Backends:
  --ollama       Use Ollama REST API (fast, for local Mac smoke tests)
  (default)      Use HuggingFace transformers (for GPU machine full runs)

Output:
  data/patterns/patterns.jsonl   ← written after every problem (crash-safe)
  data/patterns/patterns.json    ← compiled at the end

Usage:
    python scripts/generate_patterns.py --max-problems 5 --ollama   # Mac smoke test
    python scripts/generate_patterns.py                              # GPU full run
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "patterns"
LOGS_DIR = ROOT / "logs"

JSONL_OUT = DATA_DIR / "patterns.jsonl"
JSON_OUT  = DATA_DIR / "patterns.json"

# ── Config ────────────────────────────────────────────────────────────────────

TEACHER_MODEL_HF     = "Qwen/Qwen3-8B"
TEACHER_MODEL_OLLAMA = "qwen3:8b"
DATASET              = "HuggingFaceH4/MATH-500"
TRAIN_SLICE          = slice(0, 400)
MAX_NEW_TOKENS       = 8192
OLLAMA_URL           = "http://localhost:11434"

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert math tutor who specialises in extracting reusable reasoning patterns.

For each problem you will produce three sections — use these tags exactly:

<cot>
Step-by-step solution to the specific problem. Show all working.
</cot>

<pattern>
A 1-2 sentence abstract rule that generalises the key reasoning move used above.
Write it as an action-oriented instruction starting with "When" or "To".
Focus on the METHOD, not the specific numbers or variables in this problem.
Example format: "When solving [problem type], first [step 1], then [step 2]."
</pattern>

<answer>
The final numerical or algebraic answer only. No prose.
</answer>

Important:
- The <pattern> must be reusable for similar problems, not just a description of this one.
- Use precise mathematical language in the <pattern>.
- Do not include the original problem statement inside any tag.\
"""

USER_TEMPLATE = "Problem: {problem}"

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"generate_patterns_{timestamp}.log"

    logger = logging.getLogger("gen_patterns")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log     : {log_file}")
    return logger

# ── Parsing ───────────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 internal reasoning)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>. Returns '' if not found."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_response(raw: str) -> tuple[str, str, str]:
    """Returns (cot, pattern, answer) extracted from the model's raw output."""
    visible = strip_thinking(raw)
    cot     = extract_tag(visible, "cot")
    pattern = extract_tag(visible, "pattern")
    answer  = extract_tag(visible, "answer")
    return cot, pattern, answer

# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Ollama backend ─────────────────────────────────────────────────────────────

def infer_ollama(problem: str) -> tuple[str, float, int]:
    """Call Ollama chat API. Returns (raw_text, latency_s, approx_tokens)."""
    payload = {
        "model": TEACHER_MODEL_OLLAMA,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": USER_TEMPLATE.format(problem=problem)},
        ],
        "stream": False,
        "options": {"num_predict": MAX_NEW_TOKENS},
    }
    t0   = time.perf_counter()
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    latency = time.perf_counter() - t0

    data   = resp.json()
    raw    = data["message"]["content"]
    n_tok  = data.get("eval_count", len(raw.split()))   # approx if missing
    return raw, latency, n_tok

# ── Transformers backend ───────────────────────────────────────────────────────

def load_teacher_hf(log: logging.Logger):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device  : {device}")
    if device == "cuda":
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Loading teacher: {TEACHER_MODEL_HF}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)

    if device == "cuda":
        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        # 8B in bfloat16 ~16GB — use full precision if VRAM allows, else 4-bit quant
        if free_vram >= 18:
            log.info(f"  Free VRAM {free_vram:.1f}GB — loading in bfloat16 (full precision)")
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
            )
        else:
            log.info(f"  Free VRAM {free_vram:.1f}GB — loading in 4-bit quant")
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF,
                quantization_config=bnb,
                device_map="cuda:0",
            )
    else:
        log.warning("  No CUDA — loading in float32 on CPU (slow)")
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_HF,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    model.eval()
    log.info("Teacher loaded")
    return model, tokenizer


def infer_hf(model, tokenizer, problem: str) -> tuple[str, float, int]:
    import torch
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER_TEMPLATE.format(problem=problem)},
    ]
    text      = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    max_new   = min(MAX_NEW_TOKENS, 32768 - input_len)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency    = time.perf_counter() - t0
    new_tokens = outputs[0][input_len:]
    raw        = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, latency, len(new_tokens)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit to N problems (smoke test)")
    parser.add_argument("--ollama", action="store_true",
                        help="Use Ollama REST API instead of transformers (for Mac smoke tests)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Only generate patterns for these subjects (e.g. 'Intermediate Algebra' Precalculus)")
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"Backend : {'ollama' if args.ollama else 'transformers'}")

    # Load dataset
    log.info(f"Dataset : {DATASET}")
    ds       = load_dataset(DATASET, split="test")
    problems = list(ds)[TRAIN_SLICE]

    # Filter by subject if requested
    if args.subjects:
        before = len(problems)
        problems_indexed = [(i, p) for i, p in enumerate(problems) if p.get("subject","") in args.subjects]
        log.info(f"Subject filter: {args.subjects} → {len(problems_indexed)}/{before} problems")
    else:
        problems_indexed = list(enumerate(problems))

    if args.max_problems:
        problems_indexed = problems_indexed[: args.max_problems]
        log.info(f"Smoke-test mode: {len(problems_indexed)} problems")
    else:
        log.info(f"Problems: {len(problems_indexed)}")

    # Resume
    done = load_done_ids(JSONL_OUT)
    if done:
        log.info(f"Resuming — {len(done)} already done")

    remaining = [(i, item) for i, item in problems_indexed if i not in done]
    log.info(f"To do   : {len(remaining)}")

    if not remaining:
        log.info("All problems already done — compiling JSON.")
    else:
        # Set up backend
        if args.ollama:
            # Verify Ollama is reachable
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
                log.info(f"Ollama OK at {OLLAMA_URL}")
            except Exception as e:
                raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
            model = tokenizer = None
        else:
            model, tokenizer = load_teacher_hf(log)

        missing_pattern = 0

        for idx, (i, item) in enumerate(remaining):
            problem = item["problem"]
            subject = item.get("subject", "")
            level   = item.get("level", "")

            if args.ollama:
                raw, latency, n_tok = infer_ollama(problem)
            else:
                raw, latency, n_tok = infer_hf(model, tokenizer, problem)

            cot, pattern, answer = parse_response(raw)

            if not pattern:
                missing_pattern += 1
                log.warning(f"  [WARN] No <pattern> tag for problem {i}")

            record = {
                "id":             i,
                "problem":        problem,
                "subject":        subject,
                "level":          level,
                "gold_answer":    item.get("answer", ""),
                "cot":            cot,
                "pattern":        pattern,
                "teacher_answer": answer,
                "has_pattern":    bool(pattern),
                "latency_s":      round(latency, 3),
                "output_tokens":  n_tok,
            }

            append_jsonl(JSONL_OUT, record)

            tps = n_tok / latency if latency > 0 else 0
            log.info(
                f"  [{idx+1}/{len(remaining)}] id={i} {subject} L{level} | "
                f"pattern={'✓' if pattern else '✗'} | "
                f"lat={latency:.1f}s tps={tps:.0f} | tok={n_tok}"
            )

        if not args.ollama and model is not None:
            del model, tokenizer
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log.info(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

        log.info(f"\nDone. Missing <pattern>: {missing_pattern}/{len(remaining)}")

    # ── Compile JSONL → JSON ──────────────────────────────────────────────────

    all_records = []
    if JSONL_OUT.exists():
        with open(JSONL_OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    all_records.sort(key=lambda r: r["id"])

    with open(JSON_OUT, "w") as f:
        json.dump(all_records, f, indent=2)

    n_with_pattern = sum(1 for r in all_records if r.get("has_pattern"))
    log.info(f"\nWrote {len(all_records)} records → {JSON_OUT}")
    log.info(f"  Patterns extracted : {n_with_pattern}/{len(all_records)}")
    log.info(f"  Missing patterns   : {len(all_records) - n_with_pattern}")

    by_subject = {}
    for r in all_records:
        s = r.get("subject", "unknown")
        by_subject.setdefault(s, {"total": 0, "with_pattern": 0})
        by_subject[s]["total"] += 1
        if r.get("has_pattern"):
            by_subject[s]["with_pattern"] += 1

    log.info("\n  By subject:")
    for subj, v in sorted(by_subject.items()):
        log.info(f"    {subj:<35} {v['with_pattern']}/{v['total']} patterns")


if __name__ == "__main__":
    main()
