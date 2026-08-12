"""
Phase 2D — Full Dataset Pattern Generation (All Subjects, Step-Level)

Extends the Phase 2C approach to ALL 400 training problems across all subjects:
  - Step-level numbered procedures (not 1-sentence abstract rules)
  - /no_think appended to every prompt (prevents Qwen3 spending tokens on <think>)
  - <example> tag (worked snippet) captured for ALL subjects
  - MAX_NEW_TOKENS=8192 (sufficient with /no_think: avg ~300-600 tokens)
  - Teacher on cuda:1 (leaves cuda:0 free for student during eval)
  - Resume-safe via JSONL (crash/restart safe)

Key lessons from Phase 2C:
  - /no_think reduces token usage from 8000-12000 to 200-600 per problem
  - Step-level patterns work: Prealgebra +13.4pp, L1 +11.1pp in Phase 2C
  - <pattern> tag placed FIRST so model writes it before running out of tokens on CoT

Output:
  data/patterns/patterns.jsonl   <- incremental crash-safe output
  data/patterns/patterns.json    <- compiled at end

Usage:
    python generate_patterns.py --ollama --max-problems 5   # smoke test
    python generate_patterns.py                              # GPU full run
    python generate_patterns.py --subject "Algebra"         # single subject test
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

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "patterns"
LOGS_DIR = ROOT / "logs"

JSONL_OUT = DATA_DIR / "patterns.jsonl"
JSON_OUT  = DATA_DIR / "patterns.json"

# ── Config ────────────────────────────────────────────────────────────────────

TEACHER_MODEL_HF     = "Qwen/Qwen3-8B"
TEACHER_MODEL_OLLAMA = "qwen3:8b"
DATASET              = "HuggingFaceH4/MATH-500"
TRAIN_SLICE          = slice(0, 400)
MAX_NEW_TOKENS       = 16384
OLLAMA_URL           = "http://localhost:11434"

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert math tutor. For each problem produce four sections using these exact tags IN THIS ORDER:

<pattern>
A numbered step-level procedure generalising the key method for this type of problem.
Write it FIRST before solving — as executable steps a student can follow on a NEW similar problem:

Step 1: [concrete mathematical action — what to set up or identify]
Step 2: [concrete mathematical action — what operation to perform]
Step 3: [concrete mathematical action — how to simplify or conclude]
Note: [one key insight or common pitfall, if applicable]

Rules for the <pattern>:
- Each step must be a concrete mathematical action, not a description
- Do NOT reference specific numbers, variables, or values from this problem
- Must be applicable to similar problems of this type, not just this one
- Keep each step to one sentence
</pattern>

<cot>
Concise step-by-step solution. Show key steps only, skip obvious arithmetic.
</cot>

<example>
A condensed 3-line worked snippet showing the key algebraic/mathematical moves from your
solution above. Show actual symbolic manipulation (not prose). This will be shown to a
student alongside a new problem as a reference example.
</example>

<answer>
Final answer only. No prose.
</answer>\
"""

# /no_think is CRITICAL — prevents Qwen3's internal <think> block from consuming
# thousands of tokens before writing any output tags
USER_TEMPLATE = "Problem: {problem} /no_think"

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
    logger.info(f"Log: {log_file}")
    return logger

# ── Parsing ───────────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def parse_response(raw: str) -> tuple[str, str, str, str]:
    visible = strip_thinking(raw)
    pattern = extract_tag(visible, "pattern")
    cot     = extract_tag(visible, "cot")
    example = extract_tag(visible, "example")
    answer  = extract_tag(visible, "answer")
    return pattern, cot, example, answer

def is_step_level(pattern: str) -> bool:
    return bool(re.search(r"Step\s*1\s*:", pattern, re.IGNORECASE))

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

# ── Backends ──────────────────────────────────────────────────────────────────

def infer_ollama(problem: str) -> tuple[str, float, int]:
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
    data    = resp.json()
    raw     = data["message"]["content"]
    n_tok   = data.get("eval_count", len(raw.split()))
    return raw, latency, n_tok

def load_teacher_hf(log: logging.Logger):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Prefer cuda:1 to leave cuda:0 free for student, but fall back if unavailable
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = "cpu"
    if n_gpus >= 2:
        try:
            torch.cuda.get_device_properties(1)
            device_map = "cuda:1"
            log.info(f"GPUs available: {n_gpus} — teacher on cuda:1")
        except Exception:
            device_map = "cuda:0"
            log.info(f"GPUs available: {n_gpus} but cuda:1 not usable — teacher on cuda:0")
    elif n_gpus == 1:
        device_map = "cuda:0"
        log.info(f"GPUs available: 1 — teacher on cuda:0")
    else:
        log.info("No GPU — teacher on CPU")

    if torch.cuda.is_available() and "cuda" in device_map:
        gpu_idx = int(device_map.split(":")[-1])
        log.info(f"GPU name : {torch.cuda.get_device_name(gpu_idx)}")
        log.info(f"VRAM     : {torch.cuda.get_device_properties(gpu_idx).total_memory / 1e9:.1f} GB")

    log.info(f"Loading teacher: {TEACHER_MODEL_HF}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)

    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        if free_vram >= 18:
            log.info(f"  Free VRAM {free_vram:.1f}GB — bfloat16")
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF, torch_dtype=torch.bfloat16, device_map=device_map)
        else:
            log.info(f"  Free VRAM {free_vram:.1f}GB — 4-bit quant")
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF, quantization_config=bnb,
                device_map=device_map, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_HF, torch_dtype=torch.float32, device_map="cpu")

    model.eval()
    log.info("Teacher loaded")
    return model, tokenizer

def infer_hf(model, tokenizer, problem: str) -> tuple[str, float, int]:
    import torch
    messages  = [{"role": "system", "content": SYSTEM},
                 {"role": "user",   "content": USER_TEMPLATE.format(problem=problem)}]
    text      = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs    = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    max_new   = min(MAX_NEW_TOKENS, 32768 - input_len)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    latency    = time.perf_counter() - t0
    new_tokens = outputs[0][input_len:]
    raw        = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, latency, len(new_tokens)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--subject", type=str, default=None,
                        help="Only process problems from this subject (for testing)")
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"Backend      : {'ollama' if args.ollama else 'transformers'}")
    log.info(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}  (with /no_think: avg ~300-600 tokens)")
    log.info(f"Dataset      : {DATASET} train[0:400]")

    ds       = load_dataset(DATASET, split="test")
    problems = list(ds)[TRAIN_SLICE]

    # Build indexed list of (original_id, problem_item)
    all_problems = [(i, p) for i, p in enumerate(problems)]

    if args.subject:
        all_problems = [(i, p) for i, p in all_problems
                        if p.get("subject", "") == args.subject]
        log.info(f"Subject filter: {args.subject} → {len(all_problems)} problems")

    if args.max_problems:
        all_problems = all_problems[:args.max_problems]
        log.info(f"Capped at: {len(all_problems)}")

    done      = load_done_ids(JSONL_OUT)
    remaining = [(i, p) for i, p in all_problems if i not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"Total problems: {len(all_problems)} | Remaining: {len(remaining)}")

    if not remaining:
        log.info("All problems already done.")
    else:
        if args.ollama:
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
                log.info(f"Ollama OK at {OLLAMA_URL}")
            except Exception as e:
                raise RuntimeError(f"Cannot reach Ollama: {e}")
            model = tokenizer = None
        else:
            model, tokenizer = load_teacher_hf(log)

        missing_pattern = 0
        non_step        = 0
        subject_counts  = {}

        for idx, (i, item) in enumerate(remaining):
            problem = item["problem"]
            subject = item.get("subject", "")
            level   = item.get("level", "")

            if args.ollama:
                raw, latency, n_tok = infer_ollama(problem)
            else:
                raw, latency, n_tok = infer_hf(model, tokenizer, problem)

            pattern, cot, example, answer = parse_response(raw)

            if not pattern:
                missing_pattern += 1
                log.warning(f"  [WARN] No <pattern> for id={i} ({subject} L{level})")
            elif not is_step_level(pattern):
                non_step += 1
                log.warning(f"  [WARN] Pattern not step-level for id={i} — keeping anyway")

            record = {
                "id":             i,
                "problem":        problem,
                "subject":        subject,
                "level":          level,
                "gold_answer":    item.get("answer", ""),
                "pattern":        pattern,
                "cot":            cot,
                "worked_example": example,
                "teacher_answer": answer,
                "has_pattern":    bool(pattern),
                "is_step_level":  is_step_level(pattern),
                "latency_s":      round(latency, 3),
                "output_tokens":  n_tok,
            }
            append_jsonl(JSONL_OUT, record)

            subject_counts[subject] = subject_counts.get(subject, 0) + 1

            log.info(
                f"  [{idx+1}/{len(remaining)}] id={i} {subject} L{level} | "
                f"pat={'✓' if pattern else '✗'} "
                f"step={'✓' if is_step_level(pattern) else '✗'} "
                f"ex={'✓' if example else '✗'} | "
                f"lat={latency:.1f}s tok={n_tok}"
            )

        if not args.ollama and model is not None:
            del model, tokenizer
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info(f"VRAM free after cleanup: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

        log.info(f"\nRun complete:")
        log.info(f"  Missing pattern : {missing_pattern}")
        log.info(f"  Non-step-level  : {non_step}")
        log.info(f"  By subject this run: {subject_counts}")

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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(all_records, f, indent=2)

    total      = len(all_records)
    with_pat   = sum(1 for r in all_records if r.get("has_pattern"))
    step_level = sum(1 for r in all_records if r.get("is_step_level"))
    with_ex    = sum(1 for r in all_records if r.get("worked_example"))

    log.info(f"\n{'='*50}")
    log.info(f"Compiled {total} records → {JSON_OUT}")
    log.info(f"  Has pattern    : {with_pat}/{total}")
    log.info(f"  Step-level     : {step_level}/{total}")
    log.info(f"  Has example    : {with_ex}/{total}")

    by_subject = {}
    for r in all_records:
        s = r.get("subject", "unknown")
        if s not in by_subject:
            by_subject[s] = {"total": 0, "has_pattern": 0, "step_level": 0}
        by_subject[s]["total"] += 1
        if r.get("has_pattern"):
            by_subject[s]["has_pattern"] += 1
        if r.get("is_step_level"):
            by_subject[s]["step_level"] += 1

    log.info(f"\n  By subject:")
    for subj in sorted(by_subject):
        v = by_subject[subj]
        log.info(f"    {subj:<35} total={v['total']}  "
                 f"pattern={v['has_pattern']}  step={v['step_level']}")

    log.info(f"\nNext: run build_memory_bank.py to build FAISS index")


if __name__ == "__main__":
    main()
