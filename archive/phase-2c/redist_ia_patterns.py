"""
Phase 2C — Step-Level IA Pattern Re-distillation

Re-distills patterns ONLY for Intermediate Algebra problems using a new prompt
that forces the teacher model (Qwen3-8B) to produce numbered, step-level
procedures instead of 1-sentence abstract rules.

Root cause from Phase 2/2B: IA patterns were too abstract for a 1.7B student
model to act on (e.g. "reindex by letting n = j+k"). The new format produces
concrete numbered steps the student can follow directly.

New pattern format:
  Step 1: [concrete mathematical operation]
  Step 2: [concrete mathematical operation]
  Step 3: [concrete mathematical operation]
  Note: [any key insight or watch-out]

Also captures a worked_example field — a condensed 3-step solution snippet
from the CoT — used for richer injection in inject_and_eval.py.

Output:
  data/patterns/ia_patterns.jsonl  ← crash-safe incremental output
  data/patterns/ia_patterns.json   ← compiled at end
  data/patterns/patterns.json      ← merged: new IA + existing non-IA patterns

Usage:
    python redist_ia_patterns.py --ollama --max-problems 3   # smoke test
    python redist_ia_patterns.py                              # GPU full run
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

ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data" / "patterns"
LOGS_DIR      = ROOT / "logs"

JSONL_OUT     = DATA_DIR / "ia_patterns.jsonl"
JSON_OUT      = DATA_DIR / "ia_patterns.json"
PHASE2_JSON   = DATA_DIR / "patterns_phase2.json"   # non-IA patterns from phase-2
MERGED_JSON   = DATA_DIR / "patterns.json"           # final merged bank

# ── Config ────────────────────────────────────────────────────────────────────

TEACHER_MODEL_HF     = "Qwen/Qwen3-8B"
TEACHER_MODEL_OLLAMA = "qwen3:8b"
DATASET              = "HuggingFaceH4/MATH-500"
TRAIN_SLICE          = slice(0, 400)
TARGET_SUBJECT       = "Intermediate Algebra"
MAX_NEW_TOKENS       = 8192
OLLAMA_URL           = "http://localhost:11434"

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert math tutor specialising in Intermediate Algebra at competition level.

For each problem produce four sections using these exact tags IN THIS ORDER:

<pattern>
A numbered procedure generalising the key method for this type of problem.
Write it FIRST before solving, as executable steps a student can follow on a NEW similar problem:

Step 1: [concrete mathematical action — what to set up or identify]
Step 2: [concrete mathematical action — what operation to perform]
Step 3: [concrete mathematical action — how to simplify or conclude]
Note: [one key insight or common pitfall, if applicable]

Rules for the <pattern>:
- Each step must be a concrete mathematical action, not a description
- Do NOT reference specific numbers, variables, or values from this problem
- Must be applicable to similar Intermediate Algebra problems, not just this one
- Keep each step to one sentence
</pattern>

<cot>
Full step-by-step solution showing all working. Be thorough.
</cot>

<example>
A condensed 3-line worked snippet showing the key algebraic moves from your
solution above. Show actual symbolic manipulation (not prose). This will be
shown to a student alongside a new problem as a reference example.
</example>

<answer>
Final answer only. No prose.
</answer>\
"""

USER_TEMPLATE = "Problem: {problem}"

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"redist_ia_{timestamp}.log"

    logger = logging.getLogger("redist_ia")
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
    visible  = strip_thinking(raw)
    cot      = extract_tag(visible, "cot")
    pattern  = extract_tag(visible, "pattern")
    example  = extract_tag(visible, "example")
    answer   = extract_tag(visible, "answer")
    return cot, pattern, example, answer

def is_step_level(pattern: str) -> bool:
    """Check if pattern follows the numbered step format."""
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
        if free_vram >= 18:
            log.info(f"  Free VRAM {free_vram:.1f}GB — bfloat16")
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF, torch_dtype=torch.bfloat16, device_map="cuda:0")
        else:
            log.info(f"  Free VRAM {free_vram:.1f}GB — 4-bit quant")
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF, quantization_config=bnb, device_map="cuda:0")
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

# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_patterns(ia_records: list, log: logging.Logger):
    """Merge new IA patterns with non-IA patterns from phase-2."""
    if not PHASE2_JSON.exists():
        log.warning(f"Phase-2 patterns not found at {PHASE2_JSON} — merged file will be IA-only")
        non_ia = []
    else:
        with open(PHASE2_JSON) as f:
            all_p2 = json.load(f)
        non_ia = [r for r in all_p2 if r.get("subject") != TARGET_SUBJECT]
        log.info(f"Loaded {len(non_ia)} non-IA patterns from phase-2")

    # Build id→record map for IA (use new re-distilled version)
    ia_map = {r["id"]: r for r in ia_records}

    merged = non_ia + list(ia_map.values())
    merged.sort(key=lambda r: r["id"])

    with open(MERGED_JSON, "w") as f:
        json.dump(merged, f, indent=2)

    ia_with_pat = sum(1 for r in ia_map.values() if r.get("has_pattern"))
    ia_step     = sum(1 for r in ia_map.values() if is_step_level(r.get("pattern", "")))
    log.info(f"\nMerged patterns.json: {len(merged)} total")
    log.info(f"  Non-IA (phase-2)   : {len(non_ia)}")
    log.info(f"  IA re-distilled    : {len(ia_map)} ({ia_with_pat} with pattern, {ia_step} step-level)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--ollama", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"Backend : {'ollama' if args.ollama else 'transformers'}")
    log.info(f"Target  : {TARGET_SUBJECT} only")

    ds       = load_dataset(DATASET, split="test")
    problems = list(ds)[TRAIN_SLICE]

    ia_problems = [(i, p) for i, p in enumerate(problems)
                   if p.get("subject", "") == TARGET_SUBJECT]
    log.info(f"IA problems in training set: {len(ia_problems)}")

    if args.max_problems:
        ia_problems = ia_problems[:args.max_problems]
        log.info(f"Capped at: {len(ia_problems)}")

    done      = load_done_ids(JSONL_OUT)
    remaining = [(i, p) for i, p in ia_problems if i not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"To do   : {len(remaining)}")

    if remaining:
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

        for idx, (i, item) in enumerate(remaining):
            problem = item["problem"]
            level   = item.get("level", "")

            if args.ollama:
                raw, latency, n_tok = infer_ollama(problem)
            else:
                raw, latency, n_tok = infer_hf(model, tokenizer, problem)

            cot, pattern, example, answer = parse_response(raw)

            if not pattern:
                missing_pattern += 1
                log.warning(f"  [WARN] No <pattern> for id={i}")
            elif not is_step_level(pattern):
                non_step += 1
                log.warning(f"  [WARN] Pattern not step-level for id={i} — keeping anyway")

            record = {
                "id":             i,
                "problem":        problem,
                "subject":        TARGET_SUBJECT,
                "level":          level,
                "gold_answer":    item.get("answer", ""),
                "cot":            cot,
                "pattern":        pattern,
                "worked_example": example,
                "teacher_answer": answer,
                "has_pattern":    bool(pattern),
                "is_step_level":  is_step_level(pattern),
                "latency_s":      round(latency, 3),
                "output_tokens":  n_tok,
            }
            append_jsonl(JSONL_OUT, record)

            log.info(
                f"  [{idx+1}/{len(remaining)}] id={i} L{level} | "
                f"pattern={'✓' if pattern else '✗'} "
                f"step={'✓' if is_step_level(pattern) else '✗'} | "
                f"lat={latency:.1f}s tok={n_tok}"
            )

        if not args.ollama and model is not None:
            del model, tokenizer
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log.info(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

        log.info(f"\nDone. Missing pattern: {missing_pattern} | Non-step-level: {non_step}")

    # Compile JSONL → JSON
    ia_records = []
    if JSONL_OUT.exists():
        with open(JSONL_OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ia_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    ia_records.sort(key=lambda r: r["id"])
    with open(JSON_OUT, "w") as f:
        json.dump(ia_records, f, indent=2)
    log.info(f"Wrote {len(ia_records)} IA records → {JSON_OUT}")

    # Merge with non-IA patterns from phase-2
    merge_patterns(ia_records, log)
    log.info(f"Merged patterns → {MERGED_JSON}")
    log.info("Next: run build_memory_bank.py to rebuild FAISS index")


if __name__ == "__main__":
    main()
