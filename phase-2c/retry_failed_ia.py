"""
Phase 2C — Retry Failed IA Patterns (Token Limit Increase)

Re-runs only the 18 IA problems that failed due to hitting the 8192 token limit.
Uses MAX_NEW_TOKENS=12288 and a more concise prompt to reduce CoT verbosity.

Writes results back into ia_patterns.jsonl, then re-merges patterns.json.

Usage:
    python retry_failed_ia.py
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "patterns"
LOGS_DIR    = ROOT / "logs"

JSONL_OUT   = DATA_DIR / "ia_patterns.jsonl"
JSON_OUT    = DATA_DIR / "ia_patterns.json"
PHASE2_JSON = DATA_DIR / "patterns_phase2.json"
MERGED_JSON = DATA_DIR / "patterns.json"

TEACHER_MODEL_HF = "Qwen/Qwen3-8B"
DATASET          = "HuggingFaceH4/MATH-500"
TRAIN_SLICE      = slice(0, 400)
TARGET_SUBJECT   = "Intermediate Algebra"
MAX_NEW_TOKENS   = 16384

FAILED_IDS = [43, 80, 96, 101, 103, 110, 120, 145, 147, 184, 204, 213, 232, 286, 306, 333, 355, 371]

# More concise prompt — less verbose CoT to save tokens for the pattern
SYSTEM = """\
You are an expert math tutor. For each problem produce three sections IN THIS ORDER:

<pattern>
A numbered step-level procedure generalising the key method. Be concise.

Step 1: [concrete mathematical action]
Step 2: [concrete mathematical action]
Step 3: [concrete mathematical action]
Note: [key insight, optional]

Rules: concrete actions only, no specific values from this problem, reusable for similar problems.
</pattern>

<cot>
Concise step-by-step solution. Show key steps only, skip obvious arithmetic.
</cot>

<answer>
Final answer only.
</answer>\
"""

USER_TEMPLATE = "Problem: {problem} /no_think"

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"retry_failed_ia_{timestamp}.log"
    logger = logging.getLogger("retry_ia")
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

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def is_step_level(pattern):
    return bool(re.search(r"Step\s*1\s*:", pattern, re.IGNORECASE))

def load_teacher(log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)
    free_vram = torch.cuda.mem_get_info()[0] / 1e9
    log.info(f"Loading {TEACHER_MODEL_HF} | Free VRAM {free_vram:.1f}GB — 4-bit quant")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_HF,
        quantization_config=bnb,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    log.info("Teacher loaded")
    return model, tokenizer

def infer(model, tokenizer, problem):
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

def merge_patterns(ia_records, log):
    if not PHASE2_JSON.exists():
        non_ia = []
    else:
        with open(PHASE2_JSON) as f:
            all_p2 = json.load(f)
        non_ia = [r for r in all_p2 if r.get("subject") != TARGET_SUBJECT]

    ia_map = {r["id"]: r for r in ia_records}
    merged = non_ia + list(ia_map.values())
    merged.sort(key=lambda r: r["id"])
    with open(MERGED_JSON, "w") as f:
        json.dump(merged, f, indent=2)
    ia_step = sum(1 for r in ia_map.values() if r.get("is_step_level"))
    log.info(f"Merged patterns.json: {len(merged)} total | IA step-level: {ia_step}/{len(ia_map)}")

def main():
    log = setup_logging()
    log.info(f"Retrying {len(FAILED_IDS)} failed IA patterns with MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

    ds       = load_dataset(DATASET, split="test")
    problems = {i: p for i, p in enumerate(list(ds)[TRAIN_SLICE])
                if i in FAILED_IDS}

    # Load existing jsonl (keep successful ones)
    existing = []
    if JSONL_OUT.exists():
        with open(JSONL_OUT) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    existing_ids = {r["id"] for r in existing if r.get("has_pattern")}
    to_retry = [i for i in FAILED_IDS if i not in existing_ids]
    log.info(f"To retry: {len(to_retry)} (already have {len(existing_ids)} good patterns)")

    if not to_retry:
        log.info("Nothing to retry — all already succeeded.")
        return

    model, tokenizer = load_teacher(log)
    new_records = []

    for idx, i in enumerate(to_retry):
        item    = problems[i]
        problem = item["problem"]
        level   = item.get("level", "")

        raw, latency, n_tok = infer(model, tokenizer, problem)
        visible  = strip_thinking(raw)
        pattern  = extract_tag(visible, "pattern")
        cot      = extract_tag(visible, "cot")
        example  = extract_tag(visible, "example")
        answer   = extract_tag(visible, "answer")

        if not pattern:
            log.warning(f"  [WARN] Still no <pattern> for id={i} (tok={n_tok})")

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
        new_records.append(record)
        log.info(f"  [{idx+1}/{len(to_retry)}] id={i} L{level} | "
                 f"pattern={'✓' if pattern else '✗'} step={'✓' if is_step_level(pattern) else '✗'} | "
                 f"lat={latency:.1f}s tok={n_tok}")

    del model, tokenizer
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # Merge new records into jsonl (replace failed entries)
    all_records = {r["id"]: r for r in existing}
    for r in new_records:
        all_records[r["id"]] = r   # overwrite failed with new attempt

    with open(JSONL_OUT, "w") as f:
        for r in sorted(all_records.values(), key=lambda x: x["id"]):
            f.write(json.dumps(r) + "\n")

    ia_list = sorted(all_records.values(), key=lambda x: x["id"])
    with open(JSON_OUT, "w") as f:
        json.dump(ia_list, f, indent=2)

    good = sum(1 for r in ia_list if r.get("has_pattern"))
    step = sum(1 for r in ia_list if r.get("is_step_level"))
    log.info(f"\nFinal IA patterns: {len(ia_list)} total | {good} with pattern | {step} step-level")

    merge_patterns(ia_list, log)
    log.info("Done. Next: run build_memory_bank.py then inject_and_eval.py")

if __name__ == "__main__":
    main()
