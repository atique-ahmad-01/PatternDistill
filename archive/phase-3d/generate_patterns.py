"""
Phase 3D — AIME Pattern Generation

Uses Qwen3-8B teacher to generate step-level reasoning patterns for AIME
problems (first 400 pre-2021 problems).

Dataset: qq8933/AIME_1983_2024
  - Pre-2021[0:400]: pattern generation (400 problems, matching MATH-500 scale)
  - 2021-2024 (103 problems): evaluation (held out)

Usage:
    python generate_patterns.py --max-problems 5   # smoke test
    python generate_patterns.py                     # full run
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

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "patterns"
LOGS_DIR = ROOT / "logs"

JSONL_OUT = DATA_DIR / "patterns.jsonl"
JSON_OUT  = DATA_DIR / "patterns.json"

TEACHER_MODEL_HF     = "Qwen/Qwen3-8B"
DATASET              = "qq8933/AIME_1983_2024"
MAX_NEW_TOKENS       = 16384

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

USER_TEMPLATE = "Problem: {problem} /no_think"


def setup_logging():
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


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def parse_response(raw):
    visible = strip_thinking(raw)
    return extract_tag(visible, "pattern"), extract_tag(visible, "cot"), \
           extract_tag(visible, "example"), extract_tag(visible, "answer")

def is_step_level(pattern):
    return bool(re.search(r"Step\s*1\s*:", pattern, re.IGNORECASE))

def load_done_ids(path):
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

def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def infer_hf(model, tokenizer, system, user):
    import torch
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = tokenizer(text, return_tensors="pt").to(model.device)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3D — AIME Pattern Generation")
    log.info("=" * 60)

    ds = load_dataset(DATASET, split="train")
    # 80% for training (746 problems), 20% for eval (187 problems)
    all_problems = list(ds)
    n_train = int(len(all_problems) * 0.8)  # 746
    problems = all_problems[:n_train]
    log.info(f"AIME total: {len(all_problems)}, using first {n_train} (80%) for patterns")

    if args.max_problems:
        problems = problems[:args.max_problems]
        log.info(f"Capped at: {len(problems)}")

    done = load_done_ids(JSONL_OUT)
    remaining = [(i, p) for i, p in enumerate(problems) if p["ID"] not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"To do: {len(remaining)}")

    if not remaining:
        log.info("All done — skipping to compile")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {device}")
        if torch.cuda.is_available():
            log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            free_vram = torch.cuda.mem_get_info()[0] / 1e9
            log.info(f"Free VRAM: {free_vram:.1f} GB")

        log.info(f"Loading teacher: {TEACHER_MODEL_HF}")
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)

        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] / 1e9
            if free_vram >= 18:
                model = AutoModelForCausalLM.from_pretrained(
                    TEACHER_MODEL_HF, torch_dtype=torch.bfloat16, device_map=device)
            else:
                log.info("  VRAM limited — 4-bit quantization")
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                model = AutoModelForCausalLM.from_pretrained(
                    TEACHER_MODEL_HF, quantization_config=bnb,
                    device_map=device, torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                TEACHER_MODEL_HF, torch_dtype=torch.float32, device_map="cpu")
        model.eval()
        log.info("Teacher loaded")

        for idx, (i, item) in enumerate(remaining):
            aime_id  = item["ID"]
            question = item["Question"]
            gold     = str(item["Answer"]).strip()
            year     = item["Year"]

            user_msg = USER_TEMPLATE.format(problem=question)
            try:
                raw, latency, n_tok = infer_hf(model, tokenizer, SYSTEM, user_msg)
            except Exception as e:
                log.error(f"  [{idx+1}/{len(remaining)}] {aime_id} ERROR: {e}")
                continue

            pattern, cot, example, answer = parse_response(raw)

            record = {
                "id":              aime_id,
                "index":           i,
                "problem":         question,
                "gold_answer":     gold,
                "year":            year,
                "domain":          "math",
                "subject":         "AIME",
                "has_pattern":     bool(pattern),
                "pattern":         pattern,
                "cot":             cot,
                "worked_example":  example,
                "teacher_answer":  answer,
                "is_step_level":   is_step_level(pattern),
                "latency_s":       round(latency, 2),
                "output_tokens":   n_tok,
            }
            append_jsonl(JSONL_OUT, record)

            step_tag = "✓step" if record["is_step_level"] else "✗flat"
            log.info(
                f"  [{idx+1}/{len(remaining)}] {aime_id} ({year}) | "
                f"{step_tag} | pat={bool(pattern)} ex={bool(example)} | "
                f"{latency:.1f}s {n_tok}tok"
            )

    # Compile
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

    has_pat  = sum(1 for r in all_records if r.get("has_pattern"))
    step_lvl = sum(1 for r in all_records if r.get("is_step_level"))
    has_ex   = sum(1 for r in all_records if r.get("worked_example"))
    log.info(f"\nCompiled {len(all_records)} → {JSON_OUT}")
    log.info(f"  Has pattern: {has_pat}/{len(all_records)}")
    log.info(f"  Step-level:  {step_lvl}/{len(all_records)}")
    log.info(f"  Has example: {has_ex}/{len(all_records)}")
    log.info(f"\nNext: run build_memory_and_graph.py")


if __name__ == "__main__":
    main()
