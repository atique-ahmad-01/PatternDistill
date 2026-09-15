"""
Phase 3C — MBPP Pattern Generation

Uses Qwen3-8B teacher to generate step-level reasoning patterns for MBPP
(Mostly Basic Python Problems) training set.

Same approach as phase-2d but adapted for code generation:
  - Pattern: step-level procedure for solving this type of coding problem
  - CoT: concise solution walkthrough
  - Example: condensed code snippet showing the key approach
  - Answer: the complete solution code

Dataset: google-research-datasets/mbpp (full config)
  - Train: 374 problems → pattern generation
  - Test: 500 problems → evaluation (held out)

Usage:
    python generate_patterns.py --max-problems 5   # smoke test on GPU
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

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "patterns"
LOGS_DIR = ROOT / "logs"

JSONL_OUT = DATA_DIR / "patterns.jsonl"
JSON_OUT  = DATA_DIR / "patterns.json"

# ── Config ────────────────────────────────────────────────────────────────────

TEACHER_MODEL_HF     = "Qwen/Qwen3-8B"
TEACHER_MODEL_OLLAMA = "qwen3:8b"
DATASET              = "google-research-datasets/mbpp"
DATASET_CONFIG       = "full"
DATASET_SPLIT        = "train"
MAX_NEW_TOKENS       = 16384
OLLAMA_URL           = "http://localhost:11434"

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert Python programming tutor. For each coding problem produce four sections using these exact tags IN THIS ORDER:

<pattern>
A numbered step-level procedure generalising the key method for this type of coding problem.
Write it FIRST before solving — as executable steps a student can follow on a NEW similar problem:

Step 1: [concrete action — what to identify or parse from the input]
Step 2: [concrete action — what algorithm, data structure, or approach to use]
Step 3: [concrete action — how to implement the core logic]
Note: [one key insight, edge case, or common pitfall]

Rules for the <pattern>:
- Each step must be a concrete programming action, not a description
- Do NOT reference specific variable names, values, or test cases from this problem
- Must be applicable to similar coding problems of this type, not just this one
- Keep each step to one sentence
</pattern>

<cot>
Concise step-by-step solution walkthrough. Show the key logic, skip obvious details.
</cot>

<example>
A condensed code snippet (3-5 lines) showing the key approach. This will be shown to a
student alongside a new problem as a reference example. Show actual code, not prose.
</example>

<answer>
The complete solution code only. No explanation.
</answer>\
"""

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

def infer_hf(model, tokenizer, system: str, user: str) -> tuple[str, float, int]:
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

def infer_ollama(system: str, user: str) -> tuple[str, float, int]:
    payload = {
        "model": TEACHER_MODEL_OLLAMA,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "stream": False,
        "options": {"num_predict": MAX_NEW_TOKENS},
    }
    t0   = time.perf_counter()
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    latency = time.perf_counter() - t0
    data    = resp.json()
    raw     = data["message"]["content"]
    n_tok   = data.get("eval_count", len(raw.split()))
    return raw, latency, n_tok

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3C — MBPP Pattern Generation")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--ollama", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3C — MBPP Pattern Generation")
    log.info("=" * 60)
    log.info(f"Backend : {'ollama' if args.ollama else 'transformers'}")

    # Load dataset
    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)
    problems = list(ds)
    log.info(f"Loaded {len(problems)} MBPP training problems")

    if args.max_problems:
        problems = problems[:args.max_problems]
        log.info(f"Capped at: {len(problems)}")

    # Resume support
    done = load_done_ids(JSONL_OUT)
    remaining = [(i, p) for i, p in enumerate(problems) if p["task_id"] not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"To do   : {len(remaining)}")

    if not remaining:
        log.info("All done — skipping to compile")
    else:
        # Load model
        if args.ollama:
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
                log.info(f"Ollama OK at {OLLAMA_URL}")
            except Exception as e:
                raise RuntimeError(f"Cannot reach Ollama: {e}")
            model = tokenizer = None
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
                    log.info(f"  VRAM sufficient — bfloat16")
                    model = AutoModelForCausalLM.from_pretrained(
                        TEACHER_MODEL_HF, torch_dtype=torch.bfloat16, device_map=device)
                else:
                    log.info(f"  VRAM limited — 4-bit quantization")
                    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                    model = AutoModelForCausalLM.from_pretrained(
                        TEACHER_MODEL_HF, quantization_config=bnb,
                        device_map=device, torch_dtype=torch.bfloat16)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    TEACHER_MODEL_HF, torch_dtype=torch.float32, device_map="cpu")

            model.eval()
            log.info("Teacher loaded")

        # Generate
        for idx, (i, item) in enumerate(remaining):
            task_id  = item["task_id"]
            problem  = item["text"]
            gold_code = item["code"]
            test_list = item["test_list"]

            user_msg = USER_TEMPLATE.format(problem=problem)

            try:
                if args.ollama:
                    raw, latency, n_tok = infer_ollama(SYSTEM, user_msg)
                else:
                    raw, latency, n_tok = infer_hf(model, tokenizer, SYSTEM, user_msg)
            except Exception as e:
                log.error(f"  [{idx+1}/{len(remaining)}] task_id={task_id} ERROR: {e}")
                continue

            pattern, cot, example, answer = parse_response(raw)

            record = {
                "id":              task_id,
                "index":           i,
                "problem":         problem,
                "gold_code":       gold_code,
                "test_list":       test_list,
                "domain":          "code",
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
                f"  [{idx+1}/{len(remaining)}] task_id={task_id} | "
                f"{step_tag} | pat={bool(pattern)} ex={bool(example)} | "
                f"{latency:.1f}s {n_tok}tok"
            )

    # ── Compile JSONL → JSON ─────────────────────────────────────────────────
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
    log.info(f"\nCompiled {len(all_records)} records → {JSON_OUT}")

    # Stats
    has_pat = sum(1 for r in all_records if r.get("has_pattern"))
    step_lvl = sum(1 for r in all_records if r.get("is_step_level"))
    has_ex = sum(1 for r in all_records if r.get("worked_example"))
    log.info(f"  Has pattern      : {has_pat}/{len(all_records)}")
    log.info(f"  Step-level       : {step_lvl}/{len(all_records)}")
    log.info(f"  Has worked example: {has_ex}/{len(all_records)}")

    log.info(f"\nNext: run build_memory_bank.py to build MBPP memory bank")


if __name__ == "__main__":
    main()
