"""
Phase 6 — Pattern Re-distillation (new schema, gate-enforced)

Re-distills MATH-500 and AIME training patterns into the phase-6 schema:
trigger / technique / steps / pitfall / verify.

Three deliberate changes from the phase-2d/3d prompt, each fixing a measured
failure:

  1. No fixed step template. The old prompt literally showed
     "Step 1: / Step 2: / Step 3:", which is why all 1,146 patterns came out
     exactly 3 steps. Here the step count is requested as whatever the method
     needs (2-6).
  2. Abstraction is *enforced*, not requested. The old prompt already said
     "Do NOT reference specific numbers" and was ignored by 13% of AIME and
     8% of MATH-500 patterns, so every pattern is now machine-checked and
     regenerated on failure.
  3. Worked examples are dropped. 43% of MATH-500 examples contained the
     anchor's own gold answer, handing the student a finished answer to a
     different problem.

Sized for a single 16GB GPU: the 8B teacher loads in 4-bit when VRAM is
short (same policy as phase-3d).

Usage:
    python generate_patterns.py --dataset math --max-problems 3   # smoke test
    python generate_patterns.py --dataset both                    # full run
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

import validate_patterns as V

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "patterns"
LOGS_DIR = ROOT / "logs"

TEACHER_MODEL_HF = "Qwen/Qwen3-8B"
MATH_DATASET     = "HuggingFaceH4/MATH-500"
AIME_DATASET     = "qq8933/AIME_1983_2024"
MATH_TRAIN_N     = 400          # matches phase-2d
AIME_TRAIN_FRAC  = 0.8          # 746 problems, matches phase-3d
MAX_NEW_TOKENS   = 2048         # patterns are short; /no_think keeps this small
MAX_RETRIES      = 2            # regeneration attempts when gates fail

SYSTEM = """\
You are an expert math tutor building a reusable strategy library. Another
student will be shown your strategy alongside a DIFFERENT problem of the same
type, and must be able to follow it without ever seeing this problem.

Produce exactly these five sections, using these tags:

<trigger>
The structural condition that tells you this method applies. Start with "A" or
"When". Describe the shape of the problem, never its specific values.
</trigger>

<technique>
The name of the mathematical move, in under 12 words. Name the actual
technique (e.g. "index substitution to collapse a double sum"), not a generic
action like "solve the equation".
</technique>

<steps>
One line per step, each beginning with "- ". Use as many steps as the method
genuinely needs, between 2 and 6 — do not pad to a fixed count. Each step must
state a concrete mathematical operation and be at least 5 words.
</steps>

<pitfall>
The specific mistake someone applying this method is likely to make.
</pitfall>

<verify>
A concrete check confirming the result is right, without recomputing it.
</verify>

ABSOLUTE RULES — a violation makes the strategy unusable:
- Never mention any number that appears in the problem.
- Never state or imply the final answer.
- Never write "this problem" or refer to the given values.
- Every section must generalise to other problems of the same type.\
"""

USER_TEMPLATE = "Problem: {problem} /no_think"

RETRY_TEMPLATE = """\
Your previous strategy was rejected for these reasons:
{failures}

Rewrite it, fixing every issue. Remember: no numbers from the problem, no
final answer, 2-6 substantive steps. Problem: {problem} /no_think\
"""


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"generate_patterns_{ts}.log"
    logger = logging.getLogger("phase6_gen")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(lf)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {lf}")
    return logger


# ── Parsing ──────────────────────────────────────────────────────────────────

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_steps(block):
    """One step per '- ' line; tolerates numbered output as a fallback."""
    steps = [re.sub(r"^[-*]\s*", "", l).strip()
             for l in block.splitlines() if l.strip().startswith(("-", "*"))]
    if not steps:
        steps = [m.strip() for m in
                 re.findall(r"^\s*(?:Step\s*)?\d+[.:)]\s*(.+)$", block, re.M)]
    return [s for s in steps if s]


def parse_response(raw):
    v = strip_thinking(raw)
    return {
        "trigger":   extract_tag(v, "trigger"),
        "technique": extract_tag(v, "technique"),
        "steps":     parse_steps(extract_tag(v, "steps")),
        "pitfall":   extract_tag(v, "pitfall"),
        "verify":    extract_tag(v, "verify"),
    }


# ── IO ───────────────────────────────────────────────────────────────────────

def load_done_ids(*paths):
    """IDs already resolved — accepted OR permanently dropped.

    Dropped ids count as done: decoding is greedy, so re-attempting a problem
    that already failed all its retries reproduces the same output and the same
    rejection. Without this, every resume re-burns 3 attempts per prior drop.
    Delete rejected.jsonl to force a retry after changing the prompt or gates.
    """
    done = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Inference ────────────────────────────────────────────────────────────────

def load_teacher(log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL_HF)

    if not torch.cuda.is_available():
        log.warning("No CUDA — CPU inference will be very slow (smoke tests only)")
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_HF, torch_dtype=torch.float32)
        return model, tok

    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    free = torch.cuda.mem_get_info()[0] / 1e9
    log.info(f"Free VRAM: {free:.1f} GB")
    if free >= 18:
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_HF, torch_dtype=torch.bfloat16, device_map=device)
    else:
        log.info("  VRAM limited (16GB card) — loading teacher in 4-bit")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            TEACHER_MODEL_HF, quantization_config=bnb,
            device_map=device, torch_dtype=torch.bfloat16)
    return model, tok


def infer(model, tok, system, user):
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    ilen = inp["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = out[0][ilen:]
    return tok.decode(new, skip_special_tokens=True), time.perf_counter() - t0, len(new)


# ── Datasets ─────────────────────────────────────────────────────────────────

def get_problems(which, log):
    """Return [(uid, problem, gold, subject, level, source)]."""
    items = []
    if which in ("math", "both"):
        ds = list(load_dataset(MATH_DATASET, split="test"))[:MATH_TRAIN_N]
        for i, r in enumerate(ds):
            items.append((f"math_{i}", r["problem"], str(r["answer"]),
                          r.get("subject", ""), r.get("level", ""), "MATH-500"))
        log.info(f"MATH-500 train: {len(ds)}")
    if which in ("aime", "both"):
        allp = list(load_dataset(AIME_DATASET, split="train"))
        n = int(len(allp) * AIME_TRAIN_FRAC)
        for i, r in enumerate(allp[:n]):
            items.append((f"aime_{i}", r["Question"], str(r["Answer"]), "", "", "AIME"))
        log.info(f"AIME train: {n} (80% of {len(allp)})")
    return items


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "aime", "both"], default="both")
    ap.add_argument("--max-problems", type=int, default=None)
    args = ap.parse_args()

    log = setup_logging()
    log.info("=" * 64)
    log.info("Phase 6 — Pattern Re-distillation (gate-enforced)")
    log.info("=" * 64)

    jsonl_out = DATA_DIR / "patterns.jsonl"
    json_out  = DATA_DIR / "patterns.json"

    items = get_problems(args.dataset, log)
    if args.max_problems:
        items = items[:args.max_problems]
        log.info(f"Capped at {len(items)}")

    done = load_done_ids(jsonl_out, DATA_DIR / "rejected.jsonl")
    todo = [it for it in items if it[0] not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"To do: {len(todo)}")

    stats = {"accepted": 0, "dropped": 0, "retried": 0}
    gate_fails = {}

    if todo:
        model, tok = load_teacher(log)

        for idx, (uid, problem, gold, subject, level, source) in enumerate(todo, 1):
            record, failures = None, {}
            for attempt in range(MAX_RETRIES + 1):
                if attempt == 0:
                    user = USER_TEMPLATE.format(problem=problem)
                else:
                    stats["retried"] += 1
                    user = RETRY_TEMPLATE.format(
                        failures="\n".join(f"- {k}: {v}" for k, v in failures.items()),
                        problem=problem)

                raw, latency, ntok = infer(model, tok, SYSTEM, user)
                fields = parse_response(raw)
                cand = {
                    "id": uid, "problem": problem, "gold_answer": gold,
                    "subject": subject, "level": level, "source": source,
                    **fields,
                }
                ok, failures = V.check(cand, require_new_schema=True)
                if ok:
                    cand["attempts"] = attempt + 1
                    cand["latency_s"] = round(latency, 1)
                    record = cand
                    break

            if record:
                stats["accepted"] += 1
                append_jsonl(jsonl_out, record)
                log.info(f"  [{idx}/{len(todo)}] {uid} OK "
                         f"({len(record['steps'])} steps, attempt {record['attempts']})")
            else:
                stats["dropped"] += 1
                for g in failures:
                    gate_fails[g] = gate_fails.get(g, 0) + 1
                # Store the candidate itself, not just the verdict — without it
                # a false-positive gate cannot be told from a genuinely bad
                # pattern after the fact.
                append_jsonl(DATA_DIR / "rejected.jsonl",
                             {"id": uid, "problem": problem, "gold_answer": gold,
                              "failures": failures, "candidate": fields})
                log.warning(f"  [{idx}/{len(todo)}] {uid} DROPPED after "
                            f"{MAX_RETRIES + 1} attempts: {list(failures)}")

    # Compile
    if jsonl_out.exists():
        records = [json.loads(l) for l in open(jsonl_out) if l.strip()]
        with open(json_out, "w") as f:
            json.dump(records, f, indent=2)
        log.info(f"\nCompiled {len(records)} patterns -> {json_out}")

        from collections import Counter
        dist = Counter(len(r["steps"]) for r in records)
        log.info(f"step-count distribution: {dict(sorted(dist.items()))}")
        if dist:
            top, topn = dist.most_common(1)[0]
            frac = topn / len(records)
            log.info(f"uniformity: {frac*100:.1f}% share {top} steps "
                     f"(limit {V.UNIFORMITY_LIMIT*100:.0f}%)")
            if frac > V.UNIFORMITY_LIMIT:
                log.error("  FAIL — patterns are templated, prompt needs revision")

    log.info(f"\nAccepted {stats['accepted']}, dropped {stats['dropped']}, "
             f"retries {stats['retried']}")
    if gate_fails:
        log.info(f"gate failures among dropped: {gate_fails}")


if __name__ == "__main__":
    main()
