"""
Phase 2B — Memory Injection Evaluation (Subject-Filtered Retrieval)

Fix over Phase 2: retrieval is now filtered to the same subject as the query
problem. Phase 2 used global FAISS search which caused cross-subject pattern
retrieval — e.g. an Algebra pattern injected into an Intermediate Algebra
problem — leading to -18pp regression on Intermediate Algebra and -12pp on
Number Theory.

Change from Phase 2:
  retrieve() now restricts the candidate pool to bank entries whose subject
  matches the query problem's subject before finding the best cosine match.
  If no same-subject entry exists (edge case), it falls back to global search.
  Uses numpy dot product over the pre-loaded vector matrix instead of FAISS
  (bank is only ~370 entries — numpy is fast enough and avoids ID mapping).

Everything else (model, prompts, eval logic, output format) is identical to
Phase 2 so results are directly comparable.

Retrieval:
  - Embed the query problem with nomic-embed-text (sentence-transformers)
  - Filter bank to same subject
  - Retrieve top-1 by cosine similarity within that subject
  - If similarity < SIMILARITY_THRESHOLD (0.40), skip injection

Backends:
  --ollama       Use Ollama REST API (fast, for local Mac smoke tests)
  (default)      Use HuggingFace transformers (for GPU machine full runs)

Output:
  results/results.jsonl   ← one record per problem, written immediately
  results/results.json    ← compiled at the end
  results/summary.json    ← aggregated metrics

Usage:
    python inject_and_eval.py --max-problems 5 --ollama   # Mac smoke test
    python inject_and_eval.py                              # GPU full run
"""

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
MEM_DIR     = ROOT / "memory"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

BANK_JSON   = MEM_DIR / "bank.json"
BANK_INDEX  = MEM_DIR / "bank.index"
JSONL_OUT   = RESULTS_DIR / "results.jsonl"
JSON_OUT    = RESULTS_DIR / "results.json"
SUMMARY_OUT = RESULTS_DIR / "summary.json"

# ── Config ────────────────────────────────────────────────────────────────────

STUDENT_MODEL_HF     = "Qwen/Qwen3-1.7B"
STUDENT_MODEL_OLLAMA = "qwen3:1.7b"
DATASET              = "HuggingFaceH4/MATH-500"
EVAL_SLICE           = slice(400, 500)
MAX_NEW_TOKENS       = 8192
SIMILARITY_THRESHOLD = 0.40

OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# ── Prompts ───────────────────────────────────────────────────────────────────

VANILLA_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "Solve the problem step by step, then state the final answer clearly."
)
VANILLA_USER = (
    "Problem: {problem}\n\n"
    "End your response with: The answer is [your answer]."
)

INJECTION_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "When a Reasoning Strategy is provided, apply it explicitly as your "
    "first step before solving the problem. Show your work step by step."
)
INJECTION_USER = (
    "Reasoning Strategy: {pattern}\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"inject_and_eval_{timestamp}.log"

    logger = logging.getLogger("inject_eval")
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

# ── Answer parsing ────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_boxed(text: str) -> str:
    results = []
    i = 0
    while i < len(text):
        if text[i:i+7] == r"\boxed{":
            depth, j = 1, i + 7
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            results.append(text[i+7 : j-1])
            i = j
        else:
            i += 1
    return results[-1].strip() if results else ""


def extract_answer(text: str) -> str:
    text  = strip_thinking(text)
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
    ans = re.sub(r"^\$+(.+?)\$+$", r"\1", ans)
    ans = re.sub(r"^\[(.+)\]$", r"\1", ans)
    ans = re.sub(r"\*+(.+?)\*+", r"\1", ans)
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\,", "").replace("\\!", "")
    return ans.lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

# ── Memory bank ───────────────────────────────────────────────────────────────

def load_memory_bank(log: logging.Logger):
    """
    Load bank metadata and pre-extract all vectors from the FAISS index
    into a numpy matrix for subject-filtered retrieval.
    """
    import faiss

    if not BANK_JSON.exists() or not BANK_INDEX.exists():
        raise FileNotFoundError(
            f"Memory bank not found. Run build_memory_bank.py first.\n"
            f"  Expected: {BANK_JSON}\n  and:      {BANK_INDEX}"
        )

    with open(BANK_JSON) as f:
        bank_meta = json.load(f)

    index = faiss.read_index(str(BANK_INDEX))
    n, dim = index.ntotal, index.d

    # Extract all vectors from FAISS into a numpy matrix (n, dim)
    all_vectors = np.zeros((n, dim), dtype=np.float32)
    index.reconstruct_n(0, n, all_vectors)

    log.info(f"Memory bank: {len(bank_meta)} entries, dim={dim}")
    log.info(f"Subject-filtered retrieval: ON")
    return bank_meta, all_vectors


def load_embed_model(log):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model


def embed_query(embed_model, problem: str) -> np.ndarray:
    """Embed one problem. Returns L2-normalised float32 vector (1, dim)."""
    vec = embed_model.encode(
        [f"search_query: {problem}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32)


def retrieve(embed_model, problem: str, subject: str,
             bank_meta: list, all_vectors: np.ndarray) -> tuple[dict, float, bool]:
    """
    Subject-filtered retrieval.

    Finds the best cosine match within entries whose subject == subject.
    Falls back to global search if no same-subject entry exists.

    Returns (matched_entry, similarity_score, used_fallback).
    """
    # Build subject-filtered index list
    subject_indices = [i for i, m in enumerate(bank_meta) if m["subject"] == subject]
    used_fallback   = False

    if not subject_indices:
        subject_indices = list(range(len(bank_meta)))
        used_fallback   = True

    query_vec = embed_query(embed_model, problem)            # (1, dim)
    sub_vecs  = all_vectors[subject_indices]                 # (k, dim)
    scores    = (sub_vecs @ query_vec.T).flatten()           # cosine sim (L2-normed)

    best_local  = int(np.argmax(scores))
    best_global = subject_indices[best_local]
    score       = float(scores[best_local])

    return bank_meta[best_global], score, used_fallback

# ── Ollama backend ─────────────────────────────────────────────────────────────

def infer_ollama(system: str, user: str) -> tuple[str, float, int]:
    payload = {
        "model": STUDENT_MODEL_OLLAMA,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
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

# ── Transformers backend ───────────────────────────────────────────────────────

def load_student_hf(log: logging.Logger):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device  : {device}")
    if device == "cuda":
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Loading student: {STUDENT_MODEL_HF}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_HF)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_HF,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
    else:
        log.warning("  No CUDA — loading in float32 on CPU (slow)")
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_HF,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    model.eval()
    log.info("Student loaded")
    return model, tokenizer


def infer_hf(model, tokenizer, system: str, user: str) -> tuple[str, float, int]:
    import torch
    messages  = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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
                    done.add(json.loads(line)["problem_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Summarise ─────────────────────────────────────────────────────────────────

def summarize(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {}

    def acc(recs, key):
        return round(sum(r[key] for r in recs) / len(recs), 4)

    def group(recs, group_key, metric_key):
        buckets = defaultdict(list)
        for r in recs:
            buckets[r[group_key]].append(r[metric_key])
        return {
            k: {"accuracy": round(sum(v)/len(v), 4), "n": len(v)}
            for k, v in sorted(buckets.items())
        }

    injected = [r for r in records if not r.get("low_similarity")]
    low_sim  = [r for r in records if r.get("low_similarity")]

    return {
        "n": n,
        "n_injected":           len(injected),
        "n_low_sim":            len(low_sim),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "retrieval_mode":       "subject_filtered",
        "vanilla": {
            "accuracy":          acc(records, "vanilla_correct"),
            "avg_latency_s":     round(sum(r["vanilla_latency_s"] for r in records) / n, 3),
            "avg_output_tokens": round(sum(r["vanilla_tokens"] for r in records) / n, 1),
            "by_subject":        group(records, "subject", "vanilla_correct"),
            "by_level":          group(records, "level",   "vanilla_correct"),
        },
        "injection": {
            "accuracy":          acc(records, "injection_correct"),
            "avg_latency_s":     round(sum(r["injection_latency_s"] for r in records) / n, 3),
            "avg_output_tokens": round(sum(r["injection_tokens"] for r in records) / n, 1),
            "by_subject":        group(records, "subject", "injection_correct"),
            "by_level":          group(records, "level",   "injection_correct"),
        },
        "delta": {
            "accuracy": round(
                acc(records, "injection_correct") - acc(records, "vanilla_correct"), 4
            ),
        },
        "phase_1_baseline": {"student_1_7b": 0.508, "teacher_8b": 0.718, "gap": 0.210},
        "phase_2_baseline": {"vanilla": 0.700, "injection": 0.690, "delta": -0.010},
    }


def print_summary(s: dict, log: logging.Logger):
    log.info(f"\n{'='*60}")
    log.info(f"Phase 2B — Subject-Filtered Memory Injection Results")
    log.info(f"  n={s['n']}  injected={s['n_injected']}  low_sim_skipped={s['n_low_sim']}")
    log.info(f"\n  Vanilla   accuracy : {s['vanilla']['accuracy']:.3f}")
    log.info(f"  Injection accuracy : {s['injection']['accuracy']:.3f}")
    delta = s['delta']['accuracy']
    log.info(f"  Delta              : {delta:+.3f}")
    pct_closed = delta / s['phase_1_baseline']['gap'] * 100
    log.info(f"\n  Phase 1 baseline   : 1.7B={s['phase_1_baseline']['student_1_7b']}  "
             f"8B={s['phase_1_baseline']['teacher_8b']}")
    log.info(f"  Gap closed         : {pct_closed:.1f}% of the 21pp Phase 1 gap")
    p2 = s['phase_2_baseline']
    log.info(f"\n  Phase 2 comparison : van={p2['vanilla']}  inj={p2['injection']}  "
             f"delta={p2['delta']:+.3f}  (global retrieval, no subject filter)")

    log.info(f"\n  Injection accuracy by subject:")
    for subj, v in s["injection"]["by_subject"].items():
        van = s["vanilla"]["by_subject"].get(subj, {}).get("accuracy", 0)
        log.info(f"    {subj:<35} inj={v['accuracy']:.3f}  van={van:.3f}  "
                 f"delta={v['accuracy']-van:+.3f}  (n={v['n']})")

    log.info(f"\n  Injection accuracy by difficulty:")
    for lvl, v in s["injection"]["by_level"].items():
        van = s["vanilla"]["by_level"].get(lvl, {}).get("accuracy", 0)
        log.info(f"    Level {lvl}: inj={v['accuracy']:.3f}  van={van:.3f}  "
                 f"delta={v['accuracy']-van:+.3f}  (n={v['n']})")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit to N problems (smoke test)")
    parser.add_argument("--ollama", action="store_true",
                        help="Use Ollama REST API instead of transformers (for Mac smoke tests)")
    parser.add_argument("--problem-ids", nargs="+", type=int, default=None,
                        help="Only evaluate these specific problem IDs (e.g. 401 408 409)")
    args = parser.parse_args()

    log = setup_logging()
    log.info(f"Backend : {'ollama' if args.ollama else 'transformers'}")
    log.info(f"Retrieval: subject-filtered (Phase 2B fix)")

    # Load dataset
    log.info(f"Dataset : {DATASET}")
    ds       = load_dataset(DATASET, split="test")
    problems = list(ds)[EVAL_SLICE]

    if args.problem_ids:
        target_ids = set(args.problem_ids)
        problems_indexed = [(400+i, p) for i, p in enumerate(problems) if (400+i) in target_ids]
        log.info(f"Targeting {len(problems_indexed)} specific problem IDs: {sorted(target_ids)}")
    else:
        problems_indexed = [(400+i, p) for i, p in enumerate(problems)]

    if args.max_problems:
        problems_indexed = problems_indexed[: args.max_problems]
        log.info(f"Capped at: {len(problems_indexed)} problems")
    else:
        log.info(f"Eval set: {len(problems_indexed)} problems")

    # Resume
    done = load_done_ids(JSONL_OUT)
    if done:
        log.info(f"Resuming — {len(done)} already done")

    remaining = [(pid, item) for pid, item in problems_indexed if pid not in done]
    log.info(f"To do   : {len(remaining)}")

    if remaining:
        bank_meta, all_vectors = load_memory_bank(log)
        embed_model            = load_embed_model(log)

        if args.ollama:
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
                log.info(f"Ollama OK at {OLLAMA_URL}")
            except Exception as e:
                raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_URL}: {e}")
            model = tokenizer = None
        else:
            model, tokenizer = load_student_hf(log)

        for idx, (problem_id, item) in enumerate(remaining):
            problem = item["problem"]
            subject = item.get("subject", "")
            level   = str(item.get("level", ""))
            gold    = item.get("answer", "")

            # ── Subject-filtered retrieval ────────────────────────────────────
            match, sim_score, used_fallback = retrieve(
                embed_model, problem, subject, bank_meta, all_vectors
            )
            low_sim          = sim_score < SIMILARITY_THRESHOLD
            retrieved_pattern = match["pattern"] if not low_sim else ""

            if used_fallback:
                log.info(f"  [{idx+1}/{len(remaining)}] id={problem_id} — no {subject} patterns, using global fallback")
            if low_sim:
                log.info(
                    f"  [{idx+1}/{len(remaining)}] id={problem_id} {subject} L{level} | "
                    f"sim={sim_score:.3f} < {SIMILARITY_THRESHOLD} → skip injection"
                )

            # ── Vanilla ───────────────────────────────────────────────────────
            van_sys  = VANILLA_SYSTEM
            van_user = VANILLA_USER.format(problem=problem)
            if args.ollama:
                van_raw, van_lat, van_tok = infer_ollama(van_sys, van_user)
            else:
                van_raw, van_lat, van_tok = infer_hf(model, tokenizer, van_sys, van_user)

            van_pred = extract_answer(van_raw)
            van_ok   = is_correct(van_pred, gold)

            # ── Injection ─────────────────────────────────────────────────────
            if low_sim:
                inj_sys  = VANILLA_SYSTEM
                inj_user = VANILLA_USER.format(problem=problem)
            else:
                inj_sys  = INJECTION_SYSTEM
                inj_user = INJECTION_USER.format(pattern=retrieved_pattern, problem=problem)

            if args.ollama:
                inj_raw, inj_lat, inj_tok = infer_ollama(inj_sys, inj_user)
            else:
                inj_raw, inj_lat, inj_tok = infer_hf(model, tokenizer, inj_sys, inj_user)

            inj_pred = extract_answer(inj_raw)
            inj_ok   = is_correct(inj_pred, gold)

            record = {
                "problem_id":          problem_id,
                "subject":             subject,
                "level":               level,
                "gold_answer":         gold,
                "retrieved_pattern":   retrieved_pattern,
                "retrieved_from_id":   match["id"],
                "retrieved_subject":   match["subject"],
                "similarity_score":    round(sim_score, 4),
                "subject_filtered":    not used_fallback,
                "low_similarity":      low_sim,
                "vanilla_answer":      van_pred,
                "vanilla_correct":     van_ok,
                "vanilla_latency_s":   round(van_lat, 3),
                "vanilla_tokens":      van_tok,
                "injection_answer":    inj_pred,
                "injection_correct":   inj_ok,
                "injection_latency_s": round(inj_lat, 3),
                "injection_tokens":    inj_tok,
            }

            append_jsonl(JSONL_OUT, record)

            log.info(
                f"  [{idx+1}/{len(remaining)}] id={problem_id} {subject} L{level} | "
                f"sim={sim_score:.3f} | "
                f"van={'✓' if van_ok else '✗'}({van_lat:.0f}s) "
                f"inj={'✓' if inj_ok else '✗'}({inj_lat:.0f}s) | "
                f"{'↑' if inj_ok and not van_ok else '↓' if van_ok and not inj_ok else '='}"
            )

        if not args.ollama and model is not None:
            del model, tokenizer
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log.info(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # ── Compile ───────────────────────────────────────────────────────────────
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

    all_records.sort(key=lambda r: r["problem_id"])

    with open(JSON_OUT, "w") as f:
        json.dump(all_records, f, indent=2)
    log.info(f"\nWrote {len(all_records)} records → {JSON_OUT}")

    summary = summarize(all_records)
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote summary → {SUMMARY_OUT}")

    print_summary(summary, log)


if __name__ == "__main__":
    main()
