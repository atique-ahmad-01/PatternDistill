"""
Phase 3E — Adaptive Token-Budget Strategy (ATBS) Evaluation

Structural improvements over phase-3D:
  1. No graph, no worked example — strategy-only templates
  2. Early-exit StoppingCriteria on \\boxed{} completion
  3. Conciseness instruction in system prompt
  4. Dual-attempt: if 1st template produces no \\boxed{}, retry with 2nd-best template
  5. 24K token budget for all runs (up from 16K)

Per problem:
  - Vanilla: single run, 24K tokens, early-exit
  - ATBS:    attempt 1 with best template, if no answer → attempt 2 with 2nd-best template
             both attempts: 24K tokens, early-exit, conciseness instruction

Evaluates on 287 problems: 100 MATH-500 + 187 AIME (same split as phase-3D).

Usage:
    python inject_and_eval_atbs.py --max-problems 5   # smoke test
    python inject_and_eval_atbs.py                     # full run
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT        = Path(__file__).parent
BUFFER_DIR  = ROOT / "buffer"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

BUFFER_JSON  = BUFFER_DIR / "buffer.json"
BUFFER_INDEX = BUFFER_DIR / "buffer.index"

JSONL_OUT    = RESULTS_DIR / "results.jsonl"
JSON_OUT     = RESULTS_DIR / "results.json"
SUMMARY_OUT  = RESULTS_DIR / "summary.json"

STUDENT_MODEL_HF = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS   = 20480    # up from 16384 (20K — max safe for 16GB VRAM)
EMBED_MODEL      = "nomic-ai/nomic-embed-text-v1"

# Retrieval thresholds (same as phase-3D)
SIM_THRESHOLDS = {
    "MATH-500": {
        "Intermediate Algebra": 0.60, "Number Theory": 0.65,
        "Precalculus": 0.65, "Geometry": 0.55, "default": 0.50,
    },
    "AIME": {"default": 0.50},
}
SAME_SOURCE_BIAS     = 0.05
CROSS_THRESHOLD_BUMP = 0.10

# ── Prompts ──────────────────────────────────────────────────────────────────

VANILLA_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "Solve the problem step by step, then state the final answer clearly."
)
VANILLA_USER = (
    "Problem: {problem}\n\n"
    "Solve step by step. State your final numerical answer at the end."
)

# ATBS: conciseness instruction + strategy-only template
ATBS_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "A reasoning template is provided to guide your approach. "
    "Adapt the template steps to this problem. "
    "Be direct and efficient — avoid restating the problem or "
    "repeating steps. Present your answer in \\boxed{}."
)
ATBS_USER = (
    "Reasoning Template:\n{template}\n\n"
    "Problem: {problem}\n\n"
    "Adapt the template and solve. Show key steps, then give \\boxed{{answer}}."
)

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"eval_atbs_{ts}.log"
    logger = logging.getLogger("atbs")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(lf)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {lf}")
    return logger

# ── Early-exit StoppingCriteria ──────────────────────────────────────────────

class BoxedStoppingCriteria:
    """Stop generation once a complete \\boxed{...} is detected."""

    def __init__(self, tokenizer, input_length):
        self.tokenizer = tokenizer
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs):
        # Decode only the new tokens
        new_tokens = input_ids[0][self.input_length:]
        if len(new_tokens) < 10:
            return False
        # Only check the last 200 tokens to avoid repeated full decode
        check_start = max(0, len(new_tokens) - 200)
        text = self.tokenizer.decode(new_tokens[check_start:], skip_special_tokens=True)
        # Look for completed \boxed{...}
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return False
        after = text[idx + 7:]
        depth = 1
        for ch in after:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                return True
        return False

# ── Answer parsing (same as phase-3D) ───────────────────────────────────────

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def extract_boxed(text):
    results = []
    i = 0
    while i < len(text):
        if text[i:i+7] == r"\boxed{":
            depth, j = 1, i + 7
            while j < len(text) and depth > 0:
                if text[j] == "{": depth += 1
                elif text[j] == "}": depth -= 1
                j += 1
            results.append(text[i+7:j-1])
            i = j
        else:
            i += 1
    return results[-1].strip() if results else ""

def extract_number(text):
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    return numbers[-1].replace(",", "") if numbers else text.strip()

def extract_answer_math(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return _clean_math_tail(m.group(1).strip().rstrip("."))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:]\s*(.+)", text)
    if m:
        return _clean_math_tail(re.sub(r'\*+', '', m.group(1)).strip())
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _clean_math_tail(ans):
    m = re.match(r'^\$\s*(.+?)\s*\$[\s.,;:]*\w*[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    m = re.match(r'^.*?(\-?[\d,]+\.?\d*)\s+(?:feet|miles|cm|meters?|degrees?|hours?|minutes?|seconds?|square|cubic|units?)\b.*$', ans, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r'^[A-Z][a-z].*?(\-?[\d,]+\.?\d*).*$', ans)
    if m:
        return m.group(1)
    return ans

def extract_answer_aime(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        cleaned = re.sub(r'\*+', '', boxed).replace("$", "").strip()
        return extract_number(cleaned)
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return extract_number(m.group(1))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:]\s*(.+)", text)
    if m:
        return extract_number(re.sub(r'\*+', '', m.group(1)).replace("$", ""))
    bold = re.findall(r'\*\*\$?([\d,]+\.?\d*)\$?\*\*', text)
    if bold:
        return bold[-1].replace(",", "")
    return extract_number(text)

def normalize_math(ans):
    ans = ans.strip().rstrip(".")
    ans = re.sub(r"^\$+(.+?)\$+$", r"\1", ans)
    ans = ans.lstrip("$").rstrip("$")
    ans = re.sub(r"^\[(.+)\]$", r"\1", ans)
    ans = re.sub(r"\*+(.+?)\*+", r"\1", ans)
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mbox\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\^\\circ", "", ans)
    ans = re.sub(r"\\(?:circ|degree)", "", ans)
    ans = re.sub(r"[°]", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\,", "").replace("\\!", "")
    ans = ans.replace("\\$", "").replace("$", "")
    ans = re.sub(r'([\d})])\s*(?:cm|meters?|miles|feet|hours?|minutes?|seconds?|units?|square|cubic|cm2|m2)\b.*$', r'\1', ans, flags=re.IGNORECASE)
    ans = ans.replace("\\begin{pmatrix}", "(").replace("\\end{pmatrix}", ")")
    ans = ans.replace("\\\\", ",")
    if ans.startswith("0."):
        ans_alt = ans[1:]
    elif ans.startswith("."):
        ans_alt = "0" + ans
    else:
        ans_alt = None
    ans = ans.lower()
    if ans_alt:
        ans = ans_alt.lower() + "|" + ans
    return ans

def normalize_aime(ans):
    ans = ans.strip().rstrip(".")
    ans = ans.lstrip("$").rstrip("$")
    ans = re.sub(r'\*+', '', ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(",", "")
    m = re.match(r'^(-?\d+\.?\d*)', ans)
    if m:
        ans = m.group(1)
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans

def is_correct(pred, gold, source):
    if source == "MATH-500":
        np_ = normalize_math(pred)
        ng_ = normalize_math(gold)
        np_forms = set(np_.split("|"))
        ng_forms = set(ng_.split("|"))
        if np_forms & ng_forms:
            return True
        np_clean = np_.split("|")[0] if "|" in np_ else np_
        ng_clean = ng_.split("|")[0] if "|" in ng_ else ng_
        try:
            pred_parts = sorted(np_clean.replace("{","").replace("}","").split(","))
            gold_parts = sorted(ng_clean.replace("{","").replace("}","").split(","))
            if pred_parts == gold_parts and len(pred_parts) > 1:
                return True
        except:
            pass
        return False
    else:
        return normalize_aime(pred) == normalize_aime(gold)

def has_boxed(text):
    """Check if the output contains a completed \\boxed{}."""
    return bool(extract_boxed(strip_thinking(text)))

def extract_answer(text, source):
    if source == "MATH-500":
        return extract_answer_math(text)
    else:
        return extract_answer_aime(text)

# ── Load data ────────────────────────────────────────────────────────────────

def load_buffer(log):
    import faiss
    with open(BUFFER_JSON) as f:
        buffer = json.load(f)
    index = faiss.read_index(str(BUFFER_INDEX))
    n, dim = index.ntotal, index.d
    vecs = np.zeros((n, dim), dtype=np.float32)
    index.reconstruct_n(0, n, vecs)
    log.info(f"Buffer: {len(buffer)} templates, dim={dim}")
    return buffer, vecs

def load_embed(log, device="cuda"):
    from sentence_transformers import SentenceTransformer
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)

def embed_query(model, text):
    v = model.encode([f"search_query: {text}"], normalize_embeddings=True,
                     show_progress_bar=False)
    return v.astype(np.float32).flatten()

# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_top2(qvec, buffer, vecs, problem_source=""):
    """Retrieve best and 2nd-best templates with same-source bias."""
    scores = (vecs @ qvec).flatten()
    if problem_source:
        biased = np.copy(scores)
        for i, entry in enumerate(buffer):
            if entry.get("source", "") == problem_source:
                biased[i] += SAME_SOURCE_BIAS
        ranked = np.argsort(biased)[::-1]
    else:
        ranked = np.argsort(scores)[::-1]

    best_idx = int(ranked[0])
    second_idx = int(ranked[1]) if len(ranked) > 1 else best_idx

    return (buffer[best_idx], float(scores[best_idx]),
            buffer[second_idx], float(scores[second_idx]))

def get_threshold(source, subject="", is_cross_dataset=False):
    thresholds = SIM_THRESHOLDS.get(source, {"default": 0.50})
    base = thresholds.get(subject, thresholds.get("default", 0.50))
    if is_cross_dataset:
        base += CROSS_THRESHOLD_BUMP
    return base

# ── Inference ────────────────────────────────────────────────────────────────

def load_student(log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_HF)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_HF, torch_dtype=torch.bfloat16, device_map="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL_HF, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    log.info("Student loaded")
    return model, tokenizer

def infer_hf(model, tokenizer, system, user, use_early_exit=True):
    """Run inference with optional early-exit on \\boxed{} detection."""
    import torch
    from transformers import StoppingCriteriaList

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    il = inp["input_ids"].shape[1]
    mn = min(MAX_NEW_TOKENS, 32768 - il)

    stopping = None
    if use_early_exit:
        criteria = BoxedStoppingCriteria(tokenizer, il)
        stopping = StoppingCriteriaList([criteria])

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=mn, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    lat = time.perf_counter() - t0
    nt = out[0][il:]
    raw = tokenizer.decode(nt, skip_special_tokens=True)
    return raw, lat, len(nt)

# ── I/O ──────────────────────────────────────────────────────────────────────

def load_done(path):
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                try: done.add(json.loads(l)["problem_id"])
                except: pass
    return done

def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ── Build eval set (same as phase-3D) ───────────────────────────────────────

def build_eval_set(log):
    problems = []
    # MATH-500: last 100
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i, item in enumerate(list(math_ds)[400:]):
        problems.append({
            "problem_id": f"math_{400+i}",
            "source": "MATH-500",
            "subject": item.get("subject", ""),
            "level": str(item.get("level", "")),
            "problem": item["problem"],
            "gold_answer": item.get("answer", ""),
        })
    # AIME: last 187 (20%)
    aime_ds = load_dataset("qq8933/AIME_1983_2024", split="train")
    aime_all = list(aime_ds)
    n_train = int(len(aime_all) * 0.8)
    for item in aime_all[n_train:]:
        problems.append({
            "problem_id": item["ID"],
            "source": "AIME",
            "subject": "AIME",
            "level": "competition",
            "problem": item["Question"],
            "gold_answer": str(item["Answer"]).strip(),
            "year": item["Year"],
        })
    log.info(f"Eval set: {sum(1 for p in problems if p['source']=='MATH-500')} MATH + "
             f"{sum(1 for p in problems if p['source']=='AIME')} AIME = {len(problems)} total")
    return problems

# ── Summary ──────────────────────────────────────────────────────────────────

def summarize(records):
    n = len(records)
    if not n:
        return {}
    def acc(recs, k):
        return round(sum(r[k] for r in recs) / len(recs), 4) if recs else 0

    math_recs = [r for r in records if r["source"] == "MATH-500"]
    aime_recs = [r for r in records if r["source"] == "AIME"]
    injected  = [r for r in records if not r.get("low_similarity")]
    retried   = [r for r in records if r.get("used_2nd_template")]

    return {
        "n": n, "n_math": len(math_recs), "n_aime": len(aime_recs),
        "n_injected": len(injected), "n_skipped": n - len(injected),
        "n_retried": len(retried),
        "max_new_tokens": MAX_NEW_TOKENS,
        "method": "ATBS: strategy-only template + conciseness + early-exit + dual-attempt + 24K budget",
        "overall": {
            "vanilla": acc(records, "vanilla_correct"),
            "atbs": acc(records, "atbs_correct"),
            "delta": round(acc(records, "atbs_correct") - acc(records, "vanilla_correct"), 4),
        },
        "math500": {
            "vanilla": acc(math_recs, "vanilla_correct"),
            "atbs": acc(math_recs, "atbs_correct"),
            "delta": round(acc(math_recs, "atbs_correct") - acc(math_recs, "vanilla_correct"), 4) if math_recs else 0,
        },
        "aime": {
            "vanilla": acc(aime_recs, "vanilla_correct"),
            "atbs": acc(aime_recs, "atbs_correct"),
            "delta": round(acc(aime_recs, "atbs_correct") - acc(aime_recs, "vanilla_correct"), 4) if aime_recs else 0,
        },
    }

def print_summary(s, log):
    log.info(f"\n{'='*60}")
    log.info(f"Phase 3E — ATBS Results")
    log.info(f"{'='*60}")
    log.info(f"  Total: n={s['n']} (MATH={s['n_math']}, AIME={s['n_aime']})")
    log.info(f"  Injected: {s['n_injected']}  Skipped: {s['n_skipped']}  "
             f"Retried (2nd template): {s['n_retried']}")
    log.info(f"  Budget: {s['max_new_tokens']} tokens")
    for label, key in [("OVERALL", "overall"), ("MATH-500", "math500"), ("AIME", "aime")]:
        d = s[key]
        log.info(f"\n  {label}:")
        log.info(f"    Vanilla:  {d['vanilla']:.3f}")
        log.info(f"    ATBS:     {d['atbs']:.3f}")
        log.info(f"    Delta:    {d['delta']:+.3f}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3E — ATBS Evaluation")
    log.info(f"  max_new_tokens = {MAX_NEW_TOKENS}")
    log.info(f"  early-exit on \\boxed{{}}")
    log.info(f"  dual-attempt (2nd-best template on retry)")
    log.info("=" * 60)

    problems = build_eval_set(log)
    if args.max_problems:
        problems = problems[:args.max_problems]
        log.info(f"Capped at: {len(problems)}")

    done = load_done(JSONL_OUT)
    remaining = [(p["problem_id"], p) for p in problems if p["problem_id"] not in done]
    if done:
        log.info(f"Resuming — {len(done)} done")
    log.info(f"To do: {len(remaining)}")

    if remaining:
        buffer, bvecs = load_buffer(log)
        embed_model = load_embed(log, "cuda")

        del embed_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = load_student(log)
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cpu")
        log.info("Embed on CPU, student on GPU")

        for idx, (pid, item) in enumerate(remaining):
            source  = item["source"]
            problem = item["problem"]
            gold    = item["gold_answer"]
            subject = item.get("subject", "")

            # Retrieve top-2 templates
            qvec = embed_query(embed_model, problem)
            t1, t1_sim, t2, t2_sim = retrieve_top2(qvec, buffer, bvecs, problem_source=source)

            t1_src = t1.get("source", "")
            is_cross = t1_src != source
            thr = get_threshold(source, subject, is_cross_dataset=is_cross)
            low = t1_sim < thr

            # ── Vanilla (24K, early-exit) ──
            vr, vl, vt = infer_hf(model, tokenizer, VANILLA_SYSTEM,
                                   VANILLA_USER.format(problem=problem))
            vp = extract_answer(vr, source)
            vok = is_correct(vp, gold, source)

            # ── ATBS Attempt 1: best template ──
            used_2nd = False
            if low:
                # Below threshold — use vanilla prompt (same as phase-3D skip)
                a1r, a1l, a1t = infer_hf(model, tokenizer, VANILLA_SYSTEM,
                                          VANILLA_USER.format(problem=problem))
                log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                         f"sim={t1_sim:.3f} < {thr} -> skip")
            else:
                a1r, a1l, a1t = infer_hf(model, tokenizer, ATBS_SYSTEM,
                                          ATBS_USER.format(template=t1["template"],
                                                           problem=problem))

            # Check if attempt 1 produced a boxed answer
            a1_has_answer = has_boxed(a1r)

            # ── ATBS Attempt 2: 2nd-best template (only if attempt 1 failed) ──
            a2r, a2l, a2t = "", 0.0, 0
            if not a1_has_answer and not low:
                used_2nd = True
                t2_src = t2.get("source", "")
                is_cross_2 = t2_src != source
                thr_2 = get_threshold(source, subject, is_cross_dataset=is_cross_2)

                if t2_sim < thr_2:
                    # 2nd template also below threshold — vanilla prompt
                    a2r, a2l, a2t = infer_hf(model, tokenizer, VANILLA_SYSTEM,
                                              VANILLA_USER.format(problem=problem))
                else:
                    a2r, a2l, a2t = infer_hf(model, tokenizer, ATBS_SYSTEM,
                                              ATBS_USER.format(template=t2["template"],
                                                               problem=problem))

            # Pick the final ATBS answer
            if used_2nd:
                final_raw = a2r
                final_lat = a1l + a2l
                final_tok = a1t + a2t
            else:
                final_raw = a1r
                final_lat = a1l
                final_tok = a1t

            atbs_answer = extract_answer(final_raw, source)
            atbs_correct = is_correct(atbs_answer, gold, source)

            rec = {
                "problem_id": pid, "source": source, "subject": subject,
                "level": item.get("level", ""), "year": item.get("year"),
                "gold_answer": gold,
                "template1_id": t1["id"], "template1_source": t1_src,
                "template1_similarity": round(t1_sim, 4),
                "template2_id": t2["id"], "template2_source": t2.get("source", ""),
                "template2_similarity": round(t2_sim, 4),
                "sim_threshold": thr, "low_similarity": low,
                "used_2nd_template": used_2nd,
                "attempt1_has_answer": a1_has_answer,
                "attempt1_tokens": a1t, "attempt1_latency_s": round(a1l, 3),
                "attempt2_tokens": a2t, "attempt2_latency_s": round(a2l, 3),
                "vanilla_raw": vr[-300:], "vanilla_answer": vp,
                "vanilla_correct": vok, "vanilla_latency_s": round(vl, 3),
                "vanilla_tokens": vt,
                "atbs_raw": final_raw[-300:], "atbs_answer": atbs_answer,
                "atbs_correct": atbs_correct, "atbs_latency_s": round(final_lat, 3),
                "atbs_tokens": final_tok,
            }
            append_jsonl(JSONL_OUT, rec)

            out = ("^" if atbs_correct and not vok else
                   "v" if vok and not atbs_correct else "=")
            retry_tag = " R2" if used_2nd else ""
            from_tag = f" <-{t1_src}" if t1_src != source else ""
            log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                     f"sim={t1_sim:.3f}{from_tag} | "
                     f"van={'Y' if vok else 'N'}({vl:.0f}s/{vt}t) "
                     f"atbs={'Y' if atbs_correct else 'N'}({final_lat:.0f}s/{final_tok}t)"
                     f"{retry_tag} | {out}")

        del model, tokenizer
        import gc; gc.collect()

    # ── Compile ──
    recs = []
    if JSONL_OUT.exists():
        with open(JSONL_OUT) as f:
            for l in f:
                l = l.strip()
                if l:
                    try: recs.append(json.loads(l))
                    except: pass
    recs.sort(key=lambda r: r["problem_id"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w") as f:
        json.dump(recs, f, indent=2)
    log.info(f"\nWrote {len(recs)} -> {JSON_OUT}")
    s = summarize(recs)
    with open(SUMMARY_OUT, "w") as f:
        json.dump(s, f, indent=2)
    print_summary(s, log)

    # ── Comparison with phase-3D ──
    p3d_summary = ROOT.parent / "phase-3d" / "results" / "summary_combined_fixed.json"
    if p3d_summary.exists():
        with open(p3d_summary) as f:
            p3d = json.load(f)
        log.info(f"\n{'='*60}")
        log.info(f"Comparison: Phase 3E (ATBS) vs Phase 3D (ATGI)")
        log.info(f"{'='*60}")
        log.info(f"  {'':20s} {'3D ATGI':>10s} {'3E ATBS':>10s} {'Diff':>10s}")
        for label, k3d, k3e in [("Overall", "overall", "overall"),
                                 ("MATH-500", "math500", "math500"),
                                 ("AIME", "aime", "aime")]:
            d3d = p3d[k3d]["delta"] if k3d in p3d else 0
            d3e = s[k3e]["delta"] if k3e in s else 0
            log.info(f"  {label:20s} {d3d:+10.3f} {d3e:+10.3f} {d3e-d3d:+10.3f}")


if __name__ == "__main__":
    main()
