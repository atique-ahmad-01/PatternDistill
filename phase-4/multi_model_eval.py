"""
Phase 4 — Multi-Model Evaluation

Same pipeline as phase-3d (ATGI + anchor-first + thought graph),
same combined MATH+AIME bank, same 287 test problems.

Runs 3 additional student models to test generalization:
  1. Qwen/Qwen3-0.6B
  2. google/gemma-2-2b-it
  3. microsoft/Phi-3-mini-4k-instruct

Usage:
    python multi_model_eval.py --model qwen3-0.6b
    python multi_model_eval.py --model gemma-2-2b
    python multi_model_eval.py --model phi-3-mini
    python multi_model_eval.py --model all
    python multi_model_eval.py --model qwen3-0.6b --max-problems 5
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
from datasets import load_dataset

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
PHASE2D     = ROOT.parent / "phase-2d"
PHASE3D     = ROOT.parent / "phase-3d"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

# Reuse phase-3d graph and banks
GRAPH_JSON     = PHASE3D / "graph" / "thought_graph.json"
STEP_EMBED_NPY = PHASE3D / "graph" / "step_embeddings.npy"
MATH_BANK_JSON  = PHASE2D / "memory" / "bank.json"
MATH_BANK_INDEX = PHASE2D / "memory" / "bank.index"
AIME_BANK_JSON  = PHASE3D / "memory" / "bank.json"
AIME_BANK_INDEX = PHASE3D / "memory" / "bank.index"

# ── Model registry ───────────────────────────────────────────────────────────

MODELS = {
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-0.6B",
        "short_name": "Qwen3-0.6B",
        "max_context": 32768,
    },
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "short_name": "Gemma2-2B",
        "max_context": 8192,
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "short_name": "Phi3-Mini-3.8B",
        "max_context": 4096,
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "short_name": "Qwen2.5-3B",
        "max_context": 32768,
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "short_name": "Qwen3-4B",
        "max_context": 32768,
        "max_new_tokens": 8192,
    },
}

MAX_NEW_TOKENS   = 16384
EMBED_MODEL      = "nomic-ai/nomic-embed-text-v1"

MAX_BONUS_STEPS   = 1
NOVELTY_THRESHOLD = 0.85

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
ATGI_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "A Reasoning Strategy and Reference Example are provided. "
    "Follow the strategy steps explicitly, adapting them to this "
    "specific problem."
)
ATGI_USER_RICH = (
    "Reasoning Strategy:\n{strategy}\n\n"
    "Reference Example:\n{example}\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)
ATGI_USER_NO_EXAMPLE = (
    "Reasoning Strategy:\n{strategy}\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(model_key):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"eval_{model_key}_{ts}.log"
    logger = logging.getLogger(f"phase4_{model_key}")
    logger.setLevel(logging.DEBUG)
    # Clear existing handlers
    logger.handlers.clear()
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

# ── Answer parsing ───────────────────────────────────────────────────────────

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

def _clean_boxed(boxed):
    """Clean boxed content: strip '= value' suffix, \\[...\\], $$...$$ wrappers."""
    # Strip "= decimal" suffix: "\frac{9}{2} = 4.5" → "\frac{9}{2}"
    boxed = re.sub(r'\s*=\s*-?[\d.]+\s*$', '', boxed)
    # Strip \[ ... \] wrapper
    m = re.match(r'^\\\[\s*(.+?)\s*\\\]$', boxed, re.DOTALL)
    if m:
        boxed = m.group(1)
    # Strip $$ ... $$ wrapper
    m = re.match(r'^\$\$\s*(.+?)\s*\$\$$', boxed, re.DOTALL)
    if m:
        boxed = m.group(1)
    return boxed.strip()

def extract_answer_math(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        return _clean_boxed(boxed)
    # Match "the answer/remainder/result is ..."
    m = re.search(r"[Tt]he (?:answer|remainder|result|value) is[:\s]*(.+)", text)
    if m:
        return _clean_math_tail(m.group(1).strip().rstrip("."))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:]\s*(.+)", text)
    if m:
        return _clean_math_tail(re.sub(r'\*+', '', m.group(1)).strip())
    # Extract from bold: **0**, **42**
    bold = re.findall(r'\*\*\$?([^*]+?)\$?\*\*', text)
    if bold:
        return _clean_math_tail(bold[-1].strip())
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _clean_math_tail(ans):
    # Strip $...$ wrapper
    m = re.match(r'^\$\s*(.+?)\s*\$[\s.,;:]*\w*[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    # Strip $$...$$ wrapper
    m = re.match(r'^\$\$\s*(.+?)\s*\$\$[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    # Strip \[...\] wrapper
    m = re.match(r'^\\\[\s*(.+?)\s*\\\][\s.,;:]*$', ans, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Strip trailing unit words
    m = re.match(r'^.*?(\-?[\d,]+\.?\d*)\s+(?:feet|miles|cm|meters?|degrees?|hours?|minutes?|seconds?|square|cubic|units?)\b.*$', ans, re.IGNORECASE)
    if m:
        return m.group(1)
    # Strip trailing words after a number: "60 handshakes." or "3 boxes contain..."
    m = re.match(r'^(\-?[\d,]+\.?\d*)\s+[a-zA-Z].*$', ans)
    if m:
        return m.group(1)
    # Extract number from sentence: "So, the 8th grade should have 15 representatives"
    # Find the last standalone number in the sentence
    nums = re.findall(r'(?<!\d)(\-?\d+\.?\d*)(?!\d)', ans)
    if nums:
        return nums[-1]
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
    ans = ans.replace("\\tfrac", "\\frac")
    ans = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", ans)
    ans = ans.replace("\\,", "").replace("\\!", "")
    ans = ans.replace("\\$", "").replace("$", "")
    ans = re.sub(r'([\d})])\s*(?:cm|meters?|miles|feet|hours?|minutes?|seconds?|units?|square|cubic|cm2|m2)\b.*$', r'\1', ans, flags=re.IGNORECASE)
    # Strip trailing words after a plain number: "60handshakes" "3boxescontain..."
    ans = re.sub(r'^(\-?\d+\.?\d*)[a-zA-Z].*$', r'\1', ans)
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

# ── Load data ────────────────────────────────────────────────────────────────

def load_graph(log):
    if not GRAPH_JSON.exists():
        log.warning(f"Graph not found: {GRAPH_JSON}")
        return None, None, None, None
    with open(GRAPH_JSON) as f:
        g = json.load(f)
    nodes = g["nodes"]
    adj = {int(k): v for k, v in g["adjacency"].items()}
    emb = np.load(STEP_EMBED_NPY)
    m = g["metadata"]
    log.info(f"Graph: {m['num_nodes']} nodes, cross={m.get('cross_domain_pairs',0)}")
    return nodes, adj, emb, m

def load_combined_bank(log):
    import faiss
    with open(MATH_BANK_JSON) as f:
        math_bank = json.load(f)
    math_idx = faiss.read_index(str(MATH_BANK_INDEX))
    n1, dim = math_idx.ntotal, math_idx.d
    math_vecs = np.zeros((n1, dim), dtype=np.float32)
    math_idx.reconstruct_n(0, n1, math_vecs)

    with open(AIME_BANK_JSON) as f:
        aime_bank = json.load(f)
    aime_idx = faiss.read_index(str(AIME_BANK_INDEX))
    n2 = aime_idx.ntotal
    aime_vecs = np.zeros((n2, dim), dtype=np.float32)
    aime_idx.reconstruct_n(0, n2, aime_vecs)

    for e in math_bank:
        e["_source"] = "MATH-500"
        e["_bank_idx_offset"] = 0
    for e in aime_bank:
        e["_source"] = "AIME"
        e["_bank_idx_offset"] = n1

    combined_bank = math_bank + aime_bank
    combined_vecs = np.vstack([math_vecs, aime_vecs])
    log.info(f"Combined bank: {len(math_bank)} MATH + {len(aime_bank)} AIME = {len(combined_bank)}")
    return combined_bank, combined_vecs

def embed_query(model, text):
    v = model.encode([f"search_query: {text}"], normalize_embeddings=True, show_progress_bar=False)
    return v.astype(np.float32).flatten()

# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_anchor(qvec, bank, vecs, problem_source=""):
    scores = (vecs @ qvec).flatten()
    if problem_source:
        biased_scores = np.copy(scores)
        for i, entry in enumerate(bank):
            if entry.get("_source", "") == problem_source:
                biased_scores[i] += SAME_SOURCE_BIAS
        best = int(np.argmax(biased_scores))
    else:
        best = int(np.argmax(scores))
    return bank[best], float(scores[best])

def get_threshold(source, subject="", is_cross_dataset=False):
    thresholds = SIM_THRESHOLDS.get(source, {"default": 0.50})
    base = thresholds.get(subject, thresholds.get("default", 0.50))
    if is_cross_dataset:
        base += CROSS_THRESHOLD_BUMP
    return base

def find_bonus(anchor_pid, qvec, nodes, adj, emb):
    if nodes is None:
        return []
    anodes = [n for n in nodes if n["pattern_id"] == anchor_pid]
    if not anodes:
        return []
    aids = {n["node_id"] for n in anodes}
    aemb = np.array([emb[n["node_id"]] for n in anodes])
    cands, seen = [], set()
    for an in anodes:
        for edge in adj.get(an["node_id"], []):
            if edge["type"] != "semantic":
                continue
            dst = edge["dst"]
            if dst in aids or dst in seen or nodes[dst]["pattern_id"] == anchor_pid:
                continue
            seen.add(dst)
            sims = (aemb @ emb[dst]).flatten()
            if float((np.max(sims)+1)/2) >= NOVELTY_THRESHOLD:
                continue
            cands.append({"node": nodes[dst], "qsim": float(emb[dst] @ qvec)})
    cands.sort(key=lambda c: c["qsim"], reverse=True)
    return [c["node"] for c in cands[:MAX_BONUS_STEPS]]

def assemble(anodes, bonus):
    lines, sn = [], 1
    for n in anodes:
        if n["label"].lower().startswith("note"):
            lines.append(f"Note: {n['text']}")
        else:
            lines.append(f"Step {sn}: {n['text']}")
            sn += 1
    for n in bonus:
        lines.append(f"Step {sn}: {n['text']}")
        sn += 1
    return "\n".join(lines)

# ── Inference ────────────────────────────────────────────────────────────────

def load_student(model_hf_id, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Loading model: {model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    log.info(f"Student loaded: {model_hf_id}")
    return model, tokenizer

def infer_hf(model, tokenizer, system, user, max_context=32768, max_new_tokens=None):
    import torch
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    il = inp["input_ids"].shape[1]
    mn = min(max_new_tokens, max_context - il)
    if mn <= 0:
        return "(input too long)", 0.0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=mn, do_sample=False,
                             temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    lat = time.perf_counter() - t0
    nt = out[0][il:]
    raw = tokenizer.decode(nt, skip_special_tokens=True)
    return raw, lat, len(nt)

# ── JSONL helpers ────────────────────────────────────────────────────────────

def load_done(path):
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for l in f:
            l = l.strip()
            if l:
                try:
                    done.add(json.loads(l)["problem_id"])
                except:
                    pass
    return done

def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")

# ── Build eval set ───────────────────────────────────────────────────────────

def build_eval_set(log):
    problems = []
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
             f"{sum(1 for p in problems if p['source']=='AIME')} AIME = {len(problems)}")
    return problems

# ── Summary ──────────────────────────────────────────────────────────────────

def summarize(records, model_name):
    n = len(records)
    if not n:
        return {}
    def acc(recs, k):
        return round(sum(r[k] for r in recs)/len(recs), 4) if recs else 0

    math_recs = [r for r in records if r["source"] == "MATH-500"]
    aime_recs = [r for r in records if r["source"] == "AIME"]

    van_c = sum(r["vanilla_correct"] for r in records)
    inj_c = sum(r["injection_correct"] for r in records)
    van_only = sum(r["vanilla_correct"] and not r["injection_correct"] for r in records)
    inj_only = sum(not r["vanilla_correct"] and r["injection_correct"] for r in records)

    return {
        "model": model_name,
        "n": n, "n_math": len(math_recs), "n_aime": len(aime_recs),
        "overall": {
            "vanilla": acc(records, "vanilla_correct"),
            "injection": acc(records, "injection_correct"),
            "delta": round(acc(records, "injection_correct") - acc(records, "vanilla_correct"), 4),
            "vanilla_correct": van_c,
            "injection_correct": inj_c,
            "regressions": van_only,
            "improvements": inj_only,
        },
        "math500": {
            "vanilla": acc(math_recs, "vanilla_correct"),
            "injection": acc(math_recs, "injection_correct"),
            "delta": round(acc(math_recs, "injection_correct") - acc(math_recs, "vanilla_correct"), 4) if math_recs else 0,
        },
        "aime": {
            "vanilla": acc(aime_recs, "vanilla_correct"),
            "injection": acc(aime_recs, "injection_correct"),
            "delta": round(acc(aime_recs, "injection_correct") - acc(aime_recs, "vanilla_correct"), 4) if aime_recs else 0,
        },
    }

def print_summary(s, log):
    log.info(f"\n{'='*60}")
    log.info(f"Phase 4 — {s['model']} Results")
    log.info(f"{'='*60}")
    log.info(f"  Total: n={s['n']} (MATH={s['n_math']}, AIME={s['n_aime']})")
    o = s['overall']
    log.info(f"\n  OVERALL:")
    log.info(f"    Vanilla:   {o['vanilla']:.3f} ({o['vanilla_correct']}/{s['n']})")
    log.info(f"    Injection: {o['injection']:.3f} ({o['injection_correct']}/{s['n']})")
    log.info(f"    Delta:     {o['delta']:+.3f}")
    log.info(f"    Regressions: {o['regressions']}  Improvements: {o['improvements']}")
    log.info(f"\n  MATH-500 (n={s['n_math']}):")
    log.info(f"    Vanilla: {s['math500']['vanilla']:.3f}  Injection: {s['math500']['injection']:.3f}  Delta: {s['math500']['delta']:+.3f}")
    log.info(f"\n  AIME (n={s['n_aime']}):")
    log.info(f"    Vanilla: {s['aime']['vanilla']:.3f}  Injection: {s['aime']['injection']:.3f}  Delta: {s['aime']['delta']:+.3f}")

# ── Main ─────────────────────────────────────────────────────────────────────

def run_model(model_key, max_problems=None):
    cfg = MODELS[model_key]
    model_hf_id = cfg["hf_id"]
    model_name = cfg["short_name"]
    max_context = cfg["max_context"]
    model_max_new_tokens = cfg.get("max_new_tokens", MAX_NEW_TOKENS)

    log = setup_logging(model_key)
    log.info("=" * 60)
    log.info(f"Phase 4 — Multi-Model Eval: {model_name}")
    log.info(f"  HF ID: {model_hf_id}")
    log.info(f"  Max context: {max_context}")
    log.info(f"  Max new tokens: {model_max_new_tokens}")
    log.info("=" * 60)

    jsonl_out = RESULTS_DIR / f"results_{model_key}.jsonl"
    json_out = RESULTS_DIR / f"results_{model_key}.json"
    summary_out = RESULTS_DIR / f"summary_{model_key}.json"

    problems = build_eval_set(log)
    if max_problems:
        problems = problems[:max_problems]
        log.info(f"Capped at: {len(problems)}")

    done = load_done(jsonl_out)
    remaining = [(p["problem_id"], p) for p in problems if p["problem_id"] not in done]
    if done:
        log.info(f"Resuming -- {len(done)} done")
    log.info(f"To do: {len(remaining)}")

    if not remaining:
        log.info("Nothing to do!")
    else:
        bank, bvecs = load_combined_bank(log)
        nodes, adj, semb, gmeta = load_graph(log)

        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cpu")

        model, tokenizer = load_student(model_hf_id, log)
        log.info("Embed on CPU, student on GPU")

        pat2nodes = {}
        if nodes:
            for nd in nodes:
                pid = nd["pattern_id"]
                if pid not in pat2nodes:
                    pat2nodes[pid] = []
                pat2nodes[pid].append(nd)
            for pid in pat2nodes:
                pat2nodes[pid].sort(key=lambda x: x["step_index"])

        for idx, (pid, item) in enumerate(remaining):
            source  = item["source"]
            problem = item["problem"]
            gold    = item["gold_answer"]
            subject = item.get("subject", "")

            qvec = embed_query(embed_model, problem)
            anchor, asim = retrieve_anchor(qvec, bank, bvecs, problem_source=source)

            anchor_src = anchor.get("_source", "")
            is_cross_dataset = anchor_src != source
            thr = get_threshold(source, subject, is_cross_dataset=is_cross_dataset)
            low = asim < thr

            if anchor_src == "MATH-500":
                apid = f"math_{anchor['index']}"
            else:
                apid = f"aime_{anchor['index']}"

            bonus, is_cross, anodes_graph = [], False, []
            if not low and nodes:
                anodes_graph = pat2nodes.get(apid, [])
                bonus = find_bonus(apid, qvec, nodes, adj, semb)
                anchor_domain = "math_500" if anchor_src == "MATH-500" else "aime"
                is_cross = any(n["domain"] != anchor_domain for n in bonus)

            if low:
                strategy, example = "", ""
                log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                         f"sim={asim:.3f} < {thr} -> skip")
            else:
                example = anchor.get("worked_example", "")
                strategy = assemble(anodes_graph, bonus) if anodes_graph else anchor.get("pattern", "")

            # Vanilla
            vr, vl, vt = infer_hf(model, tokenizer, VANILLA_SYSTEM,
                                   VANILLA_USER.format(problem=problem),
                                   max_context=max_context,
                                   max_new_tokens=model_max_new_tokens)
            vp = extract_answer_math(vr) if source == "MATH-500" else extract_answer_aime(vr)
            vok = is_correct(vp, gold, source)

            # Injection
            if low or not strategy:
                isys, iuser = VANILLA_SYSTEM, VANILLA_USER.format(problem=problem)
            elif example:
                isys = ATGI_SYSTEM
                iuser = ATGI_USER_RICH.format(strategy=strategy, example=example, problem=problem)
            else:
                isys = ATGI_SYSTEM
                iuser = ATGI_USER_NO_EXAMPLE.format(strategy=strategy, problem=problem)

            ir, il_t, it = infer_hf(model, tokenizer, isys, iuser,
                                     max_context=max_context,
                                     max_new_tokens=model_max_new_tokens)
            ip = extract_answer_math(ir) if source == "MATH-500" else extract_answer_aime(ir)
            iok = is_correct(ip, gold, source)

            rec = {
                "model": model_name,
                "problem_id": pid, "source": source, "subject": subject,
                "level": item.get("level", ""), "year": item.get("year"),
                "gold_answer": gold,
                "anchor_source": anchor_src,
                "anchor_similarity": round(asim, 4), "sim_threshold": thr,
                "low_similarity": low,
                "anchor_steps": len(anodes_graph), "bonus_steps": len(bonus),
                "is_cross_domain": is_cross,
                "vanilla_raw": vr[-300:], "vanilla_answer": vp,
                "vanilla_correct": vok, "vanilla_latency_s": round(vl, 3),
                "vanilla_tokens": vt,
                "injection_raw": ir[-300:], "injection_answer": ip,
                "injection_correct": iok, "injection_latency_s": round(il_t, 3),
                "injection_tokens": it,
            }
            append_jsonl(jsonl_out, rec)

            out = ("^" if iok and not vok else "v" if vok and not iok else "=")
            cross_tag = "X" if is_cross else ""
            from_tag = f"<-{anchor_src}" if anchor_src != source else ""
            log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                     f"sim={asim:.3f} {from_tag}| "
                     f"van={'Y' if vok else 'N'}({vl:.0f}s/{vt}t) "
                     f"inj={'Y' if iok else 'N'}({il_t:.0f}s/{it}t) {cross_tag}| {out}")

        del model, tokenizer
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compile results
    recs = []
    if jsonl_out.exists():
        with open(jsonl_out) as f:
            for l in f:
                l = l.strip()
                if l:
                    try:
                        recs.append(json.loads(l))
                    except:
                        pass
    recs.sort(key=lambda r: r["problem_id"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(recs, f, indent=2)
    s = summarize(recs, model_name)
    with open(summary_out, "w") as f:
        json.dump(s, f, indent=2)
    print_summary(s, log)
    log.info(f"\nWrote {len(recs)} results -> {json_out}")
    return s


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Multi-model evaluation")
    parser.add_argument("--model", required=True,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Model to evaluate")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems (for testing)")
    args = parser.parse_args()

    if args.model == "all":
        summaries = []
        for key in MODELS:
            s = run_model(key, args.max_problems)
            summaries.append(s)
        # Print combined table
        print("\n" + "=" * 70)
        print("PHASE 4 — MULTI-MODEL COMPARISON")
        print("=" * 70)
        print(f"{'Model':20s} {'Van':>8s} {'Inj':>8s} {'Delta':>8s} {'Regr':>6s} {'Impr':>6s}")
        print("-" * 60)
        for s in summaries:
            o = s['overall']
            print(f"{s['model']:20s} {o['vanilla']:>8.3f} {o['injection']:>8.3f} "
                  f"{o['delta']:>+8.3f} {o['regressions']:>6d} {o['improvements']:>6d}")
    else:
        run_model(args.model, args.max_problems)


if __name__ == "__main__":
    main()
