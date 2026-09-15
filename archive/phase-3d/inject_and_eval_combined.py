"""
Phase 3D — Combined MATH-500 + AIME Evaluation

Evaluates on BOTH test sets with the combined thought graph:
  - MATH-500 test: last 100 problems (same as phase-2d/3)
  - AIME test: last 187 problems (20% of 933)
  - Total: 287 eval problems

Uses combined graph (MATH-500 + AIME patterns) for cross-dataset transfer.
For each test problem, anchor retrieval searches BOTH banks.

Fixed vanilla prompt: no [your answer] template — clean extraction.

Usage:
    python inject_and_eval_combined.py --max-problems 5
    python inject_and_eval_combined.py
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

ROOT        = Path(__file__).parent
PHASE2D     = ROOT.parent / "phase-2d"
GRAPH_DIR   = ROOT / "graph"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

GRAPH_JSON     = GRAPH_DIR / "thought_graph.json"
STEP_EMBED_NPY = GRAPH_DIR / "step_embeddings.npy"

# Both banks
MATH_BANK_JSON  = PHASE2D / "memory" / "bank.json"
MATH_BANK_INDEX = PHASE2D / "memory" / "bank.index"
AIME_BANK_JSON  = ROOT / "memory" / "bank.json"
AIME_BANK_INDEX = ROOT / "memory" / "bank.index"

JSONL_OUT   = RESULTS_DIR / "results_combined.jsonl"
JSON_OUT    = RESULTS_DIR / "results_combined.json"
SUMMARY_OUT = RESULTS_DIR / "summary_combined.json"

STUDENT_MODEL_HF = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS   = 16384   # increased for hard AIME problems
EMBED_MODEL      = "nomic-ai/nomic-embed-text-v1"

MAX_BONUS_STEPS   = 1
NOVELTY_THRESHOLD = 0.85

# Per-source thresholds
SIM_THRESHOLDS = {
    "MATH-500": {
        "Intermediate Algebra": 0.60, "Number Theory": 0.65,
        "Precalculus": 0.65, "Geometry": 0.55, "default": 0.50,
    },
    "AIME": {"default": 0.50},
}

# ── Prompts ───────────────────────────────────────────────────────────────────

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

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"eval_combined_{ts}.log"
    logger = logging.getLogger("combined")
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

# ── Answer parsing ────────────────────────────────────────────────────────────

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
    """For MATH-500: extract LaTeX or number answers."""
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return _clean_math_tail(m.group(1).strip().rstrip("."))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:：]\s*(.+)", text)
    if m:
        ans = re.sub(r'\*+', '', m.group(1)).strip()
        return _clean_math_tail(ans)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _clean_math_tail(ans):
    """Strip trailing unit words / sentences after a math expression.
    Only applied to 'the answer is ...' / 'Answer: ...' extractions,
    NOT to raw last-line fallback (which may be incomplete text)."""
    # Remove trailing sentence fragments after $ ... $ expr
    # e.g. "$ 137 \frac{1}{2} $ miles." → "137 \frac{1}{2}"
    m = re.match(r'^\$\s*(.+?)\s*\$[\s.,;:]*\w*[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    # Strip trailing unit words: "200 feet less than Martha" → "200"
    m = re.match(r'^.*?(\-?[\d,]+\.?\d*)\s+(?:feet|miles|cm|meters?|degrees?|hours?|minutes?|seconds?|square|cubic|units?)\b.*$', ans, re.IGNORECASE)
    if m:
        return m.group(1)
    # "Jim walks 200 feet less than Martha." → extract number from sentence
    m = re.match(r'^[A-Z][a-z].*?(\-?[\d,]+\.?\d*).*$', ans)
    if m:
        return m.group(1)
    return ans

def extract_answer_aime(text):
    """For AIME: extract integer 0-999."""
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        cleaned = re.sub(r'\*+', '', boxed).replace("$", "").strip()
        return extract_number(cleaned)
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return extract_number(m.group(1))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:：]\s*(.+)", text)
    if m:
        return extract_number(re.sub(r'\*+', '', m.group(1)).replace("$", ""))
    bold = re.findall(r'\*\*\$?([\d,]+\.?\d*)\$?\*\*', text)
    if bold:
        return bold[-1].replace(",", "")
    return extract_number(text)

def normalize_math(ans):
    ans = ans.strip()
    ans = ans.rstrip(".")
    ans = re.sub(r"^\$+(.+?)\$+$", r"\1", ans)
    ans = ans.lstrip("$").rstrip("$")
    ans = re.sub(r"^\[(.+)\]$", r"\1", ans)
    ans = re.sub(r"\*+(.+?)\*+", r"\1", ans)
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mbox\{([^}]*)\}", r"\1", ans)     # \mbox{cm} → cm
    ans = re.sub(r"\^\\circ", "°", ans)                   # ^\circ → °
    ans = re.sub(r"\\(?:circ|degree)", "°", ans)         # \circ → °
    ans = re.sub(r"°", "", ans)                          # drop ° entirely for comparison
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\,", "").replace("\\!", "")
    ans = ans.replace("\\$", "").replace("$", "")
    # Strip trailing unit text: "15cm²" → "15", "137.5miles" → "137.5"
    ans = re.sub(r'([\d})])\s*(?:cm|meters?|miles|feet|hours?|minutes?|seconds?|units?|square|cubic|cm²|m²)\b.*$', r'\1', ans, flags=re.IGNORECASE)
    ans = ans.replace("\\begin{pmatrix}", "(").replace("\\end{pmatrix}", ")")
    ans = ans.replace("\\\\", ",")                       # matrix row sep → comma
    # Leading zero: 0.35625 == .35625
    if ans.startswith("0."):
        ans_alt = ans[1:]  # .35625
    elif ans.startswith("."):
        ans_alt = "0" + ans  # 0.35625
    else:
        ans_alt = None
    # Store alt for comparison
    ans = ans.lower()
    if ans_alt:
        ans = ans_alt.lower() + "|" + ans  # store both forms separated by |
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
        # Handle | separated alternates (leading zero)
        np_forms = set(np_.split("|"))
        ng_forms = set(ng_.split("|"))
        if np_forms & ng_forms:
            return True
        # Handle set/tuple order: {-2,1} == {1,-2}
        np_clean = np_.split("|")[0] if "|" in np_ else np_
        ng_clean = ng_.split("|")[0] if "|" in ng_ else ng_
        # Try sorting comma-separated elements
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

# ── Load data ─────────────────────────────────────────────────────────────────

def load_graph(log):
    if not GRAPH_JSON.exists():
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
    """Load both MATH-500 and AIME banks, concatenate vectors."""
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

    # Tag each entry with source
    for e in math_bank:
        e["_source"] = "MATH-500"
        e["_bank_idx_offset"] = 0
    for e in aime_bank:
        e["_source"] = "AIME"
        e["_bank_idx_offset"] = n1

    combined_bank = math_bank + aime_bank
    combined_vecs = np.vstack([math_vecs, aime_vecs])

    log.info(f"Combined bank: {len(math_bank)} MATH + {len(aime_bank)} AIME = {len(combined_bank)} total")
    return combined_bank, combined_vecs

def load_embed(log, device="cuda"):
    from sentence_transformers import SentenceTransformer
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)

def embed_query(model, text):
    v = model.encode([f"search_query: {text}"], normalize_embeddings=True, show_progress_bar=False)
    return v.astype(np.float32).flatten()

# ── Retrieval ─────────────────────────────────────────────────────────────────

SAME_SOURCE_BIAS     = 0.05   # prefer same-dataset anchors
CROSS_THRESHOLD_BUMP = 0.10   # higher threshold for cross-dataset anchors


def retrieve_anchor(qvec, bank, vecs, problem_source=""):
    """Retrieve best anchor with same-source bias."""
    scores = (vecs @ qvec).flatten()

    # Apply same-source bias
    if problem_source:
        biased_scores = np.copy(scores)
        for i, entry in enumerate(bank):
            if entry.get("_source", "") == problem_source:
                biased_scores[i] += SAME_SOURCE_BIAS
        best = int(np.argmax(biased_scores))
    else:
        best = int(np.argmax(scores))

    # Return raw (unbiased) similarity for threshold check
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
    cands = []
    seen = set()
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
    lines = []
    sn = 1
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

# ── Backend ───────────────────────────────────────────────────────────────────

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

def infer_hf(model, tokenizer, system, user):
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    il = inp["input_ids"].shape[1]
    mn = min(MAX_NEW_TOKENS, 32768 - il)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=mn, do_sample=False,
                             temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    lat = time.perf_counter() - t0
    nt = out[0][il:]
    raw = tokenizer.decode(nt, skip_special_tokens=True)
    return raw, lat, len(nt)

# ── JSONL ─────────────────────────────────────────────────────────────────────

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

# ── Build eval set ────────────────────────────────────────────────────────────

def build_eval_set(log):
    """Build combined eval set: MATH-500 test (100) + AIME test (187)."""
    problems = []

    # MATH-500: last 100
    math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math_all = list(math_ds)
    for i, item in enumerate(math_all[400:]):
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

# ── Summary ───────────────────────────────────────────────────────────────────

def summarize(records):
    n = len(records)
    if not n:
        return {}
    def acc(recs, k):
        return round(sum(r[k] for r in recs)/len(recs), 4) if recs else 0

    math_recs = [r for r in records if r["source"] == "MATH-500"]
    aime_recs = [r for r in records if r["source"] == "AIME"]
    injected = [r for r in records if not r.get("low_similarity")]
    bonus = [r for r in injected if r.get("bonus_steps", 0) > 0]
    cross = [r for r in injected if r.get("is_cross_domain")]

    return {
        "n": n, "n_math": len(math_recs), "n_aime": len(aime_recs),
        "n_injected": len(injected), "n_skipped": n - len(injected),
        "n_bonus": len(bonus), "n_cross_domain": len(cross),
        "overall": {
            "vanilla": acc(records, "vanilla_correct"),
            "atgi": acc(records, "injection_correct"),
            "delta": round(acc(records, "injection_correct") - acc(records, "vanilla_correct"), 4),
        },
        "math500": {
            "vanilla": acc(math_recs, "vanilla_correct"),
            "atgi": acc(math_recs, "injection_correct"),
            "delta": round(acc(math_recs, "injection_correct") - acc(math_recs, "vanilla_correct"), 4) if math_recs else 0,
        },
        "aime": {
            "vanilla": acc(aime_recs, "vanilla_correct"),
            "atgi": acc(aime_recs, "injection_correct"),
            "delta": round(acc(aime_recs, "injection_correct") - acc(aime_recs, "vanilla_correct"), 4) if aime_recs else 0,
        },
    }

def print_summary(s, log):
    log.info(f"\n{'='*60}")
    log.info(f"Phase 3D — Combined MATH-500 + AIME Results")
    log.info(f"{'='*60}")
    log.info(f"  Total: n={s['n']} (MATH={s['n_math']}, AIME={s['n_aime']})")
    log.info(f"  Injected: {s['n_injected']}  Skipped: {s['n_skipped']}  "
             f"Bonus: {s['n_bonus']}  Cross-domain: {s['n_cross_domain']}")
    log.info(f"\n  OVERALL:")
    log.info(f"    Vanilla: {s['overall']['vanilla']:.3f}")
    log.info(f"    ATGI:    {s['overall']['atgi']:.3f}")
    log.info(f"    Delta:   {s['overall']['delta']:+.3f}")
    log.info(f"\n  MATH-500 (n={s['n_math']}):")
    log.info(f"    Vanilla: {s['math500']['vanilla']:.3f}")
    log.info(f"    ATGI:    {s['math500']['atgi']:.3f}")
    log.info(f"    Delta:   {s['math500']['delta']:+.3f}")
    log.info(f"\n  AIME (n={s['n_aime']}):")
    log.info(f"    Vanilla: {s['aime']['vanilla']:.3f}")
    log.info(f"    ATGI:    {s['aime']['atgi']:.3f}")
    log.info(f"    Delta:   {s['aime']['delta']:+.3f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3D — Combined MATH-500 + AIME Evaluation")
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
        bank, bvecs = load_combined_bank(log)
        nodes, adj, semb, gmeta = load_graph(log)
        embed_model = load_embed(log, "cuda")

        del embed_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = load_student(log)
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cpu")
        log.info("Embed on CPU, student on GPU")

        # Pattern lookup
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

            # Determine anchor source and cross-dataset status
            anchor_src = anchor.get("_source", "")
            is_cross_dataset = anchor_src != source
            thr = get_threshold(source, subject, is_cross_dataset=is_cross_dataset)
            low = asim < thr

            # Determine anchor pattern_id in graph
            if anchor_src == "MATH-500":
                apid = f"math_{anchor['index']}"
            else:
                apid = f"aime_{anchor['index']}"

            # Graph expansion
            bonus = []
            is_cross = False
            anodes_graph = []
            if not low and nodes:
                anodes_graph = pat2nodes.get(apid, [])
                bonus = find_bonus(apid, qvec, nodes, adj, semb)
                # Cross-domain: anchor from one dataset, bonus from another
                anchor_domain = "math_500" if anchor_src == "MATH-500" else "aime"
                is_cross = any(n["domain"] != anchor_domain for n in bonus)

            if low:
                strategy = ""
                example = ""
                log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                         f"sim={asim:.3f} < {thr} → skip")
            else:
                example = anchor.get("worked_example", "")
                if anodes_graph:
                    strategy = assemble(anodes_graph, bonus)
                else:
                    strategy = anchor.get("pattern", "")

            # Vanilla
            vr, vl, vt = infer_hf(model, tokenizer, VANILLA_SYSTEM,
                                   VANILLA_USER.format(problem=problem))
            if source == "MATH-500":
                vp = extract_answer_math(vr)
            else:
                vp = extract_answer_aime(vr)
            vok = is_correct(vp, gold, source)

            # ATGI
            if low or not strategy:
                isys = VANILLA_SYSTEM
                iuser = VANILLA_USER.format(problem=problem)
            elif example:
                isys = ATGI_SYSTEM
                iuser = ATGI_USER_RICH.format(strategy=strategy, example=example, problem=problem)
            else:
                isys = ATGI_SYSTEM
                iuser = ATGI_USER_NO_EXAMPLE.format(strategy=strategy, problem=problem)

            ir, il_t, it = infer_hf(model, tokenizer, isys, iuser)
            if source == "MATH-500":
                ip = extract_answer_math(ir)
            else:
                ip = extract_answer_aime(ir)
            iok = is_correct(ip, gold, source)

            rec = {
                "problem_id": pid, "source": source, "subject": subject,
                "level": item.get("level", ""), "year": item.get("year"),
                "gold_answer": gold,
                "anchor_source": anchor_src,
                "anchor_similarity": round(asim, 4), "sim_threshold": thr,
                "low_similarity": low,
                "anchor_steps": len(anodes_graph), "bonus_steps": len(bonus),
                "is_cross_domain": is_cross,
                "bonus_domains": [n["domain"] for n in bonus],
                "vanilla_raw": vr[-300:], "vanilla_answer": vp,
                "vanilla_correct": vok, "vanilla_latency_s": round(vl, 3),
                "vanilla_tokens": vt,
                "injection_raw": ir[-300:], "injection_answer": ip,
                "injection_correct": iok, "injection_latency_s": round(il_t, 3),
                "injection_tokens": it,
            }
            append_jsonl(JSONL_OUT, rec)

            out = ("↑" if iok and not vok else "↓" if vok and not iok else "=")
            cross_tag = "🌐" if is_cross else ""
            from_tag = f"←{anchor_src}" if anchor_src != source else ""
            log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                     f"sim={asim:.3f} {from_tag}| "
                     f"van={'✓' if vok else '✗'}({vl:.0f}s) "
                     f"atgi={'✓' if iok else '✗'}({il_t:.0f}s) {cross_tag}| {out}")

        del model, tokenizer
        import gc; gc.collect()

    # Compile
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
    log.info(f"\nWrote {len(recs)} → {JSON_OUT}")
    s = summarize(recs)
    with open(SUMMARY_OUT, "w") as f:
        json.dump(s, f, indent=2)
    print_summary(s, log)


if __name__ == "__main__":
    main()
