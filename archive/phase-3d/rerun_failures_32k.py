"""
Phase 3D — Rerun 131 FAILED cases with 32K token budget.
Only runs problems that failed in the original 16K run.
Saves to results_rerun_32k.jsonl (separate from original results).
Has resume support.
"""

import json
import logging
import re
import time
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

MATH_BANK_JSON  = PHASE2D / "memory" / "bank.json"
MATH_BANK_INDEX = PHASE2D / "memory" / "bank.index"
AIME_BANK_JSON  = ROOT / "memory" / "bank.json"
AIME_BANK_INDEX = ROOT / "memory" / "bank.index"

JSONL_OUT    = RESULTS_DIR / "results_rerun_32k.jsonl"

STUDENT_MODEL_HF = "Qwen/Qwen3-1.7B"
MAX_NEW_TOKENS   = 32768
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

# ── 131 failed problem IDs ──────────────────────────────────────────────────

FAILED_IDS = {
    "2018-II-9", "2019-I-5", "2019-II-1", "2019-II-15", "2020-I-14",
    "2022-I-5", "2022-II-11", "2022-II-13", "2022-II-7", "2023-I-10",
    "2023-I-4", "2023-I-6", "2024-II-11", "2024-II-2", "2024-II-6",
    "math_497", "2018-I-10", "2018-I-14", "2018-I-15", "2018-I-7",
    "2018-I-9", "2018-II-1", "2018-II-10", "2018-II-11", "2018-II-12",
    "2018-II-13", "2018-II-14", "2018-II-15", "2018-II-7", "2019-I-1",
    "2019-I-10", "2019-I-11", "2019-I-12", "2019-I-13", "2019-I-15",
    "2019-I-4", "2019-I-7", "2019-II-10", "2019-II-12", "2019-II-13",
    "2019-II-14", "2019-II-4", "2019-II-5", "2019-II-7", "2020-I-1",
    "2020-I-10", "2020-I-11", "2020-I-12", "2020-I-15", "2020-I-5",
    "2020-I-6", "2020-II-10", "2020-II-11", "2020-II-12", "2020-II-13",
    "2020-II-14", "2020-II-15", "2020-II-7", "2020-II-8", "2020-II-9",
    "2021-I-1", "2021-I-10", "2021-I-11", "2021-I-12", "2021-I-14",
    "2021-I-15", "2021-I-6", "2021-I-7", "2021-II-10", "2021-II-11",
    "2021-II-13", "2021-II-14", "2021-II-15", "2021-II-2", "2021-II-3",
    "2021-II-5", "2021-II-8", "2021-II-9", "2022-I-11", "2022-I-12",
    "2022-I-13", "2022-I-14", "2022-I-15", "2022-I-4", "2022-I-6",
    "2022-I-7", "2022-I-8", "2022-I-9", "2022-II-12", "2022-II-14",
    "2022-II-15", "2022-II-2", "2022-II-5", "2022-II-6", "2022-II-8",
    "2022-II-9", "2023-I-11", "2023-I-12", "2023-I-13", "2023-I-14",
    "2023-I-7", "2023-I-8", "2023-I-9", "2023-II-10", "2023-II-11",
    "2023-II-12", "2023-II-13", "2023-II-14", "2023-II-15", "2023-II-6",
    "2023-II-7", "2023-II-8", "2023-II-9", "2024-II-1", "2024-II-12",
    "2024-II-14", "2024-II-15", "2024-II-3", "2024-II-5", "2024-II-7",
    "2024-II-8", "math_400", "math_408", "math_420", "math_422",
    "math_444", "math_460", "math_462", "math_481", "math_490", "math_494",
}

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

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"rerun_32k_{ts}.log"
    logger = logging.getLogger("rerun32k")
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
    ans = ans.replace("\\tfrac", "\\frac")
    # Convert \frac{a}{b} → a/b for consistent comparison
    ans = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", ans)
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


# ── Load data ────────────────────────────────────────────────────────────────

def load_graph(log):
    if not GRAPH_JSON.exists():
        return None, None, None, None
    with open(GRAPH_JSON) as f:
        g = json.load(f)
    nodes = g["nodes"]
    adj = {int(k): v for k, v in g["adjacency"].items()}
    emb = np.load(STEP_EMBED_NPY)
    log.info(f"Graph: {g['metadata']['num_nodes']} nodes")
    return nodes, adj, emb, g["metadata"]

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
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
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
    return problems


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3D — Rerun FAILURES with 32K token budget")
    log.info(f"  MAX_NEW_TOKENS = {MAX_NEW_TOKENS}")
    log.info(f"  Failed IDs: {len(FAILED_IDS)}")
    log.info("=" * 60)

    problems = build_eval_set(log)
    problems = [p for p in problems if p["problem_id"] in FAILED_IDS]
    log.info(f"Matched: {len(problems)}")

    done = load_done(JSONL_OUT)
    remaining = [(p["problem_id"], p) for p in problems if p["problem_id"] not in done]
    if done:
        log.info(f"Resuming -- {len(done)} done")
    log.info(f"To do: {len(remaining)}")

    if not remaining:
        log.info("Nothing to do!")
        return

    bank, bvecs = load_combined_bank(log)
    nodes, adj, semb, gmeta = load_graph(log)

    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cpu")

    model, tokenizer = load_student(log)
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

        # Vanilla with 32K budget
        vr, vl, vt = infer_hf(model, tokenizer, VANILLA_SYSTEM,
                               VANILLA_USER.format(problem=problem))
        vp = extract_answer_math(vr) if source == "MATH-500" else extract_answer_aime(vr)
        vok = is_correct(vp, gold, source)

        # Injection with 32K budget
        if low or not strategy:
            isys, iuser = VANILLA_SYSTEM, VANILLA_USER.format(problem=problem)
        elif example:
            isys = ATGI_SYSTEM
            iuser = ATGI_USER_RICH.format(strategy=strategy, example=example, problem=problem)
        else:
            isys = ATGI_SYSTEM
            iuser = ATGI_USER_NO_EXAMPLE.format(strategy=strategy, problem=problem)

        ir, il_t, it = infer_hf(model, tokenizer, isys, iuser)
        ip = extract_answer_math(ir) if source == "MATH-500" else extract_answer_aime(ir)
        iok = is_correct(ip, gold, source)

        rec = {
            "problem_id": pid, "source": source, "subject": subject,
            "level": item.get("level", ""), "year": item.get("year"),
            "gold_answer": gold,
            "anchor_source": anchor_src,
            "anchor_similarity": round(asim, 4),
            "sim_threshold": thr, "low_similarity": low,
            "anchor_steps": len(anodes_graph), "bonus_steps": len(bonus),
            "is_cross_domain": is_cross,
            "vanilla_raw": vr[-300:], "vanilla_answer": vp,
            "vanilla_correct": vok, "vanilla_latency_s": round(vl, 3),
            "vanilla_tokens": vt,
            "injection_raw": ir[-300:], "injection_answer": ip,
            "injection_correct": iok, "injection_latency_s": round(il_t, 3),
            "injection_tokens": it,
        }
        append_jsonl(JSONL_OUT, rec)

        out = ("^" if iok and not vok else "v" if vok and not iok else "=")
        log.info(f"  [{idx+1}/{len(remaining)}] {pid} ({source}) | "
                 f"sim={asim:.3f} | "
                 f"van={'Y' if vok else 'N'}({vl:.0f}s/{vt}t) "
                 f"inj={'Y' if iok else 'N'}({il_t:.0f}s/{it}t) | {out}")

    # Final summary
    recs = []
    with open(JSONL_OUT) as f:
        for l in f:
            l = l.strip()
            if l:
                try:
                    recs.append(json.loads(l))
                except:
                    pass
    van_c = sum(r["vanilla_correct"] for r in recs)
    inj_c = sum(r["injection_correct"] for r in recs)
    n = len(recs)
    log.info(f"\n{'='*60}")
    log.info(f"32K Rerun Summary: {n} problems (originally all failed at 16K)")
    log.info(f"  Vanilla recovered:   {van_c}/{n} ({100*van_c/n:.1f}%)")
    log.info(f"  Injection recovered: {inj_c}/{n} ({100*inj_c/n:.1f}%)")
    log.info(f"  Net new correct: van=+{van_c} inj=+{inj_c}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
