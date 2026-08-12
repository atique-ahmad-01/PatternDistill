"""
Phase 3B — Thinking Intervention (TI)

Same as Phase 3 v2 (anchor-first + novelty gate) BUT the retrieved template
is injected INSIDE <think> tags as a prefix to the model's internal reasoning,
instead of in the user prompt.

Inspired by RoT+TI (ICLR 2026): placing templates inside <think> tags
anchors the model's reasoning from the start, reducing path switching by
up to 81.8% on Qwen3-1.7B.

Key difference from phase-3 v2:
  - v2: template in user prompt → model may ignore it in <think> block
  - TI: template prepended to <think> → model reasons WITH the template

How it works:
  1. Build prompt normally (system + user with problem only)
  2. Apply chat template to get full text
  3. Append "<think>\nHere's a strategy for this type of problem:\n{strategy}\n
     Reference example: {example}\n\nNow let me apply this...\n"
  4. Model continues generating from this anchored thinking state
  5. Model produces rest of <think> + </think> + final answer

Output:
  results/results.jsonl, results/results.json, results/summary.json

Usage:
    python inject_and_eval_ti.py --max-problems 5    # smoke test
    python inject_and_eval_ti.py                      # full run
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
PHASE2D     = ROOT.parent / "phase-2d"
PHASE3      = ROOT.parent / "phase-3"
GRAPH_DIR   = PHASE3 / "graph"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

GRAPH_JSON     = GRAPH_DIR / "thought_graph.json"
STEP_EMBED_NPY = GRAPH_DIR / "step_embeddings.npy"

BANK_JSON   = PHASE2D / "memory" / "bank.json"
BANK_INDEX  = PHASE2D / "memory" / "bank.index"

JSONL_OUT   = RESULTS_DIR / "results.jsonl"
JSON_OUT    = RESULTS_DIR / "results.json"
SUMMARY_OUT = RESULTS_DIR / "summary.json"

# ── Config ────────────────────────────────────────────────────────────────────

STUDENT_MODEL_HF     = "Qwen/Qwen3-1.7B"
STUDENT_MODEL_OLLAMA = "qwen3:1.7b"
DATASET              = "HuggingFaceH4/MATH-500"
EVAL_SLICE           = slice(400, 500)
MAX_NEW_TOKENS       = 8192
OLLAMA_URL           = "http://localhost:11434"
EMBED_MODEL          = "nomic-ai/nomic-embed-text-v1"

# ── Hyperparameters ───────────────────────────────────────────────────────────

MAX_BONUS_STEPS   = 1
NOVELTY_THRESHOLD = 0.85

SIM_THRESHOLDS = {
    "Intermediate Algebra": 0.60,
    "Number Theory":        0.65,
    "Precalculus":          0.65,
    "Geometry":             0.55,
}
SIM_THRESHOLD_DEFAULT = 0.50

# ── Prompts ───────────────────────────────────────────────────────────────────

VANILLA_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "Solve the problem step by step, then state the final answer clearly."
)
VANILLA_USER = (
    "Problem: {problem}\n\n"
    "End your response with: The answer is [your answer]."
)

# TI: simple system + user (no strategy in prompt), template goes in <think>
TI_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "Solve the problem step by step, then state the final answer clearly."
)
TI_USER = (
    "Problem: {problem}\n\n"
    "End your response with: The answer is [your answer]."
)

# Template for the <think> prefix
THINK_PREFIX_RICH = (
    "<think>\n"
    "Let me analyze this problem. I recall a strategy for this type of problem:\n\n"
    "Reasoning Strategy:\n{strategy}\n\n"
    "Reference Example:\n{example}\n\n"
    "Now let me apply this strategy to the given problem.\n\n"
)

THINK_PREFIX_NO_EXAMPLE = (
    "<think>\n"
    "Let me analyze this problem. I recall a strategy for this type of problem:\n\n"
    "Reasoning Strategy:\n{strategy}\n\n"
    "Now let me apply this strategy to the given problem.\n\n"
)

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"inject_and_eval_ti_{timestamp}.log"

    logger = logging.getLogger("ti_eval")
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
                if text[j] == "{":   depth += 1
                elif text[j] == "}": depth -= 1
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
    ans = re.sub(r"^\[(.+)\]$",    r"\1", ans)
    ans = re.sub(r"\*+(.+?)\*+",   r"\1", ans)
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\,", "").replace("\\!", "")
    return ans.lower()

def is_correct(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)

# ── Load data ─────────────────────────────────────────────────────────────────

def load_thought_graph(log):
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    adj   = {int(k): v for k, v in graph["adjacency"].items()}
    meta  = graph["metadata"]
    embeddings = np.load(STEP_EMBED_NPY)
    log.info(f"Thought graph: {meta['num_nodes']} nodes, "
             f"{meta['num_sequential']} seq + {meta['num_semantic']} sem edges")
    return nodes, adj, embeddings, meta


def load_phase2d_bank(log):
    import faiss
    with open(BANK_JSON) as f:
        bank_meta = json.load(f)
    index = faiss.read_index(str(BANK_INDEX))
    n, dim = index.ntotal, index.d
    all_vectors = np.zeros((n, dim), dtype=np.float32)
    index.reconstruct_n(0, n, all_vectors)
    log.info(f"Phase-2d bank: {len(bank_meta)} entries, dim={dim}")
    return bank_meta, all_vectors


def load_embed_model(log, device="cuda"):
    from sentence_transformers import SentenceTransformer
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model


def embed_query(embed_model, problem: str) -> np.ndarray:
    vec = embed_model.encode(
        [f"search_query: {problem}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.astype(np.float32).flatten()

# ── Retrieval (same as phase-3 v2) ───────────────────────────────────────────

def get_threshold(subject: str) -> float:
    return SIM_THRESHOLDS.get(subject, SIM_THRESHOLD_DEFAULT)


def retrieve_anchor(query_vec: np.ndarray, bank_meta: list,
                    bank_vectors: np.ndarray) -> tuple[dict, float]:
    scores = (bank_vectors @ query_vec).flatten()
    best   = int(np.argmax(scores))
    return bank_meta[best], float(scores[best])


def find_novel_bonus_steps(anchor_nodes: list, query_vec: np.ndarray,
                           nodes: list, adj: dict,
                           step_embeddings: np.ndarray) -> list[dict]:
    if not anchor_nodes:
        return []

    anchor_node_ids = {n["node_id"] for n in anchor_nodes}
    anchor_pattern_id = anchor_nodes[0]["pattern_id"]
    anchor_embeds = np.array([step_embeddings[n["node_id"]] for n in anchor_nodes])

    candidates = []
    seen = set()

    for an in anchor_nodes:
        for edge in adj.get(an["node_id"], []):
            if edge["type"] != "semantic":
                continue
            dst = edge["dst"]
            if dst in anchor_node_ids or dst in seen:
                continue
            if nodes[dst]["pattern_id"] == anchor_pattern_id:
                continue
            seen.add(dst)

            dst_embed = step_embeddings[dst]
            sims_to_anchor = (anchor_embeds @ dst_embed).flatten()
            max_sim_to_anchor = float((np.max(sims_to_anchor) + 1.0) / 2.0)

            if max_sim_to_anchor >= NOVELTY_THRESHOLD:
                continue

            query_sim = float(dst_embed @ query_vec)
            candidates.append({
                "node": nodes[dst],
                "query_sim": query_sim,
            })

    candidates.sort(key=lambda c: c["query_sim"], reverse=True)
    return [c["node"] for c in candidates[:MAX_BONUS_STEPS]]


def assemble_strategy(anchor_nodes: list, bonus_nodes: list) -> str:
    lines = []
    step_num = 1
    for node in anchor_nodes:
        if node["label"].lower().startswith("note"):
            lines.append(f"Note: {node['text']}")
        else:
            lines.append(f"Step {step_num}: {node['text']}")
            step_num += 1
    for node in bonus_nodes:
        lines.append(f"Step {step_num}: {node['text']}")
        step_num += 1
    return "\n".join(lines)


def get_traversal_info(anchor_nodes: list, bonus_nodes: list) -> dict:
    all_nodes = anchor_nodes + bonus_nodes
    pattern_ids = list(set(n["pattern_id"] for n in all_nodes))
    return {
        "template_length":    len(all_nodes),
        "anchor_steps":       len(anchor_nodes),
        "bonus_steps":        len(bonus_nodes),
        "pattern_ids_used":   pattern_ids,
        "n_unique_patterns":  len(pattern_ids),
        "is_cross_pattern":   len(pattern_ids) > 1,
        "node_ids":           [n["node_id"] for n in all_nodes],
        "step_labels":        [n["label"] for n in all_nodes],
        "bonus_texts":        [n["text"][:80] for n in bonus_nodes],
    }

# ── Backends ──────────────────────────────────────────────────────────────────

def infer_ollama(system: str, user: str, think_prefix: str = "") -> tuple[str, float, int]:
    """Ollama doesn't support think prefix injection — fallback to prompt-based."""
    if think_prefix:
        # Fallback: put strategy in user prompt for ollama
        user = think_prefix.replace("<think>\n", "").replace("\n\n", "\n") + "\n" + user
    payload = {
        "model": STUDENT_MODEL_OLLAMA,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
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


def load_student_hf(log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device  : {device}")
    if torch.cuda.is_available():
        log.info(f"GPU     : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Loading student: {STUDENT_MODEL_HF}")
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


def infer_hf(model, tokenizer, system: str, user: str,
             think_prefix: str = "") -> tuple[str, float, int]:
    """
    Standard HF inference. If think_prefix is provided, it is appended
    to the tokenized prompt so the model generates as if it already
    started thinking with that content.
    """
    import torch
    messages  = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text      = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Append think_prefix so model continues from this state
    if think_prefix:
        text += think_prefix

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

    # Prepend the think prefix to raw output for correct answer extraction
    if think_prefix:
        raw = think_prefix + raw

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
        return {k: {"accuracy": round(sum(v)/len(v), 4), "n": len(v)}
                for k, v in sorted(buckets.items())}

    injected = [r for r in records if not r.get("low_similarity")]

    return {
        "n": n,
        "n_injected":          len(injected),
        "n_low_sim_skipped":   len(records) - len(injected),
        "method":              "Thinking Intervention (TI) — template inside <think> tags",
        "retrieval_mode":      "full_problem_anchor + graph_bonus + think_injection",
        "sim_thresholds":      {**SIM_THRESHOLDS, "default": SIM_THRESHOLD_DEFAULT},
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
            "accuracy": round(acc(records, "injection_correct") - acc(records, "vanilla_correct"), 4)
        },
        "baselines": {
            "phase_1":      {"student_1_7b": 0.508, "teacher_8b": 0.718, "gap": 0.210},
            "phase_2d":     {"vanilla": 0.700, "injection": 0.760, "delta": 0.060},
            "phase_3_v2":   {"vanilla": 0.700, "injection": 0.760, "delta": 0.060},
        },
    }


def print_summary(s: dict, log):
    log.info(f"\n{'='*60}")
    log.info(f"Phase 3B — Thinking Intervention (TI) Results")
    log.info(f"{'='*60}")
    log.info(f"  n={s['n']}  injected={s['n_injected']}  skipped={s['n_low_sim_skipped']}")

    log.info(f"\n  Vanilla   accuracy : {s['vanilla']['accuracy']:.3f}")
    log.info(f"  TI        accuracy : {s['injection']['accuracy']:.3f}")
    delta = s['delta']['accuracy']
    log.info(f"  Delta              : {delta:+.3f}")

    gap = s['baselines']['phase_1']['gap']
    pct = delta / gap * 100 if gap else 0
    log.info(f"  Gap closed         : {pct:.1f}% of the 21pp Phase 1 gap")

    p2d = s['baselines']['phase_2d']['delta']
    v2  = s['baselines']['phase_3_v2']['delta']
    log.info(f"\n  vs Phase 2D (prompt injection): {p2d:+.3f}")
    log.info(f"  vs Phase 3 v2 (prompt inject): {v2:+.3f}")
    log.info(f"  TI improvement over 2D:        {delta - p2d:+.3f}")
    log.info(f"  TI improvement over v2:        {delta - v2:+.3f}")

    log.info(f"\n  By subject:")
    for subj, v in s["injection"]["by_subject"].items():
        van = s["vanilla"]["by_subject"].get(subj, {}).get("accuracy", 0)
        thr = SIM_THRESHOLDS.get(subj, SIM_THRESHOLD_DEFAULT)
        log.info(f"    {subj:<35} ti={v['accuracy']:.3f}  van={van:.3f}  "
                 f"delta={v['accuracy']-van:+.3f}  (n={v['n']}  thr={thr})")

    log.info(f"\n  By difficulty:")
    for lvl, v in s["injection"]["by_level"].items():
        van = s["vanilla"]["by_level"].get(lvl, {}).get("accuracy", 0)
        log.info(f"    Level {lvl}: ti={v['accuracy']:.3f}  van={van:.3f}  "
                 f"delta={v['accuracy']-van:+.3f}  (n={v['n']})")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3B — Thinking Intervention")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--problem-ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3B — Thinking Intervention (TI)")
    log.info("=" * 60)
    log.info(f"Backend : {'ollama' if args.ollama else 'transformers'}")
    log.info(f"Method  : template injected inside <think> tags")

    ds       = load_dataset(DATASET, split="test")
    problems = list(ds)[EVAL_SLICE]

    if args.problem_ids:
        target_ids      = set(args.problem_ids)
        problems_indexed = [(400+i, p) for i, p in enumerate(problems) if (400+i) in target_ids]
    else:
        problems_indexed = [(400+i, p) for i, p in enumerate(problems)]

    if args.max_problems:
        problems_indexed = problems_indexed[:args.max_problems]
        log.info(f"Capped at: {len(problems_indexed)} problems")
    else:
        log.info(f"Eval set: {len(problems_indexed)} problems")

    done      = load_done_ids(JSONL_OUT)
    remaining = [(pid, item) for pid, item in problems_indexed if pid not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"To do   : {len(remaining)}")

    if remaining:
        nodes, adj, step_embeddings, graph_meta = load_thought_graph(log)
        bank_meta, bank_vectors = load_phase2d_bank(log)
        embed_model = load_embed_model(log, device="cuda")

        if args.ollama:
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
                log.info(f"Ollama OK at {OLLAMA_URL}")
            except Exception as e:
                raise RuntimeError(f"Cannot reach Ollama: {e}")
            model = tokenizer = None
        else:
            del embed_model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model, tokenizer = load_student_hf(log)

            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cpu")
            log.info("Embed model reloaded on CPU (student on GPU)")

        # Build pattern_id → node mapping
        pattern_to_nodes = {}
        for n in nodes:
            pid = n["pattern_id"]
            if pid not in pattern_to_nodes:
                pattern_to_nodes[pid] = []
            pattern_to_nodes[pid].append(n)
        for pid in pattern_to_nodes:
            pattern_to_nodes[pid].sort(key=lambda x: x["step_index"])

        # ── Evaluate ─────────────────────────────────────────────────────────
        for idx, (problem_id, item) in enumerate(remaining):
            problem = item["problem"]
            subject = item.get("subject", "")
            level   = str(item.get("level", ""))
            gold    = item.get("answer", "")

            # Anchor retrieval
            query_vec = embed_query(embed_model, problem)
            anchor_match, anchor_sim = retrieve_anchor(query_vec, bank_meta, bank_vectors)
            threshold = get_threshold(subject)
            low_sim   = anchor_sim < threshold

            # Graph expansion
            if low_sim:
                anchor_nodes = []
                bonus_nodes  = []
                strategy_text = ""
                think_prefix  = ""
                trav_info = {
                    "template_length": 0, "anchor_steps": 0, "bonus_steps": 0,
                    "pattern_ids_used": [], "n_unique_patterns": 0,
                    "is_cross_pattern": False, "node_ids": [], "step_labels": [],
                    "bonus_texts": [],
                }
                log.info(
                    f"  [{idx+1}/{len(remaining)}] id={problem_id} {subject} L{level} | "
                    f"sim={anchor_sim:.3f} < thr={threshold} → skip injection"
                )
            else:
                anchor_pid = anchor_match["index"]
                anchor_nodes = pattern_to_nodes.get(anchor_pid, [])
                bonus_nodes = find_novel_bonus_steps(
                    anchor_nodes, query_vec, nodes, adj, step_embeddings)
                strategy_text = assemble_strategy(anchor_nodes, bonus_nodes)
                trav_info = get_traversal_info(anchor_nodes, bonus_nodes)

                # Build <think> prefix
                worked_example = anchor_match.get("worked_example", "")
                if worked_example:
                    think_prefix = THINK_PREFIX_RICH.format(
                        strategy=strategy_text, example=worked_example)
                else:
                    think_prefix = THINK_PREFIX_NO_EXAMPLE.format(
                        strategy=strategy_text)

            # ── Vanilla ──────────────────────────────────────────────────────
            if args.ollama:
                van_raw, van_lat, van_tok = infer_ollama(
                    VANILLA_SYSTEM, VANILLA_USER.format(problem=problem))
            else:
                van_raw, van_lat, van_tok = infer_hf(
                    model, tokenizer, VANILLA_SYSTEM,
                    VANILLA_USER.format(problem=problem))
            van_pred = extract_answer(van_raw)
            van_ok   = is_correct(van_pred, gold)

            # ── TI injection ─────────────────────────────────────────────────
            if low_sim or not strategy_text:
                # No injection — run vanilla
                if args.ollama:
                    ti_raw, ti_lat, ti_tok = infer_ollama(
                        TI_SYSTEM, TI_USER.format(problem=problem))
                else:
                    ti_raw, ti_lat, ti_tok = infer_hf(
                        model, tokenizer, TI_SYSTEM,
                        TI_USER.format(problem=problem))
                used_ti = False
            else:
                # Inject template inside <think> tags
                if args.ollama:
                    ti_raw, ti_lat, ti_tok = infer_ollama(
                        TI_SYSTEM, TI_USER.format(problem=problem),
                        think_prefix=think_prefix)
                else:
                    ti_raw, ti_lat, ti_tok = infer_hf(
                        model, tokenizer, TI_SYSTEM,
                        TI_USER.format(problem=problem),
                        think_prefix=think_prefix)
                used_ti = True

            ti_pred = extract_answer(ti_raw)
            ti_ok   = is_correct(ti_pred, gold)

            # ── Record ───────────────────────────────────────────────────────
            record = {
                "problem_id":           problem_id,
                "subject":              subject,
                "level":                level,
                "gold_answer":          gold,

                "anchor_pattern_id":    anchor_match["index"],
                "anchor_similarity":    round(anchor_sim, 4),
                "sim_threshold":        threshold,
                "low_similarity":       low_sim,

                "assembled_strategy":   strategy_text,
                "template_length":      trav_info["template_length"],
                "anchor_steps":         trav_info["anchor_steps"],
                "bonus_steps":          trav_info["bonus_steps"],
                "is_cross_pattern":     trav_info["is_cross_pattern"],
                "used_think_injection": used_ti,

                "vanilla_answer":       van_pred,
                "vanilla_correct":      van_ok,
                "vanilla_latency_s":    round(van_lat, 3),
                "vanilla_tokens":       van_tok,

                "injection_answer":     ti_pred,
                "injection_correct":    ti_ok,
                "injection_latency_s":  round(ti_lat, 3),
                "injection_tokens":     ti_tok,
            }
            append_jsonl(JSONL_OUT, record)

            outcome = ("↑" if ti_ok and not van_ok else
                       "↓" if van_ok and not ti_ok else "=")
            log.info(
                f"  [{idx+1}/{len(remaining)}] id={problem_id} {subject} L{level} | "
                f"sim={anchor_sim:.3f} thr={threshold} | "
                f"{'TI' if used_ti else 'skip'} {trav_info['anchor_steps']}steps | "
                f"van={'✓' if van_ok else '✗'}({van_lat:.0f}s) "
                f"ti={'✓' if ti_ok else '✗'}({ti_lat:.0f}s) | {outcome}"
            )

        # Cleanup
        if not args.ollama and model is not None:
            del model, tokenizer
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                log.info(f"VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # ── Compile ──────────────────────────────────────────────────────────────
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
