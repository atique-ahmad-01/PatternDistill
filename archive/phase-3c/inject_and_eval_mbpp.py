"""
Phase 3C — MBPP Evaluation with ATGI v2 (Anchor-First + Cross-Domain Graph)

Evaluates on MBPP test set (500 problems) with two conditions:
  1. MBPP-only: anchor retrieval from MBPP bank only (single-domain baseline)
  2. ATGI cross-domain: anchor from MBPP bank + bonus steps from combined graph
     (MATH+MBPP), testing whether math reasoning steps help code generation

Correctness is evaluated by running test cases against generated code.

Output:
  results/results.jsonl, results/results.json, results/summary.json

Usage:
    python inject_and_eval_mbpp.py --max-problems 5   # smoke test
    python inject_and_eval_mbpp.py                     # full run
"""

import argparse
import json
import logging
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
GRAPH_DIR   = ROOT / "graph"
MEM_DIR     = ROOT / "memory"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

GRAPH_JSON     = GRAPH_DIR / "thought_graph.json"
STEP_EMBED_NPY = GRAPH_DIR / "step_embeddings.npy"

BANK_JSON   = MEM_DIR / "bank.json"
BANK_INDEX  = MEM_DIR / "bank.index"

JSONL_OUT   = RESULTS_DIR / "results.jsonl"
JSON_OUT    = RESULTS_DIR / "results.json"
SUMMARY_OUT = RESULTS_DIR / "summary.json"

# ── Config ────────────────────────────────────────────────────────────────────

STUDENT_MODEL_HF     = "Qwen/Qwen3-1.7B"
STUDENT_MODEL_OLLAMA = "qwen3:1.7b"
DATASET              = "google-research-datasets/mbpp"
DATASET_CONFIG       = "full"
EVAL_N               = 100    # first 100 test problems (80/20: 374 train / 100 eval)
MAX_NEW_TOKENS       = 8192
OLLAMA_URL           = "http://localhost:11434"
EMBED_MODEL          = "nomic-ai/nomic-embed-text-v1"

# ── Hyperparameters ───────────────────────────────────────────────────────────

MAX_BONUS_STEPS   = 1
NOVELTY_THRESHOLD = 0.85
SIM_THRESHOLD     = 0.50   # single threshold for code problems

# ── Prompts ───────────────────────────────────────────────────────────────────

VANILLA_SYSTEM = (
    "You are a precise Python programming assistant. "
    "Write a function to solve the problem. Return ONLY the Python code, "
    "no explanation."
)
VANILLA_USER = (
    "Problem: {problem}\n\n"
    "Your function must pass these tests:\n{tests}\n\n"
    "Write the Python function. /no_think"
)

ATGI_SYSTEM = (
    "You are a precise Python programming assistant. "
    "An approach hint for solving this type of problem is provided. "
    "Return ONLY the Python code, no explanation."
)

# Light injection: approach hint only, no reference code (avoids copying wrong logic)
ATGI_USER_RICH = (
    "Approach:\n{strategy}\n\n"
    "Problem: {problem}\n\n"
    "Your function must pass these tests:\n{tests}\n\n"
    "Write the Python function. /no_think"
)

ATGI_USER_NO_EXAMPLE = (
    "Approach:\n{strategy}\n\n"
    "Problem: {problem}\n\n"
    "Your function must pass these tests:\n{tests}\n\n"
    "Write the Python function. /no_think"
)

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"inject_and_eval_mbpp_{timestamp}.log"

    logger = logging.getLogger("mbpp_eval")
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

# ── Code extraction & evaluation ──────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code(raw: str) -> str:
    """Extract Python code from model output. Searches both outside and inside <think> tags."""

    def _find_code(text: str) -> str | None:
        # Try ```python ... ``` block
        m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Try ``` ... ``` block
        m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Try to find def/import lines
        lines = text.strip().splitlines()
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("import ") or stripped.startswith("from "):
                in_code = True
            if in_code:
                # Stop at prose lines (no indentation, not blank, doesn't look like code)
                if stripped and not stripped.startswith(("#", "def ", "import ", "from ", "return ",
                    "if ", "else", "elif ", "for ", "while ", "try", "except", "with ", "class ",
                    "raise ", "assert ", "yield ", "pass", "break", "continue", "print(")) \
                    and not line.startswith((" ", "\t")) and not stripped.startswith((")", "]", "}")):
                    break
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()
        return None

    # First try outside <think> tags
    visible = strip_thinking(raw)
    code = _find_code(visible)
    if code:
        return code

    # Fallback: search inside <think> tags too (model may put code there)
    code = _find_code(raw)
    if code:
        return code

    # Last resort: return visible text
    return visible.strip()


def run_tests(code: str, test_list: list, test_setup: str = "",
              timeout_s: int = 10) -> tuple[bool, int, int, str]:
    """
    Run test cases against generated code.
    Returns (all_passed, n_passed, n_total, error_msg).
    """
    n_total = len(test_list)
    if n_total == 0:
        return True, 0, 0, ""

    # Build test script
    script_parts = []
    if test_setup:
        script_parts.append(test_setup)
    script_parts.append(code)
    script_parts.append("")  # blank line
    for test in test_list:
        script_parts.append(test)

    script = "\n".join(script_parts)

    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return True, n_total, n_total, ""
        else:
            err = result.stderr.strip()[-200:] if result.stderr else "unknown error"
            return False, 0, n_total, err
    except subprocess.TimeoutExpired:
        return False, 0, n_total, "timeout"
    except Exception as e:
        return False, 0, n_total, str(e)[:200]

# ── Load data ─────────────────────────────────────────────────────────────────

def load_thought_graph(log):
    if not GRAPH_JSON.exists():
        log.warning(f"No combined graph found at {GRAPH_JSON} — running without graph")
        return None, None, None, None
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    adj   = {int(k): v for k, v in graph["adjacency"].items()}
    meta  = graph["metadata"]
    embeddings = np.load(STEP_EMBED_NPY)
    log.info(f"Combined graph: {meta['num_nodes']} nodes "
             f"({meta.get('num_math_nodes',0)} math + {meta.get('num_mbpp_nodes',0)} code), "
             f"cross-domain={meta.get('cross_domain_pairs',0)} pairs")
    return nodes, adj, embeddings, meta


def load_mbpp_bank(log):
    import faiss
    with open(BANK_JSON) as f:
        bank_meta = json.load(f)
    index = faiss.read_index(str(BANK_INDEX))
    n, dim = index.ntotal, index.d
    all_vectors = np.zeros((n, dim), dtype=np.float32)
    index.reconstruct_n(0, n, all_vectors)
    log.info(f"MBPP bank: {len(bank_meta)} entries, dim={dim}")
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

# ── Retrieval + graph expansion ───────────────────────────────────────────────

def retrieve_anchor(query_vec, bank_meta, bank_vectors):
    scores = (bank_vectors @ query_vec).flatten()
    best   = int(np.argmax(scores))
    return bank_meta[best], float(scores[best])


def find_novel_bonus_steps(anchor_pattern_id, query_vec, nodes, adj,
                           step_embeddings):
    """Find novel cross-domain bonus steps from the combined graph."""
    if nodes is None:
        return []

    # Find anchor nodes in combined graph
    anchor_nodes = [n for n in nodes if n["pattern_id"] == anchor_pattern_id]
    if not anchor_nodes:
        return []

    anchor_node_ids = {n["node_id"] for n in anchor_nodes}
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
            max_sim = float((np.max(sims_to_anchor) + 1.0) / 2.0)

            if max_sim >= NOVELTY_THRESHOLD:
                continue

            query_sim = float(dst_embed @ query_vec)
            candidates.append({
                "node": nodes[dst],
                "query_sim": query_sim,
                "source_domain": nodes[dst]["domain"],
            })

    candidates.sort(key=lambda c: c["query_sim"], reverse=True)
    return [c["node"] for c in candidates[:MAX_BONUS_STEPS]]


def assemble_strategy(pattern_text, bonus_nodes):
    """Combine anchor pattern + bonus steps."""
    lines = [pattern_text]
    step_num = 5  # anchor already has steps 1-4
    for node in bonus_nodes:
        lines.append(f"Step {step_num}: {node['text']}")
        step_num += 1
    return "\n".join(lines)

# ── Backends ──────────────────────────────────────────────────────────────────

def infer_ollama(system, user):
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
    data = resp.json()
    return data["message"]["content"], latency, data.get("eval_count", 0)


def load_student_hf(log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
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


def infer_hf(model, tokenizer, system, user):
    import torch
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    max_new  = min(MAX_NEW_TOKENS, 32768 - input_len)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    latency    = time.perf_counter() - t0
    new_tokens = outputs[0][input_len:]
    raw        = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, latency, len(new_tokens)

# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_done_ids(path):
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done

def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Summarise ─────────────────────────────────────────────────────────────────

def summarize(records):
    n = len(records)
    if n == 0:
        return {}

    def acc(recs, key):
        return round(sum(r[key] for r in recs) / len(recs), 4)

    injected = [r for r in records if not r.get("low_similarity")]
    bonus    = [r for r in injected if r.get("has_bonus_steps")]
    cross    = [r for r in injected if r.get("has_cross_domain_bonus")]

    return {
        "n": n,
        "n_injected":          len(injected),
        "n_low_sim_skipped":   n - len(injected),
        "n_with_bonus":        len(bonus),
        "n_cross_domain":      len(cross),
        "method":              "ATGI v2 on MBPP (anchor-first + cross-domain graph)",
        "vanilla": {
            "accuracy":     acc(records, "vanilla_pass"),
            "avg_latency_s": round(sum(r["vanilla_latency_s"] for r in records) / n, 3),
        },
        "injection": {
            "accuracy":     acc(records, "injection_pass"),
            "avg_latency_s": round(sum(r["injection_latency_s"] for r in records) / n, 3),
        },
        "delta": {
            "accuracy": round(acc(records, "injection_pass") - acc(records, "vanilla_pass"), 4),
        },
    }


def print_summary(s, log):
    log.info(f"\n{'='*60}")
    log.info(f"Phase 3C — MBPP ATGI Results")
    log.info(f"{'='*60}")
    log.info(f"  n={s['n']}  injected={s['n_injected']}  skipped={s['n_low_sim_skipped']}  "
             f"bonus={s['n_with_bonus']}  cross-domain={s['n_cross_domain']}")
    log.info(f"\n  Vanilla   pass rate : {s['vanilla']['accuracy']:.3f}")
    log.info(f"  ATGI      pass rate : {s['injection']['accuracy']:.3f}")
    log.info(f"  Delta               : {s['delta']['accuracy']:+.3f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3C — MBPP ATGI Evaluation")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--ollama", action="store_true")
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3C — MBPP ATGI v2 Evaluation")
    log.info("=" * 60)

    ds = load_dataset(DATASET, DATASET_CONFIG, split="test")
    problems = list(ds)[:EVAL_N]  # first 100 test problems (80/20 split)

    if args.max_problems:
        problems = problems[:args.max_problems]
        log.info(f"Capped at: {len(problems)} problems")
    else:
        log.info(f"Eval set: {len(problems)} MBPP test problems (of {len(list(ds))} total)")

    done = load_done_ids(JSONL_OUT)
    remaining = [(p["task_id"], p) for p in problems if p["task_id"] not in done]
    if done:
        log.info(f"Resuming — {len(done)} already done")
    log.info(f"To do: {len(remaining)}")

    if remaining:
        bank_meta, bank_vectors = load_mbpp_bank(log)
        nodes, adj, step_embeddings, graph_meta = load_thought_graph(log)
        embed_model = load_embed_model(log, device="cuda")

        if args.ollama:
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
                log.info(f"Ollama OK")
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
            log.info("Embed model reloaded on CPU")

        for idx, (task_id, item) in enumerate(remaining):
            problem    = item["text"]
            test_list  = item["test_list"]
            test_setup = item.get("test_setup_code", "")

            # Retrieve anchor
            query_vec = embed_query(embed_model, problem)
            anchor, anchor_sim = retrieve_anchor(query_vec, bank_meta, bank_vectors)
            low_sim = anchor_sim < SIM_THRESHOLD

            # Graph expansion
            has_bonus = False
            has_cross_domain = False
            bonus_nodes = []

            if not low_sim and nodes is not None:
                anchor_pid = f"mbpp_{anchor['index']}"
                bonus_nodes = find_novel_bonus_steps(
                    anchor_pid, query_vec, nodes, adj, step_embeddings)
                has_bonus = len(bonus_nodes) > 0
                has_cross_domain = any(n["domain"] == "math" for n in bonus_nodes)

            # Build strategy
            if low_sim:
                strategy = ""
                worked_example = ""
            else:
                pattern = anchor.get("pattern", "")
                worked_example = anchor.get("worked_example", "")
                if bonus_nodes:
                    strategy = assemble_strategy(pattern, bonus_nodes)
                else:
                    strategy = pattern

            # Format test cases for prompt
            tests_str = "\n".join(test_list)

            # ── Vanilla ──────────────────────────────────────────────────────
            if args.ollama:
                van_raw, van_lat, van_tok = infer_ollama(
                    VANILLA_SYSTEM, VANILLA_USER.format(problem=problem, tests=tests_str))
            else:
                van_raw, van_lat, van_tok = infer_hf(
                    model, tokenizer, VANILLA_SYSTEM,
                    VANILLA_USER.format(problem=problem, tests=tests_str))

            van_code = extract_code(van_raw)
            van_pass, van_n_pass, van_n_total, van_err = run_tests(
                van_code, test_list, test_setup)

            # ── ATGI injection ───────────────────────────────────────────────
            if low_sim or not strategy:
                inj_sys  = VANILLA_SYSTEM
                inj_user = VANILLA_USER.format(problem=problem, tests=tests_str)
            elif worked_example:
                inj_sys  = ATGI_SYSTEM
                inj_user = ATGI_USER_RICH.format(
                    strategy=strategy, example=worked_example, problem=problem, tests=tests_str)
            else:
                inj_sys  = ATGI_SYSTEM
                inj_user = ATGI_USER_NO_EXAMPLE.format(
                    strategy=strategy, problem=problem, tests=tests_str)

            if args.ollama:
                inj_raw, inj_lat, inj_tok = infer_ollama(inj_sys, inj_user)
            else:
                inj_raw, inj_lat, inj_tok = infer_hf(
                    model, tokenizer, inj_sys, inj_user)

            inj_code = extract_code(inj_raw)
            inj_pass, inj_n_pass, inj_n_total, inj_err = run_tests(
                inj_code, test_list, test_setup)

            record = {
                "task_id":              task_id,
                "problem":              problem[:200],
                "anchor_similarity":    round(anchor_sim, 4),
                "low_similarity":       low_sim,
                "has_bonus_steps":      has_bonus,
                "has_cross_domain_bonus": has_cross_domain,
                "bonus_domains":        [n["domain"] for n in bonus_nodes],

                "vanilla_pass":         van_pass,
                "vanilla_latency_s":    round(van_lat, 3),
                "vanilla_tokens":       van_tok,
                "vanilla_error":        van_err[:100] if van_err else "",

                "injection_pass":       inj_pass,
                "injection_latency_s":  round(inj_lat, 3),
                "injection_tokens":     inj_tok,
                "injection_error":      inj_err[:100] if inj_err else "",
            }
            append_jsonl(JSONL_OUT, record)

            outcome = ("↑" if inj_pass and not van_pass else
                       "↓" if van_pass and not inj_pass else "=")
            cross_tag = "🌐" if has_cross_domain else ""
            log.info(
                f"  [{idx+1}/{len(remaining)}] task={task_id} | "
                f"sim={anchor_sim:.3f} | "
                f"van={'✓' if van_pass else '✗'}({van_lat:.0f}s) "
                f"atgi={'✓' if inj_pass else '✗'}({inj_lat:.0f}s) "
                f"{cross_tag}| {outcome}"
            )

        if not args.ollama and model is not None:
            del model, tokenizer
            import gc; gc.collect()

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

    all_records.sort(key=lambda r: r["task_id"])
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
