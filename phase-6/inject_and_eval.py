"""
Phase 6 — Injection Evaluation (6 ablation arms)

Same test set as phase-3d/phase-4: 100 MATH-500 + 187 AIME = 287 problems.

What changed, and why (all five fixes measured in the phase-4 4B run):

  1. Budget 8,192 -> 24,576 for EVERY arm including vanilla. Phase-4 truncated
     74% of AIME vanilla runs; 111/117 AIME both-wrong cases had both arms
     truncated. Raising only the injection arm would manufacture a win.
  2. Early-exit once a complete \\boxed{} is produced, so a finished answer
     does not burn the rest of the budget.
  3. Thresholds come from config.py, which asserts NOVELTY_THRESHOLD >
     TAU_SEMANTIC. Phase-4 had 0.85 vs 0.90, making augmentation impossible.
  4. The full injected prompt is recorded per problem, so a regression can be
     traced to the pattern that caused it. Phase-4 saved none of this.
  5. Six arms, so each design choice is separately attributable. Arm B (old
     patterns, new budget) is the control that separates "we gave the model
     room to think" from "our pattern redesign worked".

Assembly rule: the anchor's steps are the spine, kept in order. At most one
graph neighbour is APPENDED and labelled, never spliced into the middle —
phase-3 showed that merging steps from many patterns produces incoherent
scaffolds that score worse.

Usage:
    python inject_and_eval.py --model qwen3-1.7b --arms A_vanilla,C_new_patterns --max-problems 3
    python inject_and_eval.py --model qwen3-1.7b --arms all
"""

import argparse
import json
import logging
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset

import config as C
import scoring as S

ROOT        = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

EVAL_ORDER_SEED = 20260813  # fixed: prefix of a partial run stays a fair sample

NEW_BANK_JSON  = ROOT / "memory" / "bank.json"
NEW_BANK_INDEX = ROOT / "memory" / "bank.index"
GRAPH_JSON     = ROOT / "graph" / "thought_graph.json"
STEP_EMBED     = ROOT / "graph" / "step_embeddings.npy"

# Arm B reuses the *old* contaminated banks on purpose, as the control.
OLD_MATH_BANK  = ROOT.parent / "phase-2d" / "memory" / "bank.json"
OLD_MATH_INDEX = ROOT.parent / "phase-2d" / "memory" / "bank.index"
OLD_AIME_BANK  = ROOT.parent / "phase-3d" / "memory" / "bank.json"
OLD_AIME_INDEX = ROOT.parent / "phase-3d" / "memory" / "bank.index"

MODELS = {
    "qwen3-1.7b": {"hf_id": "Qwen/Qwen3-1.7B", "short": "Qwen3-1.7B", "max_context": 32768},
    "qwen3-4b":   {"hf_id": "Qwen/Qwen3-4B",   "short": "Qwen3-4B",   "max_context": 32768},
}

# ── Prompts ──────────────────────────────────────────────────────────────────

VANILLA_SYSTEM = (
    "You are a precise math problem-solving assistant. "
    "Solve the problem step by step, then state the final answer clearly."
)
VANILLA_USER = (
    "Problem: {problem}\n\n"
    "Solve step by step. Put your final answer in \\boxed{{}}."
)

INJECT_SYSTEM = (
    "You are a precise math problem-solving assistant. A strategy from a "
    "similar solved problem is provided as guidance. Adapt it to the problem "
    "you are given."
)

# The escape hatch. Phase-3d/4 said "Follow the strategy steps explicitly",
# which forces compliance even when retrieval returns a poor match — the most
# likely cause of the 15 regressions in phase-4. Arm F removes this line to
# measure what it is worth.
ESCAPE_LINE = ("If the strategy does not fit this problem, say so briefly in one "
               "sentence and solve the problem directly instead.\n\n")

INJECT_USER = (
    "Strategy from a similar solved problem — adapt it to the numbers below.\n"
    "{escape}"
    "{strategy}\n\n"
    "Problem: {problem}\n\n"
    "Be direct and do not restate the problem. "
    "Put your final answer in \\boxed{{}}."
)


def setup_logging(tag):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"eval_{tag}_{ts}.log"
    logger = logging.getLogger(f"phase6_{tag}")
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


# ── Early exit ───────────────────────────────────────────────────────────────

def has_complete_boxed(text):
    """True once a \\boxed{...} with balanced braces has been emitted."""
    i = text.rfind(r"\boxed{")
    if i == -1:
        return False
    depth = 0
    for ch in text[i + 6:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


def make_stopping_criteria(tok, prompt_len, window=96):
    """Stop generating once the answer is boxed.

    Only a trailing token window is decoded each step, so the check is O(1)
    rather than O(n) in generated length. If a boxed expression is longer than
    the window the check simply misses and generation runs to EOS — safe, just
    less efficient.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class BoxedStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            gen = input_ids[0][prompt_len:]
            if len(gen) < 8:
                return False
            tail = tok.decode(gen[-window:], skip_special_tokens=True)
            return has_complete_boxed(tail)

    return StoppingCriteriaList([BoxedStop()])


# ── Model ────────────────────────────────────────────────────────────────────

def load_student(model_key, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODELS[model_key]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tok, cfg


def infer(model, tok, system, user, max_context):
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    ilen = inp["input_ids"].shape[1]
    budget = min(C.MAX_NEW_TOKENS, max_context - ilen)
    crit = make_stopping_criteria(tok, ilen) if C.EARLY_EXIT_ON_BOXED else None
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id, stopping_criteria=crit)
    new = out[0][ilen:]
    raw = tok.decode(new, skip_special_tokens=True)
    return raw, time.perf_counter() - t0, len(new), budget


# ── Banks / graph ────────────────────────────────────────────────────────────

def _read_index(path, dim=None):
    import faiss
    idx = faiss.read_index(str(path))
    v = np.zeros((idx.ntotal, idx.d), dtype=np.float32)
    idx.reconstruct_n(0, idx.ntotal, v)
    return v


def load_new_bank(log):
    bank = json.load(open(NEW_BANK_JSON))
    vecs = _read_index(NEW_BANK_INDEX)
    for e in bank:
        e["_source"] = e.get("source", "")
    log.info(f"New bank: {len(bank)} entries")
    return bank, vecs


def load_old_bank(log):
    """Combined phase-2d + phase-3d banks — the contaminated control."""
    mb = json.load(open(OLD_MATH_BANK))
    ab = json.load(open(OLD_AIME_BANK))
    mv, av = _read_index(OLD_MATH_INDEX), _read_index(OLD_AIME_INDEX)
    for e in mb:
        e["_source"] = "MATH-500"
    for e in ab:
        e["_source"] = "AIME"
    log.info(f"Old bank (control): {len(mb)} MATH + {len(ab)} AIME")
    return mb + ab, np.vstack([mv, av])


def load_graph(log):
    if not GRAPH_JSON.exists():
        log.warning("No graph — graph arms will be skipped")
        return None, None, None
    g = json.load(open(GRAPH_JSON))
    meta = g["metadata"]
    # Guard against evaluating a graph built with an incompatible threshold.
    if meta.get("novelty_threshold") != C.NOVELTY_THRESHOLD:
        log.warning(f"Graph novelty_threshold={meta.get('novelty_threshold')} "
                    f"but config has {C.NOVELTY_THRESHOLD}")
    if meta.get("augmentation_rate", 0) <= 0:
        raise SystemExit("Graph has augmentation_rate 0 — rebuild it (phase-4 bug)")
    nodes = g["nodes"]
    adj = {int(k): v for k, v in g["adjacency"].items()}
    emb = np.load(STEP_EMBED)
    log.info(f"Graph: {meta['num_nodes']} nodes, "
             f"augmentation_rate={meta.get('augmentation_rate')}")
    return nodes, adj, emb


def load_embed(log):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(C.EMBED_MODEL, trust_remote_code=True, device=dev)


def embed_query(model, text):
    v = model.encode([f"search_query: {text}"], normalize_embeddings=True,
                     show_progress_bar=False)
    return np.asarray(v, dtype=np.float32).flatten()


# ── Retrieval + assembly ─────────────────────────────────────────────────────

def retrieve_anchor(qvec, bank, vecs, source):
    scores = (vecs @ qvec).flatten()
    biased = scores.copy()
    for i, e in enumerate(bank):
        if e.get("_source") == source:
            biased[i] += C.SAME_SOURCE_BIAS
    best = int(np.argmax(biased))
    return bank[best], float(scores[best]), best


def get_threshold(source, subject="", cross=False):
    t = C.SIM_THRESHOLDS.get(source, {"default": 0.50})
    base = t.get(subject, t.get("default", 0.50))
    return base + (C.CROSS_THRESHOLD_BUMP if cross else 0.0)


def find_augmentation(anchor_pid, qvec, nodes, adj, emb):
    """At most one appended step from the anchor's 1-hop graph neighbours."""
    if nodes is None:
        return []
    anodes = [n for n in nodes if n["pattern_id"] == anchor_pid]
    if not anodes:
        return []
    aids = {n["node_id"] for n in anodes}
    aemb = np.array([emb[n["node_id"]] for n in anodes])
    cands, seen = [], set()
    for an in anodes:
        for e in adj.get(an["node_id"], []):
            if e["type"] != "semantic":
                continue
            d = e["dst"]
            if d in aids or d in seen or nodes[d]["pattern_id"] == anchor_pid:
                continue
            seen.add(d)
            if float((np.max(aemb @ emb[d]) + 1) / 2) >= C.NOVELTY_THRESHOLD:
                continue
            # ranked by relevance to the NEW problem, not to the anchor
            cands.append({"node": nodes[d], "qsim": float(emb[d] @ qvec)})
    cands.sort(key=lambda c: c["qsim"], reverse=True)
    return [c["node"] for c in cands[:C.MAX_BONUS_STEPS]]


def assemble_new(anchor, augmentation, include_example):
    """Spine order preserved; augmentation appended and labelled."""
    lines = []
    if anchor.get("trigger"):
        lines.append(f"Applies when: {anchor['trigger']}")
    if anchor.get("technique"):
        lines.append(f"Technique: {anchor['technique']}")
    lines.append("Plan:")
    for i, s in enumerate(anchor.get("steps", []), 1):
        lines.append(f"  {i}. {s}")
    for n in augmentation:
        lines.append(f"Also consider: {n['text']}")
    if anchor.get("pitfall"):
        lines.append(f"Watch out: {anchor['pitfall']}")
    if anchor.get("verify"):
        lines.append(f"Check: {anchor['verify']}")
    if include_example and anchor.get("worked_example"):
        lines.append(f"Reference (symbolic): {anchor['worked_example']}")
    return "\n".join(lines)


def assemble_old(anchor, include_example):
    """Legacy flat pattern, for the arm-B control."""
    lines = [anchor.get("pattern", "").strip()]
    if include_example and anchor.get("worked_example"):
        lines.append(f"Reference Example:\n{anchor['worked_example']}")
    return "\n".join(l for l in lines if l)


# ── Eval set ─────────────────────────────────────────────────────────────────

def build_eval_set(log):
    items = []
    math = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[400:500]
    for i, r in enumerate(math):
        items.append({"pid": f"math_{400+i}", "source": "MATH-500",
                      "problem": r["problem"], "gold": str(r["answer"]),
                      "subject": r.get("subject", ""), "level": r.get("level", ""),
                      "year": None})
    allp = list(load_dataset("qq8933/AIME_1983_2024", split="train"))
    n_train = int(len(allp) * 0.8)
    for r in allp[n_train:]:
        items.append({"pid": str(r.get("ID", "")), "source": "AIME",
                      "problem": r["Question"], "gold": str(r["Answer"]),
                      "subject": "", "level": "", "year": r.get("Year")})
    log.info(f"Eval set: {len(items)} ({len(math)} MATH-500 + {len(items)-len(math)} AIME)")
    # Shuffle so MATH-500 and AIME interleave: build_eval_set emits all MATH
    # first, so an unshuffled --max-problems (or a partial run) would be pure
    # MATH-500 and miss the hard half entirely. Fixed seed keeps the order
    # identical across resumes and across arms.
    random.Random(EVAL_ORDER_SEED).shuffle(items)
    return items


def load_done(path):
    done = set()
    if path.exists():
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["problem_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_jsonl(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="qwen3-1.7b")
    ap.add_argument("--arms", default="all",
                    help="comma-separated arm names, or 'all'")
    ap.add_argument("--max-problems", type=int, default=None)
    args = ap.parse_args()

    arms = list(C.ARMS) if args.arms == "all" else args.arms.split(",")
    for a in arms:
        if a not in C.ARMS:
            raise SystemExit(f"Unknown arm {a!r}; choose from {list(C.ARMS)}")

    log = setup_logging(args.model)
    log.info("=" * 66)
    log.info(f"Phase 6 — Injection Eval | model={args.model} arms={arms}")
    log.info(f"budget={C.MAX_NEW_TOKENS} early_exit={C.EARLY_EXIT_ON_BOXED} "
             f"tau={C.TAU_SEMANTIC} novelty={C.NOVELTY_THRESHOLD}")
    log.info("=" * 66)

    out_path = RESULTS_DIR / f"results_{args.model}.jsonl"
    items = build_eval_set(log)
    if args.max_problems:
        items = items[:args.max_problems]
        log.info(f"Capped at {len(items)}")

    needs_new = any(C.ARMS[a].get("bank") == "new" for a in arms)
    needs_old = any(C.ARMS[a].get("bank") == "old" for a in arms)
    needs_graph = any(C.ARMS[a].get("graph") for a in arms)

    embed_model = load_embed(log) if (needs_new or needs_old) else None
    new_bank = new_vecs = old_bank = old_vecs = None
    if needs_new:
        new_bank, new_vecs = load_new_bank(log)
    if needs_old:
        old_bank, old_vecs = load_old_bank(log)
    nodes = adj = semb = None
    if needs_graph:
        nodes, adj, semb = load_graph(log)

    model, tok, cfg = load_student(args.model, log)
    done = load_done(out_path)
    if done:
        log.info(f"Resuming — {len(done)} (arm, problem) pairs already done")

    stats = defaultdict(lambda: defaultdict(int))

    for idx, it in enumerate(items, 1):
        qvec = embed_query(embed_model, it["problem"]) if embed_model is not None else None

        for arm in arms:
            if (arm, it["pid"]) in done:
                continue
            spec = C.ARMS[arm]

            strategy, anchor_info = "", {}
            if spec.get("inject"):
                bank, vecs = ((new_bank, new_vecs) if spec["bank"] == "new"
                              else (old_bank, old_vecs))
                anchor, asim, aidx = retrieve_anchor(qvec, bank, vecs, it["source"])
                cross = anchor.get("_source") != it["source"]
                thr = get_threshold(it["source"], it["subject"], cross)
                aug = []
                if spec.get("graph") and nodes is not None:
                    aug = find_augmentation(anchor["id"], qvec, nodes, adj, semb)
                if spec["bank"] == "new":
                    strategy = assemble_new(anchor, aug, spec.get("example"))
                else:
                    strategy = assemble_old(anchor, spec.get("example"))
                anchor_info = {
                    "anchor_id": anchor.get("id", anchor.get("index")),
                    "anchor_index": aidx,
                    "anchor_source": anchor.get("_source", ""),
                    "anchor_similarity": round(asim, 4),
                    "sim_threshold": thr,
                    "low_similarity": asim < thr,
                    "n_anchor_steps": len(anchor.get("steps", [])) or None,
                    "n_augmentation": len(aug),
                    "augmentation_text": [n["text"] for n in aug],
                }

            if spec.get("inject") and strategy:
                system = INJECT_SYSTEM
                user = INJECT_USER.format(
                    escape=ESCAPE_LINE if spec.get("escape") else "",
                    strategy=strategy, problem=it["problem"])
            else:
                system = VANILLA_SYSTEM
                user = VANILLA_USER.format(problem=it["problem"])

            raw, lat, ntok, budget = infer(model, tok, system, user, cfg["max_context"])
            pred = (S.extract_answer_math(raw) if it["source"] == "MATH-500"
                    else S.extract_answer_aime(raw))
            ok = S.is_correct(pred, it["gold"], it["source"])
            truncated = ntok >= budget

            rec = {
                "model": cfg["short"], "arm": arm,
                "problem_id": it["pid"], "source": it["source"],
                "subject": it["subject"], "level": it["level"], "year": it["year"],
                "gold_answer": it["gold"],
                **anchor_info,
                # full injected prompt, so any regression is traceable
                "injected_strategy": strategy,
                "system_prompt": system,
                "user_prompt": user,
                "escape_hatch": bool(spec.get("escape")) if spec.get("inject") else None,
                "raw": raw,
                "answer": pred,
                "correct": ok,
                "latency_s": round(lat, 1),
                "tokens": ntok,
                "budget": budget,
                "truncated": truncated,
            }
            append_jsonl(out_path, rec)

            stats[arm]["n"] += 1
            stats[arm]["correct"] += int(ok)
            stats[arm]["truncated"] += int(truncated)
            stats[arm]["augmented"] += anchor_info.get("n_augmentation", 0)

            log.info(f"  [{idx}/{len(items)}] {it['pid']:>12s} {arm:<16s} "
                     f"{'Y' if ok else 'N'} {ntok:>6d}tok {lat:>6.1f}s"
                     f"{' TRUNC' if truncated else ''}")

    log.info("\n" + "=" * 66)
    for arm in arms:
        s = stats[arm]
        if not s["n"]:
            continue
        log.info(f"{arm:<18s} n={s['n']:4d} acc={s['correct']/s['n']*100:5.1f}% "
                 f"trunc={s['truncated']/s['n']*100:5.1f}% aug_steps={s['augmented']}")
    log.info(f"\nResults -> {out_path}")
    log.info("Run analyze.py for McNemar tests and confidence intervals.")


if __name__ == "__main__":
    main()
