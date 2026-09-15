"""
Phase 6 — Memory Bank + Thought Graph (new schema)

Builds the retrieval bank and step graph from phase-6 patterns.

Differences from phase-3d:
  * Reads the new schema (`steps` is a real list, so no "Step N:" re-parsing
    and no fixed 3-step assumption).
  * Thresholds come from config.py, whose import-time assertion guarantees
    NOVELTY_THRESHOLD > TAU_SEMANTIC. In phase-4 those two lived in separate
    files, contradicted each other, and silently disabled augmentation.
  * Writes the thresholds into graph metadata and verifies at build time that
    a real fraction of anchors would actually receive an augmentation, so a
    dead graph fails loudly here instead of after a multi-hour eval.

Usage:
    python build_memory_and_graph.py                 # full build
    python build_memory_and_graph.py --max-patterns 20   # smoke test
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import config as C

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data" / "patterns"
MEM_DIR   = ROOT / "memory"
GRAPH_DIR = ROOT / "graph"
LOGS_DIR  = ROOT / "logs"

PATTERNS_JSON = DATA_DIR / "patterns.json"
BANK_JSON     = MEM_DIR / "bank.json"
BANK_INDEX    = MEM_DIR / "bank.index"
GRAPH_JSON    = GRAPH_DIR / "thought_graph.json"
STEP_EMBED    = GRAPH_DIR / "step_embeddings.npy"

BATCH_SIZE = 64
MIN_AUGMENT_RATE = 0.05   # build fails below this — the phase-4 bug was 0.0


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"build_graph_{ts}.log"
    logger = logging.getLogger("phase6_build")
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


def make_summary(problem, max_len=120):
    s = problem[:max_len].strip()
    if len(problem) > max_len:
        sp = s.rfind(" ")
        if sp > 60:
            s = s[:sp]
        s += "..."
    return s


def load_embed_model(log):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embed model: {C.EMBED_MODEL} on {device}")
    return SentenceTransformer(C.EMBED_MODEL, trust_remote_code=True, device=device)


def embed_texts(model, texts, log, label=""):
    prefixed = [f"search_document: {t}" for t in texts]
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(prefixed), BATCH_SIZE):
        out.append(model.encode(prefixed[i:i + BATCH_SIZE],
                                normalize_embeddings=True, show_progress_bar=False))
    vecs = np.vstack(out).astype(np.float32)
    log.info(f"  {label}: {vecs.shape} in {time.perf_counter() - t0:.1f}s")
    return vecs


# ── Stage 1: bank ────────────────────────────────────────────────────────────

def build_bank(records, embed_model, log):
    log.info("\n" + "=" * 60)
    log.info("Stage 1: Memory bank")
    log.info("=" * 60)

    vectors = embed_texts(embed_model, [r["problem"] for r in records], log, "problems")

    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    MEM_DIR.mkdir(parents=True, exist_ok=True)
    bank = []
    for i, r in enumerate(records):
        bank.append({
            "index": i,
            "id": r["id"],
            "problem": r["problem"],
            "source": r.get("source", ""),
            "subject": r.get("subject", ""),
            "level": r.get("level", ""),
            "gold_answer": r.get("gold_answer", ""),
            "trigger": r.get("trigger", ""),
            "technique": r.get("technique", ""),
            "steps": r.get("steps", []),
            "pitfall": r.get("pitfall", ""),
            "verify": r.get("verify", ""),
        })

    with open(BANK_JSON, "w") as f:
        json.dump(bank, f, indent=2)
    faiss.write_index(index, str(BANK_INDEX))
    log.info(f"Bank: {len(bank)} entries -> {BANK_JSON}")
    return bank, vectors


# ── Stage 2: graph ───────────────────────────────────────────────────────────

def build_graph(bank, embed_model, log, max_patterns=None, smoke=False):
    log.info("\n" + "=" * 60)
    log.info(f"Stage 2: Thought graph (tau={C.TAU_SEMANTIC})")
    log.info("=" * 60)

    nodes = []
    nid = 0
    for entry in (bank[:max_patterns] if max_patterns else bank):
        steps = entry.get("steps") or []
        if not steps:
            continue
        summary = make_summary(entry["problem"])
        domain = "math_500" if entry.get("source") == "MATH-500" else "aime"
        for si, text in enumerate(steps):
            nodes.append({
                "node_id": nid,
                "pattern_id": entry["id"],
                "step_index": si,
                "text": text,
                "context_text": f"{summary} | {text}",
                "domain": domain,
                "subject": entry.get("subject", ""),
                "total_steps": len(steps),
            })
            nid += 1

    if not nodes:
        raise SystemExit("No nodes built — patterns.json has no usable steps")

    log.info(f"Nodes: {len(nodes)} from "
             f"{len({n['pattern_id'] for n in nodes})} patterns")
    dist = defaultdict(int)
    for n in nodes:
        dist[n["total_steps"]] += 1
    log.info(f"  step-count spread across nodes: {dict(sorted(dist.items()))}")

    emb = embed_texts(embed_model, [n["context_text"] for n in nodes], log, "steps")

    # sequential edges within a pattern
    by_pat = defaultdict(list)
    for n in nodes:
        by_pat[n["pattern_id"]].append(n)
    edges = []
    for pn in by_pat.values():
        pn.sort(key=lambda x: x["step_index"])
        for a, b in zip(pn, pn[1:]):
            edges.append({"src": a["node_id"], "dst": b["node_id"],
                          "type": "sequential", "weight": 1.0})
    n_seq = len(edges)

    # semantic edges across patterns
    sim = (emb @ emb.T + 1.0) / 2.0
    pat_of = [n["pattern_id"] for n in nodes]
    dom_of = [n["domain"] for n in nodes]
    cross = same = 0
    weights = []
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if pat_of[i] == pat_of[j] or sim[i, j] < C.TAU_SEMANTIC:
                continue
            w = float(sim[i, j])
            weights.append(w)
            edges.append({"src": i, "dst": j, "type": "semantic", "weight": w})
            edges.append({"src": j, "dst": i, "type": "semantic", "weight": w})
            if dom_of[i] != dom_of[j]:
                cross += 1
            else:
                same += 1

    log.info(f"Edges: {n_seq} sequential, {len(weights)} unique semantic "
             f"(same-domain {same}, cross-domain {cross})")
    if weights:
        log.info(f"  semantic sim range [{min(weights):.3f}, {max(weights):.3f}], "
                 f"mean {np.mean(weights):.3f}")

    adj = defaultdict(list)
    for e in edges:
        adj[e["src"]].append({"dst": e["dst"], "type": e["type"], "weight": e["weight"]})

    # Verify augmentation is actually reachable before anything expensive runs.
    rate, total = augmentation_rate(nodes, adj, emb)
    log.info(f"\nAugmentation reachability check "
             f"(novelty={C.NOVELTY_THRESHOLD}):")
    log.info(f"  {total} candidates survive; {rate*100:.1f}% of anchors would "
             f"receive >=1 bonus step")
    if rate < MIN_AUGMENT_RATE:
        msg = (f"Augmentation rate {rate*100:.1f}% < {MIN_AUGMENT_RATE*100:.0f}% — "
               f"the graph would contribute nothing (this is the phase-4 failure). "
               f"Check NOVELTY_THRESHOLD ({C.NOVELTY_THRESHOLD}) vs "
               f"TAU_SEMANTIC ({C.TAU_SEMANTIC}).")
        if smoke:
            # Too few patterns to form cross-pattern edges is expected in a
            # smoke test; only a full build treats this as fatal.
            log.warning(f"  SMOKE MODE — ignoring: {msg}")
        else:
            raise SystemExit(msg)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    np.save(STEP_EMBED, emb)
    meta = {
        "created": datetime.now().isoformat(),
        "version": "v3-phase6",
        "num_nodes": len(nodes),
        "num_sequential": n_seq,
        "num_semantic_unique": len(weights),
        "cross_domain_pairs": cross,
        "same_domain_pairs": same,
        # persisted so the evaluator can assert it is using a compatible graph
        "tau_semantic": C.TAU_SEMANTIC,
        "novelty_threshold": C.NOVELTY_THRESHOLD,
        "augmentation_rate": round(rate, 4),
        "embed_model": C.EMBED_MODEL,
    }
    with open(GRAPH_JSON, "w") as f:
        json.dump({"metadata": meta, "nodes": nodes,
                   "adjacency": {str(k): v for k, v in adj.items()}}, f)
    log.info(f"\nGraph -> {GRAPH_JSON}")
    log.info(f"Step embeddings -> {STEP_EMBED}")
    return nodes, adj, emb


def augmentation_rate(nodes, adj, emb):
    """Fraction of anchors that would receive >=1 augmentation candidate."""
    pat2 = defaultdict(list)
    for n in nodes:
        pat2[n["pattern_id"]].append(n)
    total = augmented = 0
    for pid, anodes in pat2.items():
        aids = {n["node_id"] for n in anodes}
        aemb = np.array([emb[n["node_id"]] for n in anodes])
        seen, surv = set(), 0
        for an in anodes:
            for e in adj.get(an["node_id"], []):
                if e["type"] != "semantic":
                    continue
                d = e["dst"]
                if d in aids or d in seen or nodes[d]["pattern_id"] == pid:
                    continue
                seen.add(d)
                if float((np.max(aemb @ emb[d]) + 1) / 2) >= C.NOVELTY_THRESHOLD:
                    continue
                surv += 1
        total += surv
        if surv:
            augmented += 1
    return (augmented / max(len(pat2), 1)), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-patterns", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="warn instead of failing when the graph has no augmentation "
                         "(expected with few patterns)")
    args = ap.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 6 — Bank + Graph")
    log.info("=" * 60)
    log.info(f"tau={C.TAU_SEMANTIC} novelty={C.NOVELTY_THRESHOLD} "
             f"(invariant asserted in config.py)")

    if not PATTERNS_JSON.exists():
        raise SystemExit(f"Missing {PATTERNS_JSON} — run generate_patterns.py first")

    records = json.load(open(PATTERNS_JSON))
    if args.max_patterns:
        records = records[:args.max_patterns]
    log.info(f"Patterns: {len(records)}")

    embed_model = load_embed_model(log)
    bank, _ = build_bank(records, embed_model, log)
    build_graph(bank, embed_model, log, args.max_patterns, args.smoke)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
