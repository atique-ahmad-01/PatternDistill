"""
Phase 3C — Build MBPP Memory Bank + Combined Thought Graph

Two stages:
  1. Build MBPP memory bank (embed MBPP patterns, create FAISS index)
  2. Build COMBINED thought graph (MATH-500 + MBPP nodes, cross-domain edges)

The combined graph enables cross-domain step transfer:
  - MATH steps like "set up equation from constraints" can help code problems
  - Code steps like "iterate through list and filter" can help math problems

Input:
  phase-3c/data/patterns/patterns.json  (MBPP patterns from generate_patterns.py)
  phase-2d/memory/bank.json             (MATH-500 patterns from phase-2d)

Output:
  phase-3c/memory/bank.json + bank.index  (MBPP-only memory bank)
  phase-3c/graph/thought_graph.json       (combined MATH+MBPP graph)
  phase-3c/graph/step_embeddings.npy      (combined embeddings)
  phase-3c/graph/step_faiss.index         (combined FAISS index)

Usage:
    python build_memory_and_graph.py                     # full build
    python build_memory_and_graph.py --max-patterns 10   # smoke test
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent
PHASE2D   = ROOT.parent / "phase-2d"

# MBPP patterns
MBPP_PATTERNS = ROOT / "data" / "patterns" / "patterns.json"

# MATH-500 patterns (from phase-2d)
MATH_BANK     = PHASE2D / "memory" / "bank.json"

# Output
MEM_DIR    = ROOT / "memory"
GRAPH_DIR  = ROOT / "graph"
LOGS_DIR   = ROOT / "logs"

BANK_JSON      = MEM_DIR / "bank.json"
BANK_INDEX     = MEM_DIR / "bank.index"
GRAPH_JSON     = GRAPH_DIR / "thought_graph.json"
STEP_EMBED_NPY = GRAPH_DIR / "step_embeddings.npy"
STEP_FAISS_IDX = GRAPH_DIR / "step_faiss.index"

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE  = 64
TAU_SEMANTIC_DEFAULT = 0.90

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"build_memory_and_graph_{timestamp}.log"
    logger = logging.getLogger("build_all")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {log_file}")
    return logger

# ── Step parsing ──────────────────────────────────────────────────────────────

STEP_RE = re.compile(r"(Step\s*\d+\s*[:.]|Note\s*[:.])", re.IGNORECASE)


def parse_steps(pattern_text: str) -> list[dict]:
    if not pattern_text or not pattern_text.strip():
        return []
    splits = list(STEP_RE.finditer(pattern_text))
    if not splits:
        return [{"label": "Step 1", "text": pattern_text.strip()}]
    steps = []
    for i, match in enumerate(splits):
        label = match.group(0).rstrip(":. ").strip()
        start = match.end()
        end   = splits[i + 1].start() if i + 1 < len(splits) else len(pattern_text)
        text  = pattern_text[start:end].strip()
        if text:
            steps.append({"label": label, "text": text})
    return steps


def make_problem_summary(problem: str, max_len: int = 120) -> str:
    summary = problem[:max_len].strip()
    if len(problem) > max_len:
        last_space = summary.rfind(" ")
        if last_space > 60:
            summary = summary[:last_space]
        summary += "..."
    return summary

# ── Embedding ─────────────────────────────────────────────────────────────────

def load_embed_model(log):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model


def embed_texts(model, texts: list, log, label: str = "") -> np.ndarray:
    prefixed = [f"search_document: {t}" for t in texts]
    all_vecs = []
    t0_total = time.perf_counter()
    for i in range(0, len(prefixed), BATCH_SIZE):
        batch = prefixed[i : i + BATCH_SIZE]
        t0 = time.perf_counter()
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        all_vecs.append(vecs)
        log.debug(f"  {label} batch {i // BATCH_SIZE + 1} "
                  f"({i + 1}–{min(i + len(batch), len(texts))}/{len(texts)}) {elapsed:.1f}s")
    total = time.perf_counter() - t0_total
    result = np.vstack(all_vecs).astype(np.float32)
    log.info(f"  {label}: {result.shape} in {total:.1f}s")
    return result

# ── Stage 1: MBPP Memory Bank ────────────────────────────────────────────────

def build_mbpp_bank(mbpp_records: list, embed_model, log):
    """Build MBPP-only memory bank (like phase-2d's bank for MATH)."""
    log.info(f"\n{'='*60}")
    log.info(f"Stage 1: Building MBPP Memory Bank")
    log.info(f"{'='*60}")

    valid = [r for r in mbpp_records if r.get("has_pattern") and r.get("pattern")]
    log.info(f"Valid patterns: {len(valid)}/{len(mbpp_records)}")

    texts = [r["problem"] for r in valid]
    vectors = embed_texts(embed_model, texts, log, "MBPP problems")

    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    MEM_DIR.mkdir(parents=True, exist_ok=True)

    bank_meta = []
    for i, r in enumerate(valid):
        bank_meta.append({
            "index":           i,
            "id":              r["id"],
            "problem":         r["problem"],
            "domain":          "code",
            "subject":         "Python",
            "gold_code":       r.get("gold_code", ""),
            "test_list":       r.get("test_list", []),
            "cot":             r.get("cot", ""),
            "pattern":         r["pattern"],
            "worked_example":  r.get("worked_example", ""),
            "is_step_level":   r.get("is_step_level", False),
            "teacher_answer":  r.get("teacher_answer", ""),
        })

    with open(BANK_JSON, "w") as f:
        json.dump(bank_meta, f, indent=2)
    log.info(f"Saved MBPP bank → {BANK_JSON} ({len(bank_meta)} entries)")

    faiss.write_index(index, str(BANK_INDEX))
    log.info(f"Saved MBPP FAISS → {BANK_INDEX}")

    return bank_meta, vectors

# ── Stage 2: Combined Thought Graph ──────────────────────────────────────────

def build_combined_graph(math_bank: list, mbpp_bank: list,
                         embed_model, tau: float, max_patterns: int | None, log):
    """Build combined MATH + MBPP thought graph with cross-domain edges."""
    log.info(f"\n{'='*60}")
    log.info(f"Stage 2: Building Combined Thought Graph")
    log.info(f"{'='*60}")

    # Parse all patterns into nodes
    nodes = []
    node_id = 0

    # MATH nodes
    math_count = 0
    for entry in math_bank:
        if max_patterns and math_count >= max_patterns:
            break
        steps = parse_steps(entry.get("pattern", ""))
        if not steps:
            continue
        summary = make_problem_summary(entry["problem"])
        for step_idx, step in enumerate(steps):
            nodes.append({
                "node_id":         node_id,
                "pattern_id":      f"math_{entry['index']}",
                "problem_id":      entry["id"],
                "step_index":      step_idx,
                "label":           step["label"],
                "text":            step["text"],
                "context_text":    f"{summary} | {step['text']}",
                "problem_summary": summary,
                "domain":          "math",
                "subject":         entry.get("subject", "Math"),
                "total_steps":     len(steps),
            })
            node_id += 1
        math_count += 1

    # MBPP nodes
    mbpp_count = 0
    for entry in mbpp_bank:
        if max_patterns and mbpp_count >= max_patterns:
            break
        steps = parse_steps(entry.get("pattern", ""))
        if not steps:
            continue
        summary = make_problem_summary(entry["problem"])
        for step_idx, step in enumerate(steps):
            nodes.append({
                "node_id":         node_id,
                "pattern_id":      f"mbpp_{entry['index']}",
                "problem_id":      entry["id"],
                "step_index":      step_idx,
                "label":           step["label"],
                "text":            step["text"],
                "context_text":    f"{summary} | {step['text']}",
                "problem_summary": summary,
                "domain":          "code",
                "subject":         "Python",
                "total_steps":     len(steps),
            })
            node_id += 1
        mbpp_count += 1

    log.info(f"Total nodes: {len(nodes)} (MATH: {sum(1 for n in nodes if n['domain']=='math')}, "
             f"MBPP: {sum(1 for n in nodes if n['domain']=='code')})")

    # Embed all nodes (context-enriched)
    log.info("\nEmbedding all nodes (context-enriched)...")
    texts = [n["context_text"] for n in nodes]
    embeddings = embed_texts(embed_model, texts, log, "combined")

    # Sequential edges
    log.info("\nBuilding sequential edges...")
    by_pattern = {}
    for n in nodes:
        pid = n["pattern_id"]
        if pid not in by_pattern:
            by_pattern[pid] = []
        by_pattern[pid].append(n)

    seq_edges = []
    for pid, pnodes in by_pattern.items():
        pnodes.sort(key=lambda x: x["step_index"])
        for i in range(len(pnodes) - 1):
            seq_edges.append({
                "src": pnodes[i]["node_id"], "dst": pnodes[i+1]["node_id"],
                "type": "sequential", "weight": 1.0,
            })
    log.info(f"Sequential edges: {len(seq_edges)}")

    # Semantic edges
    log.info(f"\nBuilding semantic edges (τ={tau})...")
    n = len(nodes)
    t0 = time.perf_counter()
    sim_matrix = embeddings @ embeddings.T
    sim_norm = (sim_matrix + 1.0) / 2.0

    sem_edges = []
    edge_set = set()
    pattern_of = {nd["node_id"]: nd["pattern_id"] for nd in nodes}
    domain_of  = {nd["node_id"]: nd["domain"] for nd in nodes}

    cross_domain_pairs = 0
    same_domain_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if pattern_of[i] == pattern_of[j]:
                continue
            if sim_norm[i, j] >= tau:
                pair = (i, j)
                if pair not in edge_set:
                    edge_set.add(pair)
                    w = float(sim_norm[i, j])
                    sem_edges.append({"src": i, "dst": j, "type": "semantic", "weight": w})
                    sem_edges.append({"src": j, "dst": i, "type": "semantic", "weight": w})
                    if domain_of[i] != domain_of[j]:
                        cross_domain_pairs += 1
                    else:
                        same_domain_pairs += 1

    elapsed = time.perf_counter() - t0
    log.info(f"Semantic edges: {len(sem_edges)} ({len(edge_set)} unique pairs) in {elapsed:.1f}s")
    log.info(f"  Same-domain pairs:  {same_domain_pairs}")
    log.info(f"  Cross-domain pairs: {cross_domain_pairs} (math↔code)")

    if edge_set:
        weights = [float(sim_norm[i, j]) for i, j in edge_set]
        log.info(f"  Sim range: [{min(weights):.3f}, {max(weights):.3f}], mean={np.mean(weights):.3f}")

    # Build adjacency + save
    all_edges = seq_edges + sem_edges
    adj = {nd["node_id"]: [] for nd in nodes}
    for e in all_edges:
        adj[e["src"]].append({"dst": e["dst"], "type": e["type"], "weight": e["weight"]})

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    degrees = [len(adj[nd["node_id"]]) for nd in nodes]

    graph_data = {
        "metadata": {
            "created":           datetime.now().isoformat(),
            "version":           "v2-combined-math-mbpp",
            "num_nodes":         len(nodes),
            "num_math_nodes":    sum(1 for nd in nodes if nd["domain"] == "math"),
            "num_mbpp_nodes":    sum(1 for nd in nodes if nd["domain"] == "code"),
            "num_sequential":    len(seq_edges),
            "num_semantic":      len(sem_edges),
            "num_edges_total":   len(all_edges),
            "cross_domain_pairs": cross_domain_pairs,
            "same_domain_pairs":  same_domain_pairs,
            "tau_semantic":      tau,
            "embed_model":       EMBED_MODEL,
            "embed_dim":         int(embeddings.shape[1]),
            "embed_format":      "context_enriched: problem_summary | step_text",
            "avg_degree":        float(np.mean(degrees)),
            "max_degree":        int(max(degrees)) if degrees else 0,
            "isolated_nodes":    sum(1 for d in degrees if d == 0),
        },
        "nodes": nodes,
        "adjacency": {str(k): v for k, v in adj.items()},
    }

    with open(GRAPH_JSON, "w") as f:
        json.dump(graph_data, f, indent=2)
    log.info(f"\nSaved graph → {GRAPH_JSON}")

    np.save(STEP_EMBED_NPY, embeddings)
    log.info(f"Saved embeddings → {STEP_EMBED_NPY} {embeddings.shape}")

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(STEP_FAISS_IDX))
    log.info(f"Saved FAISS → {STEP_FAISS_IDX} ({index.ntotal} vectors)")

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"COMBINED THOUGHT GRAPH SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  MATH nodes:          {sum(1 for nd in nodes if nd['domain']=='math')}")
    log.info(f"  MBPP nodes:          {sum(1 for nd in nodes if nd['domain']=='code')}")
    log.info(f"  Total nodes:         {len(nodes)}")
    log.info(f"  Sequential edges:    {len(seq_edges)}")
    log.info(f"  Semantic edges:      {len(sem_edges)} ({len(edge_set)} unique)")
    log.info(f"    Same-domain:       {same_domain_pairs}")
    log.info(f"    Cross-domain:      {cross_domain_pairs}")
    log.info(f"  Avg degree:          {np.mean(degrees):.1f}")
    log.info(f"  Max degree:          {max(degrees)}")

    return graph_data

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3C — Build MBPP Bank + Combined Graph")
    parser.add_argument("--tau", type=float, default=TAU_SEMANTIC_DEFAULT)
    parser.add_argument("--max-patterns", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3C — Build MBPP Memory Bank + Combined Graph")
    log.info("=" * 60)

    # Load MBPP patterns
    with open(MBPP_PATTERNS) as f:
        mbpp_records = json.load(f)
    log.info(f"MBPP patterns: {len(mbpp_records)}")

    # Load MATH-500 bank
    with open(MATH_BANK) as f:
        math_bank = json.load(f)
    log.info(f"MATH-500 bank: {len(math_bank)}")

    embed_model = load_embed_model(log)

    # Stage 1: MBPP bank
    mbpp_bank, _ = build_mbpp_bank(mbpp_records, embed_model, log)

    # Stage 2: Combined graph
    build_combined_graph(math_bank, mbpp_bank, embed_model, args.tau, args.max_patterns, log)

    # Cleanup
    del embed_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"\nDone! Next: run inject_and_eval_mbpp.py")


if __name__ == "__main__":
    main()
