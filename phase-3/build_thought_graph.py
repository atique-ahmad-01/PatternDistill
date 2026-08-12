"""
Phase 3 — Build Thought Graph v2 (ATGI)

Changes from v1:
  1. CONTEXT-ENRICHED EMBEDDINGS: each step is embedded as
     "{problem_summary} | {step_text}" instead of just step_text.
     This gives step embeddings the discriminative power of full-problem
     embeddings while retaining step-level granularity.

  2. Stores problem summary per node for richer context.

Input:  ../phase-2d/memory/bank.json  (400 patterns with Step 1/2/3/Note format)
Output: graph/thought_graph.json      (nodes + adjacency lists)
        graph/step_embeddings.npy     (N x 768 float32 matrix)
        graph/step_faiss.index        (FAISS IndexFlatIP for step retrieval)

Usage:
    python build_thought_graph.py                          # full build
    python build_thought_graph.py --tau 0.85               # stricter semantic edges
    python build_thought_graph.py --max-patterns 10        # smoke test
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
BANK_JSON = PHASE2D / "memory" / "bank.json"
GRAPH_DIR = ROOT / "graph"
LOGS_DIR  = ROOT / "logs"

GRAPH_JSON     = GRAPH_DIR / "thought_graph.json"
STEP_EMBED_NPY = GRAPH_DIR / "step_embeddings.npy"
STEP_FAISS_IDX = GRAPH_DIR / "step_faiss.index"

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBED_DIM   = 768
BATCH_SIZE  = 64

TAU_SEMANTIC_DEFAULT = 0.90   # semantic edge threshold (normalized cosine)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"build_thought_graph_{timestamp}.log"
    logger = logging.getLogger("build_graph")
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

STEP_PATTERN = re.compile(
    r"(Step\s*\d+\s*[:.]|Note\s*[:.])",
    re.IGNORECASE,
)


def parse_steps(pattern_text: str) -> list[dict]:
    if not pattern_text or not pattern_text.strip():
        return []

    splits = list(STEP_PATTERN.finditer(pattern_text))
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
    """Create a short summary of the problem for context-enriched embeddings."""
    # Take first max_len chars, cut at last space
    summary = problem[:max_len].strip()
    if len(problem) > max_len:
        last_space = summary.rfind(" ")
        if last_space > 60:
            summary = summary[:last_space]
        summary += "..."
    return summary


def build_nodes(bank: list, max_patterns: int | None, log) -> list[dict]:
    nodes = []
    node_id = 0
    patterns_used = 0
    skipped_empty = 0

    for entry in bank:
        if max_patterns and patterns_used >= max_patterns:
            break

        pattern_text = entry.get("pattern", "")
        steps = parse_steps(pattern_text)

        if not steps:
            skipped_empty += 1
            log.debug(f"  Skipped pattern_id={entry['index']} (no parseable steps)")
            continue

        problem_summary = make_problem_summary(entry["problem"])

        for step_idx, step in enumerate(steps):
            # Context-enriched text for embedding
            context_text = f"{problem_summary} | {step['text']}"

            nodes.append({
                "node_id":         node_id,
                "pattern_id":      entry["index"],
                "problem_id":      entry["id"],
                "step_index":      step_idx,
                "label":           step["label"],
                "text":            step["text"],
                "context_text":    context_text,
                "problem_summary": problem_summary,
                "subject":         entry["subject"],
                "level":           entry["level"],
                "total_steps":     len(steps),
            })
            node_id += 1

        patterns_used += 1

    log.info(f"Parsed {patterns_used} patterns → {len(nodes)} step nodes")
    if skipped_empty:
        log.warning(f"  Skipped {skipped_empty} patterns with no parseable steps")

    counts = [n["total_steps"] for n in nodes[::4]] if nodes else [0]
    log.info(f"  Steps/pattern: min={min(counts)}, max={max(counts)}, "
             f"avg={np.mean(counts):.1f}, median={np.median(counts):.0f}")

    by_subject = {}
    for n in nodes:
        s = n["subject"]
        by_subject[s] = by_subject.get(s, 0) + 1
    log.info(f"  By subject:")
    for subj, cnt in sorted(by_subject.items()):
        log.info(f"    {subj:<30} {cnt} nodes")

    return nodes


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_embed_model(log):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model


def embed_steps(model, nodes: list, log) -> np.ndarray:
    """Embed context-enriched step texts (problem_summary | step_text)."""
    texts = [f"search_document: {n['context_text']}" for n in nodes]
    all_vecs = []

    t0_total = time.perf_counter()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        t0 = time.perf_counter()
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        all_vecs.append(vecs)
        log.info(f"  Embedded batch {i // BATCH_SIZE + 1} "
                 f"({i + 1}–{min(i + len(batch), len(texts))}/{len(texts)}) "
                 f"{elapsed:.1f}s")

    total_time = time.perf_counter() - t0_total
    embeddings = np.vstack(all_vecs).astype(np.float32)
    log.info(f"Embedding complete: {embeddings.shape} in {total_time:.1f}s")
    return embeddings


# ── Edge construction ─────────────────────────────────────────────────────────

def build_sequential_edges(nodes: list, log) -> list[dict]:
    by_pattern = {}
    for n in nodes:
        pid = n["pattern_id"]
        if pid not in by_pattern:
            by_pattern[pid] = []
        by_pattern[pid].append(n)

    edges = []
    for pid, pnodes in by_pattern.items():
        pnodes.sort(key=lambda x: x["step_index"])
        for i in range(len(pnodes) - 1):
            edges.append({
                "src":    pnodes[i]["node_id"],
                "dst":    pnodes[i + 1]["node_id"],
                "type":   "sequential",
                "weight": 1.0,
            })

    log.info(f"Sequential edges: {len(edges)}")
    return edges


def build_semantic_edges(nodes: list, embeddings: np.ndarray,
                         tau: float, log) -> list[dict]:
    n = len(nodes)
    log.info(f"Computing semantic edges (τ={tau}, {n} nodes, "
             f"{n * (n - 1) // 2} pairs to check)...")

    t0 = time.perf_counter()
    sim_matrix = embeddings @ embeddings.T
    sim_norm = (sim_matrix + 1.0) / 2.0

    edges = []
    edge_set = set()
    pattern_of = {n_["node_id"]: n_["pattern_id"] for n_ in nodes}

    for i in range(n):
        for j in range(i + 1, n):
            if pattern_of[i] == pattern_of[j]:
                continue
            if sim_norm[i, j] >= tau:
                pair = (i, j)
                if pair not in edge_set:
                    edge_set.add(pair)
                    w = float(sim_norm[i, j])
                    edges.append({"src": i, "dst": j, "type": "semantic", "weight": w})
                    edges.append({"src": j, "dst": i, "type": "semantic", "weight": w})

    elapsed = time.perf_counter() - t0
    unique_pairs = len(edge_set)
    log.info(f"Semantic edges: {len(edges)} (from {unique_pairs} unique pairs) "
             f"in {elapsed:.1f}s")

    if edge_set:
        weights = [float(sim_norm[i, j]) for i, j in edge_set]
        log.info(f"  Similarity range: [{min(weights):.3f}, {max(weights):.3f}]")
        log.info(f"  Mean: {np.mean(weights):.3f}, Median: {np.median(weights):.3f}")

    cross_subj = sum(1 for i, j in edge_set if nodes[i]["subject"] != nodes[j]["subject"])
    same_subj = unique_pairs - cross_subj
    log.info(f"  Same-subject pairs: {same_subj}, Cross-subject pairs: {cross_subj}")

    return edges


# ── Save ──────────────────────────────────────────────────────────────────────

def build_adjacency(nodes: list, edges: list) -> dict[int, list]:
    adj = {n["node_id"]: [] for n in nodes}
    for e in edges:
        adj[e["src"]].append({
            "dst":    e["dst"],
            "type":   e["type"],
            "weight": e["weight"],
        })
    return adj


def save_graph(nodes: list, adj: dict, edges: list,
               embeddings: np.ndarray, tau: float, log):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    seq_count = sum(1 for e in edges if e["type"] == "sequential")
    sem_count = sum(1 for e in edges if e["type"] == "semantic")
    degrees = [len(adj[n["node_id"]]) for n in nodes]
    isolated = sum(1 for d in degrees if d == 0)

    graph_data = {
        "metadata": {
            "created":           datetime.now().isoformat(),
            "version":           "v2-context-enriched",
            "source":            str(BANK_JSON),
            "num_nodes":         len(nodes),
            "num_sequential":    seq_count,
            "num_semantic":      sem_count,
            "num_edges_total":   len(edges),
            "tau_semantic":      tau,
            "embed_model":       EMBED_MODEL,
            "embed_dim":         int(embeddings.shape[1]),
            "embed_format":      "context_enriched: problem_summary | step_text",
            "isolated_nodes":    isolated,
            "avg_degree":        float(np.mean(degrees)),
            "max_degree":        int(max(degrees)) if degrees else 0,
        },
        "nodes": nodes,
        "adjacency": {str(k): v for k, v in adj.items()},
    }

    with open(GRAPH_JSON, "w") as f:
        json.dump(graph_data, f, indent=2)
    log.info(f"Saved graph → {GRAPH_JSON} ({len(nodes)} nodes, {len(edges)} edges)")

    np.save(STEP_EMBED_NPY, embeddings)
    log.info(f"Saved embeddings → {STEP_EMBED_NPY} {embeddings.shape}")

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(STEP_FAISS_IDX))
    log.info(f"Saved FAISS index → {STEP_FAISS_IDX} ({index.ntotal} vectors)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build thought graph v2 (context-enriched)")
    parser.add_argument("--bank", type=Path, default=BANK_JSON)
    parser.add_argument("--tau", type=float, default=TAU_SEMANTIC_DEFAULT)
    parser.add_argument("--max-patterns", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3 — Build Thought Graph v2 (Context-Enriched)")
    log.info("=" * 60)
    log.info(f"Bank:         {args.bank}")
    log.info(f"τ_semantic:   {args.tau}")
    log.info(f"max_patterns: {args.max_patterns or 'all'}")
    log.info(f"Embed format: problem_summary | step_text")

    with open(args.bank) as f:
        bank = json.load(f)
    log.info(f"Loaded {len(bank)} entries from memory bank")

    log.info("\n── Step 1.1: Parsing patterns into steps ──")
    nodes = build_nodes(bank, args.max_patterns, log)
    if not nodes:
        log.error("No nodes parsed — check bank.json format")
        return

    log.info("\n── Step 1.2: Embedding steps (context-enriched) ──")
    embed_model = load_embed_model(log)
    embeddings  = embed_steps(embed_model, nodes, log)

    del embed_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("\n── Step 1.3: Building sequential edges ──")
    seq_edges = build_sequential_edges(nodes, log)

    log.info("\n── Step 1.4: Building semantic edges ──")
    sem_edges = build_semantic_edges(nodes, embeddings, args.tau, log)

    all_edges = seq_edges + sem_edges
    adj = build_adjacency(nodes, all_edges)

    log.info("\n── Step 1.5: Saving graph ──")
    save_graph(nodes, adj, all_edges, embeddings, args.tau, log)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("THOUGHT GRAPH v2 SUMMARY")
    log.info("=" * 60)
    log.info(f"  Nodes:            {len(nodes)}")
    log.info(f"  Sequential edges: {len(seq_edges)}")
    log.info(f"  Semantic edges:   {len(sem_edges)} "
             f"({len(sem_edges) // 2} unique pairs)")
    log.info(f"  Total edges:      {len(all_edges)}")
    log.info(f"  τ_semantic:       {args.tau}")

    degrees = [len(adj[n["node_id"]]) for n in nodes]
    log.info(f"  Avg degree:       {np.mean(degrees):.1f}")
    log.info(f"  Max degree:       {max(degrees)}")
    log.info(f"  Isolated nodes:   {sum(1 for d in degrees if d == 0)}")

    log.info(f"\n  Steps by subject:")
    by_subj = {}
    for n in nodes:
        by_subj.setdefault(n["subject"], []).append(n)
    for subj in sorted(by_subj):
        sn = by_subj[subj]
        log.info(f"    {subj:<30} {len(sn):>4} steps "
                 f"({len(set(n['pattern_id'] for n in sn)):>3} patterns)")

    log.info(f"\nOutputs:")
    log.info(f"  {GRAPH_JSON}")
    log.info(f"  {STEP_EMBED_NPY}")
    log.info(f"  {STEP_FAISS_IDX}")
    log.info(f"\nNext: run inject_and_eval_atgi.py")


if __name__ == "__main__":
    main()
