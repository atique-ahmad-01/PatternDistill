"""
Phase 3D — Build AIME Memory Bank + Combined Thought Graph (MATH-500 + AIME)

Both datasets are competition math → high cross-dataset transfer expected.
MATH-500 patterns (Algebra, NT, Precalc, etc.) should directly help AIME.

Input:
  phase-3d/data/patterns/patterns.json   (400 AIME patterns)
  phase-2d/memory/bank.json              (400 MATH-500 patterns)

Output:
  phase-3d/memory/bank.json + bank.index  (AIME-only memory bank)
  phase-3d/graph/thought_graph.json       (combined MATH+AIME graph)
  phase-3d/graph/step_embeddings.npy
  phase-3d/graph/step_faiss.index

Usage:
    python build_memory_and_graph.py
    python build_memory_and_graph.py --max-patterns 10
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent
PHASE2D   = ROOT.parent / "phase-2d"

AIME_PATTERNS = ROOT / "data" / "patterns" / "patterns.json"
MATH_BANK     = PHASE2D / "memory" / "bank.json"

MEM_DIR    = ROOT / "memory"
GRAPH_DIR  = ROOT / "graph"
LOGS_DIR   = ROOT / "logs"

BANK_JSON      = MEM_DIR / "bank.json"
BANK_INDEX     = MEM_DIR / "bank.index"
GRAPH_JSON     = GRAPH_DIR / "thought_graph.json"
STEP_EMBED_NPY = GRAPH_DIR / "step_embeddings.npy"
STEP_FAISS_IDX = GRAPH_DIR / "step_faiss.index"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE  = 64
TAU_SEMANTIC_DEFAULT = 0.90


def setup_logging():
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


STEP_RE = re.compile(r"(Step\s*\d+\s*[:.]|Note\s*[:.])", re.IGNORECASE)

def parse_steps(text):
    if not text or not text.strip():
        return []
    splits = list(STEP_RE.finditer(text))
    if not splits:
        return [{"label": "Step 1", "text": text.strip()}]
    steps = []
    for i, m in enumerate(splits):
        label = m.group(0).rstrip(":. ").strip()
        start = m.end()
        end = splits[i+1].start() if i+1 < len(splits) else len(text)
        t = text[start:end].strip()
        if t:
            steps.append({"label": label, "text": t})
    return steps

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
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model

def embed_texts(model, texts, log, label=""):
    prefixed = [f"search_document: {t}" for t in texts]
    all_vecs = []
    t0 = time.perf_counter()
    for i in range(0, len(prefixed), BATCH_SIZE):
        batch = prefixed[i:i+BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)
    total = time.perf_counter() - t0
    result = np.vstack(all_vecs).astype(np.float32)
    log.info(f"  {label}: {result.shape} in {total:.1f}s")
    return result


def build_aime_bank(records, embed_model, log):
    log.info(f"\n{'='*60}")
    log.info(f"Stage 1: Building AIME Memory Bank")
    log.info(f"{'='*60}")
    valid = [r for r in records if r.get("has_pattern") and r.get("pattern")]
    log.info(f"Valid: {len(valid)}/{len(records)}")

    texts = [r["problem"] for r in valid]
    vectors = embed_texts(embed_model, texts, log, "AIME problems")

    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    MEM_DIR.mkdir(parents=True, exist_ok=True)
    bank = []
    for i, r in enumerate(valid):
        bank.append({
            "index": i, "id": r["id"], "problem": r["problem"],
            "domain": "math", "subject": "AIME", "year": r.get("year"),
            "gold_answer": r.get("gold_answer", ""),
            "cot": r.get("cot", ""), "pattern": r["pattern"],
            "worked_example": r.get("worked_example", ""),
            "is_step_level": r.get("is_step_level", False),
            "teacher_answer": r.get("teacher_answer", ""),
        })

    with open(BANK_JSON, "w") as f:
        json.dump(bank, f, indent=2)
    log.info(f"Saved AIME bank → {BANK_JSON} ({len(bank)} entries)")
    faiss.write_index(index, str(BANK_INDEX))
    log.info(f"Saved FAISS → {BANK_INDEX}")
    return bank, vectors


def build_combined_graph(math_bank, aime_bank, embed_model, tau, max_patterns, log):
    log.info(f"\n{'='*60}")
    log.info(f"Stage 2: Combined Thought Graph (MATH-500 + AIME)")
    log.info(f"{'='*60}")

    nodes = []
    node_id = 0

    # MATH-500 nodes
    mc = 0
    for entry in math_bank:
        if max_patterns and mc >= max_patterns:
            break
        steps = parse_steps(entry.get("pattern", ""))
        if not steps:
            continue
        summary = make_summary(entry["problem"])
        for si, step in enumerate(steps):
            nodes.append({
                "node_id": node_id, "pattern_id": f"math_{entry['index']}",
                "problem_id": entry["id"], "step_index": si,
                "label": step["label"], "text": step["text"],
                "context_text": f"{summary} | {step['text']}",
                "problem_summary": summary,
                "domain": "math_500", "subject": entry.get("subject", "MATH-500"),
                "total_steps": len(steps),
            })
            node_id += 1
        mc += 1

    # AIME nodes
    ac = 0
    for entry in aime_bank:
        if max_patterns and ac >= max_patterns:
            break
        steps = parse_steps(entry.get("pattern", ""))
        if not steps:
            continue
        summary = make_summary(entry["problem"])
        for si, step in enumerate(steps):
            nodes.append({
                "node_id": node_id, "pattern_id": f"aime_{entry['index']}",
                "problem_id": entry["id"], "step_index": si,
                "label": step["label"], "text": step["text"],
                "context_text": f"{summary} | {step['text']}",
                "problem_summary": summary,
                "domain": "aime", "subject": "AIME",
                "total_steps": len(steps),
            })
            node_id += 1
        ac += 1

    n_math = sum(1 for n in nodes if n["domain"] == "math_500")
    n_aime = sum(1 for n in nodes if n["domain"] == "aime")
    log.info(f"Total nodes: {len(nodes)} (MATH-500: {n_math}, AIME: {n_aime})")

    # Embed
    texts = [n["context_text"] for n in nodes]
    embeddings = embed_texts(embed_model, texts, log, "combined")

    # Sequential edges
    by_pat = {}
    for n in nodes:
        pid = n["pattern_id"]
        if pid not in by_pat:
            by_pat[pid] = []
        by_pat[pid].append(n)

    seq_edges = []
    for pid, pn in by_pat.items():
        pn.sort(key=lambda x: x["step_index"])
        for i in range(len(pn) - 1):
            seq_edges.append({"src": pn[i]["node_id"], "dst": pn[i+1]["node_id"],
                              "type": "sequential", "weight": 1.0})
    log.info(f"Sequential edges: {len(seq_edges)}")

    # Semantic edges
    log.info(f"Building semantic edges (τ={tau})...")
    n = len(nodes)
    t0 = time.perf_counter()
    sim = embeddings @ embeddings.T
    sim_norm = (sim + 1.0) / 2.0

    sem_edges = []
    edge_set = set()
    pat_of = {nd["node_id"]: nd["pattern_id"] for nd in nodes}
    dom_of = {nd["node_id"]: nd["domain"] for nd in nodes}
    cross = 0
    same = 0

    for i in range(n):
        for j in range(i+1, n):
            if pat_of[i] == pat_of[j]:
                continue
            if sim_norm[i, j] >= tau:
                if (i, j) not in edge_set:
                    edge_set.add((i, j))
                    w = float(sim_norm[i, j])
                    sem_edges.append({"src": i, "dst": j, "type": "semantic", "weight": w})
                    sem_edges.append({"src": j, "dst": i, "type": "semantic", "weight": w})
                    if dom_of[i] != dom_of[j]:
                        cross += 1
                    else:
                        same += 1

    elapsed = time.perf_counter() - t0
    log.info(f"Semantic edges: {len(sem_edges)} ({len(edge_set)} unique) in {elapsed:.1f}s")
    log.info(f"  Same-domain:   {same}")
    log.info(f"  Cross-domain:  {cross} (MATH-500 ↔ AIME)")

    if edge_set:
        weights = [float(sim_norm[i, j]) for i, j in edge_set]
        log.info(f"  Sim range: [{min(weights):.3f}, {max(weights):.3f}], mean={np.mean(weights):.3f}")

    # Save
    all_edges = seq_edges + sem_edges
    adj = {nd["node_id"]: [] for nd in nodes}
    for e in all_edges:
        adj[e["src"]].append({"dst": e["dst"], "type": e["type"], "weight": e["weight"]})

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    degrees = [len(adj[nd["node_id"]]) for nd in nodes]

    graph_data = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "version": "v2-combined-math500-aime",
            "num_nodes": len(nodes), "num_math500_nodes": n_math,
            "num_aime_nodes": n_aime, "num_sequential": len(seq_edges),
            "num_semantic": len(sem_edges), "num_edges_total": len(all_edges),
            "cross_domain_pairs": cross, "same_domain_pairs": same,
            "tau_semantic": tau, "embed_model": EMBED_MODEL,
            "embed_dim": int(embeddings.shape[1]),
            "embed_format": "context_enriched: problem_summary | step_text",
            "avg_degree": float(np.mean(degrees)),
            "max_degree": int(max(degrees)) if degrees else 0,
            "isolated_nodes": sum(1 for d in degrees if d == 0),
        },
        "nodes": nodes,
        "adjacency": {str(k): v for k, v in adj.items()},
    }

    with open(GRAPH_JSON, "w") as f:
        json.dump(graph_data, f, indent=2)
    log.info(f"Saved graph → {GRAPH_JSON}")

    np.save(STEP_EMBED_NPY, embeddings)
    log.info(f"Saved embeddings → {STEP_EMBED_NPY} {embeddings.shape}")

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(STEP_FAISS_IDX))
    log.info(f"Saved FAISS → {STEP_FAISS_IDX}")

    log.info(f"\n{'='*60}")
    log.info(f"COMBINED GRAPH SUMMARY (MATH-500 + AIME)")
    log.info(f"{'='*60}")
    log.info(f"  MATH-500 nodes:  {n_math}")
    log.info(f"  AIME nodes:      {n_aime}")
    log.info(f"  Total:           {len(nodes)}")
    log.info(f"  Sequential:      {len(seq_edges)}")
    log.info(f"  Semantic:        {len(sem_edges)} ({len(edge_set)} unique)")
    log.info(f"    Same-domain:   {same}")
    log.info(f"    Cross-domain:  {cross} (MATH-500 ↔ AIME)")
    log.info(f"  Avg degree:      {np.mean(degrees):.1f}")
    log.info(f"  Max degree:      {max(degrees)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=TAU_SEMANTIC_DEFAULT)
    parser.add_argument("--max-patterns", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3D — AIME Bank + Combined Graph (MATH-500 + AIME)")
    log.info("=" * 60)

    with open(AIME_PATTERNS) as f:
        aime_records = json.load(f)
    log.info(f"AIME patterns: {len(aime_records)}")

    with open(MATH_BANK) as f:
        math_bank = json.load(f)
    log.info(f"MATH-500 bank: {len(math_bank)}")

    embed_model = load_embed_model(log)
    aime_bank, _ = build_aime_bank(aime_records, embed_model, log)
    build_combined_graph(math_bank, aime_bank, embed_model, args.tau, args.max_patterns, log)

    del embed_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(f"\nDone! Next: run inject_and_eval_aime.py")


if __name__ == "__main__":
    main()
