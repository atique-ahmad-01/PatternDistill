"""Stage 4 -- the approach / thought graph.

Deliberately small. Two node types and two edge types:

    nodes   approach  (one per memory entry)
            subtask   (one per reasoning step)

    edges   belongs_to  subtask -> its approach node        (structural)
            similar_to  node <-> node, cosine >= tau        (semantic)

Semantic edges are cross-problem only: an edge between two steps of the same
solution says nothing useful, it just restates that they were written together.

The graph is stored as plain JSON with an adjacency list, and vectors as a
.npy. There is no graph library and no database -- at this scale a dict of
lists is faster to read, faster to debug, and easier to swap out, which is what
"do not over-engineer" asks for. Retrieval reads these files and nothing else,
so the retrieval strategy can be replaced without touching this module.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import data_path


class MockEmbedder:
    """Deterministic hash-based embeddings for offline smoke tests.

    Produces stable, L2-normalised vectors with no model download and no
    network. Texts sharing vocabulary land near each other, which is enough to
    exercise edge building and top-k ordering -- but it is NOT semantic, so a
    retrieval quality measured with it means nothing. Runs are stamped mock.
    """

    DIM = 256

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import hashlib
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-9)
        return out


def load_embedder(cfg, log):
    if cfg.get("_mock"):
        log.warning("Embedding model: MOCK (hash-based, not semantic)")
        return MockEmbedder()
    from sentence_transformers import SentenceTransformer
    dev = cfg["embedding"]["device"]
    if dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
    log.info(f"Embedding model: {cfg['embedding']['model']} on {dev}")
    return SentenceTransformer(cfg["embedding"]["model"],
                               trust_remote_code=True, device=dev)


def embed(cfg, model, texts, prefix=None, log=None, label=""):
    """Encode texts to L2-normalised float32 vectors.

    Normalised, so an inner-product index is a cosine index. The prefix is the
    nomic-embed convention and must match between indexing and querying or
    every score is quietly wrong.
    """
    if prefix is None:
        prefix = cfg["embedding"]["doc_prefix"]
    bs = cfg["embedding"]["batch_size"]
    payload = [f"{prefix}{t}" for t in texts]
    out = []
    for i in range(0, len(payload), bs):
        out.append(model.encode(payload[i:i + bs], normalize_embeddings=True,
                                show_progress_bar=False))
    dim = out[0].shape[1] if out else 768
    vecs = np.vstack(out).astype(np.float32) if out else np.zeros((0, dim), "float32")
    if log:
        log.info(f"  embedded {label}: {vecs.shape}")
    return vecs


def build(cfg, entries, log):
    """Build nodes, vectors, and edges from memory-bank entries."""
    if not entries:
        raise ValueError("memory bank is empty -- run `main.py build-memory`")

    nodes = []
    for e in entries:
        a_id = len(nodes)
        nodes.append({
            "node_id": a_id,
            "type": "approach",
            "text": e["approach"],
            "entry_id": e["entry_id"],
            "problem_id": e["problem_id"],
            "group": e.get("group"),
            "parent_node": None,
            "step_index": None,
        })
        for si, st in enumerate(e["subtasks"]):
            nodes.append({
                "node_id": len(nodes),
                "type": "subtask",
                "text": st,
                "entry_id": e["entry_id"],
                "problem_id": e["problem_id"],
                "group": e.get("group"),
                "parent_node": a_id,
                "step_index": si,
            })

    log.info(f"Graph: {len(nodes)} nodes "
             f"({sum(1 for n in nodes if n['type']=='approach')} approach, "
             f"{sum(1 for n in nodes if n['type']=='subtask')} subtask)")

    model = load_embedder(cfg, log)
    # A subtask is embedded with its approach as context. In isolation "Apply
    # the formula" is near-meaningless and matches everything; with the
    # approach attached it carries the technique it belongs to.
    texts = []
    approach_by_node = {n["node_id"]: n for n in nodes if n["type"] == "approach"}
    for n in nodes:
        if n["type"] == "approach":
            texts.append(n["text"])
        else:
            ctx = approach_by_node[n["parent_node"]]["text"]
            texts.append(f"{ctx} | {n['text']}")
    vecs = embed(cfg, model, texts, log=log, label="nodes")

    # Structural edges.
    adjacency = defaultdict(list)
    for n in nodes:
        if n["parent_node"] is not None:
            adjacency[str(n["node_id"])].append(
                {"to": n["parent_node"], "type": "belongs_to", "score": 1.0})
            adjacency[str(n["parent_node"])].append(
                {"to": n["node_id"], "type": "has_step", "score": 1.0})

    # Semantic edges, within node type, cross-problem only.
    tau = cfg["graph"]["tau_semantic"]
    cap = cfg["graph"]["max_neighbors"]
    n_sem = 0
    for ntype, enabled in (("approach", cfg["graph"]["link_approaches"]),
                           ("subtask", cfg["graph"]["link_subtasks"])):
        if not enabled:
            continue
        idx = [i for i, n in enumerate(nodes) if n["type"] == ntype]
        if len(idx) < 2:
            continue
        sub = vecs[idx]
        sim = sub @ sub.T
        for a_pos, a in enumerate(idx):
            row = sim[a_pos]
            order = np.argsort(-row)
            added = 0
            for b_pos in order:
                b = idx[b_pos]
                if b == a:
                    continue
                if cfg["graph"]["cross_problem_only"] and \
                        nodes[b]["problem_id"] == nodes[a]["problem_id"]:
                    continue
                score = float(row[b_pos])
                if score < tau or added >= cap:
                    break
                adjacency[str(a)].append(
                    {"to": int(b), "type": "similar_to", "score": round(score, 4)})
                added += 1
                n_sem += 1

    cross_group = sum(
        1 for a, es in adjacency.items() for e in es
        if e["type"] == "similar_to"
        and nodes[int(a)].get("group") != nodes[e["to"]].get("group"))

    graph = {
        "metadata": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_nodes": len(nodes),
            "n_semantic_edges": n_sem,
            "n_cross_group_edges": cross_group,
            "tau_semantic": tau,
            "max_neighbors": cap,
            "embed_model": cfg["embedding"]["model"],
            "cross_problem_only": cfg["graph"]["cross_problem_only"],
        },
        "nodes": nodes,
        "adjacency": dict(adjacency),
    }

    gpath = data_path(cfg, "thought_graph.json")
    vpath = data_path(cfg, "node_embeddings.npy")
    Path(gpath).write_text(json.dumps(graph, ensure_ascii=False))
    np.save(vpath, vecs)

    # An index over bank problems, used to find similar TRAIN problems for a
    # test query. Separate from the node vectors on purpose: the query is a
    # problem, and problems match problems better than they match step text.
    entry_vecs = embed(cfg, model, [e["problem"] for e in entries],
                       log=log, label="bank problems")
    np.save(data_path(cfg, "entry_embeddings.npy"), entry_vecs)

    log.info(f"  semantic edges: {n_sem} ({cross_group} cross-family)")
    if n_sem == 0:
        log.warning(f"  no semantic edges at tau={tau} -- the graph adds "
                    f"nothing. Lower graph.tau_semantic.")
    log.info(f"  -> {gpath}")

    report = dict(graph["metadata"])
    report["n_entries_indexed"] = len(entries)
    return graph, vecs, report


def load(cfg):
    g = json.loads(Path(data_path(cfg, "thought_graph.json")).read_text())
    vecs = np.load(data_path(cfg, "node_embeddings.npy"))
    entry_vecs = np.load(data_path(cfg, "entry_embeddings.npy"))
    return g, vecs, entry_vecs
