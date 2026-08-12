"""
Phase 2B — Build Memory Bank

Reads the patterns generated in Phase 2A and builds a FAISS index for fast
cosine-similarity retrieval at inference time.

Each pattern is embedded using nomic-ai/nomic-embed-text-v1 via sentence-transformers
(no Ollama needed — runs directly on GPU/CPU like Phase 1).

The embedding input is the problem text — not the pattern — so that at eval time
we can query with a new problem and retrieve the most similar training problem's pattern.

Output:
  memory/bank.json    ← pattern metadata (id, subject, level, pattern, problem, ...)
  memory/bank.index   ← FAISS flat index (L2-normalised vectors → cosine sim)

Usage:
    python scripts/build_memory_bank.py
    python scripts/build_memory_bank.py --input data/patterns/patterns.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "patterns"
MEM_DIR  = ROOT / "memory"
LOGS_DIR = ROOT / "logs"

JSON_IN    = DATA_DIR / "patterns.json"
BANK_JSON  = MEM_DIR / "bank.json"
BANK_INDEX = MEM_DIR / "bank.index"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE  = 32

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = LOGS_DIR / f"build_memory_bank_{timestamp}.log"

    logger = logging.getLogger("build_bank")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log     : {log_file}")
    return logger

# ── Embedding ─────────────────────────────────────────────────────────────────

def load_embed_model(log: logging.Logger):
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    # trust_remote_code required for nomic-embed-text-v1
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model


def embed_texts(model, texts: list[str], log: logging.Logger) -> np.ndarray:
    """
    Embed texts in batches. Returns L2-normalised float32 ndarray (n, dim).
    nomic-embed-text requires a task prefix for best results.
    """
    import time
    # Use search_document prefix for the bank (problems being stored)
    prefixed = [f"search_document: {t}" for t in texts]

    all_vecs = []
    for i in range(0, len(prefixed), BATCH_SIZE):
        batch = prefixed[i : i + BATCH_SIZE]
        t0    = time.perf_counter()
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        all_vecs.append(vecs)
        log.info(
            f"  Embedded batch {i//BATCH_SIZE + 1} "
            f"({i+1}–{min(i+len(batch), len(texts))}/{len(texts)}) "
            f"in {elapsed:.1f}s"
        )

    return np.vstack(all_vecs).astype(np.float32)

# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(vectors: np.ndarray, log: logging.Logger):
    import faiss
    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on unit vectors = cosine sim
    index.add(vectors)
    log.info(f"  FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=JSON_IN,
                        help="Path to patterns.json")
    args = parser.parse_args()

    log = setup_logging()

    # Load patterns
    if not args.input.exists():
        raise FileNotFoundError(
            f"patterns.json not found at {args.input}. Run generate_patterns.py first."
        )

    with open(args.input) as f:
        records = json.load(f)

    log.info(f"Loaded {len(records)} records from {args.input}")

    valid   = [r for r in records if r.get("has_pattern") and r.get("pattern")]
    skipped = len(records) - len(valid)
    if skipped:
        log.warning(f"  Skipping {skipped} records with missing patterns")
    log.info(f"  Embedding {len(valid)} records")

    # Load embed model and embed
    embed_model = load_embed_model(log)

    log.info("\nEmbedding problems ...")
    texts   = [r["problem"] for r in valid]
    vectors = embed_texts(embed_model, texts, log)
    log.info(f"Vectors shape: {vectors.shape}")

    # Build FAISS index
    log.info("\nBuilding FAISS index ...")
    index = build_faiss_index(vectors, log)

    # Save
    MEM_DIR.mkdir(parents=True, exist_ok=True)

    bank_meta = []
    for i, r in enumerate(valid):
        bank_meta.append({
            "index":          i,
            "id":             r["id"],
            "problem":        r["problem"],
            "subject":        r["subject"],
            "level":          r["level"],
            "gold_answer":    r["gold_answer"],
            "cot":            r["cot"],
            "pattern":        r["pattern"],
            "teacher_answer": r["teacher_answer"],
        })

    with open(BANK_JSON, "w") as f:
        json.dump(bank_meta, f, indent=2)
    log.info(f"Saved metadata → {BANK_JSON}  ({len(bank_meta)} entries)")

    import faiss as faiss_module
    faiss_module.write_index(index, str(BANK_INDEX))
    log.info(f"Saved FAISS index → {BANK_INDEX}")

    log.info("\n── Memory Bank Summary ──────────────────────────────────────")
    log.info(f"  Total entries : {len(bank_meta)}")
    log.info(f"  Vector dim    : {vectors.shape[1]}")
    log.info(f"  Embed model   : {EMBED_MODEL}")
    log.info(f"  Index type    : IndexFlatIP (cosine sim via L2-norm)")

    by_subject = {}
    for r in bank_meta:
        s = r.get("subject", "unknown")
        by_subject[s] = by_subject.get(s, 0) + 1
    log.info("\n  By subject:")
    for subj, n in sorted(by_subject.items()):
        log.info(f"    {subj:<35} {n}")


if __name__ == "__main__":
    main()
