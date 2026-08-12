"""
Phase 2D — Build Memory Bank

Embeds all valid patterns from patterns.json into a FAISS IndexFlatIP index.
Stores worked_example and is_step_level for every subject (not just IA).

Usage:
    python build_memory_bank.py
    python build_memory_bank.py --input data/patterns/patterns.json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "patterns"
MEM_DIR  = ROOT / "memory"
LOGS_DIR = ROOT / "logs"

JSON_IN    = DATA_DIR / "patterns.json"
BANK_JSON  = MEM_DIR / "bank.json"
BANK_INDEX = MEM_DIR / "bank.index"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE  = 32


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
    logger.info(f"Log: {log_file}")
    return logger


def load_embed_model(log):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embed model: {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)
    log.info("Embed model loaded")
    return model


def embed_texts(model, texts: list, log) -> np.ndarray:
    import time
    prefixed = [f"search_document: {t}" for t in texts]
    all_vecs = []
    for i in range(0, len(prefixed), BATCH_SIZE):
        batch   = prefixed[i : i + BATCH_SIZE]
        t0      = time.perf_counter()
        vecs    = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        all_vecs.append(vecs)
        log.info(f"  Batch {i//BATCH_SIZE+1} ({i+1}–{min(i+len(batch), len(texts))}/{len(texts)}) {elapsed:.1f}s")
    return np.vstack(all_vecs).astype(np.float32)


def build_faiss_index(vectors: np.ndarray, log):
    import faiss
    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    log.info(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=JSON_IN)
    args = parser.parse_args()

    log = setup_logging()

    with open(args.input) as f:
        records = json.load(f)
    log.info(f"Loaded {len(records)} records from {args.input}")

    valid   = [r for r in records if r.get("has_pattern") and r.get("pattern")]
    skipped = len(records) - len(valid)
    if skipped:
        log.warning(f"Skipping {skipped} records with missing patterns")
    log.info(f"Embedding {len(valid)} records")

    embed_model = load_embed_model(log)
    texts       = [r["problem"] for r in valid]
    vectors     = embed_texts(embed_model, texts, log)

    index = build_faiss_index(vectors, log)

    MEM_DIR.mkdir(parents=True, exist_ok=True)

    bank_meta = []
    for i, r in enumerate(valid):
        bank_meta.append({
            "index":           i,
            "id":              r["id"],
            "problem":         r["problem"],
            "subject":         r["subject"],
            "level":           r["level"],
            "gold_answer":     r["gold_answer"],
            "cot":             r.get("cot", ""),
            "pattern":         r["pattern"],
            "worked_example":  r.get("worked_example", ""),
            "is_step_level":   r.get("is_step_level", False),
            "teacher_answer":  r.get("teacher_answer", ""),
        })

    with open(BANK_JSON, "w") as f:
        json.dump(bank_meta, f, indent=2)
    log.info(f"Saved metadata → {BANK_JSON}  ({len(bank_meta)} entries)")

    import faiss as faiss_module
    faiss_module.write_index(index, str(BANK_INDEX))
    log.info(f"Saved FAISS index → {BANK_INDEX}")

    by_subject = {}
    step_count = 0
    example_count = 0
    for r in bank_meta:
        s = r.get("subject", "unknown")
        by_subject[s] = by_subject.get(s, 0) + 1
        if r.get("is_step_level"):
            step_count += 1
        if r.get("worked_example"):
            example_count += 1

    log.info(f"\nMemory Bank Summary:")
    log.info(f"  Total entries    : {len(bank_meta)}")
    log.info(f"  Step-level       : {step_count}  ({100*step_count//len(bank_meta)}%)")
    log.info(f"  Has worked ex.   : {example_count}  ({100*example_count//len(bank_meta)}%)")
    log.info(f"  Vector dim       : {vectors.shape[1]}")
    log.info(f"\n  By subject:")
    for subj, n in sorted(by_subject.items()):
        log.info(f"    {subj:<35} {n}")

    log.info(f"\nNext: run inject_and_eval.py to evaluate injection")


if __name__ == "__main__":
    main()
