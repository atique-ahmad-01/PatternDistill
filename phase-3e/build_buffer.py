"""
Phase 3E — Build Meta-Buffer (FAISS, no graph)

Loads existing patterns from phase-2d (MATH-500) and phase-3d (AIME),
builds a flat FAISS buffer over reasoning strategies. No graph, no
worked examples — strategy templates only.

Input:
  phase-2d/data/patterns/patterns.jsonl  (400 MATH-500)
  phase-3d/data/patterns/patterns.json   (746 AIME)

Output:
  phase-3e/buffer/buffer.json + buffer.index

Usage:
    python build_buffer.py
    python build_buffer.py --max-patterns 10
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).parent
PHASE2D = ROOT.parent / "phase-2d"
PHASE3D = ROOT.parent / "phase-3d"

MATH_PATTERNS = PHASE2D / "data" / "patterns" / "patterns.jsonl"
AIME_PATTERNS = PHASE3D / "data" / "patterns" / "patterns.json"

BUFFER_DIR   = ROOT / "buffer"
LOGS_DIR     = ROOT / "logs"
BUFFER_JSON  = BUFFER_DIR / "buffer.json"
BUFFER_INDEX = BUFFER_DIR / "buffer.index"

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
BATCH_SIZE  = 64


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = LOGS_DIR / f"build_buffer_{ts}.log"
    logger = logging.getLogger("build")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(lf)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {lf}")
    return logger


def load_all_patterns(log):
    patterns = []

    if MATH_PATTERNS.exists():
        with open(MATH_PATTERNS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not r.get("has_pattern") or not r.get("pattern"):
                    continue
                patterns.append({
                    "id": f"math_{r['id']}",
                    "source": "MATH-500",
                    "subject": r.get("subject", ""),
                    "problem": r["problem"],
                    "template": r["pattern"],
                    "gold_answer": r.get("gold_answer", ""),
                })
        log.info(f"MATH-500 patterns: {sum(1 for p in patterns if p['source']=='MATH-500')}")

    if AIME_PATTERNS.exists():
        with open(AIME_PATTERNS) as f:
            aime_recs = json.load(f)
        for r in aime_recs:
            if not r.get("has_pattern") or not r.get("pattern"):
                continue
            patterns.append({
                "id": f"aime_{r['id']}",
                "source": "AIME",
                "subject": "AIME",
                "problem": r["problem"],
                "template": r["pattern"],
                "gold_answer": r.get("gold_answer", ""),
                "year": r.get("year"),
            })
        log.info(f"AIME patterns: {sum(1 for p in patterns if p['source']=='AIME')}")

    log.info(f"Total patterns: {len(patterns)}")
    return patterns


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-patterns", type=int, default=None)
    args = parser.parse_args()

    log = setup_logging()
    log.info("=" * 60)
    log.info("Phase 3E — Build Meta-Buffer (no graph)")
    log.info("=" * 60)

    patterns = load_all_patterns(log)
    if args.max_patterns:
        patterns = patterns[:args.max_patterns]
        log.info(f"Capped at: {len(patterns)}")

    embed_model = load_embed_model(log)

    # Embed pattern strategies (not problem text)
    texts = [p["template"] for p in patterns]
    vectors = embed_texts(embed_model, texts, log, "templates")

    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    BUFFER_DIR.mkdir(parents=True, exist_ok=True)
    buffer = []
    for i, p in enumerate(patterns):
        buffer.append({
            "index": i,
            "id": p["id"],
            "source": p["source"],
            "subject": p.get("subject", ""),
            "year": p.get("year"),
            "problem": p["problem"],
            "template": p["template"],
            "gold_answer": p.get("gold_answer", ""),
        })

    with open(BUFFER_JSON, "w") as f:
        json.dump(buffer, f, indent=2)
    faiss.write_index(index, str(BUFFER_INDEX))

    n_math = sum(1 for b in buffer if b["source"] == "MATH-500")
    n_aime = sum(1 for b in buffer if b["source"] == "AIME")
    log.info(f"\nBuffer: {len(buffer)} templates ({n_math} MATH + {n_aime} AIME)")
    log.info(f"Saved -> {BUFFER_JSON}")
    log.info(f"Saved -> {BUFFER_INDEX}")
    log.info(f"\nNext: python inject_and_eval_atbs.py")


if __name__ == "__main__":
    main()
