"""Embeddings -- the retriever's problem vectors and the query encoder.

The memory-bank vectors (data/artifacts/entry_embeddings.npy) are copied from
the RoT project, where they were built with this same encoder and prefix, so a
query embedded here scores against them correctly.
"""

from __future__ import annotations

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


def load_entry_vecs(cfg):
    return np.load(data_path(cfg, "entry_embeddings.npy"))
