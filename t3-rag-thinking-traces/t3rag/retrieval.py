"""Retrieval -- find the solved TRAIN problems most similar to a test problem.

Given a test problem, find the most useful memory from TRAIN problems.

Retrieval is two-step:

  1. Embed the test problem, score it against every bank problem.
  2. Drop anything that would leak, then take top-k.

Leakage guards are enforced in code and cannot be switched off in config:

  * the bank only ever contains train problems (asserted at build time)
  * an entry whose problem_id equals the query's is dropped
  * an entry whose normalised problem text equals the query's is dropped --
    catches a duplicate problem appearing under a different id

One entry per source problem, so the k examples are k different problems --
as in the paper, whose examples are distinct solved problems. random_memory is
the matching control: k examples chosen without looking at the query.
"""

from __future__ import annotations

import random
import re
import threading

import numpy as np


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class Retriever:
    def __init__(self, cfg, entries, entry_vecs, embedder, log=None):
        self.cfg = cfg
        self.entries = entries
        self.vecs = entry_vecs
        self.embedder = embedder
        self.log = log
        self._norm_problems = [_norm(e["problem"]) for e in entries]
        # The embedder is NOT thread-safe and the arms are evaluated with
        # `concurrency` worker threads. nomic-bert keeps a rotary cos/sin cache
        # on the module and regrows it in-place as sequence length changes, so
        # two threads encoding different-length problems at once corrupt it:
        # one thread reads a buffer sized for the other's sequence and the
        # forward dies with a shape mismatch. That is not hypothetical -- it
        # killed arm C 23 problems in. Encoding is milliseconds, so serialising
        # it costs nothing; the LLM calls stay fully concurrent.
        self._embed_lock = threading.Lock()
        # Every arm retrieves for the SAME test problems, so a second arm
        # would otherwise re-encode all 48 identically.
        self._qcache = {}
        if len(entries) != len(entry_vecs):
            raise ValueError(
                f"bank/vector mismatch: {len(entries)} entries vs "
                f"{len(entry_vecs)} vectors -- rebuild the graph")

    def _embed_query(self, problem: str):
        from .embedding import embed
        key = _norm(problem)
        with self._embed_lock:
            if key not in self._qcache:
                self._qcache[key] = embed(
                    self.cfg, self.embedder, [problem],
                    prefix=self.cfg["embedding"]["query_prefix"])[0]
            return self._qcache[key]

    def retrieve(self, problem: str, problem_id: str, k=None):
        """Return up to k scored entries, leakage-filtered."""
        k = k or self.cfg["retrieval"]["top_k"]
        if not self.entries:
            return []
        q = self._embed_query(problem)
        scores = self.vecs @ q

        qnorm = _norm(problem)
        order = np.argsort(-scores)
        hits, seen_problems = [], set()
        for i in order:
            e = self.entries[int(i)]
            if e["problem_id"] == problem_id:
                continue                      # guard 1: same id
            if self._norm_problems[int(i)] == qnorm:
                continue                      # guard 2: same text, other id
            if e["problem_id"] in seen_problems:
                continue                      # one entry per source problem
            seen_problems.add(e["problem_id"])
            hits.append({**e, "score": round(float(scores[int(i)]), 4)})
            if len(hits) >= k:
                break
        return hits

    def random_memory(self, problem_id: str, k=None, rng=None):
        """The control: k random entries, same leakage guards, no similarity.

        The control that separates "retrieval found the right memory" from
        "any worked example helps". Without it a gain over baseline cannot be
        attributed to retrieval at all.
        """
        k = k or self.cfg["retrieval"]["top_k"]
        rng = rng or random.Random(self.cfg["seed"])
        pool = [e for e in self.entries if e["problem_id"] != problem_id]
        if not pool:
            return []
        by_problem = {}
        for e in pool:
            by_problem.setdefault(e["problem_id"], []).append(e)
        pids = sorted(by_problem)
        rng.shuffle(pids)
        hits = []
        for pid in pids[:k]:
            e = by_problem[pid][0]
            hits.append({**e, "score": None})
        return hits


def hit_metadata(hits, query_group=None):
    """Provenance recorded with every evaluation row."""
    return {
        "retrieved_ids": [h["entry_id"] for h in hits],
        "retrieved_problem_ids": [h["problem_id"] for h in hits],
        "retrieval_scores": [h.get("score") for h in hits],
        "retrieved_groups": [h.get("group") for h in hits],
        # Diagnostic, never a filter. RB-DERIV families share one technique,
        # so "did retrieval land in the right family?" explains most of any
        # effect, and reporting it prevents a family-matching win being
        # mistaken for semantic understanding.
        "retrieved_same_group": [h.get("group") == query_group for h in hits],
        "n_retrieved": len(hits),
    }
