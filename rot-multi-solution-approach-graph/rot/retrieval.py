"""Stage 5 -- retrieval.

Given a test problem, find the most useful memory from TRAIN problems.

Retrieval is two-step:

  1. Embed the test problem, score it against every bank problem.
  2. Drop anything that would leak, then take top-k.

Leakage guards are enforced in code and cannot be switched off in config:

  * the bank only ever contains train problems (asserted at build time)
  * an entry whose problem_id equals the query's is dropped
  * an entry whose normalised problem text equals the query's is dropped --
    catches a duplicate problem appearing under a different id

Four presentation modes. They retrieve the SAME entries and differ only in what
is shown to the student, so a difference between arms C and D is a difference
in presentation, not in retrieval quality.
"""

from __future__ import annotations

import random
import re

import numpy as np

from .memory import _render_solution


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
        if len(entries) != len(entry_vecs):
            raise ValueError(
                f"bank/vector mismatch: {len(entries)} entries vs "
                f"{len(entry_vecs)} vectors -- rebuild the graph")

    def _embed_query(self, problem: str):
        from .graph import embed
        return embed(self.cfg, self.embedder, [problem],
                     prefix=self.cfg["embedding"]["query_prefix"])[0]

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
        """Arm B: k random entries, same leakage guards, no similarity.

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


# ── Presentation ─────────────────────────────────────────────────────────────

def format_memory(hits, mode, max_subtasks=8) -> str:
    """Render retrieved entries for the student prompt."""
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, 1):
        if mode == "full_solution":
            body = _render_solution(h["approach"], h["subtasks"][:max_subtasks])
            blocks.append(f"[Memory {i}] Problem: {h['problem']}\n{body}")
        elif mode == "approach":
            blocks.append(f"[Memory {i}] APPROACH: {h['approach']}")
        elif mode == "subtasks":
            steps = "\n".join(f"  - {s}" for s in h["subtasks"][:max_subtasks])
            blocks.append(f"[Memory {i}] Reasoning steps:\n{steps}")
        elif mode == "approach_subtasks":
            steps = "\n".join(f"  {j}. {s}" for j, s in
                              enumerate(h["subtasks"][:max_subtasks], 1))
            blocks.append(f"[Memory {i}] APPROACH: {h['approach']}\n"
                          f"Reasoning steps:\n{steps}")
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
    return "\n\n".join(blocks)


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
