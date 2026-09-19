"""RoT inference-time retrieval -- arXiv:2509.21743 Algorithm 1, as implemented
in the authors' `graph_handler.generate_dynamic_template`.

    1  candidates  <- template_type match AND knowledge-tag overlap
    2  first step  <- argmax  w_fq * cos(q, v) + w_fs0 * [v is step 0]
    3  traversal   <- argmax  w_tq * cos(q, v) + w_tf * [v is sequential child]
    4  stop when   len(T) >= l_max
                   OR no unvisited successors
                   OR (best cos(q,v) < thr AND best combined < thr)

FOUR PLACES WHERE THE PAPER'S TEXT AND THE AUTHORS' CODE DISAGREE. The code is
what produced the published numbers, so the code is what is implemented here.

  paper                              reference implementation
  ---------------------------------  ------------------------------------------
  sim~ = (cos+1)/2, tau = 0.85       raw cosine throughout, 0.85 for edges
  R_Initial = 0.8*R_Q + 0.2*R_S      0.75*sim + 0.25*s0          (VARS.py)
  R = R_Q + R_F                      0.3*sim + 0.7*flow          (graph_handler)
  halt when max(R) < tau             halt when sim < 0.8 AND combined < 0.8

The last one is the consequential difference. Under the paper's rule a
sequential step scores R = R_Q + 1 >= 1 > tau and the walk can never stop early;
under the code's rule the walk stops as soon as the next step has drifted from
the query, which is what makes the template query-specific rather than a verbatim
replay of one source solution.

SEQUENTIAL EDGES ALWAYS WIN THE ARGMAX, in the paper's formulation and in the
reference alike: a sequential child scores 0.7 + 0.3*sim while any semantic
neighbour scores at most 0.3. So traversal walks its starting template until the
steps stop matching the query, and only crosses into another template at a
template's final step. Recombination happens at the seams, not step by step --
worth knowing before reading `rot_crossed_templates`.
"""

from __future__ import annotations

import threading

import numpy as np


class RoTRetriever:
    def __init__(self, cfg, graph, vecs, embedder, log=None):
        self.cfg = cfg
        self.rc = cfg["rot"]
        self.nodes = graph["nodes"]
        self.adj = graph["adjacency"]
        self.seq = {int(k): v for k, v in graph["sequential"].items()}
        self.vecs = vecs
        self.embedder = embedder
        self.log = log
        self.thr = self.rc["sim_threshold"]
        self.l_max = self.rc["l_max"]
        self.w_fq, self.w_fs0 = self.rc["w_first_query"], self.rc["w_first_s0"]
        self.w_tq, self.w_tf = self.rc["w_trav_query"], self.rc["w_trav_flow"]
        # The embedder is not thread-safe and arms run with `concurrency`
        # workers; see the same guard in retrieval.Retriever.
        self._lock = threading.Lock()
        self._qcache = {}
        if len(self.nodes) != len(vecs):
            raise ValueError(f"graph/vector mismatch: {len(self.nodes)} nodes "
                             f"vs {len(vecs)} vectors -- rebuild the RoT graph")

    def _embed_query(self, text):
        from .rot_graph import embed
        with self._lock:
            if text not in self._qcache:
                q = embed(self.cfg, self.embedder, [text])[0]
                if q.shape[0] != self.vecs.shape[1]:
                    # Hit in practice by running --mock against a graph built
                    # with a real embedder: the query lands in a different
                    # space and every similarity is meaningless. Fail loudly
                    # rather than produce scored nonsense.
                    raise SystemExit(
                        f"embedding dimension mismatch: query is "
                        f"{q.shape[0]}-d but the RoT graph is "
                        f"{self.vecs.shape[1]}-d. The graph and this run were "
                        f"built with different embedders -- rebuild the graph "
                        f"with `build-rot-graph` under the same --mock setting.")
                self._qcache[text] = q
            return self._qcache[text]

    # ── step 1: candidate filter ─────────────────────────────────────────────
    def _candidates(self, query_tag, exclude_problem_id):
        """template_type match AND knowledge-tag overlap, then leakage guard.

        The reference requires an exact template_type match and then keeps a
        node if ANY query tag appears in its comma-separated knowledge_tag
        list. RB-DERIV carries one label that serves as both: all twelve
        instances of a family are solved by one technique, and the family is a
        property of the problem statement, never of its answer.

        NOTE FOR THE WRITE-UP. With tag_source=gold this arm is handed the
        query's family, which arms C and D had to find by embedding similarity
        alone. The reference does the same -- it annotates query metadata by
        hand -- but it is an advantage over the other arms and must be reported
        as one. tag_source=predicted puts it back on equal footing.
        """
        ok = []
        for i, n in enumerate(self.nodes):
            if n["problem_id"] == exclude_problem_id:
                continue                       # leakage guard
            if query_tag is not None and n["knowledge_tag"] != query_tag:
                continue
            ok.append(i)
        return ok

    def _query_tag(self, problem, query_group):
        src = self.rc["tag_source"]
        if src == "none":
            return None
        if src == "gold":
            return query_group
        if src == "predicted":
            q = self._embed_query(problem)
            return self.nodes[int(np.argmax(self.vecs @ q))]["knowledge_tag"]
        raise ValueError(f"unknown rot.tag_source {src!r}")

    # ── steps 2-4 ────────────────────────────────────────────────────────────
    def assemble(self, problem, problem_id, query_group=None):
        """Return (template_steps, metadata). Empty list if nothing qualifies."""
        tag = self._query_tag(problem, query_group)
        cand = self._candidates(tag, problem_id)
        meta = {
            "rot_query_tag": tag,
            "rot_n_candidates": len(cand),
            "rot_template_node_ids": [],
            "rot_template_templates": [],
            "rot_template_tags": [],
            "rot_step_scores": [],
            "rot_step_query_sims": [],
            "rot_edge_types": [],
            "rot_stop_reason": None,
            "rot_template_len": 0,
            "rot_n_source_templates": 0,
            "rot_crossed_templates": False,
        }
        if not cand:
            meta["rot_stop_reason"] = "no_candidates_at_start"
            return [], meta

        q = self._embed_query(problem)
        sim = self.vecs @ q                                 # raw cosine

        # step 2 -- first node.
        cand_arr = np.array(cand)
        s0 = np.array([1.0 if self.nodes[i]["step_index"] == 0 else 0.0
                       for i in cand], dtype=np.float32)
        first = self.w_fq * sim[cand_arr] + self.w_fs0 * s0
        pick = int(np.argmax(first))
        v0 = int(cand_arr[pick])

        T = [v0]
        cand_set = set(cand)
        meta["rot_step_scores"].append(round(float(first[pick]), 4))
        meta["rot_step_query_sims"].append(round(float(sim[v0]), 4))
        meta["rot_edge_types"].append("initial")

        # step 3 -- traversal.
        while len(T) < self.l_max:
            cur = T[-1]
            seq_child = self.seq.get(cur)
            nbrs = {}
            for e in self.adj.get(str(cur), []):
                b = e["to"]
                if b in cand_set and b not in T:
                    nbrs[b] = e["type"]
            if not nbrs:
                meta["rot_stop_reason"] = "no_successors"
                break

            ids = list(nbrs)
            flow = np.array([1.0 if b == seq_child else 0.0 for b in ids],
                            dtype=np.float32)
            comb = self.w_tq * sim[ids] + self.w_tf * flow
            best = int(np.argmax(comb))
            best_sim, best_comb = float(sim[ids[best]]), float(comb[best])

            # The reference's disjunctive guard: a step survives on query
            # alignment OR on flow. Requiring both would cut every template at
            # its first slightly-off step; requiring neither never stops.
            if best_sim < self.thr and best_comb < self.thr:
                meta["rot_stop_reason"] = "below_threshold"
                break

            v = ids[best]
            T.append(v)
            meta["rot_step_scores"].append(round(best_comb, 4))
            meta["rot_step_query_sims"].append(round(best_sim, 4))
            meta["rot_edge_types"].append(nbrs[v])
        else:
            meta["rot_stop_reason"] = "l_max"

        steps = [self.nodes[i]["text"] for i in T]
        tmpls = [self.nodes[i]["template_id"] for i in T]
        meta.update({
            "rot_template_node_ids": T,
            "rot_template_templates": tmpls,
            "rot_template_tags": [self.nodes[i]["knowledge_tag"] for i in T],
            "rot_template_len": len(T),
            "rot_n_source_templates": len(set(tmpls)),
            # If this is never True, traversal never left its starting solution
            # and the arm has degenerated into "show one retrieved solution",
            # which is arm D with extra machinery.
            "rot_crossed_templates": len(set(tmpls)) > 1,
        })
        return steps, meta


# ── step 5: prompt construction ──────────────────────────────────────────────

def format_template(steps) -> str:
    """Step numbering exactly as `llm_handler_qwen3.build_prompt` writes it."""
    import re
    out = []
    for i, step in enumerate(steps, 1):
        s = str(step).strip()
        if s.lower().startswith(f"step {i}"):
            out.append(s)
        elif re.match(r"^\d+\.", s):
            out.append(re.sub(r"^\d+\.", f"Step {i}:", s, count=1))
        else:
            out.append(f"Step {i}: {s}")
    return "\n".join(out)
