"""Student evaluation -- Qwen3-1.7B answers each TEST problem.

    A_baseline   problem only            (copied from the RoT project's f_exact)
    G_t3         problem + top-k retrieved examples, each the example problem
                 and its T3-transformed trace, under the paper's instruction
    R_random_t3  same, but k RANDOM examples                         (control)

The baseline is not re-run. It is the same 48 problems, the same student, the
same system prompt (SYSTEM below is verbatim from the run that produced it) and
the same decoding settings, so the injected examples are the only difference
between it and G_t3. Re-running it would only add sampling noise to the
comparison point every earlier RoT result was measured against.

R_random_t3 separates "retrieval found the right example" from "any worked
example helps". RB-DERIV families share one technique, so without it a G_t3
gain cannot be credited to retrieval.
"""

from __future__ import annotations

import random
import threading
import time
import zlib
from pathlib import Path

from .config import read_jsonl, results_path, write_jsonl
from .retrieval import hit_metadata
from .scoring import extract_answer, is_correct
from .t3 import build_user_prompt, format_examples

# Verbatim from rot-multi-solution-approach-graph/rot/student.py, which ran the
# baseline. Changing a word here changes the comparison.
SYSTEM = """You are solving one mathematics problem. The answer is a single integer.

The quantities involved are far too large to reach by step-by-step computation.
The problem is solvable only through a closed form, a cycle, a period, or an
equivalent structural shortcut.

Show only the reasoning you need. Do not restate the problem. Do not present
alternative solutions.

End your reply with exactly one line, and nothing after it:

ANSWER: <integer>

Write the integer in plain decimal digits: no commas, no spaces, no units, no
LaTeX, no words."""

ARMS = {"G_t3": "retrieved", "R_random_t3": "random"}


def run_arm(cfg, arm, client, test_problems, retriever, texts, log, resume=True):
    """Evaluate one arm over the test set. Resumable per problem."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {list(ARMS)} "
                         f"(A_baseline is copied, not run)")
    mode = ARMS[arm]
    out_path = results_path(cfg, f"eval_{arm}.jsonl")
    if not resume:
        # Rows are appended per problem, so a stale file must go or --no-resume
        # would add a second pass and the report would average the two.
        Path(out_path).unlink(missing_ok=True)
    existing = read_jsonl(out_path) if resume else []
    have = {r["problem_id"] for r in existing}
    todo = [p for p in test_problems if p["problem_id"] not in have]
    if existing:
        log.info(f"  [{arm}] resuming: {len(have)} done, {len(todo)} to go")
    if not todo:
        log.info(f"  [{arm}] already complete ({len(existing)} rows)")
        return existing

    k = cfg["retrieval"]["top_k"]
    variant = cfg["t3"]["variant"]
    log.info(f"  [{arm}] t3.variant={variant} top_k={k} "
             f"max_tok={cfg['student']['max_tokens']}")
    write_lock = threading.Lock()

    def _one(p):
        if mode == "random":
            # crc32, not hash(): str hashes are salted per process, so hash()
            # would draw different "random" examples on every resume.
            seed = cfg["seed"] + zlib.crc32(p["problem_id"].encode())
            hits = retriever.random_memory(p["problem_id"], k=k,
                                           rng=random.Random(seed))
        else:
            hits = retriever.retrieve(p["problem"], p["problem_id"], k=k)
        user = build_user_prompt(p["problem"], format_examples(hits, texts))

        t0 = time.perf_counter()
        reply = client.chat(SYSTEM, user, n=1)[0]
        wall = time.perf_counter() - t0

        pred = extract_answer(reply.text)
        think_tok, answer_tok = reply.split_thinking()
        row = {
            "arm": arm,
            "problem_id": p["problem_id"],
            "group": p.get("group"),
            "problem": p["problem"],
            "gold_answer": p["answer"],
            "predicted_answer": pred,
            "correct": is_correct(pred, p["answer"]),
            "unparseable": pred is None,
            "student_model": cfg["student"]["model"],
            "retrieval_mode": f"t3_{variant}_{mode}",
            "input_tokens": reply.prompt_tokens,
            "output_tokens": reply.completion_tokens,
            "thinking_tokens": think_tok,
            "answer_tokens": answer_tok,
            "latency_s": round(wall, 3),
            "finish_reason": reply.finish_reason,
            "truncated": reply.truncated,
            "error": reply.error,
            "raw_reply": reply.text,
            "user_prompt": user,
        }
        row.update(hit_metadata(hits, query_group=p.get("group")))
        with write_lock:
            write_jsonl(out_path, [row], mode="a")
        return row

    client.map(todo, _one, desc=arm, log=log)

    allrows = read_jsonl(out_path)
    n_ok = sum(1 for r in allrows if r["correct"])
    log.info(f"  [{arm}] {n_ok}/{len(allrows)} correct "
             f"({n_ok / max(len(allrows), 1):.1%})")
    return allrows
