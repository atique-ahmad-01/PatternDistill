"""Stage 6 -- student evaluation.

Qwen3-1.7B answers each TEST problem under four conditions:

    A_baseline            problem only
    B_random_memory       problem + k random memories        (control)
    C_full_solution       problem + k retrieved full solutions
    D_approach_subtasks   problem + k retrieved approaches and steps

Everything except the injected block is held constant: same system prompt,
same decoding parameters, same answer contract. A token or accuracy difference
between arms is therefore attributable to the injection and not to prompt
formatting -- which is the only way the comparison means anything.

Arm B is the one that makes the result interpretable. Without it, a gain in C
or D could be "any worked example helps" rather than "retrieval found the right
memory", and those are entirely different claims.
"""

from __future__ import annotations

import random
import time

from .config import read_jsonl, results_path, write_jsonl
from .retrieval import format_memory, hit_metadata
from .scoring import extract_answer, is_correct

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

MEMORY_PREAMBLE = """Here is how similar problems were solved. The approaches may or may
not apply to the problem below -- judge that yourself. If an approach does not
fit, ignore it and solve the problem directly.

{memory}

Now solve this problem."""

ARM_MODES = {
    "A_baseline": None,
    "B_random_memory": "random",
    "C_full_solution": "full_solution",
    "D_approach_subtasks": "approach_subtasks",
}


def build_user_prompt(problem: str, memory_block: str) -> str:
    if not memory_block:
        return problem
    return MEMORY_PREAMBLE.format(memory=memory_block) + f"\n\n{problem}"


def run_arm(cfg, arm, client, test_problems, retriever, log, resume=True):
    """Evaluate one arm over the test set. Resumable per problem."""
    if arm not in ARM_MODES:
        raise ValueError(f"unknown arm {arm!r}; known: {list(ARM_MODES)}")
    mode = ARM_MODES[arm]
    out_path = results_path(cfg, f"eval_{arm}.jsonl")

    existing = read_jsonl(out_path) if resume else []
    have = {r["problem_id"] for r in existing}
    todo = [p for p in test_problems if p["problem_id"] not in have]
    if existing:
        log.info(f"  [{arm}] resuming: {len(have)} done, {len(todo)} to go")
    if not todo:
        log.info(f"  [{arm}] already complete ({len(existing)} rows)")
        return existing

    k = cfg["retrieval"]["top_k"]
    max_sub = cfg["retrieval"]["max_subtasks_shown"]
    rmode = cfg["retrieval"]["mode"]
    rng = random.Random(cfg["seed"])

    def _one(p):
        hits, memory_block = [], ""
        if mode == "random":
            hits = retriever.random_memory(p["problem_id"], k=k,
                                           rng=random.Random(
                                               cfg["seed"] + hash(p["problem_id"]) % 10000))
            memory_block = format_memory(hits, rmode, max_sub)
        elif mode is not None:
            hits = retriever.retrieve(p["problem"], p["problem_id"], k=k)
            memory_block = format_memory(hits, mode, max_sub)

        user = build_user_prompt(p["problem"], memory_block)
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
            "retrieval_mode": (mode if mode not in (None, "random") else mode),
            "input_tokens": reply.prompt_tokens,
            "output_tokens": reply.completion_tokens,
            # The headline quantity: reasoning tokens are where "searching for
            # the strategy" lives, so a retrieved approach should shrink this
            # specifically, not the answer.
            "thinking_tokens": think_tok,
            "answer_tokens": answer_tok,
            "latency_s": round(wall, 3),
            "finish_reason": reply.finish_reason,
            "truncated": reply.truncated,
            "error": reply.error,
            "raw_reply": reply.text,
        }
        row.update(hit_metadata(hits, query_group=p.get("group")))
        return row

    rows = client.map(todo, _one, desc=f"{arm}", log=log)
    write_jsonl(out_path, rows, mode="a")

    allrows = read_jsonl(out_path)
    n_ok = sum(1 for r in allrows if r["correct"])
    log.info(f"  [{arm}] {n_ok}/{len(allrows)} correct "
             f"({n_ok / max(len(allrows), 1):.1%})")
    return allrows
