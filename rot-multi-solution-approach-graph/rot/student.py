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
import threading
import time

from .config import read_jsonl, results_path, write_jsonl
from .llm import split_thinking_tokens
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

# Gated arms: {arm: (first-pass arm, injection mode for the second pass)}.
#
# WHY THIS ARM EXISTS. Unconditional injection nets to zero because it does two
# opposite things at once. Measured on D against A: it FIXED 7 of the 12
# problems the baseline got wrong -- 5 of them problems the baseline truncated
# on, where the retrieved approach ended the search and let the model finish --
# and it BROKE 7 of the 36 the baseline already had right. The breakages were
# not retrieval misses: they retrieved the correct family (3/3 same-family on
# most) from a bank that is 100% gold-verified. Handing a working solver someone
# else's strategy displaces the one it already had.
#
# So the injection should be conditional. Solve unaided first; inject only where
# the first pass visibly failed to produce an answer.
GATED_ARMS = {
    "E_gated": ("A_baseline", "approach_subtasks"),
}


def gate_needs_memory(row):
    """The gate signal. MUST be computable without the gold answer.

    `truncated` (hit max_tokens mid-trace) and `unparseable` (no ANSWER line)
    both say the same thing: the student did not produce an answer at all. Both
    are visible at inference time.

    It deliberately does NOT look at `correct`. Gating on correctness would be
    an oracle, and an oracle gate cannot lose -- it reproduces
    max(baseline, retrieval) on every problem by construction. It would post a
    large number that means nothing at all, because no deployable system can
    consult the gold answer to decide whether to retrieve.
    """
    return bool(row.get("truncated")) or bool(row.get("unparseable"))


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

    # See the note in teacher.generate: rows are appended as each problem
    # finishes, so a killed arm resumes instead of restarting.
    write_lock = threading.Lock()

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
        with write_lock:
            write_jsonl(out_path, [row], mode="a")
        return row

    client.map(todo, _one, desc=f"{arm}", log=log)

    allrows = read_jsonl(out_path)
    n_ok = sum(1 for r in allrows if r["correct"])
    log.info(f"  [{arm}] {n_ok}/{len(allrows)} correct "
             f"({n_ok / max(len(allrows), 1):.1%})")
    return allrows


def run_gated_arm(cfg, arm, client, test_problems, retriever, log, resume=True):
    """Second pass over a completed arm, run only where the gate fires.

    Cost accounting is the part that is easy to get wrong. A gated problem is
    generated TWICE, and this arm charges itself for both passes: its token and
    latency figures are pass1 + pass2. The gate buys accuracy with tokens, and
    a table that billed only the second pass would hide the price and make the
    arm look strictly better than the baseline when it is a trade.

    The outcome fields (predicted_answer, correct, truncated) come from pass 2,
    because that is the answer that stands.
    """
    src_arm, mode = GATED_ARMS[arm]
    src_path = results_path(cfg, f"eval_{src_arm}.jsonl")
    src_rows = read_jsonl(src_path)
    if not src_rows:
        raise FileNotFoundError(
            f"{arm} is a second pass over {src_arm}, but {src_path} is empty. "
            f"Run {src_arm} first.")
    pass1 = {r["problem_id"]: r for r in src_rows}
    missing = [p["problem_id"] for p in test_problems
               if p["problem_id"] not in pass1]
    if missing:
        raise ValueError(
            f"{src_arm} is missing {len(missing)} of this arm's test problems "
            f"({missing[:5]}). A gated arm needs a COMPLETE first pass, or the "
            f"gate silently skips the problems it cannot see.")

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

    fires = [p for p in todo if gate_needs_memory(pass1[p["problem_id"]])]
    keeps = [p for p in todo if not gate_needs_memory(pass1[p["problem_id"]])]
    log.info(f"  [{arm}] gate fires on {len(fires)}/{len(todo)} problems; "
             f"{len(keeps)} kept from {src_arm} unchanged")

    write_lock = threading.Lock()

    def _pass1_split(r):
        """Recompute pass 1's thinking/answer split.

        Rows written before the fix in llm.split_thinking_tokens booked an
        unterminated <think> as pure answer. Gated rows are exactly the
        truncated ones, so relying on the stored split here would import that
        bug into every single gated row.
        """
        return split_thinking_tokens(r.get("raw_reply") or "",
                                     r.get("output_tokens") or 0)

    # Problems the gate did not fire on: the first pass answer stands, at the
    # first pass price. No second call is made.
    for p in keeps:
        r1 = pass1[p["problem_id"]]
        t1, a1 = _pass1_split(r1)
        row = dict(r1)
        row.update({
            "arm": arm,
            "thinking_tokens": t1,
            "answer_tokens": a1,
            "gated": False,
            "gate_reason": None,
            "pass1_output_tokens": r1.get("output_tokens") or 0,
            "pass2_output_tokens": 0,
        })
        write_jsonl(out_path, [row], mode="a")

    def _one(p):
        r1 = pass1[p["problem_id"]]
        t1, a1 = _pass1_split(r1)
        reason = "truncated" if r1.get("truncated") else "unparseable"

        hits = retriever.retrieve(p["problem"], p["problem_id"], k=k)
        memory_block = format_memory(hits, mode, max_sub)
        user = build_user_prompt(p["problem"], memory_block)

        t0 = time.perf_counter()
        reply = client.chat(SYSTEM, user, n=1)[0]
        wall = time.perf_counter() - t0

        pred = extract_answer(reply.text)
        t2, a2 = reply.split_thinking()
        row = {
            "arm": arm,
            "problem_id": p["problem_id"],
            "group": p.get("group"),
            "problem": p["problem"],
            "gold_answer": p["answer"],
            # outcome: pass 2 is the answer that stands
            "predicted_answer": pred,
            "correct": is_correct(pred, p["answer"]),
            "unparseable": pred is None,
            "truncated": reply.truncated,
            "finish_reason": reply.finish_reason,
            "error": reply.error,
            "raw_reply": reply.text,
            "student_model": cfg["student"]["model"],
            "retrieval_mode": mode,
            # cost: BOTH passes
            "input_tokens": (r1.get("input_tokens") or 0) + reply.prompt_tokens,
            "output_tokens": (r1.get("output_tokens") or 0) + reply.completion_tokens,
            "thinking_tokens": t1 + t2,
            "answer_tokens": a1 + a2,
            "latency_s": round((r1.get("latency_s") or 0) + wall, 3),
            "gated": True,
            "gate_reason": reason,
            "pass1_output_tokens": r1.get("output_tokens") or 0,
            "pass2_output_tokens": reply.completion_tokens,
        }
        row.update(hit_metadata(hits, query_group=p.get("group")))
        with write_lock:
            write_jsonl(out_path, [row], mode="a")
        return row

    if fires:
        client.map(fires, _one, desc=f"{arm}", log=log)

    allrows = read_jsonl(out_path)
    n_ok = sum(1 for r in allrows if r["correct"])
    n_gated = sum(1 for r in allrows if r.get("gated"))
    log.info(f"  [{arm}] {n_ok}/{len(allrows)} correct "
             f"({n_ok / max(len(allrows), 1):.1%}); {n_gated} gated")
    return allrows
