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

from pathlib import Path

from .config import read_jsonl, results_path, write_jsonl
from .llm import split_thinking_continued, split_thinking_tokens
from .retrieval import format_memory, hit_metadata
from .rot_retrieval import format_template as format_rot_template
from .scoring import extract_answer, extract_answer_open_trace, is_correct

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

    if not resume:
        # --no-resume must START OVER, not add a second pass. Rows are appended
        # as each problem finishes, so leaving the old file in place produced
        # 2N rows for N problems -- silently, and the report then averaged a
        # run against itself. Truncate first.
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
    if not resume:
        # --no-resume must START OVER, not add a second pass. Rows are appended
        # as each problem finishes, so leaving the old file in place produced
        # 2N rows for N problems -- silently, and the report then averaged a
        # run against itself. Truncate first.
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


# ── RoT arm (arXiv:2509.21743) ───────────────────────────────────────────────

ROT_ARMS = {"F_rot"}

# Two variants of the reference's `build_prompt` preamble
# (vllm_experiments/src/llm_handler_qwen3.py).
#
# REFERENCE is verbatim, including its \boxed answer contract. Selected by
# rot.system_prompt=reference, together with the reference's own system message.
# That pair reproduces the published setup exactly and is what to run when the
# claim is "we reproduced arXiv:2509.21743".
#
# PROJECT drops only the contract sentence, so the arm ends with the ANSWER:
# line that SYSTEM asks of arms A-E. Use it when the claim is "RoT compared
# against our other arms" -- a different answer format across arms would put a
# parsing difference inside the comparison. scoring handles \boxed either way.
ROT_REFERENCE_TEMPLATE_PROMPT = (
    "Solve the math problem and always conclude with:\n"
    "Therefore, the final answer is: $\\boxed{answer}$. Where [answer] is just "
    "the final numerical answer that solves the problem. Ensure the number is "
    "clearly identifiable within the box.\n"
    "You are given a template to solve the problem. Use the given steps if "
    "applicable otherwise use them to guide your reasoning and present the "
    "solution steps logically to solve this problem:\n"
)

ROT_TEMPLATE_PROMPT = (
    "You are given a template to solve the problem. Use the given steps if "
    "applicable otherwise use them to guide your reasoning and present the "
    "solution steps logically to solve this problem:\n"
)

# Verbatim from the same function's thinking-intervention branch. The reference
# splices raw <|im_end|>/<|im_start|> markers into the user string and lets
# continue_final_message resume from there; this sends a real assistant message
# instead, which renders to the same tokens without hand-writing the chat
# template.
#
# The wording matters and is not decoration. "modify the steps to make them
# match the problem" is the only thing standing between the student and a
# template that names another problem's constants -- which is 37% of our step
# texts against the reference library's 0.2%.
ROT_PREFILL = (
    "<think>I need to follow the given steps to guide me in reasoning to solve "
    "the problem. I will follow each step and modify the steps to make them "
    "match the problem and then solve the problem accordingly. The template is. "
    "{template} Strictly following the given steps for guidance, I will now "
    "solve the problem starting from the step 1. Using step 1"
)

# The reference swaps the system prompt too. Ours is held constant across arms
# by design -- see this module's docstring -- so `project` is the default and
# `reference` exists for a strict reproduction.
ROT_REFERENCE_SYSTEM = (
    "You are asked to solve the mathematical problem using a step-by-step "
    "template. Follow each step exactly and output your reasoning at each step. "
    "Use only the steps given in the template for your reasoning"
)


def run_rot_arm(cfg, arm, client, test_problems, rot_retriever, log, resume=True):
    """RoT: assemble a template by graph traversal, inject it via TI.

    Differs from run_arm in two ways that matter for the comparison:

      * the template is built by reward-guided traversal over step nodes, so it
        can recombine steps from several source solutions into a path that
        exists in no single one of them;

      * it is injected as an already-open reasoning block that the model
        CONTINUES, rather than as reference material in the user turn.

    Everything else -- system prompt, decoding parameters, answer contract,
    scoring -- is identical to the other arms, so a difference is attributable
    to those two changes.
    """
    rc = cfg["rot"]
    out_path = results_path(cfg, f"eval_{arm}.jsonl")
    if not resume:
        # --no-resume must START OVER, not add a second pass. Rows are appended
        # as each problem finishes, so leaving the old file in place produced
        # 2N rows for N problems -- silently, and the report then averaged a
        # run against itself. Truncate first.
        Path(out_path).unlink(missing_ok=True)
    existing = read_jsonl(out_path) if resume else []
    have = {r["problem_id"] for r in existing}
    todo = [p for p in test_problems if p["problem_id"] not in have]
    if existing:
        log.info(f"  [{arm}] resuming: {len(have)} done, {len(todo)} to go")
    if not todo:
        log.info(f"  [{arm}] already complete ({len(existing)} rows)")
        return existing

    use_ti = rc["injection"] == "ti_prefill"
    exact = rc.get("system_prompt") == "reference"
    system = ROT_REFERENCE_SYSTEM if exact else SYSTEM
    preamble = ROT_REFERENCE_TEMPLATE_PROMPT if exact else ROT_TEMPLATE_PROMPT
    log.info(f"  [{arm}] injection={rc['injection']} tag_source={rc['tag_source']} "
             f"system={rc.get('system_prompt', 'project')} "
             f"templates={rc.get('template_source', 'retrieved')} "
             f"max_tok={cfg['student']['max_tokens']} "
             f"thr={rc['sim_threshold']} l_max={rc['l_max']}")
    # The control that makes a gain in this arm interpretable, and the exact
    # analogue of arm B for arms C/D. `shuffled` assembles each problem's
    # template from a DIFFERENT test problem's query, so the template is real,
    # coherent, and drawn from the same length and source-count distribution --
    # it is simply not the one retrieval chose for this problem. Without it,
    # "+4.2pp" cannot be separated from "any eight-step plan helps", which is a
    # much weaker claim.
    #
    # A derangement, so no problem can be handed its own template. The leakage
    # guard still keys off the REAL problem_id, so a donor template can never
    # contain the query's own solution.
    donor = {}
    if rc.get("template_source") == "shuffled":
        ids = [q["problem_id"] for q in test_problems]
        by_id = {q["problem_id"]: q for q in test_problems}
        if len(ids) < 2:
            raise SystemExit("rot.template_source=shuffled needs >= 2 test "
                             "problems to form a derangement")
        rng = random.Random(cfg["seed"])
        perm = list(ids)
        for _ in range(1000):
            rng.shuffle(perm)
            if all(a != b for a, b in zip(ids, perm)):
                break
        else:
            raise SystemExit("could not derange the test set")
        donor = {i: by_id[j] for i, j in zip(ids, perm)}
        log.info(f"  [{arm}] SHUFFLED control: templates reassigned across "
                 f"{len(donor)} problems")

    write_lock = threading.Lock()

    def _one(p):
        # The query the template is built FOR; the problem_id stays this
        # problem's so the leakage guard is unaffected.
        src = donor.get(p["problem_id"], p)
        steps, meta = rot_retriever.assemble(src["problem"], p["problem_id"],
                                             query_group=src.get("group"))
        meta["rot_donor_problem_id"] = (src["problem_id"]
                                        if src is not p else None)
        template = format_rot_template(steps)

        prefill, user = None, p["problem"]
        if template and use_ti:
            # The template is flattened to one line, as the reference does:
            # `template_str.replace('\n', ' ')`.
            user = f"{preamble}\nProblem: {p['problem']}."
            prefill = ROT_PREFILL.format(template=template.replace("\n", " ") + " ")
        elif template:
            # Ablation path: same template, delivered as user-turn reference
            # material -- the reference's own non-TI branch. Kept because this
            # project's phase-3B found <think> injection did nothing on a 1.7B
            # while prompt injection helped, which contradicts the paper.
            user = (f"{preamble}{template}\n\n"
                    f"Problem: {p['problem']}")

        t0 = time.perf_counter()
        reply = client.chat(system, user, n=1, prefill=prefill)[0]
        wall = time.perf_counter() - t0

        # The prefill is not echoed back, so parsing and the thinking split
        # both have to account for a trace that was already open.
        if prefill:
            full_text = prefill + reply.text
            think_tok, answer_tok = split_thinking_continued(
                reply.text, reply.completion_tokens)
        else:
            full_text = reply.text
            think_tok, answer_tok = reply.split_thinking()

        pred = (extract_answer_open_trace(reply.text, reply.truncated)
                if prefill else extract_answer(full_text))
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
            "retrieval_mode": f"rot_{rc['injection']}",
            "input_tokens": reply.prompt_tokens,
            "output_tokens": reply.completion_tokens,
            "thinking_tokens": think_tok,
            "answer_tokens": answer_tok,
            "latency_s": round(wall, 3),
            "finish_reason": reply.finish_reason,
            "truncated": reply.truncated,
            "error": reply.error,
            "raw_reply": full_text,
            # raw_reply is prefill + generation. report.summarize recomputes the
            # thinking split from raw_reply, and it MUST skip these characters:
            # they are prompt, not completion, so counting them as reasoning
            # inflates thinking_tokens toward output_tokens and drives
            # answer_tokens to zero. A reply that closes </think> immediately
            # inverts completely without this.
            "prefill_chars": len(prefill) if prefill else 0,
            "rot_template": template,
        }
        row.update(meta)
        # Same-family accounting, so this arm lines up with the others in the
        # report. Every template step carries its source template's tag.
        tags = meta.get("rot_template_tags") or []
        row["retrieved_same_group"] = [t == p.get("group") for t in tags]
        row["n_retrieved"] = meta.get("rot_n_source_templates", 0)
        row["retrieved_ids"] = meta.get("rot_template_templates") or []
        with write_lock:
            write_jsonl(out_path, [row], mode="a")
        return row

    client.map(todo, _one, desc=f"{arm}", log=log)

    allrows = read_jsonl(out_path)
    n_ok = sum(1 for r in allrows if r["correct"])
    crossed = sum(1 for r in allrows if r.get("rot_crossed_templates"))
    mean_len = (sum(r.get("rot_template_len", 0) for r in allrows)
                / max(len(allrows), 1))
    log.info(f"  [{arm}] {n_ok}/{len(allrows)} correct "
             f"({n_ok / max(len(allrows), 1):.1%}); "
             f"mean template {mean_len:.1f} steps; "
             f"{crossed}/{len(allrows)} templates crossed >1 source solution")
    if crossed == 0:
        log.warning(f"  [{arm}] NO template ever left its starting solution. "
                    f"Traversal followed only sequential edges, so this arm is "
                    f"replaying one retrieved solution -- not recombining. "
                    f"Check rot.tau_semantic and the semantic edge count.")
    return allrows
