"""Stage 1 -- teacher generation.

Qwen3-8B produces N independent solutions per TRAIN problem. Each solution is
parsed into an approach, an ordered list of mini-subtasks, and a final answer.

Two design choices worth stating, because the obvious alternatives are worse:

* N *independent samples*, not "give me 5 solutions in one reply". A single
  reply degrades after the second solution into reworded copies of the first,
  and one malformed reply loses all five. Independent samples also let a single
  bad solution be discarded without touching the others. vLLM shares the
  prefill across the n samples, so this costs barely more than one call.

* The gold is NEVER in the prompt. A gold-conditioned teacher rationalises
  backwards from the answer and writes an approach that cannot be reapplied to
  an unseen instance -- which is exactly what the student needs it for.

Output: data/teacher_solutions.jsonl, one record per solution, appended as it
is produced so an interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .config import read_jsonl, write_jsonl
from .scoring import extract_answer

SYSTEM = """You are an expert mathematician solving a hard problem.

The quantities involved are far too large to reach by step-by-step computation.
The problem is solvable only through a closed form, a cycle, a period, or an
equivalent structural shortcut. Find that shortcut.

Write your solution in exactly this format:

APPROACH: <the mathematical move, under 12 words>
STEP 1: <one meaningful reasoning action>
STEP 2: <one meaningful reasoning action>
STEP 3: <...as many steps as the method genuinely needs>
ANSWER: <integer>

Rules:
- Name the actual technique in APPROACH, not a generic action like "compute the
  value". "Josephus closed form" is an approach; "solve the problem" is not.
- Each STEP must be one logical sub-task, stated so it could be followed on a
  DIFFERENT problem of the same type.
- Give the final answer as plain decimal digits: no commas, no units, no LaTeX.
- Do not restate the problem. Do not write anything after the ANSWER line."""

# Nudges applied per sample index to push the samples apart. Requirement: the
# solutions should differ in reasoning strategy, not just in wording.
DIVERSITY_HINTS = [
    "",
    "\n\nSolve this using a different strategy than the most obvious one.",
    "\n\nLook for a periodicity, cycle, or modular structure you can exploit.",
    "\n\nDerive a closed form first, then substitute. Avoid iteration entirely.",
    "\n\nUse a counting, symmetry, or invariance argument if one applies.",
    "\n\nReframe the problem in terms of a standard named theorem or formula.",
]

_APPROACH = re.compile(
    r"^\s*\**\s*APPROACH\s*\**\s*[:\-]\s*\**\s*(.+?)\s*\**\s*$",
    re.IGNORECASE | re.MULTILINE)
_STEP = re.compile(
    r"^\s*\**\s*(?:STEP\s*(\d+)|(\d+)\s*[.)])\s*\**\s*[:\-]?\s*\**\s*(.+?)\s*\**\s*$",
    re.IGNORECASE | re.MULTILINE)


def parse_solution(text: str) -> dict:
    """Pull approach / subtasks / answer out of a teacher reply.

    Returns whatever it can find; `parse_ok` says whether the pieces the rest
    of the pipeline depends on are all present. Nothing is discarded here --
    filtering is a separate, configurable stage.
    """
    from .scoring import strip_thinking
    body = strip_thinking(text or "")

    m = _APPROACH.search(body)
    approach = m.group(1).strip() if m else ""
    approach = re.sub(r"\s+", " ", approach).rstrip(".")

    subtasks = []
    for m in _STEP.finditer(body):
        step = (m.group(3) or "").strip()
        step = re.sub(r"\s+", " ", step)
        if step and not step.upper().startswith("ANSWER"):
            subtasks.append(step)

    answer = extract_answer(body)

    return {
        "approach": approach,
        "subtasks": subtasks,
        "answer": answer,
        "parse_ok": bool(approach and len(subtasks) >= 1 and answer is not None),
    }


def build_prompt(problem: str, sample_index: int) -> str:
    hint = DIVERSITY_HINTS[sample_index % len(DIVERSITY_HINTS)]
    return f"{problem}{hint}"


def generate(cfg, client, problems, log, resume=True):
    """Generate N solutions for each problem. Returns the full record list."""
    from .config import data_path
    out_path = data_path(cfg, "teacher_solutions.jsonl")

    existing = read_jsonl(out_path) if resume else []
    have = {r["problem_id"] for r in existing}
    todo = [p for p in problems if p["problem_id"] not in have]

    if existing:
        log.info(f"Resuming: {len(have)} problems already generated, "
                 f"{len(todo)} to go")
    if not todo:
        log.info("Nothing to generate.")
        return existing

    n = cfg["teacher"]["n_solutions"]
    model = cfg["teacher"]["model"]

    def _one(problem):
        # One request per sample index, so each sample gets its own diversity
        # hint. Samples within an index still share a prefill.
        records = []
        for i in range(n):
            t0 = time.perf_counter()
            replies = client.chat(SYSTEM, build_prompt(problem["problem"], i), n=1)
            r = replies[0]
            parsed = parse_solution(r.text)
            records.append({
                "problem_id": problem["problem_id"],
                "problem": problem["problem"],
                "group": problem.get("group"),
                "gold_answer": problem["answer"],
                "solution_id": f"{problem['problem_id']}_s{i}",
                "sample_index": i,
                "approach": parsed["approach"],
                "subtasks": parsed["subtasks"],
                "n_subtasks": len(parsed["subtasks"]),
                "final_answer": parsed["answer"],
                "parse_ok": parsed["parse_ok"],
                "reasoning": r.text,
                "teacher_model": model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "thinking_tokens": r.split_thinking()[0],
                "latency_s": round(time.perf_counter() - t0, 3),
                "finish_reason": r.finish_reason,
                "truncated": r.truncated,
                "error": r.error,
            })
        return records

    results = client.map(todo, _one, desc="teacher", log=log)

    # Append per problem so a kill mid-run keeps everything already produced.
    for recs in results:
        write_jsonl(out_path, recs, mode="a")

    allrecs = read_jsonl(out_path)
    n_ok = sum(1 for r in allrecs if r["parse_ok"])
    n_trunc = sum(1 for r in allrecs if r.get("truncated"))
    n_err = sum(1 for r in allrecs if r.get("error"))
    log.info(f"Teacher: {len(allrecs)} solutions over "
             f"{len({r['problem_id'] for r in allrecs})} problems")
    log.info(f"  parsed cleanly: {n_ok}/{len(allrecs)}")
    if n_trunc:
        log.warning(f"  truncated (hit max_tokens): {n_trunc} "
                    f"-- consider raising teacher.max_tokens")
    if n_err:
        log.warning(f"  request errors: {n_err}")
    log.info(f"  -> {out_path}")
    return allrecs
