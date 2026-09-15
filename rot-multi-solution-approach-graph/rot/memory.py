"""Stage 3 -- the RoT memory bank.

Structure, preserved explicitly rather than flattened:

    problem  ->  solution  ->  approach
                          ->  subtasks[]  (ordered reasoning steps)

Every node keeps the ids of its parents, so any retrieved fragment can be
traced back to the solution and the problem it came from. That traceability is
what makes it possible to check, after the fact, whether a win came from the
retrieved content or from an accident of the split.

Leakage rule, enforced here and not negotiable: the bank is built ONLY from
train problems. `build` refuses to run if a test id reaches it.
"""

from __future__ import annotations

from collections import defaultdict

from .config import data_path, read_jsonl, write_jsonl


def build(cfg, log, train_ids=None):
    src = data_path(cfg, "teacher_solutions_validated.jsonl")
    rows = read_jsonl(src)
    if not rows:
        raise FileNotFoundError(
            f"no validated solutions at {src} -- run `main.py validate` first")

    if train_ids is not None:
        train_ids = set(train_ids)
        leaked = {r["problem_id"] for r in rows} - train_ids
        if leaked:
            raise AssertionError(
                f"teacher solutions contain {len(leaked)} non-train problems "
                f"({sorted(leaked)[:5]}). The memory bank must never see a "
                f"test problem's solution. Regenerate with the train split.")

    include = cfg["memory"]["include"]
    lo = cfg["memory"]["min_subtasks"]
    hi = cfg["memory"]["max_subtasks"]

    kept, skipped = [], defaultdict(int)
    for r in rows:
        if not r.get("parse_ok"):
            skipped["unparseable"] += 1
            continue
        if include == "correct_only" and not r.get("correct"):
            skipped["incorrect"] += 1
            continue
        if r["n_subtasks"] < lo:
            skipped["too_few_subtasks"] += 1
            continue
        kept.append(r)

    entries = []
    for r in kept:
        subtasks = r["subtasks"][:hi]
        entries.append({
            "entry_id": r["solution_id"],
            "problem_id": r["problem_id"],
            "problem": r["problem"],
            "group": r.get("group"),
            "gold_answer": r["gold_answer"],
            "approach": r["approach"],
            "subtasks": subtasks,
            "n_subtasks": len(subtasks),
            "final_answer": r["final_answer"],
            "correct": r.get("correct", False),
            "teacher_model": r.get("teacher_model"),
            "solution_text": _render_solution(r["approach"], subtasks),
        })

    out = data_path(cfg, "memory_bank.jsonl")
    write_jsonl(out, entries)

    by_problem = defaultdict(int)
    for e in entries:
        by_problem[e["problem_id"]] += 1
    by_group = defaultdict(int)
    for e in entries:
        by_group[str(e.get("group"))] += 1

    report = {
        "include_policy": include,
        "n_solutions_considered": len(rows),
        "n_entries": len(entries),
        "n_problems_represented": len(by_problem),
        "skipped": dict(skipped),
        "mean_entries_per_problem": round(
            len(entries) / max(len(by_problem), 1), 2),
        "total_subtasks": sum(e["n_subtasks"] for e in entries),
        "by_group": dict(sorted(by_group.items())),
    }

    log.info(f"Memory bank: {len(entries)} entries from "
             f"{len(by_problem)} problems (policy={include})")
    if skipped:
        log.info(f"  skipped: {dict(skipped)}")
    log.info(f"  subtask nodes available: {report['total_subtasks']}")

    empty = [g for g, c in by_group.items() if c == 0]
    absent = set()
    if train_ids is not None:
        absent = {str(r.get("group")) for r in rows} - set(by_group)
    if absent:
        log.warning(f"  families with no bank entry: {sorted(absent)} "
                    f"-- nothing can be retrieved for them")
    report["groups_absent"] = sorted(absent)

    log.info(f"  -> {out}")
    return entries, report


def _render_solution(approach: str, subtasks: list) -> str:
    """The full-solution form shown to the student in arm C."""
    lines = [f"APPROACH: {approach}"]
    for i, s in enumerate(subtasks, 1):
        lines.append(f"STEP {i}: {s}")
    return "\n".join(lines)


def load(cfg):
    return read_jsonl(data_path(cfg, "memory_bank.jsonl"))
