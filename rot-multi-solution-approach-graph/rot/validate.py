"""Stage 2 -- validate teacher solutions against ground truth.

Nothing is deleted. Every generated solution keeps its record and gains
correctness metadata, so a later filtering strategy (correct-only, majority
vote, all) can be tried without regenerating anything. A pipeline that drops
its failures cannot tell you how often the teacher failed, and that number is
itself a result.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import data_path, read_jsonl, write_jsonl
from .scoring import is_correct


def validate(cfg, log):
    src = data_path(cfg, "teacher_solutions.jsonl")
    rows = read_jsonl(src)
    if not rows:
        raise FileNotFoundError(
            f"no teacher solutions at {src} -- run `main.py generate` first")

    by_problem = defaultdict(list)
    for r in rows:
        by_problem[r["problem_id"]].append(r)

    # Majority answer per problem, for a filtering strategy that needs no gold.
    majority = {}
    for pid, group in by_problem.items():
        votes = Counter(r["final_answer"] for r in group
                        if r["final_answer"] is not None)
        majority[pid] = votes.most_common(1)[0][0] if votes else None

    for r in rows:
        gold = r["gold_answer"]
        pred = r["final_answer"]
        r["correct"] = is_correct(pred, gold)
        r["agrees_with_majority"] = (pred is not None
                                     and pred == majority[r["problem_id"]])
        r["unparseable"] = pred is None

    out = data_path(cfg, "teacher_solutions_validated.jsonl")
    write_jsonl(out, rows)

    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    n_unparse = sum(1 for r in rows if r["unparseable"])

    solved = {pid for pid, g in by_problem.items()
              if any(is_correct(r["final_answer"], r["gold_answer"]) for r in g)}
    per_problem_correct = {pid: sum(1 for r in g if is_correct(
        r["final_answer"], r["gold_answer"])) for pid, g in by_problem.items()}

    by_group = defaultdict(lambda: {"solutions": 0, "correct": 0,
                                    "problems": set(), "solved": set()})
    for r in rows:
        g = by_group[r.get("group")]
        g["solutions"] += 1
        g["correct"] += int(r["correct"])
        g["problems"].add(r["problem_id"])
        if r["correct"]:
            g["solved"].add(r["problem_id"])

    group_report = {}
    for gname, g in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        group_report[str(gname)] = {
            "solutions": g["solutions"],
            "correct": g["correct"],
            "solution_accuracy": round(g["correct"] / max(g["solutions"], 1), 4),
            "problems": len(g["problems"]),
            "problems_with_a_correct_solution": len(g["solved"]),
        }

    report = {
        "n_solutions": n,
        "n_correct": n_correct,
        "solution_accuracy": round(n_correct / max(n, 1), 4),
        "n_unparseable": n_unparse,
        "n_problems": len(by_problem),
        "n_problems_with_a_correct_solution": len(solved),
        "problem_coverage": round(len(solved) / max(len(by_problem), 1), 4),
        "mean_correct_per_problem": round(
            sum(per_problem_correct.values()) / max(len(by_problem), 1), 3),
        "by_group": group_report,
    }

    log.info(f"Validation: {n_correct}/{n} solutions correct "
             f"({report['solution_accuracy']:.1%})")
    log.info(f"  problems with >=1 correct solution: {len(solved)}/"
             f"{len(by_problem)} ({report['problem_coverage']:.1%})")
    if n_unparse:
        log.warning(f"  unparseable replies: {n_unparse}")

    # A family the teacher cannot solve contributes nothing usable to memory.
    # Naming it here stops it being discovered as a mystery later.
    barren = [g for g, v in group_report.items()
              if v["problems_with_a_correct_solution"] == 0]
    if barren:
        log.warning(f"  families with NO correct solution at all: {barren}")
        log.warning("  these contribute nothing to the memory bank")
    report["barren_groups"] = barren

    log.info(f"  -> {out}")
    return rows, report
