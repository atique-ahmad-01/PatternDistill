"""Generic step templates -- rewrite each memory-bank solution as a method skeleton.

The reference RoT library is made of generic steps ("Standard Transformation:
Rearrange the inequality to the form ax^2+bx+c>0"); only 0.2% of its node texts
carry a 4+ digit literal. Ours are worked solutions to one problem each, and
37% do. Under Thinking Intervention that difference is fatal: the template is
read as reasoning the student has already done, so it inherits the donor
problem's values and results. In f_exact, 23 of the 40 wrong answers are a
number that appears verbatim in the injected template.

This stage asks the teacher to restate every solution's approach and steps
with the problem's values replaced by named roles ("the modulus m", "the
exponent N"), one step in, one step out, so the step graph keeps its shape.
`rot.step_text: generic` then builds the RoT graph from these texts instead.

Output: data/artifacts/rot_generic_steps.jsonl, one record per memory entry,
appended as produced so a killed run resumes.
"""

from __future__ import annotations

import re
import threading

from .config import data_path, read_jsonl, write_jsonl
from .teacher import _APPROACH, _STEP

SYSTEM = """You turn one worked solution into a reusable method template.

You are given a problem and a solution written for it: an APPROACH line and
numbered steps. Rewrite them so the same template could be followed on ANY
other problem of the same type, with different numbers.

Rules:
- Remove every value that belongs to this problem: the given numbers AND every
  intermediate or final result computed from them. Refer to quantities by their
  role instead ("the modulus m", "the exponent N", "the cycle length L",
  "the number of beads n").
- Keep a number only if it is part of the method itself and would be the same
  for every problem of this type (for example "mod 2" in a parity argument).
- Keep the operation each step performs. Do not merge, split, add or drop steps:
  output exactly as many steps as you were given, in the same order.
- Each step stays one short imperative sentence.

Write exactly this format and nothing else:

APPROACH: <the technique, under 12 words, no problem-specific numbers>
STEP 1: <generic step>
STEP 2: <generic step>
..."""

# Any literal of 3+ digits; also used for the post-rewrite audit.
_BIG = re.compile(r"\d{3,}")


def _prompt(entry):
    steps = "\n".join(f"STEP {i + 1}: {s}" for i, s in enumerate(entry["subtasks"]))
    return (f"Problem: {entry['problem']}\n\n"
            f"APPROACH: {entry['approach']}\n{steps}\n\n"
            f"Rewrite this as a generic template with exactly "
            f"{len(entry['subtasks'])} steps.")


def _parse(text):
    from .scoring import strip_thinking
    body = strip_thinking(text or "")
    m = _APPROACH.search(body)
    approach = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".") if m else ""
    steps = [re.sub(r"\s+", " ", m.group(3)).strip() for m in _STEP.finditer(body)]
    return approach, [s for s in steps if s]


def generalize(cfg, client, entries, log, resume=True, attempts=3):
    """Rewrite every entry's approach + subtasks. Returns {entry_id: record}."""
    out_path = data_path(cfg, "rot_generic_steps.jsonl")
    existing = read_jsonl(out_path) if resume else []
    if not resume and out_path.exists():
        out_path.unlink()
    have = {r["entry_id"] for r in existing}
    todo = [e for e in entries if e["entry_id"] not in have]
    if existing:
        log.info(f"Resuming: {len(have)} entries already rewritten, {len(todo)} to go")

    write_lock = threading.Lock()

    def _one(e):
        n = len(e["subtasks"])
        approach, steps, raw = "", [], ""
        for k in range(attempts):
            # Same seed every call, so a retry must change the sampling or it
            # would reproduce the same malformed reply.
            r = client.chat(SYSTEM, _prompt(e), n=1,
                            temperature=client.temperature + 0.3 * k)[0]
            raw = r.text
            approach, steps = _parse(r.text)
            if approach and len(steps) == n:
                break
        ok = bool(approach) and len(steps) == n
        rec = {
            "entry_id": e["entry_id"],
            "problem_id": e["problem_id"],
            "ok": ok,
            "attempts": k + 1,
            # On failure the original text is kept, so the graph stays complete
            # and the failure count says how much of it is still instance-bound.
            "approach_generic": approach if ok else e["approach"],
            "subtasks_generic": steps if ok else list(e["subtasks"]),
            "approach_original": e["approach"],
            "subtasks_original": list(e["subtasks"]),
            "raw": raw,
        }
        with write_lock:
            write_jsonl(out_path, [rec], mode="a")
        return rec

    if todo:
        client.map(todo, _one, desc="generalize", log=log)

    recs = {r["entry_id"]: r for r in read_jsonl(out_path)}
    rep = audit(recs)
    log.info(f"Generic templates: {rep['n_entries']} entries, "
             f"{rep['n_failed']} kept original (step count or format mismatch)")
    log.info(f"  steps with a 3+ digit literal: {rep['before_3plus']:.1%} -> "
             f"{rep['after_3plus']:.1%}; with any digit: "
             f"{rep['before_any_digit']:.1%} -> {rep['after_any_digit']:.1%}")
    log.info(f"  -> {out_path}")
    return recs, rep


def audit(recs):
    before = [s for r in recs.values() for s in r["subtasks_original"]]
    after = [s for r in recs.values() for s in r["subtasks_generic"]]
    share = lambda xs, pat: sum(1 for s in xs if pat.search(s)) / max(len(xs), 1)
    digit = re.compile(r"\d")
    return {
        "n_entries": len(recs),
        "n_failed": sum(1 for r in recs.values() if not r["ok"]),
        "n_steps": len(after),
        "before_3plus": round(share(before, _BIG), 4),
        "after_3plus": round(share(after, _BIG), 4),
        "before_any_digit": round(share(before, digit), 4),
        "after_any_digit": round(share(after, digit), 4),
    }


def load(cfg):
    """{entry_id: record}; raises if the stage has not been run."""
    path = data_path(cfg, "rot_generic_steps.jsonl")
    if not path.exists():
        raise SystemExit("rot.step_text=generic but no generic templates -- "
                         "run `python main.py generalize` first")
    return {r["entry_id"]: r for r in read_jsonl(path)}
