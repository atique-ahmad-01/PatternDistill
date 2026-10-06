"""T3 -- transform thinking traces for RAG (arXiv:2605.03344).

"RAG over Thinking Traces Can Improve Reasoning Tasks" retrieves whole solved
examples, not step templates, and puts them in the USER prompt as hints. Its
one non-obvious ingredient is T3: an offline rewrite of each raw thinking trace
(~1,600 words, full of dead ends) into a short, clean form. Raw traces helped
too, but every T3 form beat them, and long raw traces were "too verbose and
noisy".

Three forms, as in the paper:

    struct     concise step-by-step cheatsheet, at most 7 steps
               (best for the paper's small models on MATH)
    semantic   three passes of rising abstraction: full -> core moves -> insight
    reflect    the trace's traps and how to avoid them, then the correct
               approach (best overall in the paper, GPT-5 on AIME)

The source is the teacher's FULL reply -- the <think> trace plus its final
APPROACH/STEP block -- not the parsed steps, because the dead ends and checks
in the trace are what `reflect` is built from. Every form keeps the example's
final answer, as the paper's do.

The paper's exact prompt texts are not reproduced here; these follow its
stated instructions and output formats.

Output: data/artifacts/t3_<variant>.jsonl, one record per memory entry,
appended as produced so a killed run resumes.
"""

from __future__ import annotations

import re
import threading

from .config import data_path, read_jsonl, write_jsonl

VARIANTS = ("struct", "semantic", "reflect")

_COMMON = """You are given a math problem and a reasoning trace that solved it
correctly. The trace is long and contains false starts, checks and repetition.

Write ONLY the requested output, in plain text, and nothing else. Keep the
mathematics exact. Do not add methods that are not in the trace. End with the
line:

Answer: <the trace's final answer>"""

SYSTEM = {
    "struct": _COMMON + """

Task: convert the reasoning trace into a concise step-by-step cheatsheet that
someone could follow to solve this problem and others like it. Drop dead ends,
repetition and verification chatter. Use at most 7 steps.

Format:
Approach: <the strategy, one line>
Step 1: <one concrete action>
Step 2: <...>
...
Answer: <final answer>""",

    "semantic": _COMMON + """

Task: compress the reasoning trace into progressively more abstract
representations. Provide three levels of abstraction.

Format:
Pass 1 (full): <the complete derivation, condensed to its essential lines>
Pass 2 (core moves): <the 2-4 key decisions that made the solution work>
Pass 3 (insight): <the one structural insight, in a sentence or two>
Answer: <final answer>""",

    "reflect": _COMMON + """

Task: extract failure patterns and negative knowledge from the reasoning trace.
Focus on the mistakes that are easy to make on this type of problem -- wrong
turns the trace took or nearly took -- and how to avoid them. Then give the
correct approach.

Format:
Trap profile: <what makes this type of problem easy to get wrong, one line>
✗ <a pitfall>
✓ <the safeguard against it>
(2-4 pitfall/safeguard pairs)
Correct approach: <the working method in a few short lines>
Answer: <final answer>""",
}

# The paper's inference instruction (its Figure 7), followed by the examples.
RAG_INSTRUCTION = ("Solve the main problem by using useful hints and strategies "
                   "from the retrieved examples.")

_ANSWER = re.compile(r"^\s*\**\s*answer\s*\**\s*:", re.IGNORECASE | re.MULTILINE)
_PASS3 = re.compile(r"pass\s*3", re.IGNORECASE)
_STEP = re.compile(r"^\s*\**\s*step\s*\d", re.IGNORECASE | re.MULTILINE)
_PITFALL = re.compile(r"✗")


def _prompt(entry, trace):
    return (f"Problem: {entry['problem']}\n\n"
            f"Reasoning trace:\n{trace}\n\n"
            f"Final answer of the trace: {entry['final_answer']}")


def _clean(text):
    from .scoring import strip_thinking
    return strip_thinking(text or "").strip()


def _well_formed(variant, text):
    """Cheap format check; a failing reply is retried, then kept with ok=False."""
    if not text or not _ANSWER.search(text):
        return False
    if variant == "struct":
        return bool(_STEP.search(text))
    if variant == "semantic":
        return bool(_PASS3.search(text))
    return bool(_PITFALL.search(text))


def transform(cfg, client, entries, traces, variant, log, resume=True, attempts=3):
    """Rewrite every entry's trace into `variant`. Returns {entry_id: record}.

    `traces` maps entry_id -> the teacher's full reply for that solution.
    """
    if variant not in VARIANTS:
        raise ValueError(f"t3 variant must be one of {VARIANTS}, got {variant!r}")
    out_path = data_path(cfg, f"t3_{variant}.jsonl")
    existing = read_jsonl(out_path) if resume else []
    if not resume and out_path.exists():
        out_path.unlink()
    have = {r["entry_id"] for r in existing}
    missing = [e["entry_id"] for e in entries if e["entry_id"] not in traces]
    if missing:
        raise SystemExit(f"{len(missing)} memory entries have no teacher trace "
                         f"({missing[:3]}) -- memory bank and teacher solutions "
                         f"are out of sync")
    todo = [e for e in entries if e["entry_id"] not in have]
    if existing:
        log.info(f"Resuming: {len(have)} entries already transformed, {len(todo)} to go")

    write_lock = threading.Lock()

    def _one(e):
        text, raw = "", ""
        for k in range(attempts):
            # Same seed every call, so a retry must change the sampling or it
            # would reproduce the same malformed reply.
            r = client.chat(SYSTEM[variant], _prompt(e, traces[e["entry_id"]]),
                            n=1, temperature=client.temperature + 0.3 * k)[0]
            raw = r.text
            text = _clean(r.text)
            if _well_formed(variant, text):
                break
        rec = {
            "entry_id": e["entry_id"],
            "problem_id": e["problem_id"],
            "variant": variant,
            "ok": _well_formed(variant, text),
            "attempts": k + 1,
            # A malformed rewrite is still kept when it has text: the student
            # sees it as-is, and the ok count says how many there were.
            "text": text,
            "n_words": len(text.split()),
            "trace_words": len(traces[e["entry_id"]].split()),
            "raw": raw,
        }
        with write_lock:
            write_jsonl(out_path, [rec], mode="a")
        return rec

    if todo:
        client.map(todo, _one, desc=f"t3-{variant}", log=log)

    recs = {r["entry_id"]: r for r in read_jsonl(out_path)}
    n = max(len(recs), 1)
    words = sorted(r["n_words"] for r in recs.values())
    trace_words = sorted(r["trace_words"] for r in recs.values())
    rep = {
        "variant": variant,
        "n_entries": len(recs),
        "n_malformed": sum(1 for r in recs.values() if not r["ok"]),
        "n_empty": sum(1 for r in recs.values() if not r["text"]),
        "median_words": words[len(words) // 2] if words else 0,
        "median_trace_words": trace_words[len(trace_words) // 2] if trace_words else 0,
    }
    log.info(f"T3 {variant}: {rep['n_entries']} entries, "
             f"{rep['n_malformed']} malformed, {rep['n_empty']} empty; "
             f"median {rep['median_words']} words "
             f"(raw trace {rep['median_trace_words']})")
    log.info(f"  -> {out_path}")
    return recs, rep


def load(cfg, variant):
    """{entry_id: text}; raises if the stage has not been run for `variant`."""
    path = data_path(cfg, f"t3_{variant}.jsonl")
    if not path.exists():
        raise SystemExit(f"no T3 {variant} texts at {path} -- "
                         f"run `python main.py t3` with t3.variant={variant} first")
    return {r["entry_id"]: r["text"] for r in read_jsonl(path)}


def format_examples(hits, texts):
    """The paper's prompt body: Example i = the problem + its transformed trace.

    An entry with no usable T3 text falls back to its parsed APPROACH/STEP
    solution, so a malformed rewrite cannot silently drop an example.
    """
    blocks = []
    for i, h in enumerate(hits, 1):
        body = texts.get(h["entry_id"]) or h["solution_text"]
        blocks.append(f"Example {i}:\nProblem: {h['problem']}\n{body}")
    return "\n\n".join(blocks)


def build_user_prompt(problem, examples):
    if not examples:
        return problem
    return f"{RAG_INSTRUCTION}\n\n{examples}\n\nMain problem: {problem}"
