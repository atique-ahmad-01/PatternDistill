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

Each also has a `_long` form (struct_long, ...): same format, but asked to keep
the concrete formulas and intermediate values and to aim for 250-450 words.
With the short prompts Qwen3-8B wrote 61 / 152 / 100 words (median), well
under the paper's 239-454; the long forms test whether that gap matters.

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

BASE_VARIANTS = ("struct", "semantic", "reflect")
VARIANTS = BASE_VARIANTS + tuple(f"{v}_long" for v in BASE_VARIANTS)

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

# The long forms keep each short form's output format (so the same format
# check applies) and differ only in the task text: detail instead of brevity.
_LONG_TASK = {
    "struct": """

Task: convert the reasoning trace into a detailed step-by-step worked solution
that someone could follow to solve this problem and others like it. Drop dead
ends, repetition and verification chatter, but keep every computation that
matters: each step states the action AND the concrete formula, intermediate
values or result it produces, in 2-4 sentences. Use at most 7 steps. Aim for
250-400 words in total.""",

    "semantic": """

Task: compress the reasoning trace into progressively more abstract
representations. Provide three levels of abstraction. Pass 1 is the complete
derivation with its key formulas and intermediate values (150-250 words);
Pass 2 gives the 2-4 key decisions, each with a sentence on why it works;
Pass 3 states the structural insight in 2-3 sentences. Aim for 300-450 words
in total.""",

    "reflect": """

Task: extract failure patterns and negative knowledge from the reasoning trace.
Focus on the mistakes that are easy to make on this type of problem -- wrong
turns the trace took or nearly took -- and how to avoid them. Then give the
correct approach in full, with its key formulas and intermediate values. Aim
for 250-400 words in total.""",
}


def _format_block(variant):
    """The 'Format:' tail of a short-form system prompt."""
    return SYSTEM[variant][SYSTEM[variant].index("\n\nFormat:"):]


for _v in BASE_VARIANTS:
    SYSTEM[f"{_v}_long"] = _COMMON + _LONG_TASK[_v] + _format_block(_v)
SYSTEM["struct_long"] = SYSTEM["struct_long"].replace(
    "Step 1: <one concrete action>", "Step 1: <the action and the values it produces>")
SYSTEM["semantic_long"] = SYSTEM["semantic_long"].replace(
    "<the complete derivation, condensed to its essential lines>",
    "<the complete derivation with its key formulas and values>")
SYSTEM["reflect_long"] = SYSTEM["reflect_long"].replace(
    "(2-4 pitfall/safeguard pairs)", "(3-5 pitfall/safeguard pairs)").replace(
    "Correct approach: <the working method in a few short lines>",
    "Correct approach: <the working method, 5-8 lines with its key values>")

# The paper's inference instruction (its Figure 7), followed by the examples.
RAG_INSTRUCTION = ("Solve the main problem by using useful hints and strategies "
                   "from the retrieved examples.")

_ANSWER = re.compile(r"^\s*\**\s*answer\s*\**\s*:", re.IGNORECASE | re.MULTILINE)
_PASS3 = re.compile(r"pass\s*3", re.IGNORECASE)
_STEP = re.compile(r"^\s*\**\s*step\s*\d", re.IGNORECASE | re.MULTILINE)
_PITFALL = re.compile(r"✗")

# Appended to the user turn on a retry only. A long trace (~5,800 tokens) ends
# in the teacher's own APPROACH/STEP/ANSWER block, which already looks like a
# struct cheatsheet; with the instructions that far back, the teacher answers
# with just "Answer: N", deterministically, at any temperature. Restating the
# task after the trace fixed 30/30 such cases. First attempts never see it, so
# every rewrite that passed first time is unchanged.
REMINDER = {
    "struct": "Now write the step-by-step cheatsheet in the required format "
              "(Approach line, then Step 1, Step 2, ..., then Answer). Do not "
              "just copy the final answer.",
    "semantic": "Now write the three passes in the required format (Pass 1, "
                "Pass 2, Pass 3, then Answer). Do not just copy the final answer.",
    "reflect": "Now write the trap profile, the ✗/✓ pitfall pairs and the "
               "correct approach in the required format, then Answer. Do not "
               "just copy the final answer.",
}
for _v in BASE_VARIANTS:
    REMINDER[f"{_v}_long"] = REMINDER[_v].replace(
        "Do not just", "Keep the concrete values; aim for 250-400 words. Do not just")


def _prompt(entry, trace, variant=None):
    p = (f"Problem: {entry['problem']}\n\n"
         f"Reasoning trace:\n{trace}\n\n"
         f"Final answer of the trace: {entry['final_answer']}")
    return p + f"\n\n{REMINDER[variant]}" if variant else p


def _clean(text):
    from .scoring import strip_thinking
    return strip_thinking(text or "").strip()


def _well_formed(variant, text):
    """Cheap format check; a failing reply is retried, then kept with ok=False."""
    if not text or not _ANSWER.search(text):
        return False
    variant = variant.removesuffix("_long")
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
    # A malformed rewrite is redone on resume; the new record is appended and
    # wins, since the file is read back keyed by entry_id, last one kept.
    have = {r["entry_id"] for r in existing if r["ok"]}
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
            # A retry also restates the task after the trace (see REMINDER).
            # The long forms always do: with the length target only in the
            # system prompt, struct_long came out at a median 98 words (some
            # 2); restated after the trace, 186 (125-306), on 12 samples.
            again = k or variant.endswith("_long")
            user = _prompt(e, traces[e["entry_id"]], variant if again else None)
            r = client.chat(SYSTEM[variant], user,
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
