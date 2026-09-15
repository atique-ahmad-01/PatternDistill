"""Answer extraction and correctness for RB-DERIV.

Self-contained: no dependency on any other phase directory.

Every RB-DERIV gold is a single exact integer computed in Python, so scoring
is exact -- no grading model, no partial credit, no LaTeX equivalence. That is
a real simplification over the MATH-500 scorer earlier phases needed, which had
to reconcile \\frac{23}{90} against 23/90. Here a prediction either is the gold
integer or it is not.

Returning None for an unparseable reply rather than guessing keeps "the model
ignored the format" distinguishable from "the model was wrong" -- those mean
different things and must not be silently merged.
"""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_INT = re.compile(r"-?\d[\d,]*")


# ── Extraction ───────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove Qwen3 reasoning blocks before parsing.

    An unclosed <think> means a truncated reply; everything after it is
    reasoning, so a mid-thought number must not be read as the final answer.
    """
    text = _THINK.sub("", text)
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
    return text.strip()


def extract_boxed(text: str) -> str:
    """Last \\boxed{...}, brace-matched so nested braces survive."""
    results, i = [], 0
    while i < len(text):
        if text[i:i + 7] == r"\boxed{":
            depth, j = 1, i + 7
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            results.append(text[i + 7:j - 1])
            i = j
        else:
            i += 1
    return results[-1].strip() if results else ""


def _to_int(s):
    """Parse the first integer out of a fragment. None if there isn't one."""
    if s is None:
        return None
    m = _INT.search(str(s))
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str):
    """Return the model's final answer as an int, or None if unparseable.

    Order matters: the explicit contract line the prompts ask for is tried
    first, then \\boxed, then prose. Returning None rather than guessing keeps
    "model ignored the format" distinguishable from "model was wrong" -- the
    two mean different things and must not be merged.
    """
    if not text:
        return None
    body = strip_thinking(text)

    for line in reversed(body.splitlines()):
        line = line.strip()
        if re.match(r"^\**\s*(FINAL\s+)?ANSWER\s*\**\s*[:=]", line, re.IGNORECASE):
            got = _to_int(line.split(":", 1)[-1] if ":" in line else line)
            if got is not None:
                return got

    boxed = extract_boxed(body)
    if boxed:
        got = _to_int(boxed.replace("$", ""))
        if got is not None:
            return got

    m = re.search(r"[Tt]he answer is[:\s]*([^\n.]+)", body)
    if m:
        got = _to_int(m.group(1))
        if got is not None:
            return got

    tail = body.strip().splitlines()
    if tail:
        got = _to_int(tail[-1])
        if got is not None:
            return got
    return None


# ── Comparison ───────────────────────────────────────────────────────────────

def is_correct(pred, gold) -> bool:
    """Exact integer equality. Unparseable predictions are incorrect."""
    p = pred if isinstance(pred, int) else _to_int(pred)
    g = gold if isinstance(gold, int) else _to_int(gold)
    if p is None or g is None:
        return False
    return p == g


def is_gradable(gold) -> bool:
    """True if the gold is a single exact integer."""
    if isinstance(gold, bool):
        return False
    if isinstance(gold, int):
        return True
    return bool(re.fullmatch(r"\s*-?\d+\s*", str(gold)))
