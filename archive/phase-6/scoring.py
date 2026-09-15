"""
Phase 6 — Answer extraction and scoring.

Copied VERBATIM from phase-4/multi_model_eval.py (lines 145-317). Phase-4's
version is a strict superset of phase-3d's: it adds _clean_boxed, which strips
"= 4.5" suffixes and \\[...\\] / $$...$$ wrappers from boxed answers.

Reused unchanged so phase-6 scores identically to the phase-4 Qwen3-4B run.
Phase-3d/3e results predate _clean_boxed; they can be re-scored from their
saved *_raw fields if exact cross-phase comparability is needed. Do not edit
this file — any change breaks comparability with prior runs.
"""

import re

# ── Answer parsing ───────────────────────────────────────────────────────────

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def extract_boxed(text):
    results = []
    i = 0
    while i < len(text):
        if text[i:i+7] == r"\boxed{":
            depth, j = 1, i + 7
            while j < len(text) and depth > 0:
                if text[j] == "{": depth += 1
                elif text[j] == "}": depth -= 1
                j += 1
            results.append(text[i+7:j-1])
            i = j
        else:
            i += 1
    return results[-1].strip() if results else ""

def extract_number(text):
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    return numbers[-1].replace(",", "") if numbers else text.strip()

def _clean_boxed(boxed):
    """Clean boxed content: strip '= value' suffix, \\[...\\], $$...$$ wrappers."""
    # Strip "= decimal" suffix: "\frac{9}{2} = 4.5" → "\frac{9}{2}"
    boxed = re.sub(r'\s*=\s*-?[\d.]+\s*$', '', boxed)
    # Strip \[ ... \] wrapper
    m = re.match(r'^\\\[\s*(.+?)\s*\\\]$', boxed, re.DOTALL)
    if m:
        boxed = m.group(1)
    # Strip $$ ... $$ wrapper
    m = re.match(r'^\$\$\s*(.+?)\s*\$\$$', boxed, re.DOTALL)
    if m:
        boxed = m.group(1)
    return boxed.strip()

def extract_answer_math(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        return _clean_boxed(boxed)
    # Match "the answer/remainder/result is ..."
    m = re.search(r"[Tt]he (?:answer|remainder|result|value) is[:\s]*(.+)", text)
    if m:
        return _clean_math_tail(m.group(1).strip().rstrip("."))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:]\s*(.+)", text)
    if m:
        return _clean_math_tail(re.sub(r'\*+', '', m.group(1)).strip())
    # Extract from bold: **0**, **42**
    bold = re.findall(r'\*\*\$?([^*]+?)\$?\*\*', text)
    if bold:
        return _clean_math_tail(bold[-1].strip())
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def _clean_math_tail(ans):
    # Strip $...$ wrapper
    m = re.match(r'^\$\s*(.+?)\s*\$[\s.,;:]*\w*[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    # Strip $$...$$ wrapper
    m = re.match(r'^\$\$\s*(.+?)\s*\$\$[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    # Strip \[...\] wrapper
    m = re.match(r'^\\\[\s*(.+?)\s*\\\][\s.,;:]*$', ans, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Strip trailing unit words
    m = re.match(r'^.*?(\-?[\d,]+\.?\d*)\s+(?:feet|miles|cm|meters?|degrees?|hours?|minutes?|seconds?|square|cubic|units?)\b.*$', ans, re.IGNORECASE)
    if m:
        return m.group(1)
    # Strip trailing words after a number: "60 handshakes." or "3 boxes contain..."
    m = re.match(r'^(\-?[\d,]+\.?\d*)\s+[a-zA-Z].*$', ans)
    if m:
        return m.group(1)
    # Extract number from sentence: "So, the 8th grade should have 15 representatives"
    # Find the last standalone number in the sentence
    nums = re.findall(r'(?<!\d)(\-?\d+\.?\d*)(?!\d)', ans)
    if nums:
        return nums[-1]
    return ans

def extract_answer_aime(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        cleaned = re.sub(r'\*+', '', boxed).replace("$", "").strip()
        return extract_number(cleaned)
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return extract_number(m.group(1))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:]\s*(.+)", text)
    if m:
        return extract_number(re.sub(r'\*+', '', m.group(1)).replace("$", ""))
    bold = re.findall(r'\*\*\$?([\d,]+\.?\d*)\$?\*\*', text)
    if bold:
        return bold[-1].replace(",", "")
    return extract_number(text)

def normalize_math(ans):
    ans = ans.strip().rstrip(".")
    ans = re.sub(r"^\$+(.+?)\$+$", r"\1", ans)
    ans = ans.lstrip("$").rstrip("$")
    ans = re.sub(r"^\[(.+)\]$", r"\1", ans)
    ans = re.sub(r"\*+(.+?)\*+", r"\1", ans)
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\\mbox\{([^}]*)\}", r"\1", ans)
    ans = re.sub(r"\^\\circ", "", ans)
    ans = re.sub(r"\\(?:circ|degree)", "", ans)
    ans = re.sub(r"[°]", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", ans)
    ans = ans.replace("\\,", "").replace("\\!", "")
    ans = ans.replace("\\$", "").replace("$", "")
    ans = re.sub(r'([\d})])\s*(?:cm|meters?|miles|feet|hours?|minutes?|seconds?|units?|square|cubic|cm2|m2)\b.*$', r'\1', ans, flags=re.IGNORECASE)
    # Strip trailing words after a plain number: "60handshakes" "3boxescontain..."
    ans = re.sub(r'^(\-?\d+\.?\d*)[a-zA-Z].*$', r'\1', ans)
    ans = ans.replace("\\begin{pmatrix}", "(").replace("\\end{pmatrix}", ")")
    ans = ans.replace("\\\\", ",")
    if ans.startswith("0."):
        ans_alt = ans[1:]
    elif ans.startswith("."):
        ans_alt = "0" + ans
    else:
        ans_alt = None
    ans = ans.lower()
    if ans_alt:
        ans = ans_alt.lower() + "|" + ans
    return ans

def normalize_aime(ans):
    ans = ans.strip().rstrip(".")
    ans = ans.lstrip("$").rstrip("$")
    ans = re.sub(r'\*+', '', ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace(",", "")
    m = re.match(r'^(-?\d+\.?\d*)', ans)
    if m:
        ans = m.group(1)
    if ans.endswith(".0"):
        ans = ans[:-2]
    return ans

def is_correct(pred, gold, source):
    if source == "MATH-500":
        np_ = normalize_math(pred)
        ng_ = normalize_math(gold)
        np_forms = set(np_.split("|"))
        ng_forms = set(ng_.split("|"))
        if np_forms & ng_forms:
            return True
        np_clean = np_.split("|")[0] if "|" in np_ else np_
        ng_clean = ng_.split("|")[0] if "|" in ng_ else ng_
        try:
            pred_parts = sorted(np_clean.replace("{","").replace("}","").split(","))
            gold_parts = sorted(ng_clean.replace("{","").replace("}","").split(","))
            if pred_parts == gold_parts and len(pred_parts) > 1:
                return True
        except:
            pass
        return False
    else:
        return normalize_aime(pred) == normalize_aime(gold)

