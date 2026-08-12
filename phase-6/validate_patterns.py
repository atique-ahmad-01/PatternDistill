"""
Phase 6 — Pattern Validation Gates

Machine-checks distilled patterns so the failure modes measured in phase-4
cannot silently recur:

  * 30-32% of phase-2d/3d patterns contained digits from the anchor problem
  * 45% of MATH-500 worked examples contained the anchor's own gold answer
  * All 1,146 patterns were rigidly 3 steps + 1 Note

The teacher prompt already *asked* for abstraction and was ignored, so
phase-6 enforces it here instead of trusting the instruction.

Runs standalone against any bank (old or new schema), and is imported by
generate_patterns.py to gate each pattern at distillation time.

Usage:
    python validate_patterns.py --input data/patterns/patterns.jsonl
    python validate_patterns.py --input ../phase-3d/memory/bank.json   # old bank
    python validate_patterns.py --input ../phase-2d/memory/bank.json --verbose
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ── Gate configuration ───────────────────────────────────────────────────────

MIN_STEPS = 2
MAX_STEPS = 6
# Small models degrade on long injected context (phase-2 put the knee at
# ~400 tokens for 1.7B), so this stays conservative. Raised from 200 after a
# smoke test dropped an otherwise-good pattern at 204 tokens — the new schema
# carries five fields, so 200 was too tight to be a useful signal.
MAX_STRATEGY_TOKENS = 260
MIN_STEP_WORDS = 5
UNIFORMITY_LIMIT = 0.90        # bank-level: >90% sharing one step count fails

REQUIRED_FIELDS = ("trigger", "technique", "pitfall")

# Steps that are pure restatement carry no information. Matches the phase-4
# finding that "Identify" opened a step in 698/746 AIME patterns.
VACUOUS_RE = re.compile(
    r"^(identify|determine|calculate|solve|find|compute|use|apply|perform|add)\s+"
    r"(the\s+)?\w+(\s+\w+)?\.?$",
    re.IGNORECASE,
)

GATES = (
    "required_fields",
    "step_count",
    "non_vacuous_steps",
    "no_anchor_numbers",
    "no_problem_phrases",
    "no_answer_leak",
    "token_budget",
)

# Multi-digit overlap alone misses problem-bound phrasing built from single
# digits, e.g. "a 4-digit number starting with 1" (verified to pass every
# other gate). Shared content n-grams catch that class.
PHRASE_N = 3
STOPWORDS = frozenset("""
a an the of to in on for with and or is are be that this these those it its
each all any one two both same such then than from into by as at we you
""".split())


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def numbers_in(text, min_digits=2):
    """Numbers with >= min_digits digits.

    Single digits are excluded deliberately: patterns such as "add 1 to each
    exponent" are legitimately general, so blocking all digits produces false
    positives. Multi-digit numbers from the source problem are what bind a
    pattern to its anchor.
    """
    return {n for n in re.findall(r"\d+", str(text or "")) if len(n) >= min_digits}


def contains_standalone(haystack, needle):
    """Whole-token containment, so gold "9" does not match inside "1996"."""
    needle = normalize(needle)
    if not needle:
        return False
    return re.search(rf"(?<![\w\d]){re.escape(needle)}(?![\w\d])", str(haystack or "")) is not None


def content_ngrams(text, n=PHRASE_N):
    """Content-word n-grams, lowercased, LaTeX/punctuation/hyphens flattened.

    Hyphens become spaces so "$4$ -digit" and "4-digit" tokenize alike —
    without this the phrase gate misses the very case it exists to catch.
    """
    t = re.sub(r"[$\\{}(),.;:\-]", " ", str(text or "").lower())
    words = [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOPWORDS]
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def numeric_shared_phrases(problem, blob):
    """Shared phrases that carry a number.

    Restricting to number-bearing phrases is what separates problem-bound
    text ("4 digit number") from legitimately general mathematical vocabulary
    ("prime factorization number"), which patterns are expected to share.
    """
    return {g for g in (content_ngrams(problem) & content_ngrams(blob))
            if re.search(r"\d", g)}


def parse_steps_old(pattern_text):
    """Extract 'Step N:' lines from a legacy flat pattern string."""
    return [m.strip() for m in re.findall(r"^Step\s*\d+\s*:\s*(.+)$", pattern_text or "", re.M)]


def to_common(entry):
    """Normalize old and new schema into one shape for checking."""
    if isinstance(entry.get("steps"), list):          # new schema
        return {
            "problem":   entry.get("problem", ""),
            "gold":      entry.get("gold_answer", ""),
            "trigger":   entry.get("trigger", ""),
            "technique": entry.get("technique", ""),
            "steps":     [normalize(s) for s in entry["steps"] if normalize(s)],
            "pitfall":   entry.get("pitfall", ""),
            "verify":    entry.get("verify", ""),
            "example":   entry.get("worked_example", ""),
            "schema":    "new",
        }
    pat = entry.get("pattern", "") or ""              # legacy schema
    note = re.search(r"^Note\s*:\s*(.+)$", pat, re.M)
    return {
        "problem":   entry.get("problem", ""),
        "gold":      entry.get("gold_answer", ""),
        "trigger":   "",
        "technique": "",
        "steps":     parse_steps_old(pat),
        "pitfall":   note.group(1).strip() if note else "",
        "verify":    "",
        "example":   entry.get("worked_example", ""),
        "schema":    "old",
    }


def strategy_text(c):
    """The text that actually gets injected — what the budget gate measures."""
    parts = []
    if c["trigger"]:
        parts.append(f"Applies when: {c['trigger']}")
    if c["technique"]:
        parts.append(f"Technique: {c['technique']}")
    parts += [f"{i}. {s}" for i, s in enumerate(c["steps"], 1)]
    if c["pitfall"]:
        parts.append(f"Watch out: {c['pitfall']}")
    if c["verify"]:
        parts.append(f"Check: {c['verify']}")
    return "\n".join(parts)


# ── Gates ────────────────────────────────────────────────────────────────────

def check(entry, require_new_schema=True):
    """Return (passed: bool, failures: dict[gate] = reason)."""
    c = to_common(entry)
    fields = [c["trigger"], c["technique"], c["pitfall"], c["verify"]] + c["steps"]
    blob = " ".join(fields)
    fail = {}

    # 1. required fields (only meaningful for the new schema)
    if require_new_schema:
        missing = [f for f in REQUIRED_FIELDS if not normalize(c.get(f))]
        if missing:
            fail["required_fields"] = f"missing/empty: {', '.join(missing)}"

    # 2. step count within range
    n = len(c["steps"])
    if not (MIN_STEPS <= n <= MAX_STEPS):
        fail["step_count"] = f"{n} steps (need {MIN_STEPS}-{MAX_STEPS})"

    # 3. no vacuous steps
    vac = [s for s in c["steps"]
           if VACUOUS_RE.match(s) or len(s.split()) < MIN_STEP_WORDS]
    if vac:
        fail["non_vacuous_steps"] = f"{len(vac)} vacuous: {vac[0][:60]!r}"

    # 4. no multi-digit numbers carried over from the anchor problem
    leaked = sorted(numbers_in(c["problem"]) & numbers_in(blob))
    if leaked:
        fail["no_anchor_numbers"] = f"anchor numbers in pattern: {leaked[:5]}"

    # 5. no distinctive phrasing carried over from the anchor problem
    shared = numeric_shared_phrases(c["problem"], blob)
    if shared:
        fail["no_problem_phrases"] = f"shared phrase: {sorted(shared)[0][:50]!r}"

    # 6. the anchor's own answer must not appear anywhere (incl. any example)
    gold = normalize(c["gold"])
    if gold and (contains_standalone(blob, gold) or contains_standalone(c["example"], gold)):
        where = "pattern" if contains_standalone(blob, gold) else "worked_example"
        fail["no_answer_leak"] = f"gold answer {gold!r} appears in {where}"

    # 7. injected strategy must stay small (~4 chars/token approximation)
    approx = max(1, len(strategy_text(c)) // 4)
    if approx > MAX_STRATEGY_TOKENS:
        fail["token_budget"] = f"~{approx} tokens (limit {MAX_STRATEGY_TOKENS})"

    return (not fail), fail


# ── Bank-level report ────────────────────────────────────────────────────────

def load_any(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def report(entries, require_new_schema=True, verbose=False):
    counts = Counter()
    examples = {}
    step_dist = Counter()
    passed = 0

    for e in entries:
        ok, fail = check(e, require_new_schema)
        step_dist[len(to_common(e)["steps"])] += 1
        if ok:
            passed += 1
        for gate, reason in fail.items():
            counts[gate] += 1
            examples.setdefault(gate, reason)

    n = len(entries)
    print(f"\nEntries: {n}   passed all gates: {passed} ({passed/n*100:.1f}%)\n")
    print(f"{'gate':22s} {'failed':>7s} {'rate':>7s}   example")
    print("-" * 92)
    for gate in GATES:
        c = counts.get(gate, 0)
        ex = examples.get(gate, "")
        print(f"{gate:22s} {c:7d} {c/n*100:6.1f}%   {ex[:44]}")

    print(f"\nstep-count distribution: {dict(sorted(step_dist.items()))}")
    if step_dist:
        top, topn = step_dist.most_common(1)[0]
        frac = topn / n
        verdict = "FAIL - patterns are templated" if frac > UNIFORMITY_LIMIT else "ok"
        print(f"uniformity: {frac*100:.1f}% share {top} steps "
              f"(limit {UNIFORMITY_LIMIT*100:.0f}%) -> {verdict}")

    if verbose:
        print("\n--- first 5 failing entries ---")
        shown = 0
        for e in entries:
            ok, fail = check(e, require_new_schema)
            if ok or shown >= 5:
                continue
            shown += 1
            print(f"\n[{shown}] {normalize(e.get('problem',''))[:90]}")
            for g, r in fail.items():
                print(f"    {g}: {r}")

    return passed, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="patterns .json/.jsonl or a bank.json")
    ap.add_argument("--old-schema", action="store_true",
                    help="skip new-schema-only gates (for legacy banks)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="exit 1 if pass-rate percentage falls below this")
    args = ap.parse_args()

    entries = load_any(args.input)
    print("=" * 92)
    print(f"Pattern validation — {args.input}")
    print("=" * 92)
    passed, n = report(entries, not args.old_schema, args.verbose)

    if args.fail_under is not None and (passed / n * 100) < args.fail_under:
        print(f"\nFAIL: pass-rate {passed/n*100:.1f}% < required {args.fail_under}%")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
