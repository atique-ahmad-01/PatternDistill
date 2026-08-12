#!/usr/bin/env python3
"""Re-evaluate results_combined.jsonl with fixed answer extraction.
No GPU needed — just re-parses existing raw outputs."""
import json, re, sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
JSONL_IN    = RESULTS_DIR / "results_combined.jsonl"
JSONL_OUT   = RESULTS_DIR / "results_combined_fixed.jsonl"
JSON_OUT    = RESULTS_DIR / "results_combined_fixed.json"
SUMMARY_OUT = RESULTS_DIR / "summary_combined_fixed.json"

# ── Import extraction functions from main script ──
# Inline them to avoid heavy imports (torch, transformers, etc.)

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

def _clean_math_tail(ans):
    """Strip trailing unit words / sentences after a math expression."""
    m = re.match(r'^\$\s*(.+?)\s*\$[\s.,;:]*\w*[\s.,;:]*$', ans)
    if m:
        return m.group(1).strip()
    m = re.match(r'^.*?(\-?[\d,]+\.?\d*)\s+(?:feet|miles|cm|meters?|degrees?|hours?|minutes?|seconds?|square|cubic|units?)\b.*$', ans, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r'^[A-Z][a-z].*?(\-?[\d,]+\.?\d*).*$', ans)
    if m:
        return m.group(1)
    return ans

def extract_answer_math(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return _clean_math_tail(m.group(1).strip().rstrip("."))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:：]\s*(.+)", text)
    if m:
        ans = re.sub(r'\*+', '', m.group(1)).strip()
        return _clean_math_tail(ans)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""

def extract_answer_aime(text):
    text = strip_thinking(text)
    boxed = extract_boxed(text)
    if boxed:
        cleaned = re.sub(r'\*+', '', boxed).replace("$", "").strip()
        return extract_number(cleaned)
    m = re.search(r"[Tt]he answer is[:\s]*(.+)", text)
    if m:
        return extract_number(m.group(1))
    m = re.search(r"\*?\*?\s*[Aa]nswer\s*\*?\*?\s*[:：]\s*(.+)", text)
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
    ans = re.sub(r"°", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\,", "").replace("\\!", "")
    ans = ans.replace("\\$", "").replace("$", "")
    ans = re.sub(r'([\d})])\s*(?:cm|meters?|miles|feet|hours?|minutes?|seconds?|units?|square|cubic|cm²|m²)\b.*$', r'\1', ans, flags=re.IGNORECASE)
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

# ── Main ──

records = []
with open(JSONL_IN) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} records from {JSONL_IN}")

changes = []
for r in records:
    src = r["source"]
    gold = r["gold_answer"]

    # Re-extract
    if src == "MATH-500":
        new_van = extract_answer_math(r.get("vanilla_raw", ""))
        new_atgi = extract_answer_math(r.get("injection_raw", ""))
    else:
        new_van = extract_answer_aime(r.get("vanilla_raw", ""))
        new_atgi = extract_answer_aime(r.get("injection_raw", ""))

    new_van_ok = is_correct(new_van, gold, src)
    new_atgi_ok = is_correct(new_atgi, gold, src)

    old_van_ok = r.get("vanilla_correct", False)
    old_atgi_ok = r.get("injection_correct", False)

    if new_van_ok != old_van_ok or new_atgi_ok != old_atgi_ok:
        changes.append({
            "id": r["problem_id"], "source": src,
            "gold": gold,
            "van": f"{'✓' if old_van_ok else '✗'}→{'✓' if new_van_ok else '✗'}",
            "van_old": r.get("vanilla_answer",""), "van_new": new_van,
            "atgi": f"{'✓' if old_atgi_ok else '✗'}→{'✓' if new_atgi_ok else '✗'}",
            "atgi_old": r.get("injection_answer",""), "atgi_new": new_atgi,
        })

    # Update record
    r["vanilla_answer"] = new_van
    r["vanilla_correct"] = new_van_ok
    r["injection_answer"] = new_atgi
    r["injection_correct"] = new_atgi_ok

# Print changes
print(f"\n=== CHANGED: {len(changes)} records ===")
for c in changes:
    print(f"  {c['id']:<18} {c['source']:<10} gold={str(c['gold'])[:20]:<20}  "
          f"van:{c['van']} ({str(c['van_old'])[:20]}→{str(c['van_new'])[:20]})  "
          f"atgi:{c['atgi']} ({str(c['atgi_old'])[:20]}→{str(c['atgi_new'])[:20]})")

# Save
records.sort(key=lambda r: r["problem_id"])
with open(JSONL_OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
with open(JSON_OUT, "w") as f:
    json.dump(records, f, indent=2)

# Summary
n = len(records)
math_recs = [r for r in records if r["source"] == "MATH-500"]
aime_recs = [r for r in records if r["source"] == "AIME"]
injected = [r for r in records if not r.get("low_similarity")]

def acc(recs, k):
    return round(sum(r[k] for r in recs)/len(recs), 4) if recs else 0

summary = {
    "n": n, "n_math": len(math_recs), "n_aime": len(aime_recs),
    "n_injected": len(injected), "n_skipped": n - len(injected),
    "overall": {
        "vanilla": acc(records, "vanilla_correct"),
        "atgi": acc(records, "injection_correct"),
        "delta": round(acc(records, "injection_correct") - acc(records, "vanilla_correct"), 4),
    },
    "math500": {
        "vanilla": acc(math_recs, "vanilla_correct"),
        "atgi": acc(math_recs, "injection_correct"),
        "delta": round(acc(math_recs, "injection_correct") - acc(math_recs, "vanilla_correct"), 4),
    },
    "aime": {
        "vanilla": acc(aime_recs, "vanilla_correct"),
        "atgi": acc(aime_recs, "injection_correct"),
        "delta": round(acc(aime_recs, "injection_correct") - acc(aime_recs, "vanilla_correct"), 4),
    },
}

with open(SUMMARY_OUT, "w") as f:
    json.dump(summary, f, indent=2)

# Print
wins = sum(1 for r in records if r["injection_correct"] and not r["vanilla_correct"])
losses = sum(1 for r in records if r["vanilla_correct"] and not r["injection_correct"])
van_t = sum(1 for r in records if r["vanilla_correct"])
atgi_t = sum(1 for r in records if r["injection_correct"])

print(f"\n{'='*60}")
print(f"FIXED RESULTS")
print(f"{'='*60}")
print(f"  Vanilla: {van_t}/{n} ({van_t/n*100:.1f}%)")
print(f"  ATGI:    {atgi_t}/{n} ({atgi_t/n*100:.1f}%)")
print(f"  Wins ↑:  {wins}   Losses ↓: {losses}   Net: {wins-losses:+d}")
print(f"  Delta:   {(atgi_t-van_t)/n*100:+.1f}pp")
print(f"\n  MATH-500: Van={summary['math500']['vanilla']:.3f}  ATGI={summary['math500']['atgi']:.3f}  Δ={summary['math500']['delta']:+.4f}")
print(f"  AIME:     Van={summary['aime']['vanilla']:.3f}  ATGI={summary['aime']['atgi']:.3f}  Δ={summary['aime']['delta']:+.4f}")
print(f"\nSaved: {JSONL_OUT}, {JSON_OUT}, {SUMMARY_OUT}")
