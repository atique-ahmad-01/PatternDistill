"""
Phase 6 — Analysis: significance, confidence intervals, subgroup splits

No prior phase reported a p-value, and none of their headline deltas survive
one (phase-4 +1.7pp p=0.500, phase-3d +3.5pp p=0.164, phase-3e +3.1pp
p=0.134). Every number this script prints carries a significance test.

Two reporting rules, both learned from phase-4:
  * truncated and finished problems are reported SEPARATELY, because pooling
    them is what made phase-4 uninterpretable (74% of AIME was truncated, so
    the pooled delta mostly measured who ran out of room);
  * comparisons are paired per (problem, arm) against arm A, using exact
    McNemar on discordant pairs — the correct test for paired binary outcomes.

Usage:
    python analyze.py --input results/results_qwen3-1.7b.jsonl
    python analyze.py --input results/results_qwen3-1.7b.jsonl --baseline A_vanilla
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import config as C


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = baseline-wrong->arm-right, c = the reverse."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(p, 1.0)


def bootstrap_ci(pairs, iters=10000, seed=0):
    """Percentile CI for the paired accuracy delta (arm - baseline)."""
    import random
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(pairs)
    deltas = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            bl, ar = pairs[rng.randrange(n)]
            s += ar - bl
        deltas.append(s / n * 100)
    deltas.sort()
    return (deltas[int(0.025 * iters)], deltas[int(0.975 * iters)])


def wins_losses(pairs):
    b = sum(1 for bl, ar in pairs if ar and not bl)   # arm fixed it
    c = sum(1 for bl, ar in pairs if bl and not ar)   # arm broke it
    return b, c


def fmt_block(title, pairs, indent="  "):
    if not pairs:
        print(f"{indent}{title:<26s} (no data)")
        return
    n = len(pairs)
    bl = sum(p[0] for p in pairs)
    ar = sum(p[1] for p in pairs)
    w, l = wins_losses(pairs)
    p = mcnemar_exact(w, l)
    lo, hi = bootstrap_ci(pairs)
    sig = "SIG" if p < 0.05 else "ns "
    print(f"{indent}{title:<26s} n={n:4d} base={bl/n*100:5.1f}% arm={ar/n*100:5.1f}% "
          f"d={(ar-bl)/n*100:+5.1f}pp [{lo:+5.1f},{hi:+5.1f}] "
          f"W={w:3d} L={l:3d} p={p:.3f} {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--baseline", default="A_vanilla")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if not rows:
        raise SystemExit("no rows")

    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["arm"]][r["problem_id"]] = r

    if args.baseline not in by_arm:
        raise SystemExit(f"baseline arm {args.baseline!r} not in results "
                         f"({list(by_arm)})")
    base = by_arm[args.baseline]

    print("=" * 108)
    print(f"Phase 6 analysis — {Path(args.input).name}  (baseline: {args.baseline})")
    print(f"budget={C.MAX_NEW_TOKENS}  novelty={C.NOVELTY_THRESHOLD}  tau={C.TAU_SEMANTIC}")
    print("=" * 108)

    # Health check first — this is what phase-4 lacked.
    print("\nRUN HEALTH")
    for arm in sorted(by_arm):
        rs = list(by_arm[arm].values())
        tr = sum(r["truncated"] for r in rs)
        empt = sum(1 for r in rs if not str(r["answer"]).strip())
        aug = sum(r.get("n_augmentation", 0) or 0 for r in rs)
        esc = sum(1 for r in rs if r.get("escape_hatch"))
        print(f"  {arm:<18s} n={len(rs):4d} truncated={tr:4d} ({tr/len(rs)*100:4.1f}%) "
              f"empty={empt:3d} aug_steps={aug:4d} escape_on={esc:4d}")
    worst = max((sum(r["truncated"] for r in v.values()) / len(v)
                 for v in by_arm.values()), default=0)
    if worst > 0.10:
        print(f"  WARNING: truncation {worst*100:.1f}% exceeds the 10% target — "
              f"results are budget-limited, like phase-4")

    # Paired comparisons
    for arm in sorted(by_arm):
        if arm == args.baseline:
            continue
        cur = by_arm[arm]
        shared = sorted(set(base) & set(cur))
        print(f"\n{'-'*108}\n{arm}  vs  {args.baseline}   ({len(shared)} paired problems)\n{'-'*108}")

        allp = [(base[p]["correct"], cur[p]["correct"]) for p in shared]
        fmt_block("OVERALL", allp)

        for src in ["MATH-500", "AIME"]:
            sp = [(base[p]["correct"], cur[p]["correct"]) for p in shared
                  if base[p]["source"] == src]
            fmt_block(src, sp)

        # finished vs truncated — never pool these
        fin = [(base[p]["correct"], cur[p]["correct"]) for p in shared
               if not base[p]["truncated"] and not cur[p]["truncated"]]
        tru = [(base[p]["correct"], cur[p]["correct"]) for p in shared
               if base[p]["truncated"] or cur[p]["truncated"]]
        fmt_block("finished in budget", fin)
        fmt_block("truncated", tru)

        # AIME-finished is the phase-6 target: 1.7B was -2.6pp there with
        # 30.8pp of headroom unused, which is the cleanest test of the fix.
        af = [(base[p]["correct"], cur[p]["correct"]) for p in shared
              if base[p]["source"] == "AIME"
              and not base[p]["truncated"] and not cur[p]["truncated"]]
        fmt_block("AIME finished (target)", af)

        # did the escape hatch fire, and did it help?
        fired = [p for p in shared
                 if "does not fit" in str(cur[p].get("raw", "")).lower()
                 or "doesn't fit" in str(cur[p].get("raw", "")).lower()]
        if fired:
            fp = [(base[p]["correct"], cur[p]["correct"]) for p in fired]
            fmt_block("escape hatch fired", fp)

    print("\n" + "=" * 108)
    print("W = arm fixed a baseline failure; L = arm broke a baseline success.")
    print("p from exact McNemar on discordant pairs; CI is a 10k paired bootstrap.")
    print("With ~15 losses, ~29 wins (net +14) are needed to reach p<0.05.")


if __name__ == "__main__":
    main()
