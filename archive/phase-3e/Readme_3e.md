# Phase 3E — Adaptive Token-Budget Strategy (ATBS)

## Overview

Phase 3E makes **structural improvements** over phase-3D based on deep analysis of why ATGI wins and loses. The core finding: ATGI's improvement is almost entirely explained by **keeping the model within its token budget** (69% of wins = vanilla hit 16K limit, ATGI didn't).

Evaluated on 287 problems (100 MATH-500 + 187 AIME) using Qwen3-1.7B.

---

## What Changed from Phase 3D

| Change | Phase 3D | Phase 3E | Why |
|---|---|---|---|
| Token budget | 16,384 | **24,576** | 100% of 16K hits failed — more room = more correct answers |
| Worked example | Injected (~120 tokens) | **Dropped** | Specific numbers contaminate arithmetic |
| Graph / bonus steps | 1 bonus step from RoT graph | **None** | 0/26 wins had bonus steps — dead weight |
| Early-exit | None | **Stop on \\boxed{}** | Saves 200-500 tokens after answer |
| Conciseness | None | **Explicit instruction** | "Be direct, avoid restating" reduces verbosity |
| Retry | None | **2nd-best template** | If 1st template produces no answer, try different angle |

---

## Pipeline

```
Test Problem
     |
     v
  RETRIEVE top-2 templates from FAISS buffer
     |
     v
  VANILLA RUN (24K tokens, early-exit on \boxed{})
     → baseline answer
     |
     v
  ATBS ATTEMPT 1: best template (24K tokens, early-exit)
     → conciseness instruction + strategy template
     → if \boxed{} found → DONE
     → if NO \boxed{} ↓
     |
     v
  ATBS ATTEMPT 2: 2nd-best template (24K tokens, early-exit)
     → different reasoning angle
     → record answer
```

---

## Run

### Step 1: Build Buffer (~5 min)

```bash
python build_buffer.py
```

### Step 2: Evaluate (~10-12 hours)

```bash
python inject_and_eval_atbs.py --max-problems 5   # smoke test
python inject_and_eval_atbs.py                     # full run
```

---

## Files

```
phase-3e/
├── buffer/                    # FAISS buffer (strategy templates only)
├── results/                   # Evaluation results
├── logs/                      # Run logs
├── build_buffer.py            # Step 1: build FAISS buffer
├── inject_and_eval_atbs.py    # Step 2: ATBS evaluation
└── Readme_3e.md
```

---

## ⚠️ CORRECTION — the numbers below are from an incomplete run

**The completed run does not support the conclusions in this document.** The
tables below were written from a mid-run snapshot. Re-reading the finished
`results/results.jsonl` (262 problems) gives:

| Scope | n | Vanilla | ATBS | Delta | W | L | p (exact McNemar) |
|---|---|---|---|---|---|---|---|
| **ALL** | 262 | 56.1% | 56.5% | **+0.4pp** | 22 | 21 | **1.000** |
| MATH-500 | 100 | 83.0% | 86.0% | +3.0pp | 3 | 0 | 0.250 |
| **AIME** | 162 | 39.5% | 38.3% | **−1.2pp** | 19 | 21 | 0.875 |

The snapshot showed 15 wins against 7 losses; the completed run shows **22
wins against 21 losses** — the losses caught up as more AIME problems entered
the run. The overall effect is +0.4pp, statistically indistinguishable from
zero.

Note also that `results/results_clean.jsonl` (140 problems, +2.9pp) is a
favourable subset: the other 122 problems score **−2.5pp**. Reporting the
clean subset alone overstates the result.

Nothing below is corrected in place — it is kept as a record of what the
snapshot appeared to show. Phase-6 exists partly because of this: it reports
p-values, never draws conclusions from partial runs, and never pools
truncated with finished problems.

## Results (in-progress snapshot: 255/287 — superseded, see correction above)

### Overall

| Metric | Vanilla | ATBS | Delta |
|--------|---------|------|-------|
| **Accuracy** | 190/255 (74.5%) | 198/255 (77.6%) | **+8 (+3.1pp)** |
| Gains (^) | — | 15 | — |
| Losses (v) | — | 7 | — |
| Net | — | **+8** | — |
| Avg tokens | 6,234 | 5,597 | **−10.2%** |
| Hit 20K limit | 26 | 14 | −46% |

### By Dataset

| Dataset | Vanilla | ATBS | Delta |
|---------|---------|------|-------|
| MATH-500 (199) | 167 (83.9%) | 173 (86.9%) | **+6 (+3.0pp)** |
| AIME (56) | 23 (41.1%) | 25 (44.6%) | **+2 (+3.6pp)** |

### Retry Mechanism

- 12 problems had no answer on attempt 1
- 5 triggered 2nd-template retry
- 2 of those retries produced correct answers

### Key Observations

1. **Token efficiency**: ATBS uses ~10% fewer tokens on average while being more accurate — conciseness instruction + early-exit working as intended.
2. **Budget headroom matters**: ATBS cuts token-limit hits from 26 → 14 (−46%), confirming the phase-3D finding that budget exhaustion is the primary failure mode.
3. **MATH-500 gains are clean**: +6 gains, 0 losses — pure improvement on easier problems.
4. **AIME is noisier**: +9 gains but −7 losses — harder problems show more variance, though net is still positive (+2).

### Comparison with Prior Phases

| Phase | Method | MATH-500 Δ | AIME Δ | Overall Δ |
|-------|--------|-----------|--------|-----------|
| 3 (ATGI v2) | Graph + worked example | +6.0pp | — | +6.0pp |
| 3B (TI) | `<think>` injection | +0.0pp | — | +0.0pp |
| 3D (Combined) | Graph + template + worked example | +4.6pp | +2.1pp | +3.1pp |
| **3E (ATBS)** | **Strategy-only + conciseness + retry** | **+3.0pp** | **+3.6pp** | **+3.1pp** |

Phase 3E matches 3D's overall delta (+3.1pp) with a **much simpler pipeline** — no graph, no worked examples, just strategy templates + token discipline.

> **Superseded.** On the completed 262-problem run this claim does not hold:
> the overall delta is **+0.4pp (p=1.000)**, with AIME at **−1.2pp**. See the
> correction at the top of this file. The "simpler pipeline matches the
> complex one" conclusion was an artifact of stopping the comparison early.
>
> One further caveat that applies to phase 3D as well: the graph-augmentation
> step both phases relied on was **inert**. Its novelty cutoff (0.85) sat below
> the graph's edge threshold (0.90), so every candidate step was rejected by
> construction — verified as 2,279 candidates considered, 2,279 rejected, 0
> used. Phase 3E's decision to drop graph steps as "dead weight" was therefore
> measuring a bug, not the graph.
