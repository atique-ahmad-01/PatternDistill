# Phase 3D — Combined MATH-500 + AIME Evaluation

## Overview

Phase 3D evaluates **Anchor-guided Thought Graph Injection (ATGI)** on a combined benchmark of **287 problems** (100 MATH-500 + 187 AIME) using a **Qwen3-1.7B** student model on a single RTX 5070 Ti GPU.

Unlike earlier phases that evaluated datasets separately, phase 3D uses a **combined thought graph** built from both MATH-500 and AIME training patterns, enabling cross-dataset transfer.

---

## Approach — Hybrid ATGI

Combines the best of three prior approaches:

| Source | What we take | Why it helps |
|---|---|---|
| **RoT** (Reasoning on Thought) | Step-level graph + traversal | Assembles novel step combinations across patterns instead of dumping full templates |
| **BoT** (Boosting of Thought) | Template reuse | Reuses proven reasoning structures, but we skip BoT's extra LLM distiller call |
| **Phase-2d** (our prior best) | Worked examples + per-subject thresholds | Concrete symbolic grounding + selective injection avoids hurting easy problems |

### Why this hybrid beats pure approaches

- **vs Pure Top-K (phase 3A)**: Top-K dumps 3 full patterns → overwhelms 1.7B context. ATGI selects only the relevant steps → lean, focused prompt.
- **vs Pure BoT (phase 3B)**: BoT's `<think>` injection was too rigid for 1.7B (+0.0pp). ATGI uses prompt-level injection which the model follows better.
- **vs Pure RoT (phase 3C)**: RoT assembles steps only, no worked example. Adding the best-matching worked example (proven in phase-2d) provides concrete symbolic grounding.

### Key design choices

- **Anchor-first retrieval**: search the full pattern bank first, then extract steps from the anchor's graph neighborhood (learned from phase-3: pure step-level search loses discriminative power)
- **Same-source bias** (+0.05): prefer anchors from the same dataset (MATH→MATH, AIME→AIME) while still allowing cross-dataset transfer
- **Per-subject similarity thresholds**: IA=0.60, NT=0.65, Precalc=0.65, Geom=0.55, default=0.50 (from phase-2d)
- **Novelty filter** (0.85): bonus graph steps must be sufficiently different from anchor steps

---

## Pipeline

### 1. Pattern Generation (`generate_patterns.py`)
- Extracts reasoning patterns from training splits of MATH-500 (400 problems) and AIME (746 problems)
- Each pattern has: anchor problem, step-by-step strategy, worked example
- Output: `data/patterns/patterns.jsonl` (1,146 patterns)

### 2. Memory Bank & Graph (`build_memory_and_graph.py`)
- Embeds all patterns using `nomic-embed-text-v1`
- Builds FAISS index for fast similarity search (`memory/bank.index`)
- Constructs a **thought graph** (`graph/thought_graph.json`) linking reasoning steps across patterns
- Graph nodes = individual reasoning steps; edges = semantic similarity between steps from different patterns

### 3. Evaluation (`inject_and_eval_combined.py`)

For each test problem:

```
Test Problem → Embed → Search combined bank (MATH + AIME)
                         ↓
                    Best anchor (similarity > threshold)
                         ↓
              Retrieve reasoning steps from graph
                         ↓
              Inject strategy + example into prompt
                         ↓
              Run Qwen3-1.7B (vanilla vs ATGI)
```

- **Vanilla**: plain prompt, no guidance
- **ATGI**: prompt includes retrieved reasoning strategy + reference example
- Same-source bias (+0.05) prefers anchors from the same dataset
- Similarity thresholds: MATH-500 varies by subject (0.50–0.65), AIME default 0.50
- Problems below threshold are skipped (20 problems)
- Max 16,384 new tokens per generation

---

## Results (Fixed Extraction)

### Accuracy

| Dataset | N | Vanilla | ATGI | Wins ↑ | Losses ↓ | Net | Delta |
|---|---|---|---|---|---|---|---|
| **ALL** | 287 | 146 (50.9%) | 156 (54.4%) | 26 | 16 | +10 | **+3.5pp** |
| MATH-500 | 100 | 85 (85.0%) | 89 (89.0%) | 5 | 1 | +4 | **+4.0pp** |
| AIME | 187 | 61 (32.6%) | 67 (35.8%) | 21 | 15 | +6 | **+3.2pp** |

### Outcome Breakdown

| Category | Count |
|---|---|
| Both correct | 130 |
| Both wrong | 115 |
| ATGI wins (↑) | 26 |
| ATGI losses (↓) | 16 |
| Skipped (low sim) | 20 |

### Latency & Token Usage

| Metric | Vanilla | ATGI | Diff |
|---|---|---|---|
| Avg latency | 96.2s | 88.1s | **-8.1s** |
| Median latency | 93.0s | 73.9s | **-19.1s** |
| Avg tokens | 9,937 | 9,165 | **-772** |
| Total tokens | 2,851,983 | 2,630,409 | **-221,574** |
| Hit token limit (16K) | 102 (35.5%) | 71 (24.7%) | **-31** |

### Efficiency by Outcome

| Category | N | Van lat | ATGI lat | Δ lat | Van tok | ATGI tok | Δ tok |
|---|---|---|---|---|---|---|---|
| Both correct | 130 | 39.7s | 38.7s | -1.0s | 4,578 | 4,433 | -146 |
| Both wrong | 115 | 150.1s | 140.7s | -9.5s | 14,992 | 14,137 | -855 |
| **ATGI wins** | 26 | 137.4s | **80.8s** | **-56.6s** | 13,832 | **8,754** | **-5,077** |
| ATGI losses | 16 | 100.5s | 122.5s | +22.0s | 10,817 | 12,552 | +1,735 |

When ATGI wins, it uses **5,077 fewer tokens** and finishes **56.6s faster** than vanilla (which burns the full budget and still gets it wrong).

### AIME by Year

| Year | N | Vanilla | ATGI | Net | Delta |
|---|---|---|---|---|---|
| 2018 | 24 | 29.2% | 41.7% | +3 | **+12.5pp** |
| 2019 | 30 | 40.0% | 40.0% | 0 | +0.0pp |
| 2020 | 30 | 36.7% | 43.3% | +2 | **+6.7pp** |
| 2021 | 30 | 23.3% | 40.0% | +5 | **+16.7pp** |
| 2022 | 30 | 33.3% | 26.7% | -2 | -6.7pp |
| 2023 | 29 | 34.5% | 31.0% | -1 | -3.4pp |
| 2024 | 14 | 28.6% | 21.4% | -1 | -7.1pp |

ATGI helps strongly on 2018–2021 AIME (+12.5pp to +16.7pp) but hurts on 2022–2024. The pattern bank was built from training data that is stylistically closer to older problems.

---

## ATGI Wins (26) — Correct where Vanilla failed

| ID | Source | Gold | Vanilla | ATGI |
|---|---|---|---|---|
| 2018-I-8 | AIME | 147 | 2 | 147 |
| 2018-I-13 | AIME | 126 | 21 | 126 |
| 2018-II-4 | AIME | 23 | 2 | 23 |
| 2018-II-8 | AIME | 556 | 1 | 556 |
| 2019-II-8 | AIME | 53 | 201 | 53 |
| 2019-II-9 | AIME | 472 | 372 | 472 |
| 2019-II-11 | AIME | 11 | *(empty)* | 11 |
| 2020-I-7 | AIME | 81 | 19 | 81 |
| 2020-II-4 | AIME | 108 | *(empty)* | 108 |
| 2020-II-5 | AIME | 151 | 4194301 | 151 |
| 2021-I-8 | AIME | 57 | 20 | 57 |
| 2021-I-9 | AIME | 567 | 1134 | 567 |
| 2021-I-13 | AIME | 672 | 2 | 672 |
| 2021-II-7 | AIME | 145 | 3 | 145 |
| 2021-II-12 | AIME | 47 | 2 | 47 |
| 2022-I-2 | AIME | 227 | 111 | 227 |
| 2022-I-3 | AIME | 242 | 92 | 242 |
| 2023-II-2 | AIME | 585 | 945 | 585 |
| 2023-II-5 | AIME | 719 | 5 | 719 |
| 2024-II-10 | AIME | 468 | 429 | 468 |
| 2024-II-13 | AIME | 321 | *(empty)* | 321 |
| math_425 | MATH-500 | 3R^2 | *(ramble)* | 3R^2 |
| math_454 | MATH-500 | y = 2x + 3 | 3 | y = 2x + 3 |
| math_458 | MATH-500 | 41 | *(ramble)* | 41 |
| math_473 | MATH-500 | 7 | 25.5 | 7 |
| math_491 | MATH-500 | 64 | 192 | 64 |

Common pattern: vanilla either hits the token limit and produces no answer, or computes a wildly wrong number. ATGI's injected strategy guides the model to the correct approach efficiently.

## ATGI Losses (16) — Vanilla correct, ATGI wrong

| ID | Source | Gold | Vanilla | ATGI | Reason |
|---|---|---|---|---|---|
| 2018-II-9 | AIME | 184 | 184 | 2 | Wrong computation |
| 2019-I-5 | AIME | 252 | 252 | 258 | Off by 6 |
| 2019-II-1 | AIME | 59 | 59 | 0 | Wrong approach |
| 2019-II-15 | AIME | 574 | 574 | 3345455 | Token limit, no answer |
| 2020-I-14 | AIME | 85 | 85 | 49 | Wrong formula |
| 2022-I-5 | AIME | 550 | 550 | *(empty)* | Token limit |
| 2022-II-7 | AIME | 192 | 192 | 27 | Wrong computation |
| 2022-II-11 | AIME | 180 | 180 | *(empty)* | Token limit |
| 2022-II-13 | AIME | 220 | 220 | *(empty)* | Token limit |
| 2023-I-4 | AIME | 12 | 12 | 13 | Off by 1 |
| 2023-I-6 | AIME | 51 | 51 | 13 | Wrong approach |
| 2023-I-10 | AIME | 944 | 944 | 1349 | Wrong computation |
| 2024-II-2 | AIME | 236 | 236 | 4 | Misguided strategy |
| 2024-II-6 | AIME | 55 | 55 | 675 | Wrong approach |
| 2024-II-11 | AIME | 601 | 601 | 605 | Off by 4 |
| math_497 | MATH-500 | 7/2 | 7/2 | 3 | Wrong computation |

Loss causes: 4 token limit (empty), 3 off-by-small-number, 9 wrong computation/approach. The injected strategy sometimes leads the model down a wrong path, particularly on 2022+ AIME where problem styles have shifted from the training patterns.

---

## Empty/Failed Answers

All empty answers are caused by **max token exhaustion** (16,384 tokens). The model runs out of space while still reasoning and never produces a `\boxed{}` answer.

| Category | Count |
|---|---|
| Vanilla empty, ATGI has answer | 17 |
| ATGI empty, Vanilla has answer | 7 |
| Both empty | 6 |
| **Total** | **30** |

ATGI reduces token limit hits from 102 to 71 (30% fewer), showing that pattern injection helps the model reach conclusions faster.

---

## Answer Extraction Fix

The original extraction missed some correct answers due to formatting. Fixed cases:

| Problem | Issue | Old | Fixed |
|---|---|---|---|
| math_467 | `\mbox{ cm}^2` unit not stripped | both ✗ | both ✓ |
| math_472 | `$ 137 \frac{1}{2} $ miles.` trailing text | ATGI ✗ | ATGI ✓ |
| math_476 | `Jim walks 200 feet less...` sentence | ATGI ✗ | ATGI ✓ |
| math_499 | `106^\circ` vs `106` degree mismatch | both ✗ | both ✓ |

Fixes applied: strip trailing unit words, handle `$...$` with trailing text, normalize degree symbols for comparison.

---

## Files

```
phase-3d/
├── data/patterns/          # 1,146 extracted reasoning patterns
├── memory/                 # FAISS index + pattern bank (AIME portion)
├── graph/                  # Thought graph (nodes, edges, embeddings)
├── results/
│   ├── results_combined.jsonl       # Raw results (original extraction)
│   ├── results_combined_fixed.jsonl # Results with fixed extraction
│   └── summary_combined_fixed.json  # Summary stats
├── logs/                   # Generation and eval logs
├── generate_patterns.py    # Step 1: extract patterns
├── build_memory_and_graph.py  # Step 2: build bank + graph
├── inject_and_eval_combined.py  # Step 3: run evaluation
└── reeval.py               # Re-parse results with fixed extraction
```
