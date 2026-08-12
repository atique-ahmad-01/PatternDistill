# Phase 3 — Adaptive Thought Graph Injection (ATGI)

> Can we improve reasoning transfer by dynamically assembling step-level templates from a thought graph, instead of retrieving a single static pattern?

## Results — All Phases Comparison

### Overall

| Phase | Method | Vanilla | Injection | Delta | Gains | Losses | Gap Closed |
|-------|--------|---------|-----------|-------|-------|--------|------------|
| 1 | Baseline (no injection) | 50.8% (1.7B) | — | — | — | — | — |
| 1 | Baseline (teacher) | 71.8% (8B) | — | — | — | — | — |
| 2 | Global, 1-sentence pattern | 70.0% | 69.0% | -1.0pp | — | — | — |
| 2b | Subject-filtered | 70.0% | 68.0% | -2.0pp | — | — | — |
| 2c | Step-level, IA only | 70.0% | 72.0% | +2.0pp | — | — | 9.5% |
| 2d | Step-level, all subjects | 70.0% | 76.0% | **+6.0pp** | — | — | 28.6% |
| 3 v1 | Graph traversal (RoT-style) | 70.0% | 73.0% | +3.0pp | 6 | 3 | 14.3% |
| **3 v2** | **Anchor-first + graph** | 70.0% | **76.0%** | **+6.0pp** | **10** | **4** | **28.6%** |
| 3b | Think injection (`<think>` TI) | 70.0% | 70.0% | +0.0pp | 8 | 8 | 0% |

### By Subject (all methods on 100 test problems)

| Subject | n | Vanilla | 2d | 3 v1 | **3 v2** | 3b TI |
|---------|---|---------|-----|------|----------|-------|
| Algebra | 23 | 87.0% | 91.3% | 91.3% | **91.3%** | 82.6% |
| C&P | 6 | 66.7% | 83.3% | 66.7% | **83.3%** | 66.7% |
| Geometry | 5 | 40.0% | 40.0% | 40.0% | 40.0% | **60.0%** |
| Int. Algebra | 22 | 63.6% | 68.2% | 63.6% | 59.1% | 63.6% |
| Number Theory | 16 | 87.5% | 87.5% | 87.5% | **87.5%** | 81.2% |
| Prealgebra | 15 | 53.3% | 66.7% | 66.7% | **73.3%** | 66.7% |
| Precalculus | 13 | 61.5% | 69.2% | 61.5% | **76.9%** | 53.8% |

### By Difficulty (all methods)

| Level | n | Vanilla | 2d | 3 v1 | **3 v2** | 3b TI |
|-------|---|---------|-----|------|----------|-------|
| L1 | 9 | 88.9% | 77.8% | 77.8% | 77.8% | 66.7% |
| L2 | 15 | 73.3% | 80.0% | 80.0% | 73.3% | 66.7% |
| L3 | 21 | 85.7% | 85.7% | 90.5% | **95.2%** | 85.7% |
| L4 | 21 | 71.4% | 85.7% | 85.7% | **80.9%** | 76.2% |
| L5 | 34 | 52.9% | 61.8% | 50.0% | **61.8%** | 58.8% |

### Key Findings

1. **Phase-3 v2 (anchor-first + graph) = best overall architecture** — matches phase-2d's +6.0pp while enabling future multi-dataset extension
2. **Prealgebra (+20pp) and Precalculus (+15.4pp) are v2's strongest gains** — better than phase-2d
3. **Phase-3b TI failed (+0.0pp)** — `<think>` injection too rigid for 1.7B; the model needs freedom to reason around the strategy
4. **TI's one win: Geometry L5** — problem 400 solved for the first time ever (never solved in 2d, 3v1, or 3v2)
5. **L1 regression persists across ALL methods** — injection distracts on easy problems regardless of approach

### By Subject (v2)

| Subject | n | Vanilla | ATGI v2 | Delta | vs Phase-2d |
|---------|---|---------|---------|-------|-------------|
| **Prealgebra** | 15 | 53.3% | **73.3%** | **+20.0pp** | +6.7pp better |
| **C&P** | 6 | 66.7% | **83.3%** | **+16.7pp** | same |
| **Precalculus** | 13 | 61.5% | **76.9%** | **+15.4pp** | +7.7pp better |
| Algebra | 23 | 87.0% | 91.3% | +4.3pp | same |
| Number Theory | 16 | 87.5% | 87.5% | 0.0pp | same |
| Geometry | 5 | 40.0% | 40.0% | 0.0pp | same |
| Int. Algebra | 22 | 63.6% | 59.1% | -4.5pp | was +4.5pp |

### By Difficulty (v2)

| Level | n | Vanilla | ATGI v2 | Delta |
|-------|---|---------|---------|-------|
| **L3** | 21 | 85.7% | **95.2%** | **+9.5pp** |
| **L4** | 21 | 71.4% | **80.9%** | **+9.5pp** |
| **L5** | 34 | 52.9% | **61.8%** | **+8.8pp** |
| L2 | 15 | 73.3% | 73.3% | 0.0pp |
| L1 | 9 | 88.9% | 77.8% | -11.1pp |

### Injection vs Skip (v2)

81/100 injected, **19 skipped** (identical to phase-2d). Skip breakdown:

| Subject | Skipped | Threshold | Note |
|---------|---------|-----------|------|
| Number Theory | 12 | 0.65 | Specialized topics, strictest threshold |
| Intermediate Algebra | 3 | 0.60 | Below threshold |
| Prealgebra | 2 | 0.50 | Very low similarity |
| Precalculus | 2 | 0.65 | Strictest threshold |

Number Theory dominates skips (12/19) — NT problems are highly specialized (divisibility, modular arithmetic), so the closest bank pattern often doesn't clear the 0.65 bar.

---

## Motivation

Phase-2d achieved **+6.0pp** (70% → 76%) by injecting the top-1 most similar pattern into the student's prompt. But this approach has key limitations:

- **Static retrieval**: always injects one fixed pattern, even if the best match is from a different sub-topic
- **No recombination**: can't mix Step 1 from one pattern with Step 2 from another
- **Geometry/Number Theory plateau**: 0pp improvement — the closest full pattern isn't good enough

Phase 3 addresses these by decomposing patterns into individual reasoning steps, organizing them in a graph, and enabling cross-pattern step assembly when beneficial.

## Method

Inspired by:
- **RoT** (Retrieval-of-Thought, ICLR 2026) — step-level thought graph + reward-guided traversal
- **BoT** (Buffer of Thoughts, NeurIPS 2024) — reusable thought-templates + adaptive instantiation
- **Phase-2d** — worked examples + per-subject thresholds (proven effective)

---

## v1 vs v2: What We Learned

### v1: Pure Graph Traversal (+3.0pp)

```
  Test Problem Q
       │
       ▼
  ┌─────────────┐
  │ Step-Level   │     Thought Graph (1600 nodes)
  │ Embedding    │──────────────────────────────┐
  └──────────────┘                              │
                                                ▼
                              ┌──────────────────────────┐
                              │ Subject-Filtered         │
                              │ Initial Node Retrieval   │
                              │ R = 0.8·sim + 0.2·is_s0  │
                              └──────────┬───────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │ Greedy Reward-Guided     │
                              │ Traversal (max 6 steps)  │
                              │ R = 0.6·sim + 0.4·flow   │
                              │ sequential + semantic     │
                              └──────────┬───────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │ Assembled Template       │
                              │ 5-6 steps from           │
                              │ multiple patterns        │
                              └──────────┬───────────────┘
                                         │
                    ┌────────────────────┘
                    │     + Worked Example (top-1 from phase-2d bank)
                    ▼
            ┌──────────────┐
            │  Qwen3-1.7B  │
            └──────────────┘
```

**Why it underperformed:**
1. Step embeddings too generic — "Identify the equation" matches hundreds of problems
2. Greedy traversal created Frankenstein templates — mixing irrelevant steps from different patterns
3. Always filled to 6 steps — extra steps = noise for the 1.7B student
4. All 3 losses came from cross-pattern assemblies

### v2: Anchor-First + Novelty Gate (+6.0pp)

Full-problem FAISS → top-1 anchor (phase-2d style) → enter graph at anchor's nodes → novelty-gated bonus steps.

**Why it works:**
1. Full-problem embeddings give high discriminative power (avg sim ~0.66 vs v1's ~0.55)
2. Anchor pattern provides all 4 steps as a coherent unit
3. Bonus steps only added if genuinely novel (sim < 0.85 to all anchor steps)
4. Context-enriched graph embeddings ("problem_summary | step_text") make semantic edges 10x more selective

---

## Architecture (v2)

```
  Test Problem Q
       │
       ▼
  ┌─────────────────┐
  │ Full-Problem     │     Phase-2d FAISS Index
  │ Embedding        │────────────────────────────┐
  └────────┬────────┘                             │
           │                                      ▼
           │                            ┌──────────────────┐
           │                            │ Top-1 Anchor     │
           │                            │ (pattern + example)│
           │                            └────────┬─────────┘
           │                                     │
           │              ┌──────────────────────┘
           │              │
           ▼              ▼
  ┌────────────────────────────────┐
  │  Thought Graph (1600 nodes)    │
  │  Enter at anchor's Step 1     │
  │  Follow sequential edges      │
  │  Check semantic neighbors     │
  │  for novel bonus steps        │
  └──────────┬─────────────────────┘
             │
             ▼
  ┌──────────────────────┐     ┌──────────────────┐
  │ Anchor's 4 Steps     │     │ Best Worked      │
  │ + 0-1 Bonus Steps    │  +  │ Example          │
  └──────────┬───────────┘     └────────┬─────────┘
             │                          │
             ▼                          ▼
  ┌──────────────────────────────────────┐
  │         Injection Prompt             │
  └──────────────────┬───────────────────┘
                     │
                     ▼
             ┌──────────────┐
             │  Qwen3-1.7B  │
             └──────────────┘
```

### Pipeline (3 scripts)

| Step | Script | Description | Time |
|------|--------|-------------|------|
| 1 | `build_thought_graph.py` | Parse 400 patterns → 1600 nodes, context-enriched embed, build graph | ~2s |
| 2 | `inject_and_eval_atgi.py` | Anchor retrieval + graph expansion + eval on 100 test problems | ~90 min |
| 3 | `analyze_results.py` | Per-subject, per-difficulty, traversal path analysis | <1 min |

---

## Thought Graph v2

### Context-Enriched Embeddings

Each step is embedded as `"{problem_summary} | {step_text}"` instead of just the step text. This gives step embeddings the discriminative power of full-problem embeddings while retaining step-level granularity.

### Graph Stats (v2, τ=0.90)

| Metric | v1 | v2 |
|--------|----|----|
| Nodes | 1,600 | 1,600 |
| Sequential edges | 1,200 | 1,200 |
| Semantic edges | 28,312 | **2,830** (10x fewer) |
| Avg degree | 18.4 | **2.5** |
| Max degree | 157 | **27** |
| Same-subject pairs | 4,808 | 1,037 |
| Cross-subject pairs | 9,348 | **378** |

---

## Algorithm (v2)

### Step 1: Anchor Retrieval
Full-problem embedding → phase-2d FAISS index → top-1 match. Same as phase-2d. Per-subject thresholds: IA=0.60, NT=0.65, Precalc=0.65, Geom=0.55, default=0.50.

### Step 2: Enter Graph
Find anchor pattern's 4 nodes in graph (Step 1, Step 2, Step 3, Note). These form the base template.

### Step 3: Novelty-Gated Expansion
Check semantic neighbors of anchor nodes. A bonus step is added only if:
- From a different pattern
- Novel: max sim to all anchor steps < 0.85
- Most relevant to query (sorted by query similarity)
- Max 1 bonus step

### Step 4: Prompt Construction
Anchor's 4 steps + 0-1 bonus + worked example from anchor → injection prompt.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| MAX_BONUS_STEPS | 1 | Max cross-pattern steps to add |
| NOVELTY_THRESHOLD | 0.85 | Bonus step must be < this sim to anchor steps |
| τ_semantic | 0.90 | Semantic edge threshold in graph |

---

## Usage

```bash
# On GPU server (students@<gpu-host>)
cd ~/PatternDistill/phase-3

# Step 1: Build thought graph v2
python3 build_thought_graph.py                    # full build (~2s)
python3 build_thought_graph.py --max-patterns 5   # smoke test

# Step 2: Evaluate with ATGI v2
python3 inject_and_eval_atgi.py                   # full eval (~90 min)
python3 inject_and_eval_atgi.py --max-problems 5  # smoke test
```

---

## File Structure

```
phase-3/
├── build_thought_graph.py         # Build context-enriched thought graph
├── inject_and_eval_atgi.py        # Anchor-first eval with graph expansion
├── analyze_results.py             # Analysis & diagnostics (TBD)
├── graph/
│   ├── thought_graph.json         # 1600 nodes + adjacency lists
│   ├── step_embeddings.npy        # (1600, 768) context-enriched
│   └── step_faiss.index           # FAISS IndexFlatIP
├── results/
│   ├── results.json               # per-problem evaluation data
│   ├── results.jsonl              # incremental output
│   └── summary.json               # aggregate stats
├── logs/
│   └── *.log / *.out              # timestamped run logs
├── paper_notes_and_plan.txt       # BoT/RoT paper notes
├── todo_phase3_atgi.txt           # implementation checklist
└── Readme.md                      # this file
```

---

## Phase 3B — Thinking Intervention (TI)

**Directory:** `phase-3b/`

Instead of injecting the strategy in the user prompt, TI places it inside `<think>` tags as a prefix to the model's internal reasoning.

### How it works

```
Prompt injection (v2):                Think injection (3b):

USER: "Reasoning Strategy:            USER: "Problem: ..."
  Step 1: ...
  Step 2: ...                         MODEL generates:
  Problem: ..."                        <think>              ← we inject this
                                       I recall a strategy:
MODEL generates:                       Step 1: ...
  <think>                              Step 2: ...
  [may ignore strategy]                Now let me apply...  ← model continues
  [explores wrong paths]               [follows strategy]
  </think>                             </think>
  The answer is X                      The answer is X
```

### Result: +0.0pp (8 gains, 8 losses)

TI anchors reasoning **too aggressively** for the 1.7B student. The model treats injected `<think>` content as its own thought and follows it literally — even when the strategy doesn't perfectly match. With prompt injection (v2), the model can deviate in its thinking when it recognizes the strategy doesn't fit.

### What TI did right
- **Geometry L5 (id=400)**: solved for the first time ever — TI's rigid anchoring broke through where all other methods failed
- **Latency reduction**: 71s→5s on some problems — model does less thinking (but less thinking = less self-correction)
- **L5 +5.9pp**: slight improvement on hardest problems

### What TI broke
- **Algebra -8.7pp**: model followed bad strategies too literally
- **Precalculus -23.1pp**: worst regression of any method
- **L1 -22.2pp**: easy problems badly disrupted (worst L1 regression)

### Conclusion

For 1.7B models, prompt injection > think injection. The model needs freedom to reason around the strategy, not be locked into it. TI may work better for larger models (4B+) that can integrate and adapt injected reasoning.

### Files

```
phase-3b/
├── inject_and_eval_ti.py    # TI evaluation script
├── results/                 # results.json, summary.json
└── logs/                    # run logs
```

---

## Future: Multi-Dataset Extension

The graph architecture is designed for cross-domain reasoning transfer:

```
MATH-500 patterns ──┐
GSM8K patterns ─────┼── Thought Graph ── semantic edges ── cross-domain step reuse
MBPP patterns ──────┘
```

On MATH-500 alone, anchor-first with 0 bonus steps was sufficient. The graph's semantic edges will add value when reasoning steps from different domains can complement each other (e.g., "set up equation" reused across math word problems and code generation).

---

## References

- **RoT**: Ahmed et al., "Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts", ICLR 2026. [arXiv:2509.21743](https://arxiv.org/abs/2509.21743)
- **BoT**: Yang et al., "Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models", NeurIPS 2024. [arXiv:2406.04271](https://arxiv.org/abs/2406.04271)
