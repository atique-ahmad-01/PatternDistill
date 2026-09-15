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
6. **The graph contributed nothing to v2's +6.0pp** — the novelty gate (0.85) sits below the semantic-edge threshold (τ=0.90), so every expansion candidate is rejected by construction and `bonus_steps == 0` on all 100 problems. v2 is effectively phase-2d with extra machinery. See [Readme_v2.md](Readme_v2.md#the-novelty-gate-never-fired)
7. **Injection is cheaper, not just better** — v2 cut mean tokens 5.4% and latency 5.8% while gaining accuracy; v1 gained half as much for no cost saving (+0.1% latency)

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

## v1 vs v2

Phase 3 was built twice. Both versions ran in this directory and **v2 overwrote v1 in place**
— see [Data provenance](#data-provenance) before citing either.

Each version has its own document:

| | [**v1** — Pure Graph Traversal](Readme_v1.md) | [**v2** — Anchor-First + Novelty Gate](Readme_v2.md) |
|---|---|---|
| Retrieval | step-level entry, greedy multi-hop traversal | top-1 problem-level anchor |
| Step embedding | `step_text` alone | `"{problem_summary} \| {step_text}"` |
| Semantic edges (τ=0.90) | 28,312 (avg degree 18.4) | 2,830 (avg degree 2.5) |
| Template | 5–6 steps, 2.1 patterns on average | 4 steps, always 1 pattern |
| Accuracy | 70.0% → 73.0% (**+3.0pp**) | 70.0% → 76.0% (**+6.0pp**) |
| Mean tokens | *not logged* | 3,407 → 3,223 (−5.4%) |
| Mean latency | 28.7s → 28.8s (+0.1%) | 28.7s → 27.0s (−5.8%) |
| Gains / losses | 6 / 3 | 10 / 4 |

**What changed the outcome:** only the embedding text. Same 1,600 nodes, same 1,200 sequential
edges — embedding `problem_summary | step_text` instead of bare `step_text` cut semantic edges
10× and made anchor-first retrieval viable (mean anchor sim 0.65 vs ~0.55).

**What did not change:** neither version's graph traversal actually contributed. v2's novelty
gate is dead by construction (0.85 < τ = 0.90), so `bonus_steps == 0` on all 100 problems —
v2's +6.0pp is top-1 anchor + worked example alone. Details in
[Readme_v2.md](Readme_v2.md#the-novelty-gate-never-fired).

Two caveats carried in the per-version docs: v1's "Frankenstein template" explanation is
**unestablished** (multi-pattern injections were net positive), and v1 has **two runs that
disagree** on injection rate (32/100 vs 79/100, both netting +3.0pp).

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

## Pattern anatomy

Every pattern in every bank has **exactly 4 nodes**, in this fixed shape:

```
Step 1  →  Step 2  →  Step 3  →  Note
```

The distillation prompt always produced three numbered steps plus one note, and `parse_steps`
splits on those labels, so there is no variation to observe:

| Bank | Patterns | Nodes | Nodes per pattern |
|---|---|---|---|
| MATH-500 (phase-2d) | 400 | 1,600 | 4.0 |
| AIME (phase-3d) | 746 | 2,984 | 4.0 |
| Combined graph (phase-3d, phase-4) | 1,146 | 4,584 | 4.0 |

Label counts in the phase-3 graph confirm it: `{'Step 1': 400, 'Step 2': 400, 'Step 3': 400,
'Note': 400}`, and `total_steps == 4` on all 1,600 nodes.

A real pattern (`math_0`, nodes 0–3), stored in `step_index` order and chained by `sequential`
edges:

```
idx=0  [Step 1]  Identify the formulas to convert rectangular coordinates (x, y) to polar…
idx=1  [Step 2]  Use the formulas r = √(x² + y²) and θ = tan⁻¹(…)
idx=2  [Step 3]  Calculate r and θ using the given values of x and y…
idx=3  [Note]    Ensure θ is within the range [0, 2π)
```

### What "Note" is

The Note is **not a fourth action** — it is the insight or trap the teacher flagged, and it
applies while the other three steps are being carried out. Two kinds appear:

| Kind | Examples from the bank |
|---|---|
| **Key insight** — the trick that makes the problem tractable | *"Recognize that 4 can be written as 2², which allows the bases to be matched."* · *"The key insight is that the minimum angle of rotation is determined by the order of rotational symmetry"* |
| **Caution** — the mistake that is easy to make | *"Remember that the cross product is anti-commutative, so a × b = −b × a"* · *"Be careful with the total number of coins and their individual values"* |

Because it is not an action, `assemble_strategy` renders it without a number:

```python
if label.lower().startswith("note"):
    lines.append(f"Note: {node['text']}")          # unnumbered
else:
    lines.append(f"Step {step_num}: {node['text']}")
    step_num += 1
```

So the student sees `Step 1 / Step 2 / Step 3 / Note`. Notes average 92 characters against 76
for Step 1 — consistent with a caveat needing its "why" where a step only needs its "what".

> **Open question.** The Note plausibly carries the highest-value content in the injection: it
> names a *specific* failure mode, which is the thing a 1.7B student is least likely to produce
> unaided, whereas the general procedure it often already knows. **No phase isolates this** —
> an ablation dropping the Note (or keeping only the Note) would settle it.

### Only training problems have nodes

| | Count | In the graph? |
|---|---|---|
| Bank / training problems | 1,146 | ✅ 4 nodes each — they *are* the graph |
| Test problems | 287 | ❌ none |

A test problem is embedded to a single 768-d vector, matched against the bank, and the
winner's 4 nodes are copied into the prompt. The graph is built once from training data and
stays frozen; test problems only query it.

### Two consequences of the fixed size

1. **Several reported metrics are constants, not distributions** — `anchor_steps == 4`,
   `template_length == 4`, `n_unique_patterns == 1` on every injected problem in v2, 3d and
   phase 4. There was nothing to vary.
2. **It is what forced v1's cross-pattern mixing.** With `max_steps = 6` and every pattern
   exactly 4 nodes long, v1's traversal *had* to jump to another problem's pattern to reach
   steps 5 and 6 — there is no step 5 inside one pattern. See
   [Readme_v1.md](Readme_v1.md#what-the-weights-actually-imply).

### Glossary

| Term | Plain meaning |
|---|---|
| **Anchor** | The single most similar bank problem to the test problem. Its 4 nodes become the injected strategy. |
| **Bonus step** | One extra step borrowed from a *different* problem's pattern and appended, to cover something the anchor missed. Max 1. |
| **Novelty gate** | The check deciding whether a bonus step is allowed in — it must say something genuinely new, not reword a step already present. **Rejected everything it ever saw**; see [Readme_v2.md](Readme_v2.md#the-novelty-gate-never-fired). |
| **Skip** | Anchor similarity below the per-subject threshold → no injection, vanilla prompt sent instead. |

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

⚠️ **As shipped, this step is a no-op** — the novelty bound (0.85) is tighter than the edge
threshold that produced the candidate (τ=0.90), so nothing survives. Measured: 0 bonus steps
across all 100 problems. See [Readme_v2.md](Readme_v2.md#the-novelty-gate-never-fired).

### Step 4: Prompt Construction
Anchor's 4 steps + 0-1 bonus + worked example from anchor → injection prompt.

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| MAX_BONUS_STEPS | 1 | Max cross-pattern steps to add |
| NOVELTY_THRESHOLD | 0.85 | Bonus step must be < this sim to anchor steps |
| τ_semantic | 0.90 | Semantic edge threshold in graph |

> ⚠️ **`NOVELTY_THRESHOLD` must exceed `τ_semantic`.** At 0.85 < 0.90 the expansion is dead
> code — see [Readme_v2.md](Readme_v2.md#the-novelty-gate-never-fired). Fixed in `phase-6/config.py`, which
> asserts the invariant at import so it cannot recur silently.

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
├── build_thought_graph.py         # Build context-enriched thought graph (v2)
├── inject_and_eval_atgi.py        # Anchor-first eval with graph expansion (v2)
├── analyze_results.py             # Analysis & diagnostics (TBD)
├── graph/
│   ├── thought_graph.json         # 1600 nodes + adjacency lists (v2 graph)
│   ├── step_embeddings.npy        # (1600, 768) context-enriched
│   └── step_faiss.index           # FAISS IndexFlatIP
├── results/
│   ├── results.json               # per-problem evaluation data (v2 run)
│   ├── results.jsonl              # incremental output
│   ├── summary.json               # aggregate stats
│   ├── plot_v1_architecture.py    # → phase3_v1_architecture.png
│   ├── plot_v1_results_table.py   # → phase3_v1_results_table.png (parses the v1 log)
│   ├── plot_v2_architecture.py    # → phase3_v2_architecture.png
│   ├── plot_v2_results_table.py   # → phase3_v2_results_table.png
│   └── plot_results_table.py      # → results_table_tok_lat_acc.png (all phases)
├── logs/
│   ├── inject_and_eval_atgi_20260725_223938.log   # the v1 run the README cites
│   ├── inject_and_eval_atgi_20260726_005100.log   # the v2 run
│   └── *.log / *.out              # other timestamped run logs
├── paper_notes_and_plan.txt       # BoT/RoT paper notes
├── todo_phase3_atgi.txt           # implementation checklist
└── Readme.md                      # this file
```

---

## Data provenance

v1 and v2 were developed and run in the same directory, and **v2 overwrote v1's artifacts**.
What survives of each:

| Artifact | v1 | v2 |
|---|---|---|
| Eval script | overwritten — no code on disk, no separate commit | `inject_and_eval_atgi.py` |
| Graph | overwritten (28,312 semantic edges) | `graph/thought_graph.json` (2,830, `v2-context-enriched`) |
| Per-problem results | overwritten — recovered by parsing the run log | `results/results.json` |
| Tokens per problem | never logged by the v1 script | in `results.json` |

`results/plot_v1_results_table.py` parses `logs/inject_and_eval_atgi_20260725_223938.log` and
reproduces the 70.0% → 73.0% / 6 gains / 3 losses figures cited above, which is what confirms
it is the run this README describes.

Two v1 full runs exist at the same config. The 19:20 run injected on only **32/100**
(4 gains / 1 loss); the 22:39 run injected on **79/100** (6 gains / 3 losses). Both net
+3.0pp. The gap in injection rate is unexplained and should be resolved before v1 is cited as
a baseline.

All phase-3 numbers predate the **gate-v2 pattern bank** (phase-6), so they are not directly
comparable to phase-6 results.

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

On MATH-500 alone, anchor-first reached +6.0pp with 0 bonus steps — but that was **not a
finding about MATH-500**, it was the dead novelty gate (`0.85 < τ = 0.90`, see
[Readme_v2.md](Readme_v2.md#the-novelty-gate-never-fired)). Phase-3c and phase-3d inherited the same
inverted thresholds, so they also ran with 0 bonus steps; their cross-domain transfer claims
were never actually tested. Phase-6 fixes the gate and re-measures.

---

## References

- **RoT**: Ahmed et al., "Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts", ICLR 2026. [arXiv:2509.21743](https://arxiv.org/abs/2509.21743)
- **BoT**: Yang et al., "Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models", NeurIPS 2024. [arXiv:2406.04271](https://arxiv.org/abs/2406.04271)
