# Phase 2 — Memory Injection (Method A)


  When you come back, just check progress with:                                                                                                                  
  ssh students@100.118.158.78                                                                                                                                    
  tmux attach -t phase2_hf
  
> **Hypothesis:** Injecting a retrieved reasoning pattern into the 1.7B student's prompt at inference time will meaningfully close the 21pp accuracy gap established in Phase 1.

---

## What We're Testing

Phase 1 showed a **21 percentage point gap** between Qwen 1.7B (50.8%) and Qwen 8B (71.8%) on MATH-500. Phase 2 asks:

> Can we recover some of that gap by retrieving a relevant reasoning pattern (extracted from the 8B teacher) and injecting it into the 1.7B student's prompt at inference time — without any fine-tuning?

This is a **pure inference-time intervention**. The 1.7B model's weights are never modified. If it works, it tells us the student already has the capacity but lacks the right reasoning scaffold at solve time.

---

## Approach Decision

### Why Memory Injection First

Three reasons this is the right method to run before fine-tuning (Phase 3):

1. **Fast iteration.** No training loop — results in hours, not days. Lets us validate pattern quality before committing to a fine-tuning run.
2. **Clean ablation.** Injection vs. vanilla is a controlled comparison: same model, same weights, only the prompt changes. Any accuracy delta is directly attributable to the pattern.
3. **Upper-bound signal.** If injection doesn't help much, it tells us the patterns themselves may be weak or retrieval is noisy — and we can fix that before fine-tuning on bad data.

---

## Injection Design (Research-Backed)

Several design choices here are non-obvious and matter for a 1.7B model. Here's what the research says and what we're doing.

### Prompt Structure

```
SYSTEM:
You are a precise math problem-solving assistant. When a Reasoning Strategy
is provided, apply it explicitly as your first step before solving the
problem. Show your work step by step.

USER:
Reasoning Strategy: {retrieved_pattern}

Problem: {math_problem}

Solution:
```

**Why this layout:**
- The system prompt sets a persistent behavioral directive once — not repeated per-problem.
- The pattern lands in the user turn, immediately adjacent to the problem. This is critical due to the **Lost in the Middle** effect: LLMs strongly attend to the beginning and end of their context. Anything buried mid-prompt degrades by >30% in practice. Pattern + problem together = both in high-attention zone.
- `Solution:` at the end is a lightweight generation nudge that reduces preamble and pushes the model directly into executing the strategy (tested specifically on Qwen-family models).

### Label Format

We use `Reasoning Strategy:` rather than alternatives like `Hint:` or `Note:` because:
- `Hint:` implies partial information and is associated with hedging in training data.
- `Reasoning Strategy:` is authoritative — signals a complete procedure, not a nudge.
- We avoid wrapping it in soft prose like "you may find this useful" — weak language causes small models to treat injected context as optional.

### Top-k Retrieval: Top-1 Only

We retrieve the single most similar pattern (cosine similarity via FAISS). Research on small models consistently shows:

- Performance saturates quickly as injected context grows; small models saturate faster than large ones.
- With top-3, patterns 2 and 3 land in the degraded mid-context attention zone.
- "Big Reasoning with Small Models" (arXiv:2510.13935) found top-1 retrieval outperforms top-k for SLMs — additional retrieved instructions helped 14B+ models but not small ones.

**Similarity threshold gate:** If top-1 cosine similarity < 0.70, we skip injection entirely and run vanilla. A weakly-matched pattern misleads more than it helps.

### Explicit Instruction: Once in System, Implicit at Injection

We do not repeat "please use this pattern" inline at the injection point. For 1.7B models, every extra sentence competes for a finite attention budget. The system prompt directive covers it; the injection itself is just the labeled pattern.

---

## Known Failure Modes to Watch

| Failure Mode | Description | Mitigation |
|---|---|---|
| **Pattern ignored** | Model generates confident answer, bypasses context | Explicit system directive + pattern immediately before problem |
| **Recited, not applied** | Model copies pattern phrases but solves differently | Patterns must be action-oriented ("First compute X...") not descriptive |
| **Prompt too long** | Instruction-following degrades >~400 tokens for 1.7B | Keep patterns to 1-2 sentences; monitor token count |
| **False-positive retrieval** | High cosine score but procedurally wrong match | Similarity threshold gate at 0.70; subject-level metadata filter |

---

## Pipeline

```
Phase 2A — Generate Patterns
  Run 8B teacher on MATH-500[0:400]
  Output: data/patterns/patterns.json
        ↓
Phase 2B — Build Memory Bank
  Embed patterns with nomic-embed-text → FAISS index
  Output: memory/bank.json + memory/bank.index
        ↓
Phase 2C — Inject & Evaluate
  Run on MATH-500[400:500] (100 held-out problems)
  For each: embed query → retrieve top-1 → run vanilla + injection
  Output: results/results.json + results/summary.json
```

Phases 2A → 2B → 2C must run in order. 2C produces the paired vanilla/injection comparison.

---

## Scripts

| Script | Phase | Description |
|---|---|---|
| `scripts/generate_patterns.py` | 2A | 8B teacher generates CoT + pattern + answer for 400 problems |
| `scripts/build_memory_bank.py` | 2B | Embeds patterns, builds FAISS index |
| `scripts/inject_and_eval.py` | 2C | Retrieves pattern, runs both conditions, scores results |

All scripts accept `--max-problems N` for local smoke testing (N=5) before full GPU runs.

---

## Data Format

### `data/patterns/patterns.json`

Output of Phase 2A. One entry per problem:

```json
[
  {
    "problem": "...",
    "subject": "Algebra",
    "level": 3,
    "gold_answer": "...",
    "cot": "...",
    "pattern": "When solving [X], first identify [Y], then apply [Z]...",
    "teacher_answer": "..."
  }
]
```

### `memory/bank.json`

Metadata store (mirrors patterns.json, used alongside FAISS index for retrieval).

### `memory/bank.index`

FAISS flat index (cosine similarity). Vectors from `nomic-embed-text`.

### `results/results.json`

One entry per held-out problem:

```json
[
  {
    "problem_id": 400,
    "subject": "Geometry",
    "level": 4,
    "gold_answer": "...",
    "retrieved_pattern": "...",
    "similarity_score": 0.83,
    "vanilla_answer": "...",
    "vanilla_correct": false,
    "vanilla_latency_s": 47.2,
    "injection_answer": "...",
    "injection_correct": true,
    "injection_latency_s": 51.6
  }
]
```

### `results/summary.json`

Aggregated metrics: accuracy by condition, by subject, by difficulty, latency, delta.

---

## Running

### Local smoke test (Mac, CPU)

```bash
cd phase-2

# Phase 2A
python scripts/generate_patterns.py --max-problems 5

# Phase 2B
python scripts/build_memory_bank.py

# Phase 2C
python scripts/inject_and_eval.py --max-problems 5
```

### Full run (GPU machine)

```bash
# Sync to GPU machine
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
  ~/Desktop/PatternDistill/phase-2/ \
  students@100.118.158.78:~/PatternDistill/phase-2/

# SSH in and run
ssh students@100.118.158.78
export CUDA_VISIBLE_DEVICES=0
cd ~/PatternDistill/phase-2

python scripts/generate_patterns.py   # ~3-4 hrs for 400 problems
python scripts/build_memory_bank.py   # minutes
python scripts/inject_and_eval.py     # ~2 hrs for 100 problems (runs each twice)
```

### Pull results back

```bash
rsync -avz \
  students@100.118.158.78:~/PatternDistill/phase-2/results/ \
  ~/Desktop/PatternDistill/phase-2/results/
```

---

## What Success Looks Like

| Outcome | Interpretation |
|---|---|
| Injection accuracy > vanilla by +5pp or more | Patterns are useful; injection works |
| Gap closed by >30% of the 21pp Phase 1 gap | Strong result, validates the approach |
| No improvement or regression | Patterns are too generic, retrieval is noisy, or model ignores context |
| Improvement varies strongly by subject/level | Useful signal for which problems benefit most from scaffolding |

Baseline to beat: **50.8% vanilla 1.7B** (Phase 1). Ceiling: **71.8% 8B** (Phase 1).

---

## Directory Structure

```
phase-2/
├── README.md
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── generate_patterns.py     # Phase 2A
│   ├── build_memory_bank.py     # Phase 2B
│   └── inject_and_eval.py       # Phase 2C
├── data/
│   └── patterns/
│       └── patterns.json        # Phase 2A output
├── memory/
│   ├── bank.json                # Phase 2B output
│   └── bank.index               # Phase 2B output (FAISS)
├── results/
│   ├── results.json             # Phase 2C output
│   └── summary.json             # Phase 2C output
└── logs/
```
