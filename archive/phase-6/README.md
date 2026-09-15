# Phase 6

Teach a small model to solve harder maths problems by showing it *how* a
similar problem was solved — without showing it the answer.

The idea across every phase has been the same: take a strong teacher model,
have it write down the **reasoning pattern** behind a solved problem, store
those patterns, then retrieve the closest one and paste it into the prompt
when a small student model faces a new problem.

Phase 6 exists because phases 3–5 measured that pipeline and found three
things wrong with it. See [DESIGN.md](DESIGN.md) for the full evidence; the
short version:

| Problem | What was happening |
|---|---|
| **Budget starvation** | The student was capped at 8,192 tokens. 74% of AIME runs hit the cap mid-thought. The comparison measured who ran out of room, not who reasoned better. |
| **Dead graph** | The novelty cutoff (0.85) was *below* the edge threshold (0.90), so every graph suggestion was auto-rejected. The graph had contributed nothing for three phases and nobody noticed. |
| **Contaminated patterns** | Patterns copied the original problem's numbers, labels, and — in 23% of MATH-500 cases — its actual answer. The student was being shown a finished solution to a *different* question. |

All three are fixed here, and the evaluation arms are designed so you can
tell which fix (if any) actually mattered.

---

## Architecture

Four stages. Each writes files the next one reads, so any stage can be re-run
alone.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. DISTILL          generate_patterns.py                             │
  │                                                                      │
  │    1,146 solved problems ──► teacher model (Qwen3-8B, 4-bit)         │
  │                                    │                                 │
  │                                    ▼  writes a 5-field pattern       │
  │                             ┌──────────────┐                         │
  │                             │  9 GATES     │ ◄── validate_patterns.py│
  │                             │  fail → retry│     (same rules,         │
  │                             │  3× → drop   │      runnable stand-alone)│
  │                             └──────┬───────┘                         │
  │                                    ▼                                 │
  │                      data/patterns/patterns.json   (869 kept)        │
  └──────────────────────────────────────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────┐
  │ 2. INDEX            build_memory_and_graph.py                        │
  │                                                                      │
  │    each pattern ──► embed the PROBLEM  ──► memory/bank.index         │
  │                     (how we find it later)                           │
  │                                                                      │
  │    each STEP    ──► embed separately   ──► graph/thought_graph.json  │
  │                     link steps that are similar (≥ 0.90)             │
  │                     4,148 nodes · 3,279 sequential · 9,678 semantic  │
  └──────────────────────────────────────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────┐
  │ 3. INJECT + RUN     inject_and_eval.py                               │
  │                                                                      │
  │    new problem                                                       │
  │        │                                                             │
  │        ├─► embed ─► nearest pattern in bank = ANCHOR                 │
  │        │                    │                                        │
  │        │                    ├─► its steps, in order = THE SPINE      │
  │        │                    │                                        │
  │        │                    └─► graph 1-hop neighbours               │
  │        │                          └─► at most 1 extra step,          │
  │        │                              APPENDED and labelled          │
  │        │                              (never spliced into the spine) │
  │        │                                                             │
  │        └─► assemble prompt ─► student model (Qwen3-1.7B) ─► answer   │
  │                                24,576 token budget                   │
  │                                early exit on \boxed{}                │
  │                                                                      │
  │    every run appends one row ──► results/results_<model>.jsonl       │
  └──────────────────────────────────────────────────────────────────────┘
                                       │
  ┌────────────────────────────────────▼─────────────────────────────────┐
  │ 4. ANALYSE          analyze.py                                       │
  │    exact McNemar + bootstrap CI, truncated/finished reported apart   │
  └──────────────────────────────────────────────────────────────────────┘
```

**Why the spine is preserved.** Phase 3 tried merging steps from several
patterns and it scored *worse* — the merged scaffolds were incoherent
"Frankenstein" plans. So one anchor supplies the whole plan in its original
order, and the graph may only append a single labelled extra at the end.

---

## What gets injected

A pattern has five fields. Each one earns its place by giving the student
something it demonstrably lacks:

| Field | Purpose |
|---|---|
| `trigger` | **When this applies.** The most important field — it lets the model check for itself whether the strategy fits, instead of following it blindly. |
| `technique` | Names the actual mathematical move ("index substitution to collapse a double sum"), not "solve the equation". |
| `steps` | 2–6 steps, length matched to real complexity. Earlier phases were rigidly 3 steps every time. |
| `pitfall` | The non-obvious failure mode. Cheap to state, hard for a small model to anticipate. |
| `verify` | A self-check, to catch arithmetic slips. |

Assembled into the prompt like this:

```
Strategy from a similar solved problem — adapt it to the numbers below.
If the strategy does not fit this problem, say so briefly in one sentence
and solve the problem directly instead.          ◄── the escape hatch

Applies when: {trigger}
Technique: {technique}
Plan:
  1. {step 1}
  2. {step 2}
Also consider: {graph augmentation}              ◄── only if one was accepted
Watch out: {pitfall}
Check: {verify}

Problem: {problem}

Be direct and do not restate the problem. Put your final answer in \boxed{}.
```

---

## The arms

Every arm answers **one** question, by differing from its neighbour in
exactly one respect. That's what makes the result attributable — if the
number moves, you know which change moved it.

Six were specified; **five actually run.** Arm E was withdrawn mid-sweep — it
turned out to be inert (see below).

| Arm | Inject? | Bank | Graph | Example | Escape hatch |
|---|---|---|---|---|---|
| `A_vanilla` | no | — | — | — | — |
| `B_old_patterns` | yes | **old** | no | no | yes |
| `C_new_patterns` | yes | new | no | no | yes |
| `D_new_plus_graph` | yes | new | **yes** | no | yes |
| `F_no_escape` | yes | new | no | no | **no** |

### A — Vanilla

The problem alone, no injection. Baseline. Gets the same 24,576-token budget
as everyone else, which matters: raising the budget for injection arms only
would manufacture a fake win.

### B — Old patterns, new budget

Deliberately uses the **contaminated phase-2d/3d banks** — the ones with
leaked answers and copied numbers.

This is the control that makes the whole experiment honest. Phase 6 changed
two big things at once: it raised the budget *and* rebuilt the patterns. Arm
B holds the patterns fixed at the old broken version while giving them the
new budget, so **A → B tells you what the budget fix alone was worth.**
Without it, a bigger budget could take credit for the pattern redesign.

### C — New patterns

The rebuilt 869-pattern bank, no graph, no example. **B → C is the pattern
redesign, isolated.** This is the main claim of the phase.

### D — New patterns + graph

C, plus up to one extra step pulled from the thought graph. Chosen for
relevance to the *new* problem, not to the anchor, and rejected if it's too
similar to what the spine already says.

**C → D is the first real test the graph has ever had** — it was dead code
in phases 3 through 5. Currently 61% of anchors have at least one eligible
neighbour.

### E — Withdrawn (was: new patterns + worked example)

Arm E was meant to test whether a de-contaminated example helps: examples
were dropped from the default because they were the main leak vector — 23%
of MATH-500 examples ended in the anchor's own answer — but "examples leaked
answers" and "examples don't help" are different claims.

**It never ran as designed.** The phase-6 schema has no `worked_example`
field and the distiller never emits one, so `assemble_new()`'s
`if include_example and anchor.get("worked_example")` could not fire and E
produced a prompt byte-identical to C — confirmed on 41/41 problems before
it was withdrawn. Same failure mode as the dead graph: a flag whose enabling
data doesn't exist, with nothing checking.

Testing the question needs a re-distillation pass that emits redacted
examples, then a separate two-arm run. **No C → E result exists.**

### F — New patterns, no escape hatch

C with the "if it doesn't fit, solve directly instead" line removed, which
restores the phase-3d/4 behaviour of forced compliance.

Forced compliance is the leading suspect for phase 4's 15 regressions: when
retrieval returned a poor match, the model was instructed to follow it
anyway. **C → F prices the escape hatch.** If F is meaningfully worse, one
sentence of prompt was doing real work.

### Reading the results

| Compare | Isolates |
|---|---|
| A → B | the token budget fix |
| B → C | the pattern redesign |
| C → D | graph augmentation |
| C → F | the escape hatch |

Losses matter more than wins here. Significance is gated by the regression
count, and cutting regressions is what the redaction and the escape hatch
are both aimed at.

---

## Running it

```bash
conda activate ms-atique

python generate_patterns.py                       # distil + gate  (~1.5h)
python validate_patterns.py --input data/patterns/patterns.json
python build_memory_and_graph.py                  # bank + graph   (~40s)
python inject_and_eval.py --model qwen3-1.7b --arms all
python analyze.py
```

Always run the long stages in **tmux** — an SSH drop kills a foreground run
on the shared box.

Useful flags:

- `--max-problems N` — first N problems. The eval set is shuffled under a
  fixed seed first, so any prefix is a fair mix of MATH-500 and AIME. Without
  the shuffle it would be pure MATH-500 for the first 100.
- `--arms A_vanilla,C_new_patterns` — subset of arms.

**The eval resumes.** Results append to `results/results_<model>.jsonl` and
completed `(arm, problem_id)` pairs are skipped on restart, so a dropped
connection costs only the in-flight problem.

---

## Quality gates

Patterns are machine-checked at distillation time; a failure triggers a retry,
and three failures drop the pattern. Pass rates are logged so a silently
shrinking bank can't go unnoticed.

| Gate | Rejects |
|---|---|
| `required_fields` | missing `trigger` / `technique` / `pitfall` |
| `step_count` | fewer than 2 or more than 6 steps |
| `non_vacuous_steps` | "Identify the..." with no named object |
| `no_duplicate_steps` | two steps restating one instruction |
| `no_anchor_numbers` | the source problem's numbers |
| `no_anchor_labels` | the source problem's object names (triangle **ABC**) |
| `no_problem_phrases` | number-bearing phrases copied from the problem |
| `no_answer_leak` | the source problem's gold answer |
| `token_budget` | assembled strategy over 260 tokens |

A whole-bank check also fails the build if more than 90% of patterns share one
step count — that's what caught the old banks, which were 100% three-step.

Two of these gates were added after the first phase-6 bank passed everything
and *still* shipped contamination: 9.4% echoed object labels, 12.3% had
duplicate steps. Neither was visible to the other gates — a label carries no
digits and forms no shared phrase.

Regenerating under all nine: **869 of 1,146 accepted**, all 869 passing every
gate. Residual near-duplicate steps 2.5%, label echo 2.3%.

```bash
python audit_quality.py    # measures the residual rates above
python verify_smoke.py     # dumps result rows from a short run
```

---

## Layout

```
phase-6/
├── DESIGN.md                  full specification and evidence
├── config.py                  every threshold, with the tau/novelty assertion
├── generate_patterns.py       distillation + gates
├── validate_patterns.py       stand-alone gate checker
├── build_memory_and_graph.py  bank + graph
├── inject_and_eval.py         arms, retrieval, injection, generation
├── analyze.py                 McNemar + bootstrap CI
├── scoring.py                 answer extraction (verbatim from phase-4)
├── audit_quality.py           residual contamination measurement
├── verify_smoke.py            smoke-run inspector
├── data/patterns/             patterns.json + rejected.jsonl
├── memory/                    bank.json + bank.index
├── graph/                     thought_graph.json + step_embeddings.npy
├── results/                   results_<model>.jsonl
└── logs/
```

Everything under `data/`, `memory/`, `graph/`, `results/`, `logs/` is
regenerable and git-ignored, as are the `archive_pre_gate_v2*` directories
holding the previous run for comparison.

`config.py` holds every threshold in one file with an import-time assertion
that `NOVELTY_THRESHOLD > TAU_SEMANTIC`. The dead-graph bug happened because
those two constants lived in different files, drifted into contradiction, and
nothing checked.

---

## Status

Pattern regeneration and the graph build are done. The sweep on Qwen3-1.7B is
running over all 287 problems (100 MATH-500 + 187 AIME) — 1,435 problem-arm
runs across five arms, about 34 hours on the RTX 5070 Ti.

Success is **not** defined as a bigger headline number. Phase 4's +1.7pp came
with p=0.500. Phase 6 succeeds if AIME truncation lands under 10%, the graph
demonstrably contributes for the first time, no pattern carries anchor
literals or leaked answers, and every reported delta carries a p-value and a
confidence interval — whichever way the delta goes.

A well-characterised null result with correct methodology is defensible. A
+1.7pp at p=0.500 presented as a win is not.
