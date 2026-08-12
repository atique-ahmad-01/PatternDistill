# Phase 6 — Design Document

Complete specification of every change from pattern distillation through to
injection, with the reasoning and the supporting evidence for each. Nothing
here is speculative — every problem listed was measured in the phase-4
Qwen3-4B run or verified directly against the pattern banks and graph.

**Datasets are unchanged: MATH-500 + AIME, same 287-problem test split**, so
phase-6 is directly comparable to phase-3d and phase-4.

---

## 0. Why phase-6 exists

Five problems, all verified:

| # | Problem | Evidence |
|---|---|---|
| 1 | Token starvation | 74% of AIME vanilla runs hit the 8,192 cap; 111/117 AIME both-wrong cases had **both** arms truncated; 48 answers came back empty, all capped, none containing `\boxed{}` |
| 2 | Graph expansion never ran | Novelty cutoff 0.85 < edge tau 0.90, so every neighbour is auto-rejected. Verified: 2,279 candidates across 400 anchors → 2,279 rejected → **0 survived**. `bonus_steps=0` for all 267 injected problems |
| 3 | Pattern format broken | All 1,146 patterns are exactly 3 steps + 1 Note; `Identify` opens a step in 698/746 AIME patterns; 30–32% contain digits from the anchor problem; **45% of MATH-500 worked examples contain the anchor's own gold answer** |
| 4 | No auditability | The assembled strategy and worked-example text are never saved, so no regression can be traced to its source pattern |
| 5 | No significance testing | Every phase's headline delta is within noise: phase-4 +1.7pp (p=0.500), phase-3d +3.5pp (p=0.164), phase-3e +3.1pp (p=0.134), exact McNemar |

The consequence of #5 is the one that sets the goal: **phase-6 targets a
result that survives a significance test, not a bigger headline number.**
With ~15 losses, ~29 wins (net +14) are needed for p<0.05. Since wins are
hard to manufacture, the primary lever is **cutting losses** — which is what
the pattern redesign below is aimed at.

---

## 1. The pattern problem, stated precisely

The current patterns fail in two *opposite* directions at once.

**Too generic to inform.** A real bank entry:

```
Step 1: Identify the function and substitute each value into the function.
Step 2: Perform the arithmetic operations for each substitution.
Step 3: Add the results of each substitution together.
Note: Be careful with the order of operations and simplify fractions where possible.
```

This is "do the problem" split into three parts. It names no technique, no
condition, no decision. A model that could not solve the problem is no
better off for reading it.

**Too specific to transfer.** Another real entry:

```
Step 1: Identify the structure of the number: a 4-digit number starting with 1.
```

True only of the anchor. Injected into any other counting problem, it is
actively wrong.

**And it leaks answers.** 45% of MATH-500 worked examples end in the
anchor's own gold answer:

```
Reference Example:
f(-2) = -8/-4 = 2, f(-1) = -5/-3 = 5/3, f(0) = -2/-2 = 1, so 2 + 5/3 + 1 = 14/3
                                                                      ^^^^^^
                                                    a different problem's answer
```

The model is being shown a finished solution to a different question while
being asked to solve a new one. This is a direct mechanism for regressions,
and it independently explains why higher retrieval similarity performed
*worse* (net −1 in the 0.7–0.8 band vs +4 in 0.6–0.7): a closer anchor leaks
more contaminating specifics.

**One more, in the prompt itself.** The current instruction is:

> "Follow the strategy steps explicitly, adapting them to this specific problem."

There is no escape hatch. When retrieval returns a poor match, the model is
instructed to follow it anyway. Forced compliance with a wrong strategy is
the most likely cause of the 15 regressions.

---

## 2. New pattern schema

Replaces the flat `pattern` string + `worked_example` pair.

```json
{
  "trigger":   "A double summation where the two indices appear only through a symmetric combination",
  "technique": "Index substitution to collapse a double sum into a single sum",
  "steps": [
    "Introduce a new variable for the combination the two indices share",
    "Count how many index pairs map to each value of the new variable",
    "Rewrite as a single sum over the new variable and match it to a known series"
  ],
  "pitfall":   "The substitution changes the summation bounds — recompute them rather than reusing the originals",
  "verify":    "Confirm the number of terms agrees before and after the substitution"
}
```

Each field earns its place by supplying something the student model
demonstrably lacks:

| Field | Why it exists |
|---|---|
| `trigger` | Lets the model **check applicability itself**. This is the single most important addition — it converts a blind instruction into a testable condition, so a bad retrieval can be recognised and discarded instead of blindly followed |
| `technique` | Names the actual mathematical move. This is *information*, unlike "Solve the equation" |
| `steps` | **Variable length** (2–6), matched to real problem complexity. Each step names an operation, not a generic action verb |
| `pitfall` | Encodes the non-obvious failure mode — genuinely hard for a small model to anticipate, and cheap to state |
| `verify` | Gives a self-check. Costs a few tokens, catches whole classes of arithmetic error |

**Worked examples are dropped by default.** Phase-3e already found that
removing them helped, and the 45% answer-leakage measurement explains why.
An ablation arm keeps a symbolic-only, answer-redacted variant so the choice
is tested rather than assumed.

### Before / after on a real entry

| | Current | Phase-6 |
|---|---|---|
| Structure | Always 3 steps + 1 Note | 2–6 steps, matched to complexity |
| Step 1 | "Identify the prime factorization of the number." | "Factor the input into prime powers" |
| Information | Restates the task | Names the technique + when it applies |
| Anchor literals | `196`, `4-digit`, `starting with 1` | None — validator rejects them |
| Example | `2^2 x 7^2 -> (2+1)(2+1) = 3 x 3 = 9` (leaks answer) | Dropped (or redacted, as an ablation) |
| Applicability | Not stated | `trigger` field, checkable by the model |

---

## 3. Automated validation gates

Patterns are machine-checked at distillation time. A pattern failing any gate
is regenerated (up to N retries), then dropped if it still fails. **Gate pass
rates get logged** so a silently-shrinking bank can't go unnoticed.

| Gate | Rule |
|---|---|
| No anchor literals | No numeric token appearing in the anchor problem may appear in any pattern field |
| No answer leakage | The anchor's gold answer must not appear anywhere in the pattern |
| Variable length | 2 ≤ `len(steps)` ≤ 6; a bank where >90% share one length fails the build |
| Non-vacuous steps | No step may consist solely of a generic verb phrase (`identify the`, `calculate the`, `solve the`) without a named object |
| Required fields | `trigger`, `technique`, `pitfall` non-empty |
| Token budget | Assembled strategy ≤ 200 tokens (small models degrade on long injected context) |

The "no anchor literals" gate is the precise version of the abstraction
requirement: general mathematical constants (`2`, `n`, `k`) are fine; the
anchor's specific numbers are not.

---

## 4. Assembly — how a pattern is rearranged for a *new* problem

This is the step the question was really about, and it needs a guard rail.
Phase-3 already established that naively merging steps from many patterns
produces incoherent "Frankenstein" scaffolds that scored worse. So:

**One coherent spine, optionally augmented — never a merge of equals.**

```
 Anchor (top-1)  ->  its full step sequence = the SPINE (kept in order)
                          |
                          v
       Graph 1-hop neighbours -> candidate augmentation
                          |
              accept at most 1, only if ALL hold:
                - passes the novelty check (now actually functional)
                - more similar to the NEW problem than to the anchor
                - appended as a labelled optional step, never interleaved
                          |
                          v
              Assembled scaffold (spine order preserved)
```

The augmentation is **appended and labelled**, not spliced into the middle of
the spine. That keeps the anchor's reasoning order intact — which is what
made anchor-first beat pure step-level retrieval in the first place — while
still allowing a genuinely novel move to be offered.

### Injected prompt

```
Strategy from a similar solved problem — adapt it to the numbers below.
If it does not fit this problem, say so briefly and solve directly instead.

Applies when: {trigger}
Technique: {technique}
Plan:
  1. {step 1}
  2. {step 2}
  ...
Also consider: {augmentation}        <- only when one was accepted
Watch out: {pitfall}
Check: {verify}

Problem: {problem}

Be direct. Do not restate the problem. State your final answer in \boxed{}.
```

Four deliberate changes from the current prompt:

1. **An explicit escape hatch.** "If it does not fit... solve directly
   instead." Removes forced compliance, which is the main suspected cause of
   regressions. Costs ~15 tokens; directly targets the metric that decides
   significance.
2. **`Applies when:` stated up front**, so the model checks the condition
   before committing to the plan.
3. **No worked example** in the default arm — no leaked answers.
4. **Brevity + `\boxed{}` instruction**, matching phase-3e's genuine
   finding that injection wins by *finishing*, not by reasoning harder.

---

## 5. Retrieval and gating changes

| Item | Current | Phase-6 | Why |
|---|---|---|---|
| Metric | Cosine, top-1 | Unchanged | Top-1 is right for small models |
| Novelty cutoff | 0.85 (< tau 0.90 → always rejects) | **0.95**, asserted > tau at startup | Fixes the dead-code bug |
| Similarity gate | Per-subject 0.50–0.65 | Keep, but **re-measure** | The old thresholds were tuned against contaminated patterns, so they encode the wrong tradeoff |
| Bad-match handling | Skip injection entirely | Inject **with** the escape hatch | Skipping was doing nothing useful: 20 skips, all MATH-500, 18/20 already correct |

Retrieval quality is deliberately *not* the focus. Similarity currently fails
to predict success (0.7–0.8 net −1 vs 0.6–0.7 net +4), but the leakage finding
explains that as a pattern-format artifact. Fix the format first, then
re-examine whether retrieval needs work — otherwise we would be tuning
retrieval against a broken objective.

---

## 6. Evaluation

**Budget:** 24,576 tokens for **both** arms (up from 8,192), early-exit on
`\boxed{}`. Applying it to both arms is essential — raising only the injection
arm would manufacture a fake win.

**Logged per problem:** anchor id and index, raw similarity, threshold, the
**full assembled strategy text**, the augmentation (if any), whether the
escape hatch fired, truncation flag, tokens, latency, both answers, both
verdicts.

**Statistics:** exact McNemar on discordant pairs plus a bootstrap CI on the
delta, reported alongside every headline number. Truncated-vs-finished
subgroups reported separately, since pooling them is what made phase-4
uninterpretable.

### Ablations

Each isolates one decision, so a result can be attributed rather than
assumed:

| Arm | Purpose |
|---|---|
| A. Vanilla | Baseline |
| B. Old patterns, new budget | Isolates the budget fix from everything else |
| C. New patterns, no graph | Tests the pattern redesign alone |
| D. New patterns + graph augmentation | Tests the graph for the **first time ever** |
| E. C + answer-redacted worked example | Tests whether examples help once de-contaminated |
| F. C without the escape hatch | Measures what the escape hatch is worth |

Arm B is the important control: it separates "we gave the model room to
think" from "our pattern redesign worked." Without it, the budget fix would
mask everything else.

### Model order: 1.7B first, then 4B

**Qwen3-1.7B runs the full 6-arm sweep. Qwen3-4B runs only the winning arm
plus vanilla.** The headroom measurement decides this:

| Model | Subset | Vanilla | Headroom | Current delta |
|---|---|---|---|---|
| **1.7B** @16K | AIME finished (78) | 69.2% | **30.8pp** | **−2.6pp** |
| 1.7B @16K | All finished (169) | 82.2% | 17.8pp | −0.6pp |
| 4B @8K | AIME finished (38) | 92.1% | 7.9pp | +0.0pp |
| 4B @8K | All finished (121) | 95.9% | 4.1pp | +1.7pp |

Three reasons the 1.7B is the right starting point:

1. **~4× the headroom** on problems the model can finish. On the 4B, every
   finished subset sits at 92–96%, so a null delta is indistinguishable from
   a ceiling effect — no design change can be evaluated there.
2. **It already fails exactly where the hypothesis predicts.** The 1.7B
   scores **−2.6pp on AIME-finished** — injection actively *hurts* on the 78
   problems where it had the budget to matter, with 30.8pp of headroom
   unused. That is what contamination plus forced compliance would produce,
   so phase-6 gets a clean falsifiable test: redaction + escape hatch should
   move that number positive. If it doesn't, the diagnosis is wrong and we
   learn so cheaply.
3. **Cost.** 14.7 GPU-hours per arm-pair vs 18.6, before the 24K budget
   slows both down. The full six-arm sweep is affordable on 1.7B and roughly
   75+ hours on 4B.

The 4B then runs the winning configuration only — enough to establish the
scaling trend (the effect fading as the student grows), which is likely the
more durable thesis contribution than any single delta, at about a third of
the cost of a second full sweep.

---

## 7. Expected results — calibrated

**Will improve:**

- **Validity.** AIME truncation goes from 74% to near zero. For the first
  time the AIME comparison measures reasoning rather than who ran out of
  room.
- **Regressions should fall.** Removing answer leakage and adding the escape
  hatch both target losses directly, and losses are what gate significance.
- **Auditability.** Every regression traceable to its source pattern.
- **Attribution.** Six arms mean each design choice is separately measurable.

**A real risk, stated plainly:** phase-3e found **69% of injection wins were
"vanilla hit the limit, injection didn't."** Give both arms a fair 24K budget
and that win mechanism largely disappears. **The delta may shrink rather than
grow.**

| Outcome | What it looks like | Interpretation |
|---|---|---|
| Good | Losses fall to ~5, wins hold ~20 → net ~+15, p<0.05 | A genuine, defensible result |
| Likely | Modest delta, still short of significance at n=287 | Needs more power — extend with phase-5's OpenMathReasoning split or multi-seed runs |
| Possible | Delta collapses toward zero once budget is fair | Honest null: injection does not significantly help a student that already has adequate reasoning budget |

The third outcome is **not a failed phase**. A well-characterised null result
with correct methodology, a diagnosed cause, and a scaling trend is
defensible thesis material. A +1.7pp at p=0.500 presented as a win is not.

### Success criteria

Phase-6 succeeds if, regardless of which way the delta goes:

1. AIME truncation < 10% in both arms
2. `bonus_steps > 0` on a real fraction of problems — the graph is finally tested
3. Zero patterns containing anchor literals or leaked answers (gates enforced, pass rates logged)
4. Every reported delta carries a p-value and CI
5. The budget fix is separable from the pattern redesign via arm B

---

## 8. File plan

```
phase-6/
├── DESIGN.md                 # this document
├── generate_patterns.py      # re-distillation into the new schema + validators
├── validate_patterns.py      # standalone gate checker, runnable on any bank
├── build_memory_and_graph.py # bank + graph, asserts novelty > tau at build time
├── inject_and_eval.py        # 6 arms, 24K budget, early-exit, full prompt logging
├── analyze.py                # McNemar, bootstrap CI, truncated/finished split
├── data/patterns/
├── memory/
├── graph/
├── results/
└── logs/
```

**Build order:** validators first (so bad patterns are caught before a
multi-hour distillation run), then re-distillation, then graph with the
startup assertion, then eval, then analysis. Each script smoke-tests with
`--max-problems 5` locally before any GPU run.

Two additions made during implementation:

- **`config.py`** holds every threshold. The phase-4 bug existed because
  `TAU_SEMANTIC` lived in the graph builder and `NOVELTY_THRESHOLD` in the
  evaluator; they drifted into contradiction and nothing checked. They now
  share one file with an import-time assertion.
- **`scoring.py`** is copied verbatim from `phase-4/multi_model_eval.py`
  (lines 145–317), whose scoring is a strict superset of phase-3d's — it adds
  `_clean_boxed`. Reused unchanged so phase-6 scores identically to phase-4.

---

## 9. Validation log

Everything below was verified before the full run, not assumed.

**Gates reproduce the known defects.** Run against the existing banks:

| Bank | Pass rate | Answer leak | Anchor numbers | Problem phrases | Uniformity |
|---|---|---|---|---|---|
| phase-2d (MATH-500, n=400) | 50.2% | 42.8% | 4.5% | 8.0% | 100% → **FAIL** |
| phase-3d (AIME, n=746) | 76.0% | 5.4% | 13.3% | 13.0% | 100% → **FAIL** |

The phrase gate was added after a specific miss: `"Identify the structure of
the number: a 4-digit number starting with 1"` passed every other gate because
its literals are single digits. Restricting the gate to shared *number-bearing*
n-grams catches it while leaving general vocabulary ("prime factorization")
alone — false positives dropped from 13/13 to 1/13 on a control set.

**The threshold fix is quantified.** Simulated against the phase-3d graph:

| novelty | Surviving candidates | Anchors augmented |
|---|---|---|
| 0.85 (shipped in phase-4) | **0** | 0/400 |
| 0.90 | **0** | 0/400 |
| **0.95 (phase-6)** | **2,219** | **256/400 (64%)** |

Note 0.90 also fails — the threshold must be *strictly* greater than tau, not
equal. The build now refuses to write a graph whose augmentation rate is below
5%, and the evaluator refuses to load one, so this cannot recur silently.

**Stats verified against known values.** `analyze.py`'s exact McNemar
reproduces the previously computed p-values exactly: (20W,15L)→0.500,
(26W,16L)→0.164, (15W,7L)→0.134, (5W,1L)→0.219.

**Distillation smoke test (14 MATH-500 problems, Qwen3-8B in 4-bit on the
16GB card):** 13/14 accepted, 4 retries, step counts `{3:3, 4:4, 5:7}` —
uniformity **53.8%**, versus 100% in every previous phase. Same problem,
before and after:

> **Old:** `Step 1: Identify the function and substitute each value into the
> function. / Step 2: Perform the arithmetic operations. / Step 3: Add the
> results.` plus a worked example ending `= 14/3` — the anchor's own answer.
>
> **New:** `Applies when: a function is defined algebraically and you are
> asked to evaluate it at specific points and sum them. / Technique: direct
> substitution followed by arithmetic simplification. / 1. Substitute each
> input value… 4. Express the final result as a single fraction… / Watch out:
> forgetting to simplify the fractions fully before adding. / Check: the sum
> reduces to a fraction with no common factors.` No example, no leaked answer.

`MAX_STRATEGY_TOKENS` was raised 200 → 260 after the smoke test dropped an
otherwise-sound pattern at 204 tokens; the new schema carries five fields, and
260 remains well below the ~400-token point where 1.7B instruction-following
degrades.
