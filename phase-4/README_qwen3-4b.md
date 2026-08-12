# Phase 4 — Qwen3-4B Results

## Overview

Earlier phases built a pipeline that retrieves a relevant reasoning strategy
from a bank (distilled from a larger teacher model) and injects it into a
student model's prompt before it solves a problem — no fine-tuning, just a
different prompt. That pipeline was validated on a 1.7B student. Phase 4 asks
a follow-up question: **does the same trick still help once the student
itself is bigger and already fairly capable?**

This document covers the run against **Qwen3-4B**.

## How the Injection Technique Works

**Step 1 — build the strategy bank (done once, offline):**

```
 Teacher model  --solves-->  Training problems (MATH-500 + AIME)
       |
       v
 Strategy + worked example, per problem
       |
       v
 Searchable bank  (+ individual steps linked into a graph)
```

**Step 2 — per test problem, at evaluation time:**

```
 New problem
     |
     v
 Embed it, find the closest match in the bank ("anchor")
     |
     v
 Close enough?  --no-->  Skip injection, run plain
     |
    yes
     v
 Take all of the anchor's own steps + its worked example
 (+ maybe 1 bonus step, but only from the anchor's
    immediate neighbours in the graph — see below)
     |
     v
 Add to prompt: "Strategy: ... / Reference Example: ... / Problem: ..."
     |
     v
 Student model answers
```

Every test problem is run **twice** on the identical model — once plain
("vanilla") and once with the strategy added ("injection") — so the only
difference between the two runs is what's in the prompt. The model's weights
are never touched in either case.

### The exact numbers

| Question | Answer |
|---|---|
| What similarity metric is used? | **Cosine similarity.** Problems are embedded with `nomic-embed-text-v1`, vectors are length-normalized, so a plain dot product between two vectors already *is* their cosine similarity |
| How many problems are searched to find a match? | The whole bank — 400 MATH-500 + 746 AIME training problems |
| How many neighbours are retrieved? | Just **1** — the single closest match by cosine similarity ("top-1"), not top-k |
| From how many source problems are steps taken? | Mostly **1** (the anchor). Occasionally **2**, if one extra step is pulled in from a different problem |
| How many steps come from the anchor? | **All** of that anchor's steps (typically 3–5) |
| Where does the bonus step come from? | **Not** a fresh top-k search over the whole bank. The graph already has edges connecting steps that were similar when it was built offline. The bonus candidates are only the steps directly connected to the anchor's own steps by one of those edges — i.e. the anchor's **1-hop graph neighbours**, nothing further out |
| How many bonus candidates are considered? | However many 1-hop neighbours the anchor's steps happen to have (varies per problem) — from that pool, at most **1** is actually kept |
| How is the 1 bonus step picked? | Candidates that are near-duplicates of a step already included are dropped first (novelty check below); of what's left, the one most similar to the **new test problem** (not to the anchor) is chosen |
| How is "near-duplicate" decided? | Also cosine similarity, but rescaled from [-1, 1] to a [0, 1] scale — a candidate step is rejected if it scores 0.85 or higher against any step already picked |
| What if the best match is still weak? | The raw cosine score is compared against a per-subject cutoff (e.g. 0.60 for Intermediate Algebra, 0.50 for most subjects); below that, nothing is injected and the problem runs vanilla-only |

## Experiment Setup

| | |
|---|---|
| Student model | Qwen3-4B |
| Test set | 287 held-out problems — 100 from MATH-500, 187 from AIME |
| Token budget | 8,192 tokens for **both** runs (see the caveat below — this turned out to be the binding constraint) |

## Results

| | Vanilla | Injection | Change |
|---|---|---|---|
| **Overall accuracy** (287) | 49.1% (141 correct) | 50.9% (146 correct) | **+1.7 points** |
| MATH-500 (100) | 86.0% | 90.0% | **+4.0 points** |
| AIME (187) | 29.4% | 30.0% | +0.5 points |
| Avg. tokens used | 6,282 | 6,033 | slightly fewer |
| Avg. time per problem | 118.7s | 114.5s | slightly faster |

Out of 287 problems: **20 improved**, **15 got worse**, the rest stayed the
same. 20 problems had no good match in the strategy bank (similarity below
the cutoff) and were run vanilla-only.

## Visual Breakdown

![Qwen3-4B memory injection results — overall, by subject, by difficulty level, by AIME year](results/summary_histogram_qwen3-4b.png)

By difficulty level (MATH-500), the injected strategy helps most at the
extremes — Level 1 (+11 points) and Level 5 (+9 points) — but actually hurts
at Level 4 (−5 points). On AIME, results swing both ways year to year (from
−7 to +14 points), which supports the earlier point that the AIME effect is
mostly noise rather than a consistent gain.

## Important Caveat — This Run Was Token-Starved

Digging into the raw results, the headline numbers above understate what's
going on, because most runs **never finished reasoning**:

| | MATH-500 | AIME |
|---|---|---|
| Hit the 8,192-token cap (vanilla) | 14% | **74%** |
| Answers that came out completely empty | 0 | **48 problems** |

All 48 empty answers hit the cap and contain no final answer at all — they
were cut off mid-sentence. And of the 117 AIME problems where *both* runs
were wrong, **111 had both runs truncated** — so for those, the comparison
isn't measuring whether the strategy helped, only that neither run got to
an answer.

Splitting the results by whether the run finished makes this stark:

| Subset | Vanilla | Injection | Change |
|---|---|---|---|
| Finished within budget (121) | 95.9% | 97.5% | +1.7 points |
| Truncated (166) | 15.1% | 16.9% | +1.8 points |
| AIME, finished within budget (38) | 92.1% | 92.1% | +0.0 points |
| AIME, truncated (149) | 13.4% | 14.1% | +0.7 points |

On the problems Qwen3-4B could actually finish, it was already scoring
**92–100%** — there was almost no headroom left for a strategy to add
anything. The low AIME score is therefore mostly a budget artifact, not
evidence that the strategy fails on hard problems.

### The graph expansion never actually ran

Separately: `bonus_steps` is **0 for all 267 injected problems**. This is
not a coincidence. The graph only creates an edge between two steps if
their similarity is ≥ 0.90, but the novelty filter rejects any candidate
scoring ≥ 0.85 against a step already picked. Since every edge guarantees
≥ 0.90, **every candidate is rejected by construction** — verified directly
against the graph: 2,279 candidates considered across 400 anchors, 2,279
rejected, 0 survived.

So this run (and the earlier phases sharing this code) effectively ran
**anchor-only injection**. The graph contributed nothing, which means its
value is currently *untested* rather than disproven.

## Takeaway

Injection gives Qwen3-4B a small positive lift (+1.7 points overall, +4.0 on
MATH-500), but this run can't answer the question it was meant to answer.
Two measurement problems dominate: the 8,192-token budget truncated 74% of
AIME runs, and on the problems that did finish the model was already at
92–100%, leaving nothing for the strategy to improve. A configuration bug
also meant the graph-expansion step was inert throughout.

The honest conclusion is that **this comparison is budget-limited, not
strategy-limited** — it needs a re-run with a larger budget before any claim
about how well injection scales to bigger students can be supported.
