# `f_up_exact` — the technique

Method only. No workflow, no results — those are in [`README.md`](README.md).

**In one sentence:** retrieve individual *reasoning steps* from a graph, walk that
graph to assemble a fresh ordered template for the query, and hand that template
to the model **as user-turn reference material** — but under the reference
implementation's own system prompt, `$\boxed{}$` contract and token budget.

It is `f_exact` with exactly one key changed (`injection`), which is what makes
the pair a one-variable comparison of the injection site.

---

## 1. The unit of retrieval is a step, not a solution

Ordinary retrieval scores whole worked examples by problem similarity and shows
the top-k as reference material. This technique does neither.

Every step of every stored solution becomes its own **node**, carrying:

| field | meaning |
|---|---|
| `text` | the step itself |
| `template_type` | the APPROACH line of the solution it came from |
| `knowledge_tag` | the technique family |
| `problem_id` | which problem it was written for (leakage guard) |
| `step_index` | its position in its source solution |

A step is embedded as `"{template_type} | {text}"`, not as bare text. "Apply the
formula" in isolation is near-contentless and would match every query, and since
the walk's reward is a query-to-node similarity, such a node would be selected
for everything.

Embedder: `jinaai/jina-embeddings-v2-small-en` (the paper's), no prefixes,
L2-normalised, raw cosine throughout.

## 2. The graph

Two edge types over those nodes:

* **E_sequential** — step *i* → step *i+1* of the same solution, weight 1.
* **E_semantic** — any pair with raw cosine ≥ **τ = 0.85**, cross-template only,
  neighbour count **uncapped** (as in the reference).

Semantic edges are what let a walk leave one solution and continue in another.
Sequential edges are what keep it coherent while it stays.

## 3. Template assembly — Algorithm 1

Per query, at inference time:

**Step 1 — candidate filter.**
Keep nodes whose `knowledge_tag` matches the query's (`tag_source: gold` takes it
from the query's own family label), minus every node from the query's own
problem. The reference does the same, annotating query metadata by hand.

**Step 2 — first node.**

```
argmax  0.75 · cos(q, v)  +  0.25 · [v is step 0 of its solution]
```

**Step 3 — traversal.** From the current node, over unvisited in-candidate
neighbours:

```
argmax  0.30 · cos(q, v)  +  0.70 · [v is the sequential child]
```

A sequential child scores ≥ 0.70; any semantic neighbour scores at most 0.30. So
**the walk always continues its current solution when it can**, and crosses into
another solution only where the current one runs out or drifts. Recombination
happens at the seams, not step by step.

**Step 4 — stop** when any of:

* `len(T) ≥ l_max` (8)
* no unvisited successors
* `cos(q,v) < 0.80` **AND** `combined < 0.80` — the disjunctive guard: a step
  survives on query alignment *or* on flow. Requiring both would cut every
  template at its first slightly-off step; requiring neither would never stop.

The output is an ordered list of steps, renumbered `Step 1: … Step n:`, that may
exist in no single stored solution.

**Everything above is byte-identical to `f_exact` and `f_userprompt`** — same
seed, same graph, same tags, same walks. Only what follows differs.

## 4. Injection — user-turn delivery at the reference's prompts

**This is what defines this arm.** The template is shown to the model as
reference material and the model then reasons from scratch in its own `<think>`
block. There is no prefill and no assistant message — but unlike
`f_userprompt`, everything *around* the template stays the reference's.

Two messages go to the server:

```
system   ROT_REFERENCE_SYSTEM                      (fixed — the reference's own)
user     ROT_REFERENCE_TEMPLATE_PROMPT             (fixed — incl. the \boxed contract)
         + <template, line breaks kept>            (template varies)
         + "\n\nProblem: " + <problem>             (problem varies)
```

So this arm inherits from the reference:

* its system message — *"Follow each step exactly … Use only the steps given in
  the template for your reasoning"*
* its `$\boxed{}$` answer contract
* its 15360-token budget

and departs from it in exactly one place: the template goes in the user turn
instead of into an open `<think>` block.

That single-key difference from `f_exact` is the point of the arm. Because
`rot.system_prompt`, `tag_source`, `template_source`, the graph, the walk and the
decoding parameters are all held equal, anything that moves between the two runs
is attributable to the injection site alone.

The template keeps its line breaks here. Only the Thinking-Intervention path
flattens it to one line.

**Consequence to be aware of:** inheriting the reference's contract means this
arm's replies end in `$\boxed{}$`, not in the `ANSWER:` line arms A–E use — so
this configuration is *not* comparable with those arms. `f_userprompt` is the
configuration for that.

## 5. Scoring path

Standard: strip `<think>…</think>`, look for an `ANSWER:` line, then the last
`$\boxed{}$`, then prose; exact integer match. In practice the `$\boxed{}$`
branch is what reads these replies. A reply that hits `max_tokens` mid-trace
never reaches an answer and scores wrong.

No special-case parser is needed, unlike `f_exact`: the model opens and closes
its own reasoning block, so the thinking/answer token split is meaningful. The
answer side is substantial here because the reference contract asks for a
written-out solution ending in a box, not a bare integer line.

## 6. The settings that select this technique

```yaml
rot:
  tau_semantic: 0.85        # semantic edge threshold, RAW cosine
  sim_threshold: 0.80       # traversal halt
  w_first_query: 0.75       # first-step reward
  w_first_s0: 0.25
  w_trav_query: 0.3         # traversal reward
  w_trav_flow: 0.7
  l_max: 8                  # max template length
  max_neighbors: 0          # 0 = uncapped, as the reference
  cross_template_only: true
  tag_source: gold          # candidate filter uses the query's own family
  template_source: retrieved   # written out explicitly; the control flips it
  injection: user_prompt    # ← the defining key, and the ONLY difference from f_exact
  system_prompt: reference  # ← the reference's own system message and contract
  embedding:
    model: jinaai/jina-embeddings-v2-small-en
student:
  max_tokens: 15360         # the reference's budget
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  enable_thinking: true
```

`template_source: retrieved` is spelled out because the matching control
(`config.f_shuf_exact.yaml`) differs from this config in that one key alone,
giving each problem the template assembled for a *different* problem by a
derangement — a real, coherent, same-length template that is simply not the one
retrieval chose.

## 7. Where paper and reference code disagree

The retrieval half follows the code, which is what produced the published
numbers:

| paper | reference code | here |
|---|---|---|
| `sim~ = (cos+1)/2`, τ = 0.85 | raw cosine throughout | raw |
| `R_initial = 0.8·R_Q + 0.2·R_S` | `0.75·sim + 0.25·s0` | 0.75 / 0.25 |
| `R = R_Q + R_F` | `0.3·sim + 0.7·flow` | 0.3 / 0.7 |
| halt when `max(R) < τ` | halt when `sim < 0.8` **and** `combined < 0.8` | the code's |

**The injection half is a deliberate departure.** The paper's headline mechanism
is Thinking Intervention and this arm does not use it. The branch exists in the
reference implementation, so the wording is still the authors' — but this is the
paper's retrieval delivered differently, not a reproduction of the paper.

## 8. The one deliberate adaptation

The reference retrieves from a curated library of generic method skeletons —
0.19% of its node texts contain a 4-or-more-digit literal. This project retrieves
from teacher-generated solutions to *specific* problems, where 36.6% do, so a
template can name another problem's constants. Under user-turn delivery the model
can read those as someone else's numbers and redo the arithmetic for its own
bound; that is a property of the injection site, not of the retrieval.

## 9. Code map

| stage | file |
|---|---|
| nodes, edges, embeddings | `rot/rot_graph.py` |
| candidate filter, walk, template formatting | `rot/rot_retrieval.py` |
| prompt constants and assembly | `rot/student.py:348-500` |
| shuffled-template control (derangement) | `rot/student.py:run_rot_arm`, `template_source` |
| answer extraction | `rot/scoring.py:extract_answer` |
| thinking/answer split | `rot/llm.py:split_thinking_tokens` |
