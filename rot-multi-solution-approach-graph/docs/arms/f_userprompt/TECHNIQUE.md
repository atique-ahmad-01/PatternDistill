# `f_userprompt` — the technique

Method only. No workflow, no results — those are in [`README.md`](README.md).

**In one sentence:** retrieve individual *reasoning steps* from a graph, walk that
graph to assemble a fresh ordered template for the query, and hand that template
to the model **as user-turn reference material**, under this project's own system
prompt and answer contract.

The retrieval is arXiv:2509.21743's, unchanged. The delivery is the reference
implementation's non-Thinking-Intervention branch. This is the only one of the
three arms whose prompts match arms A–E, so it is the only one that can be
compared with them.

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

**Everything above is byte-identical to `f_exact` and `f_up_exact`** — same seed,
same graph, same tags, same walks. Only what follows differs.

## 4. Injection — user-turn delivery

**This is what defines this arm.** The template is shown to the model as
reference material and the model then reasons from scratch in its own `<think>`
block. There is no prefill and no assistant message.

Two messages go to the server:

```
system   SYSTEM                                   (fixed — the same one arms A–E use)
user     ROT_TEMPLATE_PROMPT                      (fixed)
         + <template, line breaks kept>           (template varies)
         + "\n\nProblem: " + <problem>            (problem varies)
```

The preamble is the reference's own wording:

> You are given a template to solve the problem. Use the given steps if
> applicable otherwise use them to guide your reasoning and present the solution
> steps logically to solve this problem:

with **one deletion**: the reference's `$\boxed{}$` contract sentence is dropped,
so the reply ends with the `ANSWER: <integer>` line that `SYSTEM` asks of arms
A–E. A different answer format across arms would put a *parsing* difference
inside what is meant to be a *retrieval* comparison. (`scoring.py` reads
`$\boxed{}$` either way — the point is consistency, not capability.)

The template keeps its line breaks here. Only the Thinking-Intervention path
flattens it to one line.

## 5. Scoring path

Standard, identical to arms A–E: strip `<think>…</think>`, read the last
`ANSWER:` line, exact integer match; `$\boxed{}$` and prose are fallbacks. A
reply that hits `max_tokens` mid-trace never reaches an answer and scores wrong.

No special-case parser is needed, unlike `f_exact`: the model opens and closes
its own reasoning block, so the thinking/answer token split is meaningful rather
than degenerate.

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
  injection: user_prompt    # ← the defining key
  system_prompt: project    # ← this project's SYSTEM, identical to arms A–E
  embedding:
    model: jinaai/jina-embeddings-v2-small-en
student:
  max_tokens: 8192          # the budget arms A–E ran at
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  enable_thinking: true
```

`template_source` is left at its default (`retrieved`). The matching control run
(`config.f_shuf.yaml`) differs from this config in that one key alone, giving
each problem the template assembled for a *different* problem.

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
reference implementation, so the wording is still the authors' — but this is an
ablation of the technique, not a reproduction of it.

## 8. The one deliberate adaptation

The reference retrieves from a curated library of generic method skeletons —
0.19% of its node texts contain a 4-or-more-digit literal. This project retrieves
from teacher-generated solutions to *specific* problems, where 36.6% do, so a
template can name another problem's constants. Under user-turn delivery the model
can read those as someone else's numbers and override them; that is a property of
the injection site, not of the retrieval.

## 9. Code map

| stage | file |
|---|---|
| nodes, edges, embeddings | `rot/rot_graph.py` |
| candidate filter, walk, template formatting | `rot/rot_retrieval.py` |
| prompt constants and assembly | `rot/student.py:348-500` |
| answer extraction | `rot/scoring.py:extract_answer` |
| thinking/answer split | `rot/llm.py:split_thinking_tokens` |
