# `f_exact` — the technique

Method only. No workflow, no results — those are in [`README.md`](README.md).

**In one sentence:** retrieve individual *reasoning steps* from a graph, walk that
graph to assemble a fresh ordered template for the query, and inject the template
**inside an already-open `<think>` block that the model is made to continue**.

This is arXiv:2509.21743 as its authors implement it
(`github.com/ahme0599/Retrieval-of-Thought`), including their system prompt,
their `$\boxed{}$` answer contract and their prefill wording.

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

## 4. Injection — Thinking Intervention

**This is what defines this arm.** The template is not shown to the model. It is
placed inside an open reasoning block that the model continues, so it arrives in
the model's own voice rather than as something a user said.

Three messages go to the server:

```
system     ROT_REFERENCE_SYSTEM                      (fixed)
user       ROT_REFERENCE_TEMPLATE_PROMPT             (fixed)
           + "\nProblem: " + <problem> + "."         (problem varies)
assistant  ROT_PREFILL first half                    (fixed)
           + <template, flattened to one line>       (template varies)
           + ROT_PREFILL second half                 (fixed)
           ← generation starts here, mid-sentence, inside an open <think>
```

The prefill reads, verbatim:

> `<think>`I need to follow the given steps to guide me in reasoning to solve the
> problem. I will follow each step and **modify the steps to make them match the
> problem** and then solve the problem accordingly. The template is. *{template}*
> Strictly following the given steps for guidance, I will now solve the problem
> starting from the step 1. Using step 1

Two mechanical details:

* The template is flattened with `template.replace("\n", " ")`, as the reference
  does.
* vLLM is told `add_generation_prompt=False` **and** `continue_final_message=True`.
  Both are required together; setting one alone is a 400 from the server. The
  reference splices raw `<|im_end|>`/`<|im_start|>` markers into the user string
  instead; sending a real assistant message renders to the same tokens.

**Note the template is not in the user turn at all in this arm.** The user
message is preamble + problem only.

## 5. Scoring path — why it needs its own parser

Under a prefill, Qwen3-1.7B answers straight out of the open block and typically
never emits `</think>`. The standard extractor sees an unclosed `<think>` and
discards everything after it — correct for a *truncated* reply, where a
mid-thought number is not an answer, but here the model finished deliberately.

So this arm scores through `extract_answer_open_trace`, which keeps the
truncation guard and, for a finished reply, reads from the last `$\boxed{}`.
Arms A–E keep the original rules unchanged, so their published results stay
comparable.

One consequence: with no `</think>` and no `ANSWER:` line, the thinking/answer
token split books the whole completion as reasoning. `answer_tokens` is 0 by
construction in this arm — it is not a statement about answer length.

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
  template_source: retrieved
  injection: ti_prefill     # ← the defining key
  system_prompt: reference  # ← the reference's own system message and contract
  embedding:
    model: jinaai/jina-embeddings-v2-small-en
student:
  max_tokens: 15360
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  enable_thinking: true
```

## 7. Where paper and reference code disagree

The code is what produced the published numbers, so the code wins. All four are
implemented the code's way:

| paper | reference code | here |
|---|---|---|
| `sim~ = (cos+1)/2`, τ = 0.85 | raw cosine throughout | raw |
| `R_initial = 0.8·R_Q + 0.2·R_S` | `0.75·sim + 0.25·s0` | 0.75 / 0.25 |
| `R = R_Q + R_F` | `0.3·sim + 0.7·flow` | 0.3 / 0.7 |
| halt when `max(R) < τ` | halt when `sim < 0.8` **and** `combined < 0.8` | the code's |

The last is consequential: under the paper's rule a sequential step always scores
above τ, the walk could never stop early, and the template would degenerate into
a verbatim replay of one stored solution.

## 8. The one deliberate adaptation

The reference retrieves from a curated library of generic method skeletons —
0.19% of its node texts contain a 4-or-more-digit literal. This project retrieves
from teacher-generated solutions to *specific* problems, where 36.6% do. The
technique is unchanged; the template source is not the paper's, and that gap is
measured in the graph build report rather than assumed away.

## 9. Code map

| stage | file |
|---|---|
| nodes, edges, embeddings | `rot/rot_graph.py` |
| candidate filter, walk, template formatting | `rot/rot_retrieval.py` |
| prompt constants and assembly | `rot/student.py:348-500` |
| prefill transport | `rot/llm.py:ChatClient.chat` |
| open-trace answer extraction | `rot/scoring.py:extract_answer_open_trace` |
| thinking/answer split under a prefill | `rot/llm.py:split_thinking_continued` |
