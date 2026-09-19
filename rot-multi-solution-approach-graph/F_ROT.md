# `F_rot` — Retrieval-of-Thought arm

Implementation of **[arXiv:2509.21743](https://arxiv.org/abs/2509.21743)** and the
authors' reference code at `github.com/ahme0599/Retrieval-of-Thought`, run on
RB-DERIV with Qwen3-1.7B.

Scope: this document covers arm **F only**. Arms A–E are solution-level retrieval
and are documented in `README.md`.

---

## 1. What F is, and how it differs from C/D

| | arms C / D | arm F |
|---|---|---|
| unit scored by the query | whole bank **entry** (651) | individual **step node** (2450) |
| what the query is matched against | the entry's *problem* text | the step's own text |
| selection | top-k, independent | reward-guided **walk** |
| output | 3 whole solutions, verbatim | 1 ordered template, ≤ 8 steps |
| can output span several solutions? | no | **yes — 42/48 did** |
| where it is placed | user turn | **assistant turn, inside open `<think>`** |
| artifact read | `entry_embeddings.npy` | `rot_thought_graph.json` + `rot_node_embeddings.npy` |
| module | `rot/retrieval.py` | `rot/rot_retrieval.py` |

**Why a separate graph rather than reusing `thought_graph.json`:** that graph has
**0 sequential edges**, so there is no ordering relation to walk, and it thresholds
*raw* cosine at 0.82 where RoT needs 0.85. Traversal on it would have nowhere to go.

---

## 2. Architecture

### 2.1 Build path

```
data/artifacts/memory_bank.jsonl  (651 validated teacher solutions)
        │
        │  template t  := one bank entry
        │  step  (t,i) := entry.subtasks[i]
        ▼
  ┌──────────────────────────── rot_graph.build() ────────────────────────────┐
  │  1. emit nodes      2450 step nodes, id t{entry}_s{i}                     │
  │  2. emit E_seq      (t,i) → (t,i+1)                    1,799 edges        │
  │  3. embed           jina-v2-small, text = "approach | step"  → 512-d      │
  │  4. emit E_sem      raw cosine ≥ 0.85, cross-template   598,636 edges     │
  └───────────────────────────────────────────────────────────────────────────┘
        │
        ├── data/artifacts/rot_thought_graph.json    nodes + adjacency + sequential map
        └── data/artifacts/rot_node_embeddings.npy   (2450, 512) float32, L2-normalised
```

### 2.2 Node

| field | value | why it exists |
|---|---|---|
| `node_id` | 0 … 2449 | index into the embedding matrix; row *i* of the `.npy` is node *i* |
| `template_id` | e.g. `q0007_s3` | identifies the source solution; `cross_template_only` and the crossing metric both key off it |
| `step_index` | 0-based | supplies `E_sequential` **and** the first-step bonus `[step_index == 0]` |
| `n_steps` | 3 – 12 | lets a consumer tell "last step of its template" from "walk stopped early" |
| `text` | the step | what is shown to the student |
| `template_type` | the solution's approach line | the reference's coarse class; also prepended at embed time |
| `knowledge_tag` | RB-DERIV family | the candidate filter |
| `problem_id` | e.g. `q0007` | leakage guard |

**Why the step is embedded as `"{approach} | {step}"` and not bare:** a bare step
such as *"Apply the formula"* is near-contentless and lands close to everything.
Since `R_Q` is a query-to-node similarity, such a node would be selected for every
query and the template would be noise.

### 2.3 Edges

| edge | rule | weight | count | share |
|---|---|---|---|---|
| `sequential` | `(t,i) → (t,i+1)` | 1.0 | **1,799** | 0.3% |
| `semantic` | raw cosine ≥ 0.85, different template | the cosine | **598,636** | 99.7% |
| — of which cross-tag | endpoints in different families | — | **281,226** | 47.0% of semantic |
| — of which same-tag | endpoints in the same family | — | 317,410 | 53.0% of semantic |

`1,799 = 2,450 nodes − 651 templates` — every template contributes `n_steps − 1`
sequential edges, so the count is fully determined and is a build-integrity check.

| graph property | value |
|---|---|
| nodes | 2,450 |
| templates | 651 |
| steps per template | mean 3.76, min 3, max 12 |
| step-0 nodes (possible walk entry points) | 651 |
| semantic degree | mean 244.3, max 704 |
| nodes with ≥ 1 semantic neighbour | 2,448 / 2,450 |
| knowledge tags | 16, holding 63 – 257 nodes each |
| embedder | `jinaai/jina-embeddings-v2-small-en`, 512-d |
| τ (semantic) | 0.85 **raw cosine**, uncapped |

**Why τ is on raw cosine and not the paper's rescaled `sim~=(cos+1)/2`:** the
reference's shipped `knowledge_graph.graphml` has minimum semantic-edge similarity
exactly **0.8500** and every traversal calls sklearn `cosine_similarity` directly.
Applying the paper's formula literally would threshold at raw 0.70 and roughly
quadruple the edge count. `rot/config.py` rejects any `rot.tau_semantic ≤ 0.5`
because that value is almost always the rescaled figure pasted in by mistake.

**Why `max_neighbors: 0` (uncapped):** the reference caps nothing — its 2,159 nodes
carry ~107 semantic neighbours each. Ours carry 244. A cap is a deviation and is
recorded in the graph metadata when set.

### 2.4 Query path — Algorithm 1

| # | stage | formula | config key | value | why |
|---|---|---|---|---|---|
| 1 | candidate filter | `knowledge_tag == query_tag`, drop `problem_id == query` | `tag_source` | `gold` | reference filters by template type then knowledge tag, annotating the query by hand; RB-DERIV's family is both labels |
| 2 | first step | `argmax  0.75·cos + 0.25·[step_index==0]` | `w_first_query` / `w_first_s0` | 0.75 / 0.25 | a template should begin at a real template start, without abandoning query alignment |
| 3 | traversal | `argmax  0.3·cos + 0.7·[sequential child]` | `w_trav_query` / `w_trav_flow` | 0.3 / 0.7 | flow dominates so the walk follows a coherent procedure rather than hopping per-step |
| 4 | halt | `cos < 0.8 AND combined < 0.8`, or no successors, or `len ≥ 8` | `sim_threshold` / `l_max` | 0.80 / 8 | disjunctive: a step survives on query alignment **or** on flow. Requiring both cuts every template at its first slightly-off step; requiring neither never stops |
| 5 | inject | template into the prompt | `injection` | `ti_prefill` | Thinking Intervention, as published |

**Consequence of the step-3 weights — this governs the whole shape of the output:**

| candidate | score | max |
|---|---|---|
| sequential child | `0.7 + 0.3·cos` | 1.0 |
| semantic neighbour | `0.3·cos` | 0.3 |

A sequential child always outranks every semantic neighbour. **The walk therefore
follows its starting solution until that solution runs out of steps, and only then
crosses into another one.** Recombination happens at template seams, not step by
step. This holds under the paper's `R_Q + R_F` too, so it is a property of the
method rather than of the 0.3/0.7 choice.

### 2.5 What is passed where

| from | to | payload | notes |
|---|---|---|---|
| `main.py` | `RG.load(cfg)` | — | returns `(graph, vecs)`; fails with a build instruction if absent |
| `main.py` | `RoTRetriever.__init__` | graph, vecs, embedder | asserts `len(nodes) == len(vecs)` |
| `student.run_rot_arm` | `retriever.assemble()` | `problem`, `problem_id`, `query_group` | `problem_id` is the leakage key, never shown to the model |
| `assemble()` | caller | `(steps, meta)` | 18 `rot_*` diagnostic fields |
| `format_template()` | prompt | `"Step i: ..."` per line | numbering exactly as the reference's `build_prompt` |
| `student` | `llm.chat()` | `system`, `user`, `prefill` | `prefill` set only when `injection == ti_prefill` |
| `llm.chat()` | vLLM | `add_generation_prompt=False`, `continue_final_message=True` | both required together; one alone is a 400 |
| `Reply` | row | `prefill_chars` | boundary the report must skip — prefill is prompt, not completion |

**Retriever is built lazily.** A RoT-only run does not load `thought_graph.json` or
nomic-embed. On a single 16.3 GB card that allocation is the one the student needs.

### 2.6 Prompt assembly — the two injection sites

| | `ti_prefill` (published method) | `user_prompt` (ablation) |
|---|---|---|
| system | unchanged | unchanged |
| user | preamble + `Problem: …` | preamble + template + `Problem: …` |
| assistant | `<think>` + framing + **template** + `… Using step 1` | *(none — model starts fresh)* |
| model's relation to the template | its own already-committed reasoning | reference material it may reject |
| API | `continue_final_message=True` | ordinary completion |
| token accounting | `split_thinking_continued(reply[prefill_chars:])` | `split_thinking_tokens(reply)` |

**Why `ti_prefill` needs its own token split and answer parser:** the generation
begins mid-trace with no opening tag, and Qwen3 answers out of the open block
**without ever emitting `</think>`** — measured at **0/48** in the exact run. The
default parser's `strip_thinking` discards everything after an unclosed `<think>`,
which is right for a truncated reply and wrong here. Left unfixed, all 48 rows
score unparseable with every completion token booked as reasoning.
`scoring.extract_answer_open_trace` is a *separate* function, not a loosening of
the existing one, so arms A–E keep their original scoring and stay comparable to
their already-published numbers.

---

## 3. Configuration

| key | value | source | why this value |
|---|---|---|---|
| `tau_semantic` | 0.85 | shipped `.graphml`, min edge sim | raw cosine, not the paper's rescaled scale |
| `sim_threshold` | 0.80 | `VARS.py` | halt test; paper compares against τ instead |
| `w_first_query` / `w_first_s0` | 0.75 / 0.25 | `VARS.py` | paper states 0.8 / 0.2 |
| `w_trav_query` / `w_trav_flow` | 0.3 / 0.7 | `graph_handler.py` | paper states `R_Q + R_F` |
| `l_max` | 8 | `MAX_TEMPLATE_LENGTH` | — |
| `max_neighbors` | 0 (uncapped) | reference caps nothing | positive value = deviation |
| `cross_template_only` | true | — | a semantic edge inside one solution only restates that its steps were written together; `E_sequential` already carries that |
| `tag_source` | `gold` | reference annotates by hand | see §6 — this is an advantage over C/D |
| `injection` | `ti_prefill` | the published method | `user_prompt` is the ablation |
| `system_prompt` | `project` \| `reference` | — | `reference` = verbatim system message + `\boxed` contract |
| `embedding.model` | `jina-embeddings-v2-small-en` | `VARS.py` | needs `transformers<5`; see §7 |

### Paper text vs. reference code — code wins, it produced the published numbers

| | paper | reference implementation |
|---|---|---|
| similarity scale | `sim~ = (cos+1)/2`, τ = 0.85 | raw cosine, 0.85 |
| first step | `0.8·R_Q + 0.2·R_S` | `0.75 / 0.25` |
| traversal | `R = R_Q + R_F` | `0.3·cos + 0.7·flow` |
| halt | `max(R) < τ` | `cos < 0.8 AND combined < 0.8` |

The halt rule is the consequential one: under the paper's rule a sequential step
scores `R ≥ 1 > τ`, so the walk can **never** stop early and always runs to `l_max`.

---

## 4. Results — Qwen3-1.7B, frozen 48-problem test split

| run | injection | tag | prompts | max_tok | acc | Δpp | think | Δtok paired | lat | trunc | unparse |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_baseline` | — | — | project | 8192 | 75.0% | — | 2755 | — | 19.0s | 9 | — |
| **`f_exact`** | `ti_prefill` | gold | **reference** | 15360 | **16.7%** | **−58.3** | 524 | −79.7% | 3.0s | **0** | **0** |
| `default` | `ti_prefill` | gold | project | 8192 | 8.3% | −66.7 | 305 | −88.1% | 1.9s | 6 | 13 |
| `f_predicted` | `ti_prefill` | predicted | project | 8192 | 0.0% | −75.0 | 291 | −88.4% | 1.8s | 4 | 13 |
| `f_userprompt` | `user_prompt` | gold | project | 8192 | **79.2%** | **+4.2** | 2725 | −15.5% | 17.8s | 2 | 2 |

`f_exact` is the reproduction claim. `f_userprompt` is **not RoT** — it keeps RoT's
retrieval and drops RoT's injection.

### Mechanism — identical across all three `gold` runs

| metric | gold runs | `f_predicted` |
|---|---|---|
| candidates after filter | 153 | 152 |
| template length | mean 6.98 | 7.33 |
| source solutions per template | mean 2.85 | 3.35 |
| crossed > 1 solution | **42 / 48 (88%)** | 47 / 48 (98%) |
| edges taken — initial | 48 | 48 |
| edges taken — **sequential** | **196** | 189 |
| edges taken — **semantic** | **91** | 115 |
| halt: `l_max` | 34 | 37 |
| halt: `below_threshold` | 14 | 11 |
| templates carrying a foreign 4+ digit constant | 73% | 67% |

`48 + 196 + 91 = 335 = 48 × 6.98` — the edge ledger reconciles with mean length.

**The three `gold` runs assemble byte-identical templates (verified 48/48).** Their
accuracies are 16.7%, 8.3% and 79.2%. Everything separating them is prompt
placement and wording.

### `f_userprompt` vs baseline, per problem

| | count |
|---|---|
| A right → F right | 32 |
| A right → F **wrong** | 4 |
| A **wrong** → F right | 6 |
| A wrong → F wrong | 6 |
| baseline **truncations** rescued | **4 / 9** |

The gain is concentrated in truncation recovery — the assembled template ends the
search, which is RoT's own stated claim.

---

## 5. Achieved

| | evidence |
|---|---|
| Algorithm 1 implemented to the reference, not the paper's prose | 4 discrepancies resolved in favour of the code, §3 |
| The graph is actually traversed | 196 sequential + 91 semantic edges consumed per run |
| Templates genuinely recombine | 42/48 span > 1 source solution, mean 2.85 |
| Thinking Intervention works end to end on Qwen3 | prefill survives the chat template; model continues from `… Using step 1` |
| Exact reproduction run | reference system message, `\boxed` contract, `max_tokens` 15360 → 0 truncations, 0 unparseable |
| Single-variable ablation isolating the failure | identical templates, 71pp swing on placement alone |
| Reproducible graph build | CPU (mac) and GPU (box) produced identical edge counts |
| Mechanism diagnostics in every report | length, sources, crossing, edge types, halt reasons, contamination |
| 3 bugs fixed, 2 only visible live | §7 |

## 6. Missing

| gap | why it matters | cost |
|---|---|---|
| **No random-template control for F** | `f_userprompt`'s +4.2pp could be "any 8-step plan helps" rather than "the *retrieved* plan helps". This is exactly the role arm B plays for C/D — without it the headline number is contestable. | 48 calls |
| **`tag_source: gold` is an advantage** | F is handed the query's family; C/D had to find it by similarity. Hence F's 100% same-family. `predicted` recovers the tag correctly 33/48 (69%) but was only run under TI, so there is **no `user_prompt` + `predicted`** cell — the 79.2% still carries the advantage. | 48 calls |
| **Template source ≠ reference** | reference steps are generic skeletons (0.2% carry a 4+ digit literal); ours are instance-bound (36.6%). Untested whether generic templates change the TI result. | 1 teacher pass |
| **Benchmark ≠ reference** | paper uses AIME/AMC where traces run ~10k tokens; RB-DERIV baseline is 2766, far less to compress. | dataset prep |
| **No significance testing** | n = 48; one problem = 2.08pp. No CIs, no McNemar anywhere in the study. | free |
| **Single seed** | temperature 0.6, one run per cell. The 8.3% / 16.7% gap is confounded with run-to-run variance. | 3× cells |
| **No 4B/8B comparison** | the paper's headline models. Would test whether TI failure is model-scale-dependent. | needs sequential GPU slots |
| **`max_tokens` 15360 ≠ 18000** | server `max_model_len` is 16384. Not closable on this hardware. | — |

## 7. Known issues and environment

| issue | resolution |
|---|---|
| Qwen3 answers from a prefilled `<think>` without closing it (**0/48** closed) | `scoring.extract_answer_open_trace`, keeping the truncation guard; separate from `extract_answer` so A–E are untouched |
| report booked the prefill as generated reasoning | rows carry `prefill_chars`; report slices before splitting. Failure was an inversion — `(6,0)` where truth was `(0,6)` |
| `--no-resume` appended instead of replacing | pre-existing in `run_arm`/`run_gated_arm` too; all three truncate first |
| jina ships transformers-4.x modelling code; box had 5.13.1 | `~/rotenv`, a venv with `--system-site-packages` inheriting the sm_120 torch. `ms-atique` and the vLLM env untouched. `requirements.txt` pins `transformers<5` |
| embedder is not thread-safe under `concurrency` workers | `threading.Lock` around encode, plus a query cache |
| mock embedder vs real graph | explicit dimension check with a rebuild instruction |

## 8. Reading of the result

| claim | verdict |
|---|---|
| RoT reduces output tokens | **reproduced** — −79.7% (exact), −88.1% (project prompts) |
| RoT reduces latency | **reproduced** — 19.0s → 3.0s |
| RoT maintains accuracy | **not reproduced** — 75.0% → 16.7% |
| RoT's retrieval transfers to a 1.7B | **supported** — same templates, user turn: 79.2%, best non-gated arm |
| RoT's Thinking Intervention transfers to a 1.7B | **not supported** — 71pp swing on placement alone |

Under TI the model stops solving. A typical full generation in `f_exact`:

```
 to 8, I will now solve the problem accordingly.

Therefore, the final answer is: $\boxed{12}$.
```

Thinking falls 2755 → 524 tokens. The reference system prompt — *"Use only the
steps given in the template for your reasoning"* — reinforces this by instruction.
Format compliance is perfect (48/48 `\boxed`, 0 unparseable), so parsing is not the
explanation.

This is consistent with the project's earlier phase-3B finding that `<think>`
injection did nothing on a 1.7B while prompt injection helped — the same split,
far larger.

## 9. Reproducing

```bash
pip install 'transformers>=4.40,<5'        # jina ships 4.x modelling code
python main.py build-rot-graph             # 2450 nodes, ~40s on GPU
./scripts/serve_vllm.sh student            # Qwen3-1.7B on :8001

python main.py evaluate --config config.f_exact.yaml --arms F_rot   # the claim
python main.py evaluate --arms F_rot                                # project prompts
python main.py report  --config config.f_exact.yaml
```

| run | config | results |
|---|---|---|
| exact reproduction | `config.f_exact.yaml` | `results/f_exact/` |
| project prompts | `config.yaml` | `results/default/` |
| user-turn ablation | `config.f_userprompt.yaml` | `results/f_userprompt/` |
| predicted-tag ablation | `config.f_predicted.yaml` | `results/f_predicted/` |
