# `f_userprompt` — RoT templates delivered in the user turn, project prompts

**Result: 38/48 = 79.2%. Baseline 36/48 = 75.0%. +4.2 pp.**
6 problems gained, 4 lost, exact McNemar **p = 0.75** — a positive direction,
*not* a statistically distinguishable gain at n = 48.
Against its own shuffled-template control (`f_shuf`, 68.8%): 9 gained, 4 lost,
**p = 0.27**.

This is the **only one of the three arms that is legitimately comparable with
this project's arms A–E**, because it is the only one that keeps this project's
system prompt and `ANSWER:` contract.

This directory is self-contained. Nothing here depends on the other two arm
directories, and no file here is shared with them.

| file | what it is |
|---|---|
| `TECHNIQUE.md` | **the method alone** — graph, walk, injection, scoring path, settings. Start here if you only want the technique |
| `README.md` | this document — the whole A-to-Z |
| `config.yaml` | byte-identical copy of `config.f_userprompt.yaml` at the repo root |
| `run.sh` | every command, in order, runnable from the repo root |
| `example_prompt.txt` | the literal prompt sent for problem `q0028`, reconstructed from the code and the stored result row |
| `results_summary.md` | the report this arm produced (`results/f_userprompt/summary.md`) |
| `results_summary.csv`, `results_by_group.csv` | the same numbers, machine-readable |

---

## 1. The technique

Half of Retrieval-of-Thought, deliberately.

**Kept** — the paper's retrieval mechanism, unchanged and bit-for-bit identical
to the strict reproduction: reasoning *steps* are the unit of retrieval, they
form a graph with sequential and semantic edges, and a reward-guided walk
assembles an ordered template for the query that can recombine steps from
several stored solutions.

**Replaced** — the injection site. The paper places the template inside an
already-open `<think>` block that the model continues (Thinking Intervention).
This arm hands the identical template to the model as **user-turn reference
material** instead, after this project's normal system prompt, and lets the
model do its own reasoning from scratch.

This is the reference implementation's own non-TI branch, so it is an
ablation rather than an invention. It exists because this project's phase-3B
found that `<think>` injection did nothing on a 1.7B model while prompt
injection helped — which contradicts the paper. This arm tests that
contradiction on the paper's own machinery, and reproduces it: the identical
templates score **16.7% under Thinking Intervention (`f_exact`)** and **79.2%
here**.

### What makes it distinct from the other two arms

The retrieval half is **bit-identical** across `f_exact`, `f_userprompt` and
`f_up_exact` — same seed, same graph, same gold tags, same walks. Verified: all
48 templates and all 48 node-id paths match exactly across the three runs.

So the three arms differ **only in delivery**:

| | `f_exact` | `f_userprompt` | `f_up_exact` |
|---|---|---|---|
| `rot.injection` | `ti_prefill` | **`user_prompt`** | `user_prompt` |
| `rot.system_prompt` | `reference` | **`project`** | `reference` |
| answer contract | `$\boxed{}$` | **`ANSWER: <int>`** | `$\boxed{}$` |
| `student.max_tokens` | 15360 | **8192** | 15360 |
| comparable with arms A–E | no | **yes** | no |
| accuracy | 16.7% | **79.2%** | 81.2% |

`f_up_exact` is this arm plus the reference's system prompt and token budget; it
scores 81.2%, and the difference between the two is 5 gained / 4 lost,
**p = 1.00** — i.e. nothing.

---

## 2. Prerequisites

* **GPU box**: `ssh students@100.118.158.78`, one RTX 5070 Ti (16.3 GB, sm_120).
  Trust `nc -z 100.118.158.78 22`, not the Tailscale status line.
* **Always run inside `tmux`.** The box is shared; an SSH drop kills a
  foreground job, and other users do free VRAM by killing processes. Every
  stage below is resumable, which is the real protection.
* **Interpreter by absolute path**: `~/miniforge3/envs/ms-atique/bin/python`.
  Three different pythons resolve on this box depending on how you connect;
  `conda activate` and bare `python3` are not reliable.
* **`transformers >= 4.40, < 5`.** The paper's embedder
  (`jinaai/jina-embeddings-v2-small-en`) ships modelling code written against
  transformers 4.x. `rot/rot_graph.py` shims the two import-time breaks; a
  failure past that point is an environment problem, not something to shim
  further. This does not affect the vLLM servers — they run in their own env.
* **Code shipped by `rsync`, not `git clone`** — the box has no GitHub
  credentials.

---

## 3. A-to-Z workflow

`run.sh` in this directory executes all of it. What each stage does, what it
consumes, and what it leaves behind:

### Stage 0 — preflight

```bash
PY=~/miniforge3/envs/ms-atique/bin/python
./scripts/serve_vllm.sh check          # driver, GPU, vllm binary, ports 8000/8001
```

### Stage 1 — dataset and the frozen split

```bash
$PY main.py verify --config config.f_userprompt.yaml
```

**RB-DERIV**: 192 derivation-deep questions, 16 technique families × 12
instances, every gold a single exact integer computed in Python. The file is
read-only and checked against `sha256 7e043b6a…`; a silent edit fails the run
rather than the analysis.

The split is **per-family**: 3 held out from every family → **144 train / 48
test**. A global shuffle could leave a technique entirely absent from test, and
technique reuse is the whole experiment. The split is computed once and frozen
to `data/artifacts/split.json`; any later change to the split settings is a loud
error, never a quiet reshuffle. This is the guard that stops a teacher solution
to a test problem ending up in the memory bank.

### Stage 2 — teacher generation

```bash
./scripts/serve_vllm.sh teacher &      # Qwen3-8B-AWQ on :8000
$PY main.py generate --config config.f_userprompt.yaml
```

Qwen3-8B-AWQ solves each of the 144 train problems **5 independent times** (not
"give me 5 solutions in one reply" — that degrades into reworded copies after
the second). Each sample gets a different diversity hint so the five reach the
answer by different routes. Temperature 0.8, thinking on, 12288 output tokens.
**The gold answer is never in the prompt** — a gold-conditioned teacher
rationalises backwards and writes an approach that cannot be reapplied.

Each reply is parsed into `APPROACH:` + `STEP n:` lines + `ANSWER:`.
→ `data/artifacts/teacher_solutions.jsonl`, **720 records**, appended per
problem so a kill loses at most one problem.

### Stage 3 — validation

```bash
$PY main.py validate --config config.f_userprompt.yaml
```

Exact integer match against gold. Nothing is deleted — every solution keeps its
record and gains correctness metadata. **651/720 correct (90.4%)**; 141/144
problems have at least one correct solution.
→ `data/artifacts/teacher_solutions_validated.jsonl`

### Stage 4 — memory bank

```bash
$PY main.py build-memory --config config.f_userprompt.yaml
```

`correct_only`, ≥ 2 steps: **651 entries from 141 train problems, 2450 total
steps** (mean 3.76 steps per solution). The builder refuses to run if a test
problem id reaches it.
→ `data/artifacts/memory_bank.jsonl`

### Stage 5 — the RoT thought graph

```bash
$PY main.py build-rot-graph --config config.f_userprompt.yaml
```

* **Nodes (V)**: one per step — **2450 nodes** from 651 templates. Each carries
  its step text, `template_type` (the source solution's APPROACH),
  `knowledge_tag` (the RB-DERIV family) and source `problem_id`.
* **Embeddings**: `jinaai/jina-embeddings-v2-small-en`, the paper's embedder, no
  prefixes. A step is embedded **with its template type** prepended — "Apply the
  formula" alone is near-contentless and would match every query.
* **E_sequential**: step *i* → *i+1* within a solution, weight 1. **1799 edges.**
* **E_semantic**: raw cosine ≥ **0.85**, cross-template only, uncapped
  neighbours as in the reference. **598,636 edges** (281,226 cross-family).

The build also measures the adaptation's known weak point: **896/2450 = 36.6% of
step texts carry a 4-or-more-digit literal**, against **0.19%** in the
reference's curated library. Our steps name the constants of the problem they
were written for. In this arm that turns out to be survivable — see §7.

→ `data/artifacts/rot_thought_graph.json`, `rot_node_embeddings.npy`

### Stage 6 — retrieval: Algorithm 1

Per test problem, at inference time:

1. **Candidate filter.** `tag_source: gold` → keep nodes whose `knowledge_tag`
   equals the query's own family, minus any node from the query's own problem
   (leakage guard). **Mean 153 candidates** of 2450.
   *This is an advantage over arms C/D, which had to find the family by
   similarity alone. It matches the reference, which annotates query metadata by
   hand, but it must be reported as an advantage.*
2. **First step.** `argmax 0.75·cos(q,v) + 0.25·[v is step 0]`.
3. **Traversal.** `argmax 0.3·cos(q,v) + 0.7·[v is the sequential child]`.
   A sequential child scores ≥ 0.7; any semantic neighbour at most 0.3. So the
   walk follows its starting solution until the steps stop matching the query,
   and crosses into another solution only at a seam.
4. **Stop** at `l_max = 8`, or no unvisited successors, or
   `cos < 0.80 AND combined < 0.80`.

Measured over the 48 test problems (identical in all three arms):

* mean template length **6.98** steps (min 3, max 8)
* drawn from **2.85** source solutions on average; **42/48 (88%)** crossed more
  than one — the walk genuinely recombines
* edges taken: 48 initial, **196 sequential, 91 semantic**
* stopped: 34 at `l_max`, 14 `below_threshold`
* **35/48 templates carry a 4+ digit constant from another problem**
* same-family steps: **100%** (a consequence of `tag_source: gold`)

### Stage 7 — prompt assembly: **user-turn delivery**

This is the stage that defines this arm. `rot/student.py:run_rot_arm`, the
non-TI branch:

```python
user  = f"{ROT_TEMPLATE_PROMPT}{template}\n\nProblem: {problem}"
reply = client.chat(SYSTEM, user)          # no prefill, no assistant message
```

Two messages go to vLLM. There is **no assistant prefill**, so the model opens
its own `<think>` block and reasons from scratch with the template visible:

```
system:  You are solving one mathematics problem. The answer is a single integer.
         … Show only the reasoning you need. …
         End your reply with exactly one line, and nothing after it:
         ANSWER: <integer>
         ← this project's SYSTEM, identical to arms A–E

user:    You are given a template to solve the problem. Use the given steps if
         applicable otherwise use them to guide your reasoning and present the
         solution steps logically to solve this problem:
         Step 1: …
         Step 5: …

         Problem: <the problem>
```

Two details that matter:

* The preamble is the reference's own wording **minus its `$\boxed{}$` contract
  sentence**, so the reply ends with the `ANSWER:` line that arms A–E use. A
  different answer format across arms would put a *parsing* difference inside
  what is supposed to be a *retrieval* comparison. (`scoring.py` handles
  `$\boxed{}$` either way; the point is consistency, not capability.)
* The template keeps its line breaks here. Only the TI path flattens it to one
  line.

**Fixed vs. per-problem.** Only two slots change from one test problem to the
next; everything else is a string constant in `rot/student.py` and is
byte-identical across all 48 problems of this arm:

| segment | fixed / varies | source |
|---|---|---|
| system message | **fixed** | `SYSTEM` (`student.py:34`) |
| user preamble | **fixed** | `ROT_TEMPLATE_PROMPT` (`student.py:358`) |
| the template | **varies** | the graph walk, 3–8 steps |
| `"Problem: "` label | **fixed** | the f-string at `student.py:487-488` |
| the problem text | **varies** | dataset field `question` |

`example_prompt.txt` marks every segment as FIXED or VARIES, and ends with a
diff of this prompt against another problem's so the varying parts are visible
directly: [`example_prompt.txt`](example_prompt.txt).

Decoding: temperature 0.6, top_p 0.95, top_k 20, thinking enabled, seed 42,
**max_tokens 8192** — the same budget arms A–E ran at, which is what keeps the
comparison honest.

### Stage 8 — parsing and scoring

Standard path, identical to arms A–E: `extract_answer` → strip `<think>…</think>`
→ read the last `ANSWER:` line → exact integer match.

Measured: **46/48 replies closed `</think>` and emitted an `ANSWER:` line**;
2 hit the 8192-token cap mid-trace and are scored wrong (a truncated reply never
reaches an answer and is indistinguishable from a genuine failure in the
accuracy column — read `trunc` and `acc` together). No special-case parser is
needed, unlike `f_exact`.

Because the block closes normally, the thinking/answer token split is
meaningful here: **2726 thinking / 11 answer** tokens on finished replies.

### Stage 9 — report

```bash
$PY main.py report --config config.f_userprompt.yaml
```
→ `results/f_userprompt/{summary.md, summary.csv, by_group.csv, per_problem.csv,
summary.json, config.snapshot.yaml, run_manifest.json}`

---

## 4. Results

| arm | n | correct | acc | Δpp | trunc | in tok | out tok (finished) | think tok | lat s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 48 | 36 | 75.0% | — | 9 | 173 | 2766 | 2755 | 19.0 |
| **F_rot (this arm)** | 48 | **38** | **79.2%** | **+4.2** | 2 | 411 | 2737 | 2726 | 17.8 |

Paired on the same 48 problems: **6 gained, 4 lost**, exact McNemar **p = 0.75**.

**Truncation is where the gain lives.** The baseline failed to finish 9 replies;
this arm failed to finish 2. The template ends the search for a strategy, so
more replies reach an answer inside the same 8192-token budget. Paired output
tokens are **−15.5%** against the baseline on problems both arms finished.

Per-family (`results_by_group.csv`) the picture is mixed and small-n:
`wythoff` 33%→100%, `grundy_134` 0%→67%, `lcg_closed` 0%→33% against
`bitcount_range` 67%→33%, `josephus` 100%→67%, `necklace_burnside` 100%→67%,
`zeckendorf` 100%→67%. Every cell is n = 3.

### The control — is it the *retrieved* template, or would any plan do?

`f_shuf` runs this exact configuration with `template_source: shuffled`: each
problem gets the template assembled for a **different** test problem, by a
derangement, so the template is real, coherent and from the same length and
source-count distribution — it is simply not the one retrieval chose. (The
leakage guard still keys off the real `problem_id`, so a donor template can
never contain the query's own solution.)

| | accuracy |
|---|---:|
| baseline, no template | 75.0% |
| **`f_shuf` — a real but wrong template** | **68.8%** |
| **`f_userprompt` — the retrieved template** | **79.2%** |

Retrieved beats shuffled by **10.4 pp**: 9 gained, 4 lost, **p = 0.27**. The
direction is right and the control behaves as a control should (a wrong plan is
*worse* than no plan). The sample is too small to call it significant.

---

## 5. Caveats — read before quoting the delta

* **+4.2 pp is 2 problems.** At n = 48 with 6 gained and 4 lost, p = 0.75. This
  arm shows a *direction*, not a demonstrated gain. The honest one-line summary
  is: *user-turn delivery does not hurt, and may help; Thinking Intervention
  destroys the model.*
* **The baseline is shared, not re-run.** `eval_A_baseline.jsonl` here is
  byte-identical to the baseline in six other run directories. For this arm that
  is legitimate — the baseline ran at the same 8192 budget and the same project
  system prompt, so it is a genuinely matched comparison. (It is *not* matched
  for `f_exact` or `f_up_exact`; see their READMEs.)
* **`tag_source: gold`** hands this arm the query's family for free — an
  advantage over arms C/D, which had to find it by similarity.
  `config.f_predicted.yaml` removes it.

---

## 6. Faithfulness to the paper

The retrieval half is faithful. Four places where the paper's text and the
authors' code disagree, all implemented **the code's way** (it is what produced
the published numbers):

| paper | reference implementation | used here |
|---|---|---|
| `sim~ = (cos+1)/2`, τ = 0.85 | raw cosine throughout, 0.85 for edges | raw |
| `R_initial = 0.8·R_Q + 0.2·R_S` | `0.75·sim + 0.25·s0` (VARS.py) | 0.75/0.25 |
| `R = R_Q + R_F` | `0.3·sim + 0.7·flow` (graph_handler.py) | 0.3/0.7 |
| halt when `max(R) < τ` | halt when `sim < 0.8 AND combined < 0.8` | the code's |

**The injection half is a deliberate departure**, and the whole point of this
arm. The reference implementation contains this branch, so the prompt wording is
still its own — but the paper's headline mechanism is Thinking Intervention, and
this arm does not use it. Do not present this run as a reproduction of
arXiv:2509.21743. `f_exact` is that run.

---

## 7. Why it works where `f_exact` fails — a worked example

Problem `q0028` (family `grundy_134`, gold **1000001**):

> Two players alternate removing 1, 3, 4 stones from a single heap; a player who
> cannot move loses. What is the SMALLEST losing heap size strictly greater than
> **999999**?

The walk assembles 5 steps from 2 source solutions (`q0034_s1` ×4, then a
semantic hop to `q0035_s3`) — **the identical template `f_exact` received**:

```
Step 1: Compute Grundy numbers for small heap sizes to detect pattern
Step 2: Observe losing positions form sequence 0, 2, 7, 9, 14, 16, 21, 23,...
Step 3: Recognize losing positions are numbers congruent to 0 or 2 modulo 7
Step 4: Find smallest number >86420 congruent to 0 or 2 mod7
Step 5: Find the smallest n > 100,000 with n ≡ 0 or 2 mod 7 by adding 2 to 100,000
```

Steps 1–3 are the correct, reusable technique. Steps 4 and 5 carry **another
problem's constants** — `86420` and `100,000`.

In this arm the model reasons in its own voice with the template merely visible,
and audits it:

> *"But the problem says 'the smallest n > 100,000 …'. But in our case, the
> problem is about n > 999,999. So, **the steps given in the problem might be for
> a different problem**, but our reasoning shows that the answer is 1,000,001."*
> — `ANSWER: 1000001`, correct. 3122 tokens, 19.4 s.

`f_exact` receives the same five steps inside its own open `<think>` block,
applies the donor's arithmetic (add 2) to its own bound, and answers
`$\boxed{1000002}$` in 335 tokens and 2.0 s without expressing any doubt.

**That is the whole finding.** Content in the user turn is something the model
can evaluate and reject. The same content placed inside its own reasoning block
is something it has, in effect, already said — and it does not audit its own
premises. Measured across the full test set with the injection site as the only
difference (`f_exact` vs `f_up_exact`, both at reference prompt and 15360
tokens): **31 problems gained, 0 lost, p ≈ 10⁻⁹.**

This also explains why our 36.6% constant-carrying step texts are survivable
here and fatal there. The template source is a real weakness of this adaptation
— but the injection site is what decides whether the model can defend itself
against it.

---

## 8. Pitfalls

* **Do not call this a reproduction of the paper.** It is the ablation. Cite
  `f_exact` for the reproduction.
* **This is the arm to compare against A–E**, and the only one: it alone keeps
  the project system prompt, the `ANSWER:` contract and the 8192 budget.
* **2 truncated replies are scored wrong.** They are not parse failures; they
  ran out of budget. Read `trunc` alongside `acc`.
* **Never `pkill -f vllm`** on the box: the pattern matches the SSH command line
  carrying it and kills your own session. Use `./scripts/serve_vllm.sh stop`,
  which kills by exact pid from the pidfile.
* **`--attention-backend TRITON_ATTN` is not optional** on this card. vLLM
  auto-selects FLASHINFER on sm120, which JIT-compiles with the system CUDA 12.8
  and needs ≥ 12.9, then dies with a misleading "requires sm75 or higher".
* **Never `--refresh-split`** on an existing run — it moves the train/test
  boundary under already-generated data.

---

## 9. Reproduce

```bash
bash docs/arms/f_userprompt/run.sh     # from the repo root, inside tmux
```

Or, if `data/artifacts/` is already built (stages 1–5 are shared and cost ~1 h):

```bash
PY=~/miniforge3/envs/ms-atique/bin/python
./scripts/serve_vllm.sh student &
$PY main.py evaluate --config config.f_userprompt.yaml --arms F_rot   # ~2.5 min
$PY main.py report   --config config.f_userprompt.yaml

# and its control, which is what makes the number interpretable:
$PY main.py evaluate --config config.f_shuf.yaml --arms F_rot
$PY main.py report   --config config.f_shuf.yaml
```

Reproducibility notes: seed 42 propagates to Python, NumPy, Torch and the vLLM
`seed` parameter, and the dataset SHA is recorded in every result set. vLLM's
continuous batching still makes decoding **not bit-identical** across runs —
reduction order shifts with batch composition — so record the vLLM version and
do not interleave arms in one batch.
