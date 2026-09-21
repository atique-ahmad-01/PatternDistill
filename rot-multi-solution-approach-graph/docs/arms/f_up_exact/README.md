# `f_up_exact` — RoT templates in the user turn, at the reference's prompt and budget

**Result: 39/48 = 81.2%. Baseline 36/48 = 75.0%. +6.2 pp — the best score of any
arm in the study.**
6 gained, 3 lost, exact McNemar **p = 0.51** — not statistically distinguishable
from the baseline at n = 48.
Against its own shuffled-template control (`f_shuf_exact`, 75.0%): 9 gained,
6 lost, **p = 0.61**.

Read §5 before quoting the +6.2 pp. The baseline it is measured against was run
at a **smaller token budget and a different system prompt**, and part of the gap
is that difference rather than retrieval.

The result in this directory that *is* decisive is a different one: against
`f_exact` — the identical configuration with the injection site moved into the
`<think>` block — this arm gains **31 problems and loses 0** (p ≈ 10⁻⁹).

This directory is self-contained. Nothing here depends on the other two arm
directories, and no file here is shared with them.

| file | what it is |
|---|---|
| `TECHNIQUE.md` | **the method alone** — graph, walk, injection, scoring path, settings. Start here if you only want the technique |
| `README.md` | this document — the whole A-to-Z |
| `config.yaml` | byte-identical copy of `config.f_up_exact.yaml` at the repo root |
| `run.sh` | every command, in order, runnable from the repo root |
| `example_prompt.txt` | the literal prompt sent for problem `q0028`, reconstructed from the code and the stored result row |
| `results_summary.md` | the report this arm produced (`results/f_up_exact/summary.md`) |
| `results_summary.csv`, `results_by_group.csv` | the same numbers, machine-readable |

---

## 1. The technique

The paper's retrieval, the paper's prompts and token budget, but **not** the
paper's injection site.

**Kept from arXiv:2509.21743** — reasoning *steps* as the unit of retrieval, the
thought graph with sequential and semantic edges, the reward-guided walk that
assembles an ordered template for the query, the reference's own system prompt,
its `$\boxed{}$` answer contract, and a 15360-token budget large enough that
almost nothing truncates.

**Replaced** — the template is delivered as **user-turn reference material**
instead of being prefilled into an open `<think>` block. The model opens its own
reasoning block and works with the template visible rather than continuing text
it is told it already wrote.

The point of this particular combination is isolation. `f_exact` is the strict
reproduction; this arm changes **exactly one setting** from it —
`rot.injection: ti_prefill` → `user_prompt` — and nothing else. Whatever moves
between those two runs is attributable to the injection site alone.

### What makes it distinct from the other two arms

The retrieval half is **bit-identical** across `f_exact`, `f_userprompt` and
`f_up_exact` — same seed, same graph, same gold tags, same walks. Verified: all
48 templates and all 48 node-id paths match exactly across the three runs.

So the three arms differ **only in delivery**:

| | `f_exact` | `f_userprompt` | `f_up_exact` |
|---|---|---|---|
| `rot.injection` | `ti_prefill` | `user_prompt` | **`user_prompt`** |
| `rot.system_prompt` | `reference` | `project` | **`reference`** |
| answer contract | `$\boxed{}$` | `ANSWER: <int>` | **`$\boxed{}$`** |
| `student.max_tokens` | 15360 | 8192 | **15360** |
| comparable with arms A–E | no | yes | **no** |
| accuracy | 16.7% | 79.2% | **81.2%** |

Two clean one-variable contrasts run through this arm:

* **vs `f_exact`** — only the injection site differs. **31 gained, 0 lost,
  p ≈ 10⁻⁹.** This is the study's decisive result.
* **vs `f_userprompt`** — only the system prompt and token budget differ.
  **5 gained, 4 lost, p = 1.00.** The reference's prompt and larger budget buy
  essentially nothing on their own.

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
* **VRAM note.** This arm runs at `max_tokens: 15360` against a server started
  with `--max-model-len 16384`, so a long prompt plus a long trace sits close to
  the ceiling. The server is launched at `--gpu-memory-utilization 0.80`, which
  is not a comfort setting: at 0.62 the engine refuses to start, because one
  sequence at 16384 needs 1.12 GB of KV cache.

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
$PY main.py verify --config config.f_up_exact.yaml
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
$PY main.py generate --config config.f_up_exact.yaml
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
$PY main.py validate --config config.f_up_exact.yaml
```

Exact integer match against gold. Nothing is deleted — every solution keeps its
record and gains correctness metadata. **651/720 correct (90.4%)**; 141/144
problems have at least one correct solution.
→ `data/artifacts/teacher_solutions_validated.jsonl`

### Stage 4 — memory bank

```bash
$PY main.py build-memory --config config.f_up_exact.yaml
```

`correct_only`, ≥ 2 steps: **651 entries from 141 train problems, 2450 total
steps** (mean 3.76 steps per solution). The builder refuses to run if a test
problem id reaches it.
→ `data/artifacts/memory_bank.jsonl`

### Stage 5 — the RoT thought graph

```bash
$PY main.py build-rot-graph --config config.f_up_exact.yaml
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

`config.yaml` here also sets `template_source: retrieved` explicitly. That is
the default, and it is written out only because this arm's control
(`config.f_shuf_exact.yaml`) differs from it in exactly that key.

### Stage 7 — prompt assembly: **user turn, reference wording**

This is the stage that defines this arm. `rot/student.py:run_rot_arm`, the
non-TI branch with `system_prompt: reference`:

```python
user  = f"{ROT_REFERENCE_TEMPLATE_PROMPT}{template}\n\nProblem: {problem}"
reply = client.chat(ROT_REFERENCE_SYSTEM, user)     # no prefill
```

Two messages go to vLLM. There is **no assistant prefill**, so the model opens
its own `<think>` block:

```
system:  You are asked to solve the mathematical problem using a step-by-step
         template. Follow each step exactly and output your reasoning at each
         step. Use only the steps given in the template for your reasoning
         ← the reference's own system message

user:    Solve the math problem and always conclude with:
         Therefore, the final answer is: $\boxed{answer}$. Where [answer] is
         just the final numerical answer …
         You are given a template to solve the problem. Use the given steps if
         applicable otherwise use them to guide your reasoning and present the
         solution steps logically to solve this problem:
         Step 1: …
         Step 5: …

         Problem: <the problem>
```

Note what this inherits from the reference and what it does not: it keeps the
reference's system message **and** its `$\boxed{}$` contract sentence — which is
why this arm cannot be compared with arms A–E — but it does not open the
assistant turn. The template keeps its line breaks; only the TI path flattens it
to one line.

**Fixed vs. per-problem.** Only two slots change from one test problem to the
next; everything else is a string constant in `rot/student.py` and is
byte-identical across all 48 problems of this arm:

| segment | fixed / varies | source |
|---|---|---|
| system message | **fixed** | `ROT_REFERENCE_SYSTEM` (`student.py:385`) |
| user preamble | **fixed** | `ROT_REFERENCE_TEMPLATE_PROMPT` (`student.py:348`) |
| the template | **varies** | the graph walk, 3–8 steps |
| `"Problem: "` label | **fixed** | the f-string at `student.py:487-488` |
| the problem text | **varies** | dataset field `question` |

`example_prompt.txt` marks every segment as FIXED or VARIES, and ends with a
diff of this prompt against another problem's so the varying parts are visible
directly: [`example_prompt.txt`](example_prompt.txt).

Decoding: temperature 0.6, top_p 0.95, top_k 20, thinking enabled, seed 42,
**max_tokens 15360**.

### Stage 8 — parsing and scoring

Standard path: `extract_answer` → strip `<think>…</think>` → look for an
`ANSWER:` line, then the last `$\boxed{}$`, then prose → exact integer match.

Measured: **47/48 replies closed `</think>`** and 47 wrote a `$\boxed{}$`; the
`$\boxed{}$` branch of the scorer is what reads them. **1 reply hit the
15360-token cap** and is scored wrong (a truncated reply never reaches an answer
and is indistinguishable from a genuine failure in the accuracy column — read
`trunc` and `acc` together). No special-case parser is needed, unlike `f_exact`.

Because the block closes normally, the thinking/answer split is meaningful:
**3579 thinking / 576 answer** tokens on finished replies. The answer side is
large compared with `f_userprompt`'s 11 tokens because the reference contract
asks for a written-out solution ending in `$\boxed{}$`, not a bare `ANSWER:`
line.

### Stage 9 — report

```bash
$PY main.py report --config config.f_up_exact.yaml
```
→ `results/f_up_exact/{summary.md, summary.csv, by_group.csv, per_problem.csv,
summary.json, config.snapshot.yaml, run_manifest.json}`

---

## 4. Results

| arm | n | correct | acc | Δpp | trunc | in tok | out tok (finished) | think tok | lat s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 48 | 36 | 75.0% | — | 9 | 173 | 2766 | 2755 | 19.0 |
| **F_rot (this arm)** | 48 | **39** | **81.2%** | **+6.2** | 1 | 378 | 4154 | 3579 | 29.5 |

Paired on the same 48 problems: **6 gained, 3 lost**, exact McNemar **p = 0.51**.

The arm spends **+43.0% output tokens** against the baseline on problems both
finished, and 29.5 s against 19.0 s. It is not a cheaper answer; it is a longer
one that truncates less (1 vs 9).

Per-family (`results_by_group.csv`), n = 3 per cell:
`wythoff` 33%→100%, `grundy_23` 67%→100%, `fib_huge` 33%→67%,
`grundy_134` 0%→33% against `bitcount_range` 67%→33% and `perm_power` 100%→67%.
`lcg_closed` stays 0% in both.

### The control — is it the *retrieved* template, or would any plan do?

`f_shuf_exact` runs this exact configuration with `template_source: shuffled`:
each problem gets the template assembled for a **different** test problem, by a
derangement, so the template is real, coherent and from the same length and
source-count distribution — it is simply not the one retrieval chose. (The
leakage guard still keys off the real `problem_id`, so a donor template can
never contain the query's own solution.)

| | accuracy |
|---|---:|
| baseline, no template (8192, project prompt) | 75.0% |
| **`f_shuf_exact` — a real but wrong template** | **75.0%** |
| **`f_up_exact` — the retrieved template** | **81.2%** |

Retrieved beats shuffled by **6.2 pp**: 9 gained, 6 lost, **p = 0.61**.

This control does real work: it runs at the *same* 15360 budget and the *same*
reference system prompt as this arm, so it rules out "the gain is just the
bigger budget" — a wrong template at that budget lands exactly on the baseline's
75.0%. It does not fully close the question, because a wrong template may be
actively harmful and could be cancelling a budget gain.

---

## 5. Baseline caveat — read this before quoting +6.2 pp

`results/f_up_exact/eval_A_baseline.jsonl` is **byte-identical** to the baseline
in six other run directories (same md5). It was produced at
`student.max_tokens: 8192` with **this project's** system prompt — *not* at this
arm's 15360 with the reference prompt, even though
`results/f_up_exact/config.snapshot.yaml` records 15360. The snapshot records
the config the report ran under; it does not re-run the cached arm.

Why it matters here specifically:

* the baseline failed to finish **9** replies; this arm failed to finish **1**
* a truncated reply scores wrong
* so some of the 6 problems this arm "gained" may be problems the baseline would
  also have solved given 15360 tokens

`f_shuf_exact` argues against the budget explaining everything (a wrong template
at the same budget scores 75.0%, not higher). But the clean cell — **a
no-template baseline at `max_tokens: 15360` with the reference system prompt** —
**has not been run**. It is one 48-problem evaluation, roughly 4 minutes on a
warm server. Until it exists, +6.2 pp should be quoted as *"+6.2 pp against a
baseline run at a smaller token budget"*, or not quoted at all.

The 31–0 result against `f_exact` does not depend on any of this: both arms ran
at 15360 with the reference prompt, and the only difference between them is the
injection site.

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

**The injection half is a deliberate departure.** The reference implementation
contains this non-TI branch, so the prompt wording is still its own — but the
paper's headline mechanism is Thinking Intervention, and this arm does not use
it. Do not present this run as a reproduction of arXiv:2509.21743; `f_exact` is
that run. Present it as: *the paper's retrieval, delivered differently.*

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

In this arm the model takes the technique and redoes the arithmetic for its own
bound:

> *"999,999 ≡ 0 (mod 7). The next number after 999,999 that is 0 mod 7 is
> 1,000,006. The next that is 2 mod 7 is 1,000,001. Since 1,000,001 < 1,000,006
> … `$\boxed{1000001}$`"* — correct. 2552 tokens, 15.7 s.

`f_exact` receives the same five steps inside its own open `<think>` block,
applies the donor's arithmetic (add 2) to its own bound, and answers
`$\boxed{1000002}$` in 335 tokens and 2.0 s without expressing any doubt.

**That is the whole finding.** Content in the user turn is something the model
can evaluate and reject. The same content placed inside its own reasoning block
is something it has, in effect, already said — and it does not audit its own
premises. Across the full test set, with the injection site as the only
difference: **31 gained, 0 lost, p ≈ 10⁻⁹.**

This also explains why our 36.6% constant-carrying step texts are survivable
here and fatal there. The template source is a real weakness of this adaptation
— but the injection site is what decides whether the model can defend itself
against it.

---

## 8. Pitfalls

* **The +6.2 pp headline is the weakest number in this directory.** p = 0.51
  against the baseline, p = 0.61 against its own control, and the baseline is
  budget-mismatched. The 31–0 injection-site result is the one that holds.
* **Do not compare this arm's accuracy against arms A–E.** It uses the
  reference's system prompt and a `$\boxed{}$` contract instead of the `ANSWER:`
  line. Use `f_userprompt` for cross-arm comparison.
* **Do not call it a reproduction of the paper.** It is the paper's retrieval
  with the injection site changed. `f_exact` is the reproduction.
* **`tag_source: gold`** hands this arm the query's family for free — an
  advantage over arms C/D. `config.f_predicted.yaml` removes it.
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
bash docs/arms/f_up_exact/run.sh       # from the repo root, inside tmux
```

Or, if `data/artifacts/` is already built (stages 1–5 are shared and cost ~1 h):

```bash
PY=~/miniforge3/envs/ms-atique/bin/python
./scripts/serve_vllm.sh student &
$PY main.py evaluate --config config.f_up_exact.yaml --arms F_rot   # ~4 min
$PY main.py report   --config config.f_up_exact.yaml

# and its control, which is what makes the number interpretable:
$PY main.py evaluate --config config.f_shuf_exact.yaml --arms F_rot
$PY main.py report   --config config.f_shuf_exact.yaml
```

**The missing run (§5), if you want the +6.2 pp to stand:** a no-template
baseline at this arm's budget and system prompt. It needs a small config —
`config.f_up_exact.yaml` with `evaluation.arms: [A_baseline]` and a fresh
`run_name` — and about 4 minutes.

Reproducibility notes: seed 42 propagates to Python, NumPy, Torch and the vLLM
`seed` parameter, and the dataset SHA is recorded in every result set. vLLM's
continuous batching still makes decoding **not bit-identical** across runs —
reduction order shifts with batch composition — so record the vLLM version and
do not interleave arms in one batch.
