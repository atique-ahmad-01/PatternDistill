# `f_exact` — exact reproduction of arXiv:2509.21743 on RB-DERIV

**Result: 8/48 = 16.7%. Baseline 36/48 = 75.0%. −58.3 pp.**
30 problems the student solved unaided were broken by the intervention; 2 were fixed.
Exact McNemar p < 0.001 — this collapse is real, not sampling noise.

This directory is self-contained. Nothing here depends on the other two arm
directories, and no file here is shared with them.

| file | what it is |
|---|---|
| `TECHNIQUE.md` | **the method alone** — graph, walk, injection, scoring path, settings. Start here if you only want the technique |
| `README.md` | this document — the whole A-to-Z |
| `config.yaml` | byte-identical copy of `config.f_exact.yaml` at the repo root |
| `run.sh` | every command, in order, runnable from the repo root |
| `example_prompt.txt` | the literal prompt sent for problem `q0028`, reconstructed from the code and the stored result row |
| `results_summary.md` | the report this arm produced (`results/f_exact/summary.md`) |
| `results_summary.csv`, `results_by_group.csv` | the same numbers, machine-readable |

---

## 1. The technique

Retrieval-of-Thought, as the paper's authors actually implement it
(`github.com/ahme0599/Retrieval-of-Thought`). Two ideas, and this arm runs both:

**(a) The thought graph.** Reasoning steps, not whole solutions, are the unit of
retrieval. Every step of every stored solution becomes a node. Nodes are joined
by *sequential* edges (step *i* → step *i+1* of the same solution) and *semantic*
edges (cosine ≥ 0.85 between step embeddings, across different solutions). A
reward-guided walk over that graph assembles a fresh, ordered template for the
query — a path that may not exist in any single stored solution.

**(b) Thinking Intervention.** The assembled template is not shown to the model
as reference material. It is placed *inside an already-open `<think>` block that
the model is made to continue*, so the template arrives in the model's own
reasoning voice rather than as something a user said.

`f_exact` is the strict reproduction: the reference's own system prompt, its own
`$\boxed{}` answer contract, its own prefill wording, and a token budget
(15360) large enough that nothing truncates. It is the run to cite when the
claim is *"we reproduced arXiv:2509.21743."* It is **not** the run to cite when
comparing against this project's arms A–E, because its prompts and answer
contract differ from theirs — use `f_userprompt` for that.

### What makes it distinct from the other two arms

The retrieval half is **bit-identical** across `f_exact`, `f_userprompt` and
`f_up_exact` — same seed, same graph, same gold tags, same walks. Verified:
all 48 templates and all 48 node-id paths match exactly across the three runs.

So the three arms differ **only in delivery**:

| | `f_exact` | `f_userprompt` | `f_up_exact` |
|---|---|---|---|
| `rot.injection` | **`ti_prefill`** | `user_prompt` | `user_prompt` |
| `rot.system_prompt` | **`reference`** | `project` | `reference` |
| `student.max_tokens` | **15360** | 8192 | 15360 |
| accuracy | **16.7%** | 79.2% | 81.2% |

Everything this arm loses, it loses at the injection site.

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
$PY main.py verify --config config.f_exact.yaml
```

**RB-DERIV**: 192 derivation-deep questions, 16 technique families × 12
instances, every gold a single exact integer computed in Python. The file is
read-only and checked against
`sha256 7e043b6a…`; a silent edit fails the run rather than the analysis.

The split is **per-family**: 3 held out from every family → **144 train / 48
test**. A global shuffle could leave a technique entirely absent from test, and
technique reuse is the whole experiment. The split is computed once and frozen
to `data/artifacts/split.json`; any later change to the split settings is a
loud error, never a quiet reshuffle. This is the guard that stops a teacher
solution to a test problem ending up in the memory bank.

### Stage 2 — teacher generation

```bash
./scripts/serve_vllm.sh teacher &      # Qwen3-8B-AWQ on :8000
$PY main.py generate --config config.f_exact.yaml
```

Qwen3-8B-AWQ solves each of the 144 train problems **5 independent times**
(not "give me 5 solutions in one reply" — that degrades into reworded copies
after the second). Each sample gets a different diversity hint so the five
reach the answer by different routes. Temperature 0.8, thinking on, 12288
output tokens. **The gold answer is never in the prompt** — a gold-conditioned
teacher rationalises backwards and writes an approach that cannot be reapplied.

Each reply is parsed into `APPROACH:` + `STEP n:` lines + `ANSWER:`.
→ `data/artifacts/teacher_solutions.jsonl`, **720 records**, appended per
problem so a kill loses at most one problem.

### Stage 3 — validation

```bash
$PY main.py validate --config config.f_exact.yaml
```

Exact integer match against gold. Nothing is deleted — every solution keeps its
record and gains correctness metadata. **651/720 correct (90.4%)**; 141/144
problems have at least one correct solution.
→ `data/artifacts/teacher_solutions_validated.jsonl`

### Stage 4 — memory bank

```bash
$PY main.py build-memory --config config.f_exact.yaml
```

`correct_only`, ≥ 2 steps: **651 entries from 141 train problems, 2450 total
steps** (mean 3.76 steps per solution). The builder refuses to run if a test
problem id reaches it.
→ `data/artifacts/memory_bank.jsonl`

### Stage 5 — the RoT thought graph

```bash
$PY main.py build-rot-graph --config config.f_exact.yaml
```

* **Nodes (V)**: one per step — **2450 nodes** from 651 templates. Each carries
  its step text, `template_type` (the source solution's APPROACH),
  `knowledge_tag` (the RB-DERIV family) and source `problem_id`.
* **Embeddings**: `jinaai/jina-embeddings-v2-small-en`, the paper's embedder,
  no prefixes. A step is embedded **with its template type** prepended —
  "Apply the formula" alone is near-contentless and would match every query.
* **E_sequential**: step *i* → *i+1* within a solution, weight 1. **1799 edges.**
* **E_semantic**: raw cosine ≥ **0.85**, cross-template only, uncapped
  neighbours as in the reference. **598,636 edges** (281,226 of them cross-family).

The build also measures the adaptation's known weak point: **896/2450 = 36.6%
of step texts carry a 4-or-more-digit literal**, against **0.19%** in the
reference's curated template library. Our steps name the constants of the
problem they were written for. Under Thinking Intervention those constants are
asserted in the student's own voice — see §7.

→ `data/artifacts/rot_thought_graph.json`, `rot_node_embeddings.npy`

### Stage 6 — retrieval: Algorithm 1

Per test problem, at inference time:

1. **Candidate filter.** `tag_source: gold` → keep nodes whose
   `knowledge_tag` equals the query's own family, minus any node from the query's
   own problem (leakage guard). **Mean 153 candidates** of 2450.
   *This is an advantage over arms C/D, which had to find the family by
   similarity alone. The reference does the same — it annotates query metadata
   by hand — but it must be reported as an advantage.*
2. **First step.** `argmax 0.75·cos(q,v) + 0.25·[v is step 0]`.
3. **Traversal.** `argmax 0.3·cos(q,v) + 0.7·[v is the sequential child]`.
   A sequential child scores ≥ 0.7; any semantic neighbour scores at most 0.3.
   So the walk follows its starting solution until the steps stop matching the
   query, and crosses into another solution only at a seam.
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

### Stage 7 — prompt assembly: **Thinking Intervention**

This is the stage that defines this arm. `rot/student.py:run_rot_arm`:

```python
user    = f"{ROT_REFERENCE_TEMPLATE_PROMPT}\nProblem: {problem}."
prefill = ROT_PREFILL.format(template=template.replace("\n", " ") + " ")
reply   = client.chat(ROT_REFERENCE_SYSTEM, user, prefill=prefill)
```

Three messages go to vLLM — system, user, **and an assistant message that the
model continues** (`add_generation_prompt=False`, `continue_final_message=True`;
both flags are required together or the server returns 400):

```
system:    You are asked to solve the mathematical problem using a step-by-step
           template. Follow each step exactly and output your reasoning at each
           step. Use only the steps given in the template for your reasoning

user:      Solve the math problem and always conclude with:
           Therefore, the final answer is: $\boxed{answer}$. …
           You are given a template to solve the problem. …
           Problem: <the problem>.

assistant: <think>I need to follow the given steps to guide me in reasoning to
           solve the problem. … The template is. Step 1: … Step 5: …
           Strictly following the given steps for guidance, I will now solve the
           problem starting from the step 1. Using step 1
           ▲ generation starts here, mid-sentence, inside an open <think>
```

The template is flattened to one line (`template.replace("\n", " ")`) exactly as
the reference does.

**Fixed vs. per-problem.** Only two slots change from one test problem to the
next; everything else is a string constant in `rot/student.py` and is
byte-identical across all 48 problems of this arm:

| segment | fixed / varies | source |
|---|---|---|
| system message | **fixed** | `ROT_REFERENCE_SYSTEM` (`student.py:385`) |
| user preamble | **fixed** | `ROT_REFERENCE_TEMPLATE_PROMPT` (`student.py:348`) |
| `"Problem: "` label and trailing `.` | **fixed** | the f-string at `student.py:480` |
| the problem text | **varies** | dataset field `question` |
| prefill wrapper, both halves | **fixed** | `ROT_PREFILL` (`student.py:374`) |
| the template inside the prefill | **varies** | the graph walk, 3–8 steps |

`example_prompt.txt` marks every segment as FIXED or VARIES, and ends with a
diff of this prompt against another problem's so the varying parts are visible
directly: [`example_prompt.txt`](example_prompt.txt).

Decoding: temperature 0.6, top_p 0.95, top_k 20, thinking enabled, seed 42,
**max_tokens 15360**.

### Stage 8 — parsing and scoring

**The TI arm needs its own parser, and this is not a detail.** Qwen3-1.7B
answers straight out of the open reasoning block without ever emitting
`</think>` — measured here: **0 of 48 replies closed the block**. The standard
`extract_answer` sees an unclosed `<think>` and discards everything after it
(correct for a genuinely truncated reply, where a mid-thought number is not an
answer), so every row would score unparseable and the arm would measure nothing.

`extract_answer_open_trace` handles it: it keeps the truncation guard, and for a
finished reply reads from the `$\boxed{}` onward. Result: **0 unparseable, 0
truncated** — every one of the 48 rows is a real, scored answer. The 16.7% is
not a parsing artifact.

One consequence to read carefully: with no `</think>` and no `ANSWER:` line,
`split_thinking_continued` books the whole completion as reasoning. So this
arm's `think tok = 524` is really "the entire reply", and `answer_tokens = 0`
on all 48 rows. Do not read that column as "the answer was short".

### Stage 9 — report

```bash
$PY main.py report --config config.f_exact.yaml
```
→ `results/f_exact/{summary.md, summary.csv, by_group.csv, per_problem.csv,
summary.json, config.snapshot.yaml, run_manifest.json}`

---

## 4. Results

| arm | n | correct | acc | Δpp | trunc | in tok | out tok (finished) | lat s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 48 | 36 | 75.0% | — | 9 | 173 | 2766 | 19.0 |
| **F_rot (this arm)** | 48 | **8** | **16.7%** | **−58.3** | 0 | 448 | 524 | 3.0 |

Paired against the baseline on the same 48 problems: **2 gained, 30 lost**,
exact McNemar **p < 0.001**.

Per-family (`results_by_group.csv`): the collapse is not concentrated anywhere.
The arm scores 0/3 in 9 of the 16 families, and beats the baseline in none of them.

The reply is **5× shorter** than the baseline's (524 vs 2766 output tokens) and
finishes in **3.0 s vs 19.0 s**. The student is not thinking and failing — it is
not thinking at all. It follows the handed-down steps to their conclusion and
stops.

---

## 5. Baseline caveat — read before quoting the delta

`results/f_exact/eval_A_baseline.jsonl` is **byte-identical** to the baseline in
six other run directories (same md5). It was produced at
`student.max_tokens: 8192` with this project's system prompt — *not* at this
arm's 15360 with the reference prompt, even though
`results/f_exact/config.snapshot.yaml` records 15360.

For this arm it barely matters: a −58.3 pp collapse with 30 problems flipped
cannot be explained by the baseline having a smaller token budget. It matters a
great deal for `f_up_exact`, whose +6.2 pp is the same size as the effect the
budget difference could produce. The clean missing cell — a no-template baseline
at 15360 with the reference system prompt — has not been run.

---

## 6. Faithfulness to the paper

Four places where the paper's text and the authors' code disagree. **The code
wins** — it is what produced the published numbers. All four are implemented the
code's way here:

| paper | reference implementation | used here |
|---|---|---|
| `sim~ = (cos+1)/2`, τ = 0.85 | raw cosine throughout, 0.85 for edges | raw |
| `R_initial = 0.8·R_Q + 0.2·R_S` | `0.75·sim + 0.25·s0` (VARS.py) | 0.75/0.25 |
| `R = R_Q + R_F` | `0.3·sim + 0.7·flow` (graph_handler.py) | 0.3/0.7 |
| halt when `max(R) < τ` | halt when `sim < 0.8 AND combined < 0.8` | the code's |

The last is the consequential one: under the paper's rule a sequential step
always scores above τ and the walk can never stop early, which would make the
template a verbatim replay of one source solution.

**The one deliberate deviation**, and the honest threat to the reproduction: the
reference retrieves from a curated library of generic method skeletons (0.19% of
node texts contain a 4+ digit literal). This project retrieves from
teacher-generated solutions to *specific* problems (36.6%). That gap is measured
in the build report rather than assumed away — and §7 shows it is the mechanism
of the failure.

---

## 7. Why it fails — a worked example

Problem `q0028` (family `grundy_134`, gold **1000001**):

> Two players alternate removing 1, 3, 4 stones from a single heap; a player who
> cannot move loses. What is the SMALLEST losing heap size strictly greater than
> **999999**?

The walk assembles 5 steps from 2 source solutions (`q0034_s1` ×4, then a
semantic hop to `q0035_s3`):

```
Step 1: Compute Grundy numbers for small heap sizes to detect pattern
Step 2: Observe losing positions form sequence 0, 2, 7, 9, 14, 16, 21, 23,...
Step 3: Recognize losing positions are numbers congruent to 0 or 2 modulo 7
Step 4: Find smallest number >86420 congruent to 0 or 2 mod7
Step 5: Find the smallest n > 100,000 with n ≡ 0 or 2 mod 7 by adding 2 to 100,000
```

Steps 1–3 are the correct, reusable technique. Steps 4 and 5 carry **another
problem's constants** — `86420` and `100,000`.

Under Thinking Intervention the student reads all five as its own prior
reasoning and continues:

> *"Now, let's compute the smallest number > 100,000 that is congruent to 0 or 2
> mod 7. … Adding 2 to 100,000 gives 100,002. … Therefore, the smallest losing
> heap size strictly greater than 999999 is **1000002**."*
> — `$\boxed{1000002}$`, gold 1000001

It applied the *donor's* arithmetic (add 2) to its *own* bound and never checked
it. 335 output tokens, 2.0 s, no doubt expressed anywhere.

The same template delivered in the user turn (`f_userprompt`, `f_up_exact`)
produces the opposite:

> *"But the problem says 'the smallest n > 100,000 …'. But in our case, the
> problem is about n > 999,999. So, **the steps given in the problem might be for
> a different problem**, but our reasoning shows that the answer is 1,000,001."*
> — correct

**That is the whole finding.** Content in the user turn is something the model
can evaluate and reject. The same content placed inside its own reasoning block
is something it has, in effect, already said — and it does not audit its own
premises. On a 1.7B model the prefill removes the checking step that the
baseline's 2766 tokens of reasoning were spending their budget on.

---

## 8. Pitfalls

* **Do not read `answer_tokens`/`think tok` for this arm as a length claim.**
  With no `</think>`, the whole completion books as thinking. See Stage 8.
* **Do not compare this arm's accuracy against arms A–E.** It uses a different
  system prompt and a `$\boxed{}` contract instead of the `ANSWER:` line. Use
  `f_userprompt` for cross-arm comparison.
* **`tag_source: gold`** hands this arm the query's family for free. It matches
  the reference, but it is an advantage over C/D. `config.f_predicted.yaml`
  removes it — and the arm drops to **0/48**.
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
bash docs/arms/f_exact/run.sh          # from the repo root, inside tmux
```

Or, if `data/artifacts/` is already built (stages 1–5 are shared and cost ~1 h):

```bash
PY=~/miniforge3/envs/ms-atique/bin/python
./scripts/serve_vllm.sh student &
$PY main.py evaluate --config config.f_exact.yaml --arms F_rot   # ~1 min, 48 problems
$PY main.py report   --config config.f_exact.yaml
```

Reproducibility notes: seed 42 propagates to Python, NumPy, Torch and the vLLM
`seed` parameter, and the dataset SHA is recorded in every result set. vLLM's
continuous batching still makes decoding **not bit-identical** across runs —
reduction order shifts with batch composition — so record the vLLM version and
do not interleave arms in one batch.
