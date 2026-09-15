# Retrieval-of-Thought (RoT) — multi-solution approach graph

A strong teacher (Qwen3-8B) solves each training problem several different
ways. Every solution is split into an **approach** and a list of **mini-subtasks**.
Those become a memory bank and an approach graph. At test time the closest
approaches are retrieved and pasted into a small student's prompt
(Qwen3-1.7B), and the student is benchmarked with and without them.

The question the experiment answers: **does handing a small model someone
else's reasoning strategy help it solve a problem it cannot crack alone — and
is it the *retrieval* that helps, or would any worked example do?**

---

## Architecture

```
RB_DERIV/ (192 questions, 16 families)
        │
        ├── per-family split: 9 train / 3 test in EVERY family → 144 / 48
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ 1. GENERATE      teacher: Qwen3-8B via vLLM                │
  │    144 train problems × N=5 independent samples            │
  │    each parsed → APPROACH + STEP 1..k + ANSWER             │
  │    → data/teacher_solutions.jsonl                          │
  └─────┬──────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ 2. VALIDATE      exact integer match against gold          │
  │    marks correct / unparseable / agrees_with_majority      │
  │    NOTHING is deleted — filtering is a separate choice     │
  │    → data/teacher_solutions_validated.jsonl                │
  └─────┬──────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ 3. BUILD-MEMORY  problem → solution → approach → subtasks  │
  │    train problems ONLY (asserted, not assumed)             │
  │    → data/memory_bank.jsonl                                │
  └─────┬──────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ 4. BUILD-GRAPH   approach + subtask nodes                  │
  │    belongs_to (structural) · similar_to (cosine ≥ τ)       │
  │    cross-problem edges only                                │
  │    → data/thought_graph.json, *_embeddings.npy             │
  └─────┬──────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ 5. EVALUATE      student: Qwen3-1.7B via vLLM              │
  │    A baseline · B random memory · C full solution          │
  │    D approach + subtasks                                   │
  │    → results/<run>/eval_<arm>.jsonl                        │
  └─────┬──────────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────────┐
  │ 6. REPORT   accuracy · abs/rel improvement · tokens ·      │
  │             latency · per-family breakdown                 │
  │    → results/<run>/summary.{md,csv,json}                   │
  └────────────────────────────────────────────────────────────┘
```

---

## Dataset

`data/RB_DERIV/RB_DERIV_questions.json` ships with the project — the one
canonical copy of this dataset anywhere. `main.py verify`
checks it against the SHA-256 recorded in `config.yaml`, so a silent edit or a
truncated copy is caught before a run rather than after.

This project is **fully self-contained**: no file outside its own directory is
read at any point, so it can be copied, cloned, or rsynced anywhere and run.

192 questions, 16 technique families × 12 instances:

```json
{
  "query_id": "q0001",
  "family": "lcg_closed",
  "question": "Let x(0)=1 and x(k+1) = (5*x(k) + 3) mod 1000. What is x(1000000000000001)?",
  "gold_answer": 968,
  "uses_object": "lcg_closed",
  "false_friend_answer": 500,
  "false_friend_changes_answer": true
}
```

Every gold is an **exact integer**, so scoring is exact — no grading model, no
partial credit, no LaTeX equivalence. Each family is *derivation-deep*: the
work is in finding the shortcut (a closed form, a cycle, a period), not in the
arithmetic afterwards. That is exactly what makes a retrieved approach worth
measuring — it can remove the search without removing the execution.

**The split is per-family, not a global shuffle.** A global shuffle can leave a
whole technique absent from test, and technique reuse is the entire experiment.
`main.py verify` reports any family missing from test.

**The split is frozen.** It is computed once and written to
`data/artifacts/split.json`; every stage reads it from there. Without this,
each command re-derives the split from config — and a seed changed between
`generate` and `evaluate` would silently move the train/test boundary, putting
a problem the teacher already solved into the test set. The memory bank would
then hold a solution to a problem the student is scored on, invalidating the
run while every individual stage still looked fine. Changing any split setting
after freezing is a loud error, not a quiet reshuffle:

```bash
python main.py all --refresh-split    # deliberately re-split (invalidates prior data)
```

`--limit-train/--limit-test` are applied as a *view* on the frozen split, so a
trial run can never move which problems are held out.

---

## Setup

```bash
pip install -r requirements.txt
```

### Start the models with vLLM + Docker

Docker carries its own CUDA runtime, so the host's system CUDA is untouched —
only the NVIDIA driver must be recent (≥ 570 for RTX 50-series / Blackwell).

```bash
./scripts/serve_vllm.sh check       # driver, docker group, container GPU access
./scripts/serve_vllm.sh teacher     # Qwen3-8B-AWQ  → :8000
./scripts/serve_vllm.sh student     # Qwen3-1.7B    → :8001
```

The script defaults to `--gpu-memory-utilization 0.62`, **not** vLLM's 0.9.
On a shared box, holding ~15GB of 16.6GB is what gets a job killed by someone
hunting for VRAM; 0.62 leaves ~6GB visibly free and still gives a workable KV
cache at `--max-model-len 8192`.

Override anything via env vars:

```bash
TEACHER_MODEL=Qwen/Qwen3-8B GPU_UTIL=0.8 MAX_LEN=16384 ./scripts/serve_vllm.sh teacher
```

Both models can share one GPU (5.5GB + 1.2GB at 4-bit), or set `GPU=1` to split
them across cards.

---

## Running

```bash
python main.py verify        # dataset + split + server health; runs nothing
python main.py all           # every stage in order
```

Or stage by stage, each resumable and each writing files the next one reads:

```bash
python main.py generate      # teacher solutions       (the expensive stage)
python main.py validate      # correctness metadata
python main.py build-memory  # memory bank
python main.py build-graph   # nodes + semantic edges
python main.py evaluate      # student arms A/B/C/D
python main.py report        # tables from existing results
```

### Try it with no GPU at all

```bash
python main.py all --mock --run smoke --limit-train 32 --limit-test 16
```

`--mock` swaps in a deterministic fake server and a hash-based embedder, so the
whole pipeline — prompts, parsing, memory, graph, retrieval, leakage guards,
scoring, reporting — runs in about a second. Results are stamped `mock: true`.
Use it to verify wiring before spending GPU time; never to draw a conclusion.

### Useful flags

| flag | effect |
|---|---|
| `--run NAME` | results land in `results/NAME/` |
| `--limit-train N` / `--limit-test N` | cap problems, keeping every family represented |
| `--top-k K` | retrieved memories per problem |
| `--retrieval-mode M` | `full_solution` \| `approach` \| `subtasks` \| `approach_subtasks` |
| `--arms A,B` | run a subset of arms |
| `--no-resume` | ignore existing output and start over |

---

## How retrieval works

The test problem is embedded and scored against every **train** problem in the
bank; surviving entries are ranked and the top-k taken.

Three leakage guards are enforced **in code and cannot be disabled in config**:

1. the bank is built only from train problems — `build-memory` raises if a test
   id reaches it;
2. an entry whose `problem_id` matches the query is dropped;
3. an entry whose normalised problem *text* matches the query is dropped —
   this catches a duplicate problem appearing under a different id.

All four presentation modes retrieve the **same entries** and differ only in
what is shown, so a gap between arms C and D is a difference in presentation,
not in retrieval quality.

| mode | shown to the student |
|---|---|
| `full_solution` | source problem + approach + every step |
| `approach` | the approach line alone |
| `subtasks` | the reasoning steps alone |
| `approach_subtasks` | approach + numbered steps (no source problem) |

### The graph

Two node types and two edge types, stored as plain JSON plus a `.npy`:

```
nodes   approach   one per memory entry
        subtask    one per reasoning step

edges   belongs_to  subtask → its approach        (structural)
        similar_to  node ↔ node, cosine ≥ τ       (semantic, cross-problem only)
```

Subtasks are embedded **with their approach as context** — in isolation "Apply
the formula" is near-meaningless and matches everything.

There is no graph library and no database. At a few thousand nodes a dict of
adjacency lists is faster to read, faster to debug, and trivially swappable.
Retrieval reads only these files, so the strategy can be replaced without
touching graph construction.

---

## Reasoning mode

Both models run with Qwen3's **thinking mode on**, and that is a deliberate
measurement choice rather than a default.

The claim under test is that a retrieved approach removes the need to *search*
for a strategy. That search happens in the reasoning trace, not in the final
answer — so with thinking off you would be measuring answer verbosity, a much
weaker signal for the same claim. Results therefore report `thinking_tokens`
and `answer_tokens` separately; the expected effect is a collapse in the
former while accuracy holds or rises.

Three settings are coupled to that choice and must move together:

| setting | value | why |
|---|---|---|
| `max_tokens` | 8192 | reasoning traces truncate at 3072. A truncated reply has an unclosed `<think>`, no parseable answer, and scores as wrong — indistinguishable from genuine failure |
| `MAX_LEN` (vLLM) | 16384 | must hold prompt + trace + answer, and the memory arms have long prompts |
| `student.temperature` | 0.6, not 0.0 | Qwen3's guidance is explicit that greedy decoding in thinking mode degrades quality and can produce endless repetition. The seed is pinned instead, so arms stay comparable |

The teacher runs at 0.8 — slightly above Qwen3's recommended 0.6 — because its
five samples must reach the answer by *different* routes, and that diversity is
the whole point of sampling five.

If the teacher's verified-solution rate comes out low, check the truncation
count before concluding the model cannot derive the technique. `validate`
reports both.

## The arms

| arm | prompt | what it isolates |
|---|---|---|
| **A** `baseline` | problem only | the floor |
| **B** `random_memory` | problem + k *random* memories | **the control** |
| **C** `full_solution` | problem + k retrieved full solutions | does a complete worked solution help? |
| **D** `approach_subtasks` | problem + k retrieved approaches and steps | does the *strategy* alone help? |

**Arm B is what makes the result interpretable.** Without it, a gain in C or D
could be "any worked example helps" rather than "retrieval found the right
memory" — entirely different claims. Everything except the injected block is
held constant across arms: same system prompt, same decoding, same answer
contract.

---

## Reading the results

`results/<run>/summary.md` carries the headline table; `by_group.csv` the
per-family breakdown; `per_problem.csv` every problem × arm with retrieved ids,
retrieval scores, tokens and latency.

Three things the report deliberately makes hard to misread:

- **Improvement is reported both ways.** Absolute (percentage points) and
  relative (percent of baseline) answer different questions, and quoting only
  the flattering one is the usual way these numbers mislead.
- **Tokens never appear without accuracy from the same row.** A large token
  saving with collapsing accuracy is the model giving up, not a saving.
- **`same-family` retrieval rate is reported per arm.** RB-DERIV families share
  one technique, so "did retrieval land in the right family?" explains most of
  any effect. Without this column, a family-matching win reads as semantic
  understanding.

Families with few distinct golds (`digital_root`, `zeckendorf`, `xor_range`,
`fib_huge`) are flagged ⚠ in the table — guessing beats chance there.

---

## Reproducing

Every result set carries `config.snapshot.yaml` and `run_manifest.json`
recording the exact settings, the seed, and a SHA-256 fingerprint of the
dataset — so two runs cannot be compared against different data without it
being visible.

```bash
python main.py all --run baseline_k3
python main.py all --run k5_approach_only --top-k 5 --retrieval-mode approach
```

Seeds propagate to Python, NumPy and Torch, and to the vLLM `seed` parameter.
Note that vLLM's continuous batching makes greedy decoding **not bit-identical**
across runs — reduction order shifts with batch composition. The effect is
small, but since this measures token deltas between arms, record the vLLM
version and avoid interleaving arms in one batch.

---

## Layout

```
config.yaml              every knob; snapshotted with each run
main.py                  CLI
data/
  RB_DERIV/              the dataset (read-only, never written to)
  artifacts/             generated: split.json, solutions, bank, graph, vectors
rot/
  config.py              loading, validation, seeding, JSONL helpers
  dataset.py             split, gradability filter, fingerprint
  llm.py                 OpenAI-compatible client + offline mock
  teacher.py             multi-solution generation and parsing
  validate.py            correctness metadata
  memory.py              the memory bank
  graph.py               nodes, edges, embeddings
  retrieval.py           top-k, four modes, leakage guards
  student.py             arms A/B/C/D
  report.py              tables
  scoring.py             exact integer extraction and matching
scripts/serve_vllm.sh    docker/vLLM launcher
results/<run>/           per-arm JSONL, summary tables, config snapshot
logs/                    per-stage logs
```

## Cost

At N=5 over 144 train problems the teacher makes **720 calls**; the student
makes **48 × 4 = 192**. On one RTX 5070 Ti with vLLM batching, expect roughly
40–70 minutes for the teacher stage and 10–20 for evaluation. The teacher stage
is the expensive one and it is resumable — a killed run picks up where it
stopped.
