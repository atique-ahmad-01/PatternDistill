# T3 RAG over thinking traces on RB-DERIV

Reproduces the method of *RAG over Thinking Traces Can Improve Reasoning Tasks*
(Arabzadeh, Ma, Min, Zaharia — arXiv:2605.03344) at small scale, as the
comparison to the Retrieval-of-Thought work in `../rot-multi-solution-approach-graph/`.

**Goal:** match or beat the no-retrieval baseline's accuracy (75%) while using
fewer tokens and less time per problem.

## Method

1. **T3 rewrite (offline).** The Qwen3-8B teacher rewrites each of the 651
   memory-bank solutions — its *full* thinking trace, ~1,700 words — into one of
   three short forms:
   - `struct` — a step-by-step cheatsheet, at most 7 steps
   - `semantic` — full derivation → core moves → insight
   - `reflect` — pitfalls and safeguards, then the correct approach
2. **Retrieve.** For each test problem, the 3 most similar train problems
   (nomic-embed, one example per problem, leakage-guarded).
3. **Answer.** Qwen3-1.7B gets the examples in the **user prompt** under the
   paper's instruction — *"Solve the main problem by using useful hints and
   strategies from the retrieved examples."* — and reasons on its own.

| arm | what the student sees |
|---|---|
| `A_baseline` | the problem only (copied from the RoT project's `f_exact`, not re-run) |
| `G_t3` | 3 retrieved examples (problem + T3 text) + the problem |
| `R_random_t3` | 3 *random* examples — the control: does retrieval matter, or any example? |

Everything except the examples matches the baseline exactly: same 48 test
problems, frozen split, student, system prompt, decoding and 15,360-token cap.

Each form also has a `_long` version (`struct_long`, ...) asking for the
concrete formulas and values, 250-450 words; the short prompts gave 61 / 152 /
100 words, under the paper's 239-454.

## Results (2026-10-07)

48 test problems; `Δtok` = paired output tokens against the baseline.

| arm | acc | out tok | Δtok | lat s | trunc |
|---|---:|---:|---:|---:|---:|
| A_baseline | 75.0% | 2,766 | — | 19.0 | 9 |
| struct | 83.3% | 2,490 | -29.6% | 17.9 | 0 |
| semantic | 81.2% | 2,285 | -39.7% | 17.0 | 1 |
| reflect | **87.5%** | 2,366 | -24.5% | **15.6** | 0 |
| struct_long (184 words) | 85.4% | 2,477 | -25.3% | 18.4 | 0 |
| semantic_long (316 words) | 83.3% | **1,986** | **-40.5%** | 15.8 | 1 |
| reflect_long (194 words) | 83.3% | 2,594 | -24.5% | 19.1 | 0 |

Every form beats the baseline on accuracy and tokens; all but `reflect_long`
also on latency. One problem is 2.1pp, so the gaps between forms are noise;
only the gain over the baseline is consistent. Much of it is fewer runaway
traces (9 truncations to 0-1). The baseline is copied, not re-run, so latency
is not a same-session comparison, and the random-example control
(`R_random_t3`) has not been run yet.

## Run (GPU box, inside tmux)

```bash
./scripts/run_t3.sh                          # all three forms
VARIANTS=struct ./scripts/run_t3.sh          # one form
ARMS=G_t3,R_random_t3 ./scripts/run_t3.sh    # with the random control
```

Results land in `results/t3_<form>/summary.md`. Every stage resumes after a kill.

Offline check (no GPU): `python main.py t3 --mock`, then
`python main.py evaluate --mock` and `python main.py report --mock`.

## Self-contained

Nothing outside this folder is read. Copied in from the RoT project:
the dataset (`data/RB_DERIV/`), the frozen split, memory bank, teacher traces
and bank embeddings (`data/artifacts/`), and the baseline rows (`data/baseline/`).
The paper's exact T3 prompt texts are not public in full; `t3rag/t3.py` follows
its stated instructions and output formats.
