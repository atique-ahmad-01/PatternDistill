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
