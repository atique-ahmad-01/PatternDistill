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

No form is significant on its own yet (exact McNemar vs baseline: best
`reflect` 8 gained / 2 lost, p=0.11).

## Paper plan (suggestion)

**Title:** *Hints, Not Plans: Budget-Aware Retrieval of Thinking Traces for
Small Reasoning Models*

**Claim:** a small model gets more accurate *and* cheaper *and* faster when
given retrieved worked examples as hints — not when given a plan to follow.

**Findings to establish**

| # | Finding | Evidence so far |
|---|---|---|
| 1 | Plans hurt, hints help: RoT templates make the 1.7B cheap but inaccurate; T3 examples beat the baseline on all three goals | RoT 17-35%; T3 81-87.5% vs 75%, -24 to -40% tokens |
| 2 | The gain is mostly preventing runaway reasoning | 5 of `reflect`'s 8 gains were baseline truncations |
| 3 | Forms are complementary; an adaptive cascade approaches the oracle under a token budget | any-form oracle 46/48 vs best single 42/48 |
| 4 | Answer length predicts failure, so it can drive the cascade | wrong 2,989 vs right 1,717 median tokens |

**Method contribution:** adaptive form cascade — cheapest form first, stop past
a length threshold, retry with another form.

**Benchmarks**

| benchmark | why |
|---|---|
| RB-DERIV (48 test) | current results; exact integer golds |
| MATH-500 | standard; the paper's small-model setting |
| AIME 2024/25 | hard; where the paper saw the largest gains |

Skip GSM8K (too easy) and MBPP (code; RoT lost 16pp there).

**Models:** teacher Qwen3-8B; students Qwen3-0.6B, 1.7B, 4B.

**Arms**

| arm | role |
|---|---|
| baseline | no examples, same session |
| concise-only | "be concise" instruction, no examples — tests finding 2 |
| random examples | does retrieval matter? |
| RoT (`f_exact`, `p8_clean`) | plans, for finding 1 |
| T3 `struct` / `semantic` / `reflect` | hints, fixed form |
| T3 student-error `reflect` | pitfalls from the student's own mistakes |
| T3 teacher-free | notes written by the student itself |
| **cascade** | the method |

**Report:** accuracy, output tokens, latency; 3 seeds; paired McNemar /
bootstrap CIs; accuracy-vs-tokens Pareto plot.

**Order:** random + concise-only + same-session baseline on RB-DERIV (~1 h GPU)
→ cascade → MATH-500 / AIME with 3 seeds → other student sizes.

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
