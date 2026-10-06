# RoT results — p8x_clean

- teacher: `Qwen/Qwen3-8B-AWQ`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `approach_subtasks`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 36 | 75.0% | — | — | 9 | 173 | 2755 | 2766 | — | 19.0 | — |
| F_rot | 48 | 8 | 16.7% | -58.3 | -77.8 | 2 | 371 | 298 | 298 | -91.6% | 1.8 | 100% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

`trunc` = replies that hit `student.max_tokens` mid-trace. These never reach an answer, so they score as WRONG and are indistinguishable from a genuine failure in the accuracy column -- read the two together.

**Token and latency means cover FINISHED replies only.** A truncated reply is censored at the cap: its true trace is longer by an unknown amount, so averaging the cap in drags the mean toward it, by an amount that differs per arm because truncation rates differ per arm. `Δtok paired` is the honest comparison -- mean output tokens against the baseline over problems that finished in BOTH arms, so the same problems are on both sides.

## Per-family accuracy

| family | A_baseline | F_rot |
|---|---|---|
| bitcount_range | 67% | 0% |
| crt_three | 100% | 0% |
| digital_root ⚠ | 100% | 67% |
| fib_huge ⚠ | 33% | 0% |
| frobenius | 100% | 0% |
| grundy_124 | 100% | 0% |
| grundy_134 | 0% | 33% |
| grundy_23 | 67% | 100% |
| josephus | 100% | 0% |
| lcg_closed | 0% | 0% |
| necklace_burnside | 100% | 0% |
| perm_power | 100% | 0% |
| power_tower | 100% | 0% |
| wythoff | 33% | 0% |
| xor_range ⚠ | 100% | 0% |
| zeckendorf ⚠ | 100% | 67% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.

## F_rot — retrieval mechanism

- templates assembled: **48**, mean length **3.62** steps
- drawn from **1.0** source solutions on average; **0%** crossed more than one
- edges taken: {'initial': 48, 'sequential': 126}
- stopped because: {'no_successors': 48}
- carry a 4+ digit constant from another problem: **21%** (reference template library: 0.2%)
- ⚠ **no template left its starting solution.** Traversal followed only sequential edges, so this arm replayed one retrieved solution rather than recombining — it is arm D with extra machinery, and a null result says nothing about RoT.
- most templates name another problem's constants. Measured on RB-DERIV this turned out NOT to be what breaks the arm: the identical templates scored 79.2% delivered in the user turn and 8.3% under Thinking Intervention. Read a drop here against `rot.injection` before blaming the template source.