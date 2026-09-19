# RoT results — no_think

- teacher: `Qwen/Qwen3-8B-AWQ`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `approach_subtasks`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 14 | 29.2% | — | — | 2 | 177 | 0 | 229 | — | 1.3 | — |
| B_random_memory | 48 | 7 | 14.6% | -14.6 | -50.0 | 2 | 590 | 0 | 11 | -95.0% | 0.2 | 4% |
| C_full_solution | 48 | 7 | 14.6% | -14.6 | -50.0 | 0 | 723 | 0 | 9 | -96.1% | 0.2 | 88% |
| D_approach_subtasks | 48 | 7 | 14.6% | -14.6 | -50.0 | 0 | 602 | 0 | 10 | -95.7% | 0.2 | 88% |
| E_gated | 48 | 14 | 29.2% | +0.0 | +0.0 | 0 | 206 | 0 | 561 | +0.0% | 3.3 | 100% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

`trunc` = replies that hit `student.max_tokens` mid-trace. These never reach an answer, so they score as WRONG and are indistinguishable from a genuine failure in the accuracy column -- read the two together.

**Token and latency means cover FINISHED replies only.** A truncated reply is censored at the cap: its true trace is longer by an unknown amount, so averaging the cap in drags the mean toward it, by an amount that differs per arm because truncation rates differ per arm. `Δtok paired` is the honest comparison -- mean output tokens against the baseline over problems that finished in BOTH arms, so the same problems are on both sides.

## Per-family accuracy

| family | A_baseline | B_random_memory | C_full_solution | D_approach_subtasks | E_gated |
|---|---|---|---|---|---|
| bitcount_range | 0% | 0% | 0% | 0% | 0% |
| crt_three | 0% | 0% | 0% | 0% | 0% |
| digital_root ⚠ | 67% | 100% | 67% | 67% | 67% |
| fib_huge ⚠ | 0% | 0% | 0% | 0% | 0% |
| frobenius | 0% | 0% | 0% | 0% | 0% |
| grundy_124 | 33% | 33% | 33% | 33% | 33% |
| grundy_134 | 0% | 0% | 0% | 0% | 0% |
| grundy_23 | 33% | 33% | 33% | 33% | 33% |
| josephus | 33% | 0% | 0% | 0% | 33% |
| lcg_closed | 33% | 0% | 0% | 0% | 33% |
| necklace_burnside | 67% | 0% | 0% | 0% | 67% |
| perm_power | 0% | 0% | 0% | 0% | 0% |
| power_tower | 33% | 0% | 0% | 0% | 33% |
| wythoff | 33% | 0% | 0% | 0% | 33% |
| xor_range ⚠ | 100% | 67% | 67% | 67% | 100% |
| zeckendorf ⚠ | 33% | 0% | 33% | 33% | 33% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.