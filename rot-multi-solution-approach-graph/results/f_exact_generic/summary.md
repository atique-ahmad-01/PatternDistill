# RoT results — f_exact_generic

- teacher: `Qwen/Qwen3-8B-AWQ`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `approach_subtasks`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 36 | 75.0% | — | — | 9 | 173 | 2755 | 2766 | — | 19.0 | — |
| F_rot | 48 | 17 | 35.4% | -39.6 | -52.8 | 1 | 405 | 498 | 498 | -83.4% | 3.0 | 100% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

`trunc` = replies that hit `student.max_tokens` mid-trace. These never reach an answer, so they score as WRONG and are indistinguishable from a genuine failure in the accuracy column -- read the two together.

**Token and latency means cover FINISHED replies only.** A truncated reply is censored at the cap: its true trace is longer by an unknown amount, so averaging the cap in drags the mean toward it, by an amount that differs per arm because truncation rates differ per arm. `Δtok paired` is the honest comparison -- mean output tokens against the baseline over problems that finished in BOTH arms, so the same problems are on both sides.

## Per-family accuracy

| family | A_baseline | F_rot |
|---|---|---|
| bitcount_range | 67% | 0% |
| crt_three | 100% | 33% |
| digital_root ⚠ | 100% | 67% |
| fib_huge ⚠ | 33% | 0% |
| frobenius | 100% | 67% |
| grundy_124 | 100% | 0% |
| grundy_134 | 0% | 67% |
| grundy_23 | 67% | 67% |
| josephus | 100% | 0% |
| lcg_closed | 0% | 0% |
| necklace_burnside | 100% | 33% |
| perm_power | 100% | 0% |
| power_tower | 100% | 67% |
| wythoff | 33% | 33% |
| xor_range ⚠ | 100% | 100% |
| zeckendorf ⚠ | 100% | 33% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.

## F_rot — retrieval mechanism

- templates assembled: **48**, mean length **7.23** steps
- drawn from **3.15** source solutions on average; **92%** crossed more than one
- edges taken: {'initial': 48, 'sequential': 193, 'semantic': 106}
- stopped because: {'l_max': 36, 'below_threshold': 12}
- carry a 4+ digit constant from another problem: **6%** (reference template library: 0.2%)