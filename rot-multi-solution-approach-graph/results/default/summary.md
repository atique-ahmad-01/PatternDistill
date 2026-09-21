# RoT results — default

- teacher: `Qwen/Qwen3-8B-AWQ`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `approach_subtasks`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 36 | 75.0% | — | — | 9 | 173 | 2755 | 2766 | — | 19.0 | — |
| B_random_memory | 48 | 36 | 75.0% | +0.0 | +0.0 | 7 | 610 | 2602 | 2613 | -9.3% | 18.1 | 8% |
| C_full_solution | 48 | 35 | 72.9% | -2.1 | -2.8 | 8 | 719 | 2247 | 2259 | -22.1% | 15.9 | 88% |
| D_approach_subtasks | 48 | 36 | 75.0% | +0.0 | +0.0 | 4 | 598 | 2754 | 2766 | -12.5% | 19.2 | 88% |
| E_gated | 48 | 41 | 85.4% | +10.4 | +13.9 | 3 | 282 | 4103 | 4118 | +0.0% | 28.9 | 67% |
| F_rot | 48 | 4 | 8.3% | -66.7 | -88.9 | 6 | 481 | 306 | 311 | -88.1% | 1.9 | 100% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

`trunc` = replies that hit `student.max_tokens` mid-trace. These never reach an answer, so they score as WRONG and are indistinguishable from a genuine failure in the accuracy column -- read the two together.

**Token and latency means cover FINISHED replies only.** A truncated reply is censored at the cap: its true trace is longer by an unknown amount, so averaging the cap in drags the mean toward it, by an amount that differs per arm because truncation rates differ per arm. `Δtok paired` is the honest comparison -- mean output tokens against the baseline over problems that finished in BOTH arms, so the same problems are on both sides.

## Per-family accuracy

| family | A_baseline | B_random_memory | C_full_solution | D_approach_subtasks | E_gated | F_rot |
|---|---|---|---|---|---|---|
| bitcount_range | 67% | 67% | 100% | 33% | 67% | 33% |
| crt_three | 100% | 100% | 100% | 100% | 100% | 0% |
| digital_root ⚠ | 100% | 100% | 100% | 100% | 100% | 0% |
| fib_huge ⚠ | 33% | 0% | 0% | 0% | 33% | 33% |
| frobenius | 100% | 100% | 100% | 100% | 100% | 0% |
| grundy_124 | 100% | 100% | 100% | 100% | 100% | 67% |
| grundy_134 | 0% | 33% | 33% | 100% | 33% | 0% |
| grundy_23 | 67% | 67% | 0% | 0% | 100% | 0% |
| josephus | 100% | 100% | 100% | 100% | 100% | 0% |
| lcg_closed | 0% | 67% | 67% | 67% | 100% | 0% |
| necklace_burnside | 100% | 67% | 100% | 100% | 100% | 0% |
| perm_power | 100% | 33% | 33% | 33% | 100% | 0% |
| power_tower | 100% | 100% | 100% | 100% | 100% | 0% |
| wythoff | 33% | 67% | 100% | 100% | 33% | 0% |
| xor_range ⚠ | 100% | 100% | 100% | 100% | 100% | 0% |
| zeckendorf ⚠ | 100% | 100% | 33% | 67% | 100% | 0% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.

## F_rot — retrieval mechanism

- templates assembled: **48**, mean length **6.98** steps
- drawn from **2.85** source solutions on average; **88%** crossed more than one
- edges taken: {'initial': 48, 'sequential': 196, 'semantic': 91}
- stopped because: {'below_threshold': 14, 'l_max': 34}
- carry a 4+ digit constant from another problem: **73%** (reference template library: 0.2%)
- ⚠ most templates name another problem's constants. Under Thinking Intervention these are asserted in the student's own reasoning voice, so a drop here is a property of the template SOURCE, not of the retrieval method.