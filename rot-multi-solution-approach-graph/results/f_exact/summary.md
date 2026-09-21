# RoT results — f_exact

- teacher: `Qwen/Qwen3-8B-AWQ`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `approach_subtasks`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 36 | 75.0% | — | — | 9 | 173 | 2755 | 2766 | — | 19.0 | — |
| F_rot | 48 | 8 | 16.7% | -58.3 | -77.8 | 0 | 448 | 524 | 524 | -79.7% | 3.0 | 100% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

`trunc` = replies that hit `student.max_tokens` mid-trace. These never reach an answer, so they score as WRONG and are indistinguishable from a genuine failure in the accuracy column -- read the two together.

**Token and latency means cover FINISHED replies only.** A truncated reply is censored at the cap: its true trace is longer by an unknown amount, so averaging the cap in drags the mean toward it, by an amount that differs per arm because truncation rates differ per arm. `Δtok paired` is the honest comparison -- mean output tokens against the baseline over problems that finished in BOTH arms, so the same problems are on both sides.

## Per-family accuracy

| family | A_baseline | F_rot |
|---|---|---|
| bitcount_range | 67% | 33% |
| crt_three | 100% | 0% |
| digital_root ⚠ | 100% | 33% |
| fib_huge ⚠ | 33% | 33% |
| frobenius | 100% | 0% |
| grundy_124 | 100% | 33% |
| grundy_134 | 0% | 0% |
| grundy_23 | 67% | 33% |
| josephus | 100% | 0% |
| lcg_closed | 0% | 0% |
| necklace_burnside | 100% | 0% |
| perm_power | 100% | 0% |
| power_tower | 100% | 0% |
| wythoff | 33% | 0% |
| xor_range ⚠ | 100% | 67% |
| zeckendorf ⚠ | 100% | 33% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.

## F_rot — retrieval mechanism

- templates assembled: **48**, mean length **6.98** steps
- drawn from **2.85** source solutions on average; **88%** crossed more than one
- edges taken: {'initial': 48, 'sequential': 196, 'semantic': 91}
- stopped because: {'l_max': 34, 'below_threshold': 14}
- carry a 4+ digit constant from another problem: **73%** (reference template library: 0.2%)
- most templates name another problem's constants. Measured on RB-DERIV this turned out NOT to be what breaks the arm: the identical templates scored 79.2% delivered in the user turn and 8.3% under Thinking Intervention. Read a drop here against `rot.injection` before blaming the template source.