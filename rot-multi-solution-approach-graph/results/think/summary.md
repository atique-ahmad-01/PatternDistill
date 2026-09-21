# RoT results — think

- teacher: `Qwen/Qwen3-8B`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `approach_subtasks`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | in tok | think tok | out tok | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 48 | 100.0% | — | — | 30 | 42 | 88 | 0.0 | — |
| B_random_memory | 48 | 48 | 100.0% | +0.0 | +0.0 | 218 | 42 | 88 | 0.0 | 3% |
| C_full_solution | 48 | 48 | 100.0% | +0.0 | +0.0 | 312 | 42 | 88 | 0.0 | 99% |
| D_approach_subtasks | 48 | 48 | 100.0% | +0.0 | +0.0 | 218 | 42 | 88 | 0.0 | 99% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

## Per-family accuracy

| family | A_baseline | B_random_memory | C_full_solution | D_approach_subtasks |
|---|---|---|---|---|
| bitcount_range | 100% | 100% | 100% | 100% |
| crt_three | 100% | 100% | 100% | 100% |
| digital_root ⚠ | 100% | 100% | 100% | 100% |
| fib_huge ⚠ | 100% | 100% | 100% | 100% |
| frobenius | 100% | 100% | 100% | 100% |
| grundy_124 | 100% | 100% | 100% | 100% |
| grundy_134 | 100% | 100% | 100% | 100% |
| grundy_23 | 100% | 100% | 100% | 100% |
| josephus | 100% | 100% | 100% | 100% |
| lcg_closed | 100% | 100% | 100% | 100% |
| necklace_burnside | 100% | 100% | 100% | 100% |
| perm_power | 100% | 100% | 100% | 100% |
| power_tower | 100% | 100% | 100% | 100% |
| wythoff | 100% | 100% | 100% | 100% |
| xor_range ⚠ | 100% | 100% | 100% | 100% |
| zeckendorf ⚠ | 100% | 100% | 100% | 100% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.