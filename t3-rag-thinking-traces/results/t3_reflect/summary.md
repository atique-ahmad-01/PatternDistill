# T3 RAG results — t3_reflect

- teacher: `Qwen/Qwen3-8B-AWQ`
- student: `Qwen/Qwen3-1.7B`
- retrieval: `t3_reflect`, top_k=3, seed=42
- problems compared across all arms: **48**

| arm | n | correct | acc | abs Δpp | rel Δ% | trunc | in tok | think tok | out tok | Δtok paired | lat s | same-family |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_baseline** | 48 | 36 | 75.0% | — | — | 9 | 173 | 2755 | 2766 | — | 19.0 | — |
| G_t3 | 48 | 42 | 87.5% | +12.5 | +16.7 | 0 | 827 | 2354 | 2366 | -24.5% | 15.6 | 88% |

`abs Δpp` = percentage points over baseline; `rel Δ%` = percent of baseline.
`same-family` = share of retrieved memories drawn from the query's own technique family.
`think tok` = reasoning-trace tokens, apportioned from the reply by character share. This is the quantity the experiment is about: a retrieved approach should collapse the SEARCH for a strategy, which lives here rather than in the final answer.

`trunc` = replies that hit `student.max_tokens` mid-trace. These never reach an answer, so they score as WRONG and are indistinguishable from a genuine failure in the accuracy column -- read the two together.

**Token and latency means cover FINISHED replies only.** A truncated reply is censored at the cap: its true trace is longer by an unknown amount, so averaging the cap in drags the mean toward it, by an amount that differs per arm because truncation rates differ per arm. `Δtok paired` is the honest comparison -- mean output tokens against the baseline over problems that finished in BOTH arms, so the same problems are on both sides.

## Per-family accuracy

| family | A_baseline | G_t3 |
|---|---|---|
| bitcount_range | 67% | 100% |
| crt_three | 100% | 100% |
| digital_root ⚠ | 100% | 100% |
| fib_huge ⚠ | 33% | 100% |
| frobenius | 100% | 100% |
| grundy_124 | 100% | 100% |
| grundy_134 | 0% | 67% |
| grundy_23 | 67% | 33% |
| josephus | 100% | 100% |
| lcg_closed | 0% | 33% |
| necklace_burnside | 100% | 100% |
| perm_power | 100% | 67% |
| power_tower | 100% | 100% |
| wythoff | 33% | 100% |
| xor_range ⚠ | 100% | 100% |
| zeckendorf ⚠ | 100% | 100% |

⚠ = few distinct golds in this family; guessing scores above chance there, so read those cells with care.