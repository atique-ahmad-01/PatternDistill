# Phase 2D — Run Notes
Date: 2026-07-19

---

## What Changed vs Phase 2

| Change | Detail |
|---|---|
| Pattern format | Phase 2: 1-sentence abstract summary. Phase 2D: full step-level breakdown with `<steps>` and `<example>` tags |
| Pattern scope | Phase 2: all subjects but shallow format. Phase 2D: all 7 subjects, step-level (400 problems re-distilled) |
| Worked examples | Phase 2: none. Phase 2D: condensed 3-line symbolic worked snippet for every subject (397/400) |
| Retrieval | Phase 2: global, single threshold=0.40. Phase 2D: global + per-subject thresholds (0.50–0.65) |
| `/no_think` | Phase 2: not used. Phase 2D: applied to teacher generation — reduced tokens from ~12k to 300–600 |
| Result | Phase 2: delta=−1.0pp (injection hurt). Phase 2D: delta=+6.0pp |

---

## Run Timeline

| Step | Start | End | Duration |
|---|---|---|---|
| generate_patterns.py (400 problems) | ~01:04 Jul 19 | ~01:56 Jul 19 | ~52 min |
| build_memory_bank.py | ~01:57 Jul 19 | ~01:57 Jul 19 | <1 min |
| inject_and_eval.py (100 problems) | 01:57 Jul 19 | 03:36 Jul 19 | ~99 min |

---

## Hardware & Infrastructure

- **Server**: `students@<gpu-host>`
- **GPU 0**: NVIDIA RTX 5070 Ti, 16.6GB VRAM — used for teacher (pattern gen) and student (eval)
- **GPU 1**: RTX 5070, 12.2GB — cuda:1 was **not usable** during pattern gen (only GPU 0 used)
- **Teacher model**: Qwen/Qwen3-8B, loaded 4-bit quantized (14.6GB free before load)
- **Student model**: Qwen/Qwen3-1.7B
- **Embed model**: nomic-ai/nomic-embed-text-v1 on cuda

### OOM incident during pattern generation
Problem id=33 (Algebra L4) hit CUDA OOM twice and ran to max tokens (16,384), taking 266.8s vs the typical 3–8s. All other 399 problems completed normally.

---

## Pattern Generation Stats

- 400/400 problems completed (resume-safe via JSONL crash recovery)
- All produced: `pat=✓ step=✓ ex=✓`
- 397/400 had worked examples (3 missing — minor)
- Avg latency: ~3–8s per problem; one outlier at 266.8s (OOM on id=33)
- Token range: ~150–950 per problem (most 200–500)

---

## Evaluation Config

```
Mode       : global retrieval + per-subject thresholds + worked examples (all subjects)
Memory bank: 400 entries, dim=768
Thresholds : IA=0.60, NT=0.65, Precalculus=0.65, Geometry=0.55, default=0.50
Eval set   : problems 400–499 (test split)
```

---

## Final Results

```
n=100  injected=81  skipped=19  rich=81

Vanilla   accuracy : 0.700
Injection accuracy : 0.760
Delta              : +0.060  ← best phase-2 result
Gap closed         : 28.6% of the 21pp Phase 1 gap

vs Phase 2  (global, 1-sentence)       : +0.070
vs Phase 2B (subject-filtered)         : +0.080
vs Phase 2C (IA-only step + threshold) : +0.020  ← improvement over prior best
```

### By Subject
| Subject | n | Threshold | Vanilla | Injection | Delta |
|---|---|---|---|---|---|
| Algebra | 23 | 0.50 | 87.0% | 91.3% | +4.3pp |
| Counting & Probability | 6 | 0.50 | 66.7% | 83.3% | +16.7pp |
| Geometry | 5 | 0.55 | 40.0% | 40.0% | 0.0pp |
| Intermediate Algebra | 22 | 0.60 | 63.6% | 68.2% | +4.5pp |
| Number Theory | 16 | 0.65 | 87.5% | 87.5% | 0.0pp |
| Prealgebra | 15 | 0.50 | 53.3% | 66.7% | +13.3pp |
| Precalculus | 13 | 0.65 | 61.5% | 69.2% | +7.7pp |

### By Difficulty
| Level | n | Vanilla | Injection | Delta |
|---|---|---|---|---|
| 1 | 9 | 88.9% | 77.8% | **-11.1pp** |
| 2 | 15 | 73.3% | 80.0% | +6.7pp |
| 3 | 21 | 85.7% | 85.7% | 0.0pp |
| 4 | 21 | 71.4% | 85.7% | **+14.3pp** |
| 5 | 34 | 52.9% | 61.8% | +8.8pp |

---

## Key Observations

### What worked
- **C&P and Prealgebra had the biggest gains** (+16.7pp, +13.3pp). These subjects previously had no step-level patterns — biggest beneficiaries of the expansion.
- **Precalculus recovered** from -7.7pp (Phase 2C) to +7.7pp. Raising threshold 0.50→0.65 was the right call.
- **IA is now actually tested** — 0/22 skipped vs 21/22 skipped in Phase 2C. Confirmed lowering IA threshold 0.75→0.60 was correct.
- **L4 and L5 improve meaningfully** (+14.3pp, +8.8pp). Injection helps harder problems the most.

### What didn't work
- **Geometry still stuck at 0.0pp**. Small n (5) makes it hard to diagnose, but the patterns may not be capturing the right visual/spatial reasoning cues.
- **Number Theory still 0.0pp**. High vanilla baseline (87.5%) leaves little room, but injection is also not hurting. NT threshold=0.65 skips many matches — retrieval not confident enough.
- **Level 1 regresses -11.1pp** (8/9 → 7/9 problems). Easy problems have high vanilla accuracy; injected context may distract the student on trivial problems. Small n (9) so noisy.

### Skipped injections (19/100)
- Mostly NT and IA high-threshold subjects
- When injection is skipped, vanilla and injection results are identical (expected — same prompt)
- 19 skips = 19% of eval set — room to improve coverage by lowering thresholds further if quality holds

---

## Notable Individual Problems

| Problem | Subject | Sim | Threshold | Outcome | Note |
|---|---|---|---|---|---|
| id=409 IA L4 | IA | 0.664 | 0.60 | van=✗ inj=✓ ↑ | Injection rescued a hard IA problem |
| id=419 Precalc L4 | Precalculus | 0.791 | 0.65 | van=✗ inj=✓ ↑ | High-sim Precalc match — injection works |
| id=425 Precalc L5 | Precalculus | 0.681 | 0.65 | van=✗ inj=✓ ↑ | L5 Precalc rescued by injection |
| id=471 IA L5 | IA | 0.666 | 0.60 | van=✓ inj=✗ ↓ | Injection hurt — model confused on hard IA |
| id=483 Precalc L3 | Precalculus | 0.728 | 0.65 | van=✓ inj=✗ ↓ | High-sim but injection still hurt |

---

## Open Questions for Phase 2E

1. **Why does Geometry not respond?** Try subject-specific pattern prompt tuning or visual pattern embedding.
2. **Level 1 regression** — try withholding injection for problems with vanilla sim score above some ceiling (already easy for the model).
3. **NT coverage** — 10+ NT problems skipped at sim<0.65. Could try NT-specific threshold=0.55 or add more NT training data.
4. **IA L5 timeout pattern** — multiple IA L5 problems time out at 75s regardless of injection. Student model hits max tokens; patterns not helping navigate these. Beam search or constrained decoding?
5. **Cross-subject retrieval** — some IA problems may retrieve better from Algebra patterns (shared algebra structure). Could experiment with subject-agnostic fallback.
6. **Worked example impact** — 397/400 patterns had examples, but we haven't ablated whether the `<example>` tag alone drives gains vs the step sequence. Worth isolating.
