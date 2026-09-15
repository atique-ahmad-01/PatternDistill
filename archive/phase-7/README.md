# Phase 7 — RB-DERIV

Runner for the RB-DERIV reuse benchmark, with the benchmark data in
[`RB_DERIV/`](RB_DERIV/). The prompt specs still live in
[`../sep-paper/`](../sep-paper/), so two paths reach across; both resolve in
[`config.py`](config.py) and nothing else should hop `..`.

Status: scaffold. No stage implemented yet, no run has happened.

## The question

Does handing a small model a *distilled technique object* let it solve problems
whose difficulty is in the derivation rather than the execution — and does the
structure of that object (USE_WHEN / CHECK) matter beyond its content?

## Planned stages

| # | Stage | Reads | Writes |
|---|---|---|---|
| 1 | `teacher_solve.py` | 144 train questions | 720 samples, gold-verified |
| 2 | `distill.py` | verified solutions, per family | 16 technique objects |
| 3 | `gates.py` | objects | gated objects + reject log |
| 4 | `student_eval.py` | 48 test questions × 6 arms | 288 completions |
| 5 | `scoring.py` / `analyze.py` | completions | accuracy, decode tokens, rescue rate |

Stage 1 is gated: a family with fewer than `MIN_VERIFIED_PER_FAMILY` verified
solutions is **teacher-limited** and must not be distilled — the object would
encode the teacher's error and the student would inherit it.

## Open decisions (settle before running)

1. **`schema_oracle` has no template.** It is the quality ceiling in the
   architecture figure — how much of the human-authored gap distillation closed
   — but `solver-prompt.yml` defines only five arms. Add it or cut it.
2. **False-friend denominator.** The spec says 175/192; that is the full set. A
   single 9/3 split leaves 46/48. Pick the scored population.
3. **n=48 is underpowered** and collides with the per-family reporting that the
   low-diversity families require. `SPLIT_ROTATIONS = 4` fixes both at 4×
   student decode.
4. **`check_decisive` gate is not mechanizable as written** — it asks to apply
   METHOD and a corruption to CHECK's case, but the teacher emits prose. With
   16 objects, manual review is tractable.
5. **Teacher-limited policy.** If a family drops out of `schema` but survives in
   `base` and `schema_oracle`, the arms no longer cover the same questions.
   Decide now whether results report the intersection, the full set, or both.

## Running

Author locally, then (per the repo [`Readme.md`](../Readme.md)):

```bash
rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
  ~/Desktop/PatternDistill/ students@<gpu-host>:~/PatternDistill/
```

On the box: `tmux new -s phase7`, `conda activate ms-atique`. The box is shared
and jobs get killed to free VRAM — every stage must be resumable.
