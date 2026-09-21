# RoT arm documentation

Three arms, three self-contained directories. Each one holds its own
`TECHNIQUE.md` (the method alone), its own `README.md` (the full A-to-Z), its own
config, its own runnable script, its own example prompt and its own results.
**Nothing is shared between them** — open one directory and it tells you the
whole story for that arm, start to finish.

Want the method and nothing else? Read `<arm>/TECHNIQUE.md`:
[`f_exact`](f_exact/TECHNIQUE.md) ·
[`f_userprompt`](f_userprompt/TECHNIQUE.md) ·
[`f_up_exact`](f_up_exact/TECHNIQUE.md).

| directory | what it is | accuracy | vs baseline |
|---|---|---:|---|
| [`f_exact/`](f_exact/) | strict reproduction of arXiv:2509.21743 — template prefilled into an open `<think>` block | **16.7%** | −58.3 pp, p < 0.001 |
| [`f_userprompt/`](f_userprompt/) | same templates delivered in the user turn, project prompts — the only arm comparable with A–E | **79.2%** | +4.2 pp, p = 0.75 |
| [`f_up_exact/`](f_up_exact/) | same templates in the user turn, at the reference's prompt and token budget | **81.2%** | +6.2 pp, p = 0.51 |

Baseline: Qwen3-1.7B unaided, 36/48 = 75.0%, on the frozen 48-problem RB-DERIV
test split.

## The one thing to know before reading any of them

The retrieval half is **bit-identical** across all three: same seed, same graph,
same gold tags, same walks. All 48 assembled templates and all 48 node-id paths
match exactly across the three runs. The arms differ **only in how the template
is delivered to the student.**

So the accuracy spread from 16.7% to 81.2% is not about retrieval quality at
all. The cleanest statement of it is the `f_exact` → `f_up_exact` contrast,
where the injection site is the *only* setting that differs: **31 problems
gained, 0 lost, p ≈ 10⁻⁹.**

The +4.2 and +6.2 pp gains over the baseline are, by contrast, not
statistically distinguishable at n = 48 — and `f_up_exact`'s baseline is
budget-mismatched. Each README says so in its own §5.

## Controls (documented inside the arm READMEs, not here)

| control run | what it holds fixed | result |
|---|---|---|
| `results/f_shuf/` | `f_userprompt` with a *different* problem's template | 68.8% |
| `results/f_shuf_exact/` | `f_up_exact` with a *different* problem's template | 75.0% |
| `results/f_predicted/` | `f_exact` without the gold family tag | 0.0% |

## Source of truth

These directories document runs that already exist. The live artifacts are:

* configs — `config.f_exact.yaml`, `config.f_userprompt.yaml`, `config.f_up_exact.yaml`
* results — `results/<run>/`
* code — `rot/rot_graph.py`, `rot/rot_retrieval.py`, `rot/student.py`
* project-wide write-ups — `../../F_ROT.md`, `../../README.md`
