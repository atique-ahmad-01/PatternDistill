#!/usr/bin/env bash
# =============================================================================
# f_up_exact -- User-turn injection + the reference system prompt and token budget
#
# Recorded result: 39/48 = 81.2%  (baseline 36/48 = 75.0%)
#
# A-to-Z. Run it from the project root on the GPU box:
#     tmux new -s rot
#     cd ~/rot-multi-solution-approach-graph
#     bash docs/arms/f_up_exact/run.sh
#
# ALWAYS inside tmux. The box is shared and an SSH drop kills a foreground run.
# Every stage is resumable, so re-running this script after a kill continues
# from where it stopped rather than starting over.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# The interpreter MUST be named by absolute path. Three different pythons
# resolve on this box depending on how you connect, and only this env has the
# pipeline's dependencies. `conda activate` is not reliable here.
PY="${PY:-$HOME/miniforge3/envs/ms-atique/bin/python}"
CFG="config.f_up_exact.yaml"

# ---------------------------------------------------------------------------
# 0. Preflight
# ---------------------------------------------------------------------------
"$PY" - <<'PREFLIGHT'
import sys, transformers
v = transformers.__version__
if int(v.split(".")[0]) >= 5:
    sys.exit(f"transformers {v} is too new: the jina embedder ships modelling "
             "code written against 4.x. Fix the env, do not shim it further:\n"
             "    pip install 'transformers>=4.40,<5'")
print("transformers", v, "OK")
PREFLIGHT
./scripts/serve_vllm.sh check

# ---------------------------------------------------------------------------
# 1. Dataset + frozen split          192 questions -> 144 train / 48 test
#    Verifies the dataset SHA against config and freezes the split to
#    data/artifacts/split.json. Never pass --refresh-split on an existing run:
#    it would move the train/test boundary under already-generated data.
# ---------------------------------------------------------------------------
"$PY" main.py verify --config "$CFG"

# ---------------------------------------------------------------------------
# 2. Teacher generation              Qwen3-8B-AWQ on :8000, 5 solutions x 144
#    ~40-70 min. Skip stages 2-5 entirely if data/artifacts/ is already built;
#    they are shared by every arm and are NOT what distinguishes this one.
# ---------------------------------------------------------------------------
if [ ! -s data/artifacts/teacher_solutions.jsonl ]; then
  ./scripts/serve_vllm.sh teacher &
  until curl -sf -m 2 http://localhost:8000/v1/models >/dev/null; do sleep 5; done
  "$PY" main.py generate --config "$CFG"
  ./scripts/serve_vllm.sh stop
fi

# ---------------------------------------------------------------------------
# 3. Validate   exact-integer match against gold  -> 651/720 correct
# 4. Memory bank  correct_only, >=2 steps         -> 651 entries, 2450 steps
# 5. RoT graph    step nodes + sequential + semantic edges (tau=0.85 raw cos)
#                                                 -> 2450 nodes, 598636 sem
# ---------------------------------------------------------------------------
"$PY" main.py validate        --config "$CFG"
"$PY" main.py build-memory    --config "$CFG"
"$PY" main.py build-rot-graph --config "$CFG"

# ---------------------------------------------------------------------------
# 6. Student evaluation             Qwen3-1.7B on :8001
#    This arm:  injection=user_prompt  tag_source=gold
#               system_prompt=reference  template_source=retrieved
#               student.max_tokens=15360
# ---------------------------------------------------------------------------
./scripts/serve_vllm.sh student &
until curl -sf -m 2 http://localhost:8001/v1/models >/dev/null; do sleep 5; done

# The baseline this arm is scored against. It is NOT re-run when
# results/f_up_exact/eval_A_baseline.jsonl already exists -- see the README's
# "Baseline caveat" section before reading the delta.
"$PY" main.py evaluate --config "$CFG" --arms A_baseline
"$PY" main.py evaluate --config "$CFG" --arms F_rot

# ---------------------------------------------------------------------------
# 7. Report  -> results/f_up_exact/{summary.md,summary.csv,by_group.csv,
#                              per_problem.csv,summary.json}
# ---------------------------------------------------------------------------
"$PY" main.py report --config "$CFG"
cat "results/f_up_exact/summary.md"
