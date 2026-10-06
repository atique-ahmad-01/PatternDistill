#!/usr/bin/env bash
# Exact RoT on generic step templates.
#
#   f_exact_generic   f_exact with rot.step_text=generic: every RoT step node is
#                     the teacher's generic rewrite of the original step, so the
#                     template stops carrying the donor problem's values.
#
# Everything else is f_exact (Qwen3-1.7B, TI prefill, reference prompt, 15360
# cap), and its A_baseline is copied, not re-run, so the comparison point stays
# fixed. Teacher and student cannot share the card, so they run one after the
# other. Run inside tmux on the GPU box. Resumable: re-run after a kill.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/miniforge3/envs/ms-atique/bin/python"
CFG="config.f_exact_generic.yaml"
R="f_exact_generic"

"$PY" main.py verify --config "$CFG"

# 1. teacher rewrites the 651 memory entries
if ! curl -sf -m 2 http://localhost:8000/v1/models | grep -q 'Qwen3-8B-AWQ'; then
  ./scripts/serve_vllm.sh teacher > logs/vllm_teacher_generic.log 2>&1 &
  until curl -sf -m 2 http://localhost:8000/v1/models >/dev/null; do sleep 5; done
fi
"$PY" main.py generalize --config "$CFG"
./scripts/serve_vllm.sh stop

# 2. RoT graph from the generic texts (separate files; f_exact's graph is untouched)
"$PY" main.py build-rot-graph --config "$CFG"

# 3. student
if ! curl -sf -m 2 http://localhost:8001/v1/models | grep -q 'Qwen3-1.7B'; then
  ./scripts/serve_vllm.sh student > logs/vllm_student_generic.log 2>&1 &
  until curl -sf -m 2 http://localhost:8001/v1/models >/dev/null; do sleep 5; done
fi
mkdir -p "results/$R"
[ -s "results/$R/eval_A_baseline.jsonl" ] \
  || cp results/f_exact/eval_A_baseline.jsonl "results/$R/"
"$PY" main.py evaluate --config "$CFG" --arms F_rot
"$PY" main.py report   --config "$CFG"

sed -n '1,12p' "results/$R/summary.md"
./scripts/serve_vllm.sh stop
echo GENERIC_DONE
