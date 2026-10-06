#!/usr/bin/env bash
# T3 RAG over thinking traces (arXiv:2605.03344): all three T3 forms.
#
#   1. teacher (Qwen3-8B, thinking off) rewrites the 651 memory-bank traces
#      into struct, semantic and reflect forms
#   2. student (Qwen3-1.7B) answers the 48 test problems with the top-3
#      retrieved examples in the prompt -- one run per form, results/t3_<form>/
#
# A_baseline is copied from the RoT project's f_exact, not re-run. Teacher and
# student cannot share the card, so they run one after the other. Run inside
# tmux on the GPU box. Resumable: re-run after a kill.
#
#   ./scripts/run_t3.sh                   struct, semantic, reflect
#   VARIANTS="struct" ./scripts/run_t3.sh just one form
#   ARMS=G_t3,R_random_t3 ./scripts/run_t3.sh   add the random-example control
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/miniforge3/envs/ms-atique/bin/python"
VARIANTS="${VARIANTS:-struct semantic reflect}"
ARMS="${ARMS:-G_t3}"
mkdir -p logs

"$PY" main.py verify

# 1. teacher
if ! curl -sf -m 2 http://localhost:8000/v1/models | grep -q 'Qwen3-8B-AWQ'; then
  ./scripts/serve_vllm.sh teacher > logs/vllm_teacher.log 2>&1 &
  until curl -sf -m 2 http://localhost:8000/v1/models >/dev/null; do sleep 5; done
fi
for v in $VARIANTS; do "$PY" main.py t3 --variant "$v"; done
./scripts/serve_vllm.sh stop

# 2. student
if ! curl -sf -m 2 http://localhost:8001/v1/models | grep -q 'Qwen3-1.7B'; then
  ./scripts/serve_vllm.sh student > logs/vllm_student.log 2>&1 &
  until curl -sf -m 2 http://localhost:8001/v1/models >/dev/null; do sleep 5; done
fi
for v in $VARIANTS; do
  "$PY" main.py evaluate --variant "$v" --arms "A_baseline,$ARMS"
  "$PY" main.py report   --variant "$v" --arms "A_baseline,$ARMS"
done

for v in $VARIANTS; do sed -n '1,12p' "results/t3_$v/summary.md"; done
./scripts/serve_vllm.sh stop
echo T3_DONE
