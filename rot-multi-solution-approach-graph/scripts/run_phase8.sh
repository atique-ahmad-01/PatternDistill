#!/usr/bin/env bash
# Phase 8: make F_rot cheaper without losing accuracy.
#
#   p8_clean        f_userprompt + single-source templates + constant abstraction
#   p8_clean_trust  p8_clean + the "verified method, apply it directly" preamble
#
# Both are f_userprompt (Qwen3-1.7B, user-turn injection, 8192 cap) with only
# the rot.* keys above changed, so the graph, the walk's starting node and the
# A_baseline are the same ones f_userprompt was scored against. The baseline is
# copied, not re-run, for exactly that reason: the comparison point stays fixed.
# Run inside tmux on the GPU box. Resumable: re-run after a kill.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/miniforge3/envs/ms-atique/bin/python"
RUNS="p8_clean p8_clean_trust"

for r in $RUNS; do "$PY" main.py verify --config "config.$r.yaml"; done

if ! curl -sf -m 2 http://localhost:8001/v1/models | grep -q 'Qwen3-1.7B'; then
  ./scripts/serve_vllm.sh student > logs/vllm_student_p8.log 2>&1 &
  until curl -sf -m 2 http://localhost:8001/v1/models >/dev/null; do sleep 5; done
fi

for r in $RUNS; do
  mkdir -p "results/$r"
  [ -s "results/$r/eval_A_baseline.jsonl" ] \
    || cp results/f_userprompt/eval_A_baseline.jsonl "results/$r/"
  "$PY" main.py evaluate --config "config.$r.yaml" --arms F_rot
  "$PY" main.py report   --config "config.$r.yaml"
done

for r in $RUNS; do cat "results/$r/summary.md" | sed -n '1,12p'; done
./scripts/serve_vllm.sh stop
echo PHASE8_DONE
