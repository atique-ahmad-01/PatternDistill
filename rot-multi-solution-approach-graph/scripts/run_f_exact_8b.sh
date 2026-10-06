#!/usr/bin/env bash
# f_exact with the TEACHER (Qwen3-8B-AWQ) as the student.
# Identical to config.f_exact.yaml except student.model and run_name, so the
# memory bank, RoT graph and all 48 retrieved templates are the same ones the
# 1.7B saw. Run inside tmux on the GPU box. Resumable: re-run after a kill.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/miniforge3/envs/ms-atique/bin/python"
CFG="config.f_exact_8b.yaml"

"$PY" main.py verify --config "$CFG"

if ! curl -sf -m 2 http://localhost:8001/v1/models | grep -q 'Qwen3-8B-AWQ'; then
  STUDENT_MODEL=Qwen/Qwen3-8B-AWQ ./scripts/serve_vllm.sh student > logs/vllm_student_8b.log 2>&1 &
  until curl -sf -m 2 http://localhost:8001/v1/models >/dev/null; do sleep 5; done
fi

# Fresh baseline at THIS config (8B, 15360) -- not copied from another run.
"$PY" main.py evaluate --config "$CFG" --arms A_baseline
"$PY" main.py evaluate --config "$CFG" --arms F_rot
"$PY" main.py report   --config "$CFG"
cat results/f_exact_8b/summary.md
./scripts/serve_vllm.sh stop
echo F_EXACT_8B_DONE
