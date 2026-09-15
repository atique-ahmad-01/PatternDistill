#!/usr/bin/env bash
# Start the teacher or student behind an OpenAI-compatible vLLM endpoint.
#
#   ./scripts/serve_vllm.sh teacher     Qwen3-8B   on :8000
#   ./scripts/serve_vllm.sh student     Qwen3-1.7B on :8001
#   ./scripts/serve_vllm.sh check       driver / docker / GPU preflight
#
# Docker carries its own CUDA runtime, so the host's system CUDA is never
# touched -- only the NVIDIA driver has to be new enough. On Blackwell
# (RTX 50-series, sm_120) that means driver >= 570.
set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.9.2}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Deliberately NOT the 0.9 default. This box is shared: holding ~15GB of 16.6GB
# is what gets a job killed by someone else hunting for VRAM. 0.62 leaves ~6GB
# visibly free and still gives a workable KV cache at max-model-len 8192.
GPU_UTIL="${GPU_UTIL:-0.62}"
# Must fit prompt + reasoning trace + answer. Thinking mode makes the output
# side long, so 8192 total is not enough once memories are in the prompt.
MAX_LEN="${MAX_LEN:-16384}"

case "${1:-}" in
  teacher)
    MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B-AWQ}"; PORT=8000; DEVICE="${GPU:-0}" ;;
  student)
    MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}";   PORT=8001; DEVICE="${GPU:-0}" ;;
  check)
    echo "== GPU / driver =="
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv || \
      echo "nvidia-smi unavailable"
    echo
    echo "== docker access =="
    groups | tr ' ' '\n' | grep -qx docker && echo "docker group: yes" || \
      echo "docker group: NO -- you likely cannot run containers; ask about rootless podman"
    echo
    echo "== container can see the GPU =="
    docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L || \
      echo "container GPU access FAILED -- check nvidia-container-toolkit"
    exit 0 ;;
  *)
    echo "usage: $0 {teacher|student|check}" >&2; exit 1 ;;
esac

echo "Starting $MODEL on :$PORT (gpu $DEVICE, util $GPU_UTIL, max_len $MAX_LEN)"

exec docker run --rm \
  --gpus "\"device=${DEVICE}\"" \
  --ipc=host \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -p "${PORT}:8000" \
  --name "rot-${1}" \
  "${IMAGE}" \
  --model "${MODEL}" \
  --served-model-name "${MODEL}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --kv-cache-dtype fp8 \
  --disable-log-requests
