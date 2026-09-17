#!/usr/bin/env bash
# Start the teacher or student behind an OpenAI-compatible vLLM endpoint.
#
#   ./scripts/serve_vllm.sh teacher     Qwen3-8B-AWQ on :8000
#   ./scripts/serve_vllm.sh student     Qwen3-1.7B   on :8001
#   ./scripts/serve_vllm.sh check       driver / GPU / backend preflight
#   ./scripts/serve_vllm.sh stop        stop whatever this script started
#
# TWO BACKENDS. Native is the default; docker is kept because the original
# spec asked for it, but it is NOT the default any more:
#
#   VLLM_BACKEND=native   (default) the pip-installed vllm in ms-atique-vllm.
#                         Chosen because pulling the 21GB vllm/vllm-openai
#                         image over this box's ~3MB/s WiFi does not finish in
#                         a useful amount of time.
#   VLLM_BACKEND=docker   the container image. Carries its own CUDA runtime, so
#                         only the NVIDIA driver must be new enough -- on
#                         Blackwell (RTX 50-series, sm_120) that means >= 570.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${VLLM_BACKEND:-native}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.9.2}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# The interpreter MUST be named by absolute path. Three different pythons
# resolve on this box depending on how you connect (interactive tmux -> base
# miniforge, ssh 'cmd' -> /usr/bin/python3), and only this env has vllm.
VLLM_BIN="${VLLM_BIN:-$HOME/miniforge3/envs/ms-atique-vllm/bin/vllm}"

# This box is shared, so the instinct is to hold as little as possible -- 0.62
# was the original setting, chosen to leave ~6GB visibly free. It does not
# work: at 0.62 the engine refuses to start, because after ~5.6GB of AWQ
# weights and ~3.6GB of activations + CUDA-graph capture there is only 0.85GB
# left for the KV cache, and ONE sequence at max_model_len 16384 needs 1.12GB.
# 0.80 is the compromise actually in use: ~3.3GB still visibly free, and a KV
# cache big enough for several concurrent full-length sequences.
GPU_UTIL="${GPU_UTIL:-0.80}"
# Caps CUDA-graph capture sizes, which is where most of that 3.6GB goes. The
# pipeline never exceeds teacher/student `concurrency` (8) in flight, so
# capturing graphs for batches up to 512 buys nothing and costs GB.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# Must fit prompt + reasoning trace + answer. Thinking mode makes the output
# side long, so 8192 total is not enough once memories are in the prompt.
MAX_LEN="${MAX_LEN:-16384}"
# See the --attention-backend note below before changing this.
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"

# NOTE ON --reasoning-parser: deliberately NOT set. With a reasoning parser
# vLLM splits the trace out into message.reasoning_content and leaves
# message.content without it. rot/llm.py measures thinking tokens by finding
# <think>...</think> INSIDE the content, so enabling a parser would silently
# zero out every thinking-token figure -- which is the experiment's main
# measurement. Leave it off so the raw tags come through.
#
# NOTE ON --disable-log-requests: removed in vLLM 0.29 (it is now
# --enable-log-requests, off by default). The old flag aborts startup.
#
# NOTE ON --attention-backend: pinned to TRITON_ATTN, and this is NOT optional
# on this card. vLLM 0.29 auto-selects FLASHINFER on sm120 (it prefers it out
# of ['FLASHINFER','TRITON_ATTN']), and FlashInfer then JIT-compiles its
# kernels with nvcc. The nvcc it finds is the system CUDA 12.8, and FlashInfer
# needs >= 12.9 for SM 12.x -- so it drops sm120 from its target list and dies
# with "FlashInfer requires GPUs with sm75 or higher". That message is
# misleading: the card is sm120, the TOOLKIT is too old. TRITON_ATTN needs no
# nvcc at all. Same reason for VLLM_USE_FLASHINFER_SAMPLER=0 below -- the top-p
# /top-k sampler is a second, independent path into the same JIT.

case "${1:-}" in
  teacher)
    ROLE=teacher
    MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B-AWQ}"; PORT=8000; DEVICE="${GPU:-0}" ;;
  student)
    ROLE=student
    MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}";   PORT=8001; DEVICE="${GPU:-0}" ;;
  check)
    echo "== GPU / driver =="
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
      --format=csv || echo "nvidia-smi unavailable"
    echo
    echo "== backend: $BACKEND =="
    if [ "$BACKEND" = native ]; then
      if [ -x "$VLLM_BIN" ]; then
        echo "vllm binary: $VLLM_BIN"
        "$VLLM_BIN" --version 2>&1 | tail -1
        "$(dirname "$VLLM_BIN")/python" -c \
          'import torch;print("torch",torch.__version__,"cap",torch.cuda.get_device_capability(0))'
      else
        echo "NOT FOUND: $VLLM_BIN -- set VLLM_BIN or install vllm"
      fi
    else
      groups | tr ' ' '\n' | grep -qx docker && echo "docker group: yes" || \
        echo "docker group: NO -- you likely cannot run containers"
      docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi -L || \
        echo "container GPU access FAILED -- check nvidia-container-toolkit"
    fi
    echo
    echo "== ports =="
    for p in 8000 8001; do
      if curl -sf -m 2 "http://localhost:$p/v1/models" >/dev/null 2>&1; then
        echo "  :$p SERVING $(curl -s -m 2 http://localhost:$p/v1/models \
          | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)"
      else
        echo "  :$p free"
      fi
    done
    exit 0 ;;
  stop)
    # Kill by exact PID from the pidfile. NEVER pkill -f here: the pattern
    # matches the ssh command line carrying it and kills your own session.
    for f in "$HERE"/logs/vllm_*.pid; do
      [ -e "$f" ] || continue
      pid="$(cat "$f")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "stopping $(basename "$f" .pid) pid=$pid"
        kill "$pid"
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && { echo "  still up, SIGKILL"; kill -9 "$pid"; }
      else
        echo "$(basename "$f" .pid): pid $pid not running"
      fi
      rm -f "$f"
    done
    echo "GPU now:"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    exit 0 ;;
  *)
    echo "usage: $0 {teacher|student|check|stop}" >&2; exit 1 ;;
esac

mkdir -p "$HERE/logs"
echo "Starting $MODEL on :$PORT (backend $BACKEND, gpu $DEVICE, util $GPU_UTIL, max_len $MAX_LEN, max_seqs $MAX_NUM_SEQS)"

if [ "$BACKEND" = docker ]; then
  exec docker run --rm \
    --gpus "\"device=${DEVICE}\"" \
    --ipc=host \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -p "${PORT}:8000" \
    --name "rot-${ROLE}" \
    "${IMAGE}" \
    --model "${MODEL}" \
    --served-model-name "${MODEL}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_LEN}" \
    --kv-cache-dtype fp8
fi

echo $$ > "$HERE/logs/vllm_${ROLE}.pid"
export CUDA_VISIBLE_DEVICES="${DEVICE}"
export HF_HOME="${HF_CACHE}"
export VLLM_USE_FLASHINFER_SAMPLER=0
exec "$VLLM_BIN" serve "${MODEL}" \
  --served-model-name "${MODEL}" \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --attention-backend "${ATTN_BACKEND}" \
  --kv-cache-dtype fp8
