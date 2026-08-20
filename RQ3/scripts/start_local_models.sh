#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
VLLM_BIN="${VLLM_BIN:-vllm}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_HAS_FLASHINFER_CUBIN="${VLLM_HAS_FLASHINFER_CUBIN:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"

case "$MODE" in
  gemma)
    : "${GEMMA_MODEL:?set GEMMA_MODEL to the Gemma-3-12B-IT model directory}"
    exec "$VLLM_BIN" serve "$GEMMA_MODEL" \
      --host 127.0.0.1 --port "${GEMMA_PORT:-8001}" \
      --gpu-memory-utilization "${GEMMA_GPU_UTIL:-0.85}" \
      --max-model-len "${GEMMA_MAX_MODEL_LEN:-12288}"
    ;;
  embedding)
    : "${EMBED_MODEL:?set EMBED_MODEL to the Qwen3-Embedding-8B model directory}"
    exec "$VLLM_BIN" serve "$EMBED_MODEL" \
      --host 127.0.0.1 --port "${EMBED_PORT:-8002}" \
      --gpu-memory-utilization "${EMBED_GPU_UTIL:-0.55}" \
      --max-model-len "${EMBED_MAX_MODEL_LEN:-8192}"
    ;;
  *)
    echo "usage: $0 {gemma|embedding}" >&2
    exit 2
    ;;
esac
