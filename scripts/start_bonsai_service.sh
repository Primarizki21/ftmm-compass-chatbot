#!/usr/bin/env bash
set -euo pipefail

# Configuration
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
MODEL_PATH="${MODEL_PATH:-/home/primarizki/project/Bonsai-demo/models/bonsai2-gguf/4B/Ternary-Bonsai-2-4B-PQ2_0.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/home/primarizki/project/Bonsai-demo/bin/cuda/llama-server}"
CTX_SIZE="${CTX_SIZE:-2048}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

# CUDA runtime libraries
CUDA_LIB_DIR="/home/primarizki/.local/lib/ollama/cuda_v12"
BONSAI_BIN_DIR="$(dirname "$LLAMA_SERVER_BIN")"
export LD_LIBRARY_PATH="${CUDA_LIB_DIR}:${BONSAI_BIN_DIR}:${LD_LIBRARY_PATH:-}"

echo "========================================================"
echo "🚀 Starting Bonsai-2-4B llama-server (Ternary Production)"
echo "Host: ${HOST}:${PORT}"
echo "Model: ${MODEL_PATH}"
echo "Context Size: ${CTX_SIZE} | GPU Layers: ${N_GPU_LAYERS}"
echo "========================================================"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    echo "❌ Error: llama-server binary not executable at ${LLAMA_SERVER_BIN}" >&2
    exit 1
fi

exec "$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CTX_SIZE" \
    -ngl "$N_GPU_LAYERS"
