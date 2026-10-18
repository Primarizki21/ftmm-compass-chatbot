#!/usr/bin/env bash
set -euo pipefail

# Load .env jika tersedia di root repositori
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${REPO_ROOT}/.env"
    set +a
fi

# Konfigurasi dasar port dan host
PORT="${BONSAI_PORT:-${PORT_BONSAI:-8080}}"
HOST="${HOST:-0.0.0.0}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

# 1. Auto-discovery Model GGUF
RESOLVED_MODEL_PATH="${MODEL_PATH:-}"
if [ -z "$RESOLVED_MODEL_PATH" ] || [ ! -f "$RESOLVED_MODEL_PATH" ]; then
    MODEL_CANDIDATES=(
        "${MODEL_PATH:-}"
        "${REPO_ROOT}/models/Ternary-Bonsai-2-4B-PQ2_0.gguf"
        "${REPO_ROOT}/../Bonsai-demo/models/bonsai2-gguf/4B/Ternary-Bonsai-2-4B-PQ2_0.gguf"
        "/home/primarizki/project/Bonsai-demo/models/bonsai2-gguf/4B/Ternary-Bonsai-2-4B-PQ2_0.gguf"
    )
    for cand in "${MODEL_CANDIDATES[@]}"; do
        if [ -n "$cand" ] && [ -f "$cand" ]; then
            RESOLVED_MODEL_PATH="$cand"
            break
        fi
    done
fi

# 2. Auto-discovery Binary llama-server
RESOLVED_BIN_PATH="${LLAMA_SERVER_BIN:-}"
if [ -z "$RESOLVED_BIN_PATH" ] || [ ! -x "$RESOLVED_BIN_PATH" ]; then
    BIN_CANDIDATES=(
        "${LLAMA_SERVER_BIN:-}"
        "${REPO_ROOT}/bin/llama-server"
        "${REPO_ROOT}/../Bonsai-demo/bin/cuda/llama-server"
        "/home/primarizki/project/Bonsai-demo/bin/cuda/llama-server"
        "$(command -v llama-server 2>/dev/null || true)"
    )
    for cand in "${BIN_CANDIDATES[@]}"; do
        if [ -n "$cand" ] && [ -x "$cand" ]; then
            RESOLVED_BIN_PATH="$cand"
            break
        fi
    done
fi

# 3. Auto-discovery CUDA Runtime Libraries
CUDA_DIRS=(
    "${CUDA_LIB_DIR:-}"
    "${HOME}/.local/lib/ollama/cuda_v12"
    "/usr/local/cuda/lib64"
    "/usr/lib/wsl/lib"
)
for cdir in "${CUDA_DIRS[@]}"; do
    if [ -n "$cdir" ] && [ -d "$cdir" ]; then
        export LD_LIBRARY_PATH="${cdir}:${LD_LIBRARY_PATH:-}"
    fi
done
if [ -n "$RESOLVED_BIN_PATH" ]; then
    BIN_DIR="$(dirname "$RESOLVED_BIN_PATH")"
    export LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "========================================================"
echo "🚀 Menjalankan Bonsai-2-4B llama-server (Ternary Engine)"
echo "Host: ${HOST}:${PORT}"
echo "Model: ${RESOLVED_MODEL_PATH}"
echo "Binary: ${RESOLVED_BIN_PATH}"
echo "Context Size: ${CTX_SIZE} | GPU Layers: ${N_GPU_LAYERS}"
echo "========================================================"

if [ -z "$RESOLVED_MODEL_PATH" ] || [ ! -f "$RESOLVED_MODEL_PATH" ]; then
    echo "❌ Error: Model file GGUF tidak ditemukan!" >&2
    echo "   Silakan letakkan model di ./models/Ternary-Bonsai-2-4B-PQ2_0.gguf" >&2
    echo "   atau tentukan path di .env: MODEL_PATH=/path/to/Ternary-Bonsai-2-4B-PQ2_0.gguf" >&2
    echo "   Contoh download via huggingface-cli:" >&2
    echo "   uv run huggingface-cli download prism-ml/Ternary-Bonsai-2-4B-gguf Ternary-Bonsai-2-4B-PQ2_0.gguf --local-dir ./models" >&2
    exit 1
fi

if [ -z "$RESOLVED_BIN_PATH" ] || [ ! -x "$RESOLVED_BIN_PATH" ]; then
    echo "❌ Error: Binary llama-server tidak ditemukan atau tidak executable!" >&2
    echo "   Silakan letakkan binary di ./bin/llama-server (chmod +x ./bin/llama-server)" >&2
    echo "   atau tentukan path di .env: LLAMA_SERVER_BIN=/path/to/llama-server" >&2
    exit 1
fi

MODEL_PATH="$RESOLVED_MODEL_PATH"
LLAMA_SERVER_BIN="$RESOLVED_BIN_PATH"
exec "$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    -c "$CTX_SIZE" \
    -ngl "$N_GPU_LAYERS" \
    -np 1 \
    --cache-ram 0 \
    --no-cache-prompt
