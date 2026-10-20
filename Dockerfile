# Stage 1: Dependency builder with Astral UV
FROM python:3.12-slim-bookworm AS builder

# Copy UV binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation and frozen sync
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Install production dependencies with build cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


# Stage 2: Minimal runtime image
FROM python:3.12-slim-bookworm AS runtime

# Install OpenMP runtime required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Ensure python from venv is first in PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    OLLAMA_NUM_CTX=2048 \
    OLLAMA_HOST="http://host.docker.internal:11434" \
    SESSION_DB_PATH="/app/data/sessions/sessions.db" \
    SESSION_STORE_BACKEND="sqlite"

# Create writable data directory for centroids, runtime caches, and sessions
RUN mkdir -p /app/data/faiss /app/data/sessions && chown -R appuser:appgroup /app/data

# Copy application source code
COPY --chown=appuser:appgroup src/ /app/src/

USER appuser

EXPOSE 8000

# Native Python healthcheck without extra curl dependency
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--no-access-log"]
