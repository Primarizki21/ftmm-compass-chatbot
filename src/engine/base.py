"""
Definisi tipe data dan kontrak dasar engine generasi FTMM COMPASS.
"""

import os
from dataclasses import dataclass, field
from typing import Any
import ollama

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
_OLLAMA_CLIENT_POOL: dict[str, ollama.Client] = {}


def get_ollama_client() -> ollama.Client:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    normalized_host = host.rstrip("/")
    client = _OLLAMA_CLIENT_POOL.get(normalized_host)
    if client is None:
        client = ollama.Client(host=normalized_host)
        _OLLAMA_CLIENT_POOL[normalized_host] = client
    return client


@dataclass
class EngineResult:
    answer: str
    architecture: str                     # "Pure RAG" | "Pure CAG" | "Hybrid RAG-CAG"
    retrieved_chunks: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    routing_mode: str = "DEFAULT"
    metadata: dict[str, Any] = field(default_factory=dict)
