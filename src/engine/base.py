"""
Definisi tipe data dan kontrak dasar engine generasi FTMM COMPASS.
"""

from dataclasses import dataclass, field
from typing import Any


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
