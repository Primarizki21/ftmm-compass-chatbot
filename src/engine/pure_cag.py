"""
Implementasi Engine Pure CAG (Cache-Augmented Generation) Murni.
Menerapkan prinsip Chan et al. (ACM 2025):
Seluruh basis pengetahuan (SOP, regulasi akademik, profil prodi) dimuat ke context window LLM.
Proses inferensi tanpa retrieval step sama sekali (0 ms retrieval latency).
"""

import time
from typing import Any
import ollama
from src.knowledge.cag_store import get_cag_system_prompt
from src.engine.base import EngineResult, DEFAULT_NUM_CTX, get_ollama_client

DEFAULT_MODEL = "qwen2.5:3b"


class PureCAGEngine:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.system_prompt = get_cag_system_prompt(
            custom_instructions="Jawablah pertanyaan mahasiswa secara langsung dan lugas berbasis pengetahuan resmi yang telah dihafal di atas. Jangan berhalusinasi. Perhatikan secara teliti perbedaan antara batas SKS semester (Pasal 5.4) dan syarat kumulatif kelulusan/kegiatan (contoh: PKL minimal 80 SKS kumulatif, Proposal Skripsi minimal 108 SKS)."
        )

    def generate_answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> EngineResult:
        t0 = time.perf_counter()
        
        # 1. Retrieval Step: Zero retrieval (CAG paradigm)
        retrieval_latency_ms = 0.0
        chunks: list[str] = []

        # 2. Prompt Construction
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": query})

        # 3. Generation Step
        t_gen_start = time.perf_counter()
        resp = get_ollama_client().chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.2, "top_p": 0.9, "num_ctx": DEFAULT_NUM_CTX}
        )
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = resp.message.content.strip()
        tokens_eval = resp.get("eval_count", 0)

        return EngineResult(
            answer=answer,
            architecture="Pure CAG",
            retrieved_chunks=chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=tokens_eval,
            routing_mode="CAG",
            metadata={"cache_type": "pre_warmed_system_prefix"}
        )
