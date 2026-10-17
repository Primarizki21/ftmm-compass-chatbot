"""
Implementasi Engine Pure RAG (Retrieval-Augmented Generation) Murni.
Selalu menjalankan retrieval FAISS di setiap turn dialog dan menyertakan chunk hasil pencarian ke prompt LLM.
"""

import os
import time
from typing import Any
from src.knowledge.retriever import get_course_retriever
from src.engine.base import EngineResult, DEFAULT_NUM_CTX, get_llm_client, LLMClientAdapter

DEFAULT_MODEL = os.getenv("LLM_MODEL_NAME", "bonsai-2-4b")
PURE_RAG_SYSTEM_PROMPT = """Anda adalah Compass AI, asisten akademik FTMM Universitas Airlangga.
Jawab pertanyaan mahasiswa secara akurat dan informatif berdasarkan informasi konteks mata kuliah yang diberikan di bawah ini.
Jika informasi tidak ada dalam konteks, katakan dengan sopan bahwa Anda tidak memiliki informasi spesifik tersebut.

[KONTEKS MATA KULIAH RELEVAN]:
{context}
"""


class PureRAGEngine:
    def __init__(self, model: str = DEFAULT_MODEL, top_k: int = 3, llm_client: LLMClientAdapter | None = None):
        self.model = model
        self.top_k = top_k
        self.retriever = get_course_retriever()
        self.llm_client = llm_client or get_llm_client()

    def generate_answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        prodi_filter: str | None = None
    ) -> EngineResult:
        t0 = time.perf_counter()
        
        # 1. Retrieval Step
        t_ret_start = time.perf_counter()
        retrieval_results = self.retriever.retrieve(query, top_k=self.top_k, prodi_filter=prodi_filter)
        retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000.0
        
        chunks = [r["chunk"] for r in retrieval_results]
        context_str = "\n\n---\n\n".join(chunks) if chunks else "Tidak ada dokumen relevan yang ditemukan."

        # 2. Prompt Construction
        system_content = PURE_RAG_SYSTEM_PROMPT.format(context=context_str)
        messages = [{"role": "system", "content": system_content}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": query})

        # 3. Generation Step
        t_gen_start = time.perf_counter()
        resp = self.llm_client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.2, "top_p": 0.9, "num_predict": 256, "num_ctx": DEFAULT_NUM_CTX}
        )
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        answer = resp.message.content.strip()
        tokens_eval = resp.get("eval_count", 0)

        return EngineResult(
            answer=answer,
            architecture="Pure RAG",
            retrieved_chunks=chunks,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=tokens_eval,
            routing_mode="RAG",
            metadata={"top_k": self.top_k, "num_retrieved": len(chunks)}
        )
