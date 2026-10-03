"""
Implementasi Engine Hybrid RAG-CAG (Retrieval-Augmented + Cache-Augmented Generation).
Menerapkan arsitektur CAIDA (Hamza & Ajam, ACM 2026) dan riset Mindmap V3:
1. Gerbang depan NLU Router (Intent Classifier & Entity Extractor).
2. Routing dinamis:
   - CAG Path: Pertanyaan SOP & Regulasi (0ms retrieval, fixed KV cache).
   - RAG Path: Pertanyaan Silabus & Prasyarat Matkul (FAISS dense similarity search).
   - Hybrid Path: Pertanyaan Study Planning (CAG aturan SKS + RAG kandidat matkul).
   - Guarded Path: Pertanyaan out-of-scope ditolak secara aman.
"""

import time
from typing import Any
import ollama
from src.knowledge.cag_store import get_cag_system_prompt
from src.knowledge.retriever import get_course_retriever
from src.engine.router import get_router, RoutingMode, RoutingDecision
from src.engine.base import EngineResult

DEFAULT_MODEL = "qwen2.5:3b"

GUARDRAIL_REFUSAL_MESSAGE = (
    "Maaf, saya adalah **Compass AI**, asisten penasihat akademik khusus Fakultas Teknologi Maju dan "
    "Multidisiplin (FTMM) Universitas Airlangga. Saya hanya dapat membantu pertanyaan seputar kurikulum, "
    "silabus mata kuliah, regulasi akademik (SKS, PKL, Skripsi, Yudisium), dan perencanaan studi di FTMM."
)


class HybridRAGCAGEngine:
    def __init__(self, model: str = DEFAULT_MODEL, top_k: int = 3):
        self.model = model
        self.top_k = top_k
        self.router = get_router()
        self.retriever = get_course_retriever()
        self.cag_base_prompt = get_cag_system_prompt()

    def generate_answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        student_profile: dict[str, Any] | None = None,
    ) -> EngineResult:
        t0 = time.perf_counter()
        
        # 1. NLU Routing Step
        t_route_start = time.perf_counter()
        decision: RoutingDecision = self.router.route(query)
        route_latency_ms = (time.perf_counter() - t_route_start) * 1000.0

        # Case A: GUARDED (Out-of-Scope)
        if decision.routing_mode == RoutingMode.GUARDED:
            total_latency_ms = (time.perf_counter() - t0) * 1000.0
            return EngineResult(
                answer=GUARDRAIL_REFUSAL_MESSAGE,
                architecture="Hybrid RAG-CAG",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_latency_ms,
                tokens_generated=len(GUARDRAIL_REFUSAL_MESSAGE.split()),
                routing_mode="GUARDED",
                metadata={"intent": decision.intent.value, "explanation": decision.explanation}
            )

        # Case B: CAG Path (Regulation & SOP FAQ) - 0ms retrieval
        elif decision.routing_mode == RoutingMode.CAG:
            retrieval_latency_ms = 0.0
            chunks = []
            
            system_prompt = (
                f"{self.cag_base_prompt}\n\n"
                "## PERAN SAAT INI (CAG ROUTE):\n"
                "Jawab pertanyaan mahasiswa mengenai regulasi akademik atau SOP administrasi FTMM secara tepat, "
                "padat, dan lugas sesuai basis pengetahuan di atas."
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-4:])
            messages.append({"role": "user", "content": query})

            t_gen_start = time.perf_counter()
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.2, "top_p": 0.9, "num_predict": 300}
            )
            generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
            total_latency_ms = (time.perf_counter() - t0) * 1000.0
            
            return EngineResult(
                answer=resp.message.content.strip(),
                architecture="Hybrid RAG-CAG",
                retrieved_chunks=[],
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_generated=resp.get("eval_count", 0),
                routing_mode="CAG",
                metadata={"confidence": decision.confidence, "explanation": decision.explanation}
            )

        # Case C: RAG Path (Course Syllabus & Specific Code)
        elif decision.routing_mode == RoutingMode.RAG:
            t_ret_start = time.perf_counter()
            prodi_filter = decision.entities.program_studi
            if student_profile and not prodi_filter:
                prodi_filter = student_profile.get("program_studi")

            retrieval_results = self.retriever.retrieve(query, top_k=self.top_k, prodi_filter=prodi_filter)
            retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000.0
            
            chunks = [r["chunk"] for r in retrieval_results]
            context_str = "\n\n---\n\n".join(chunks) if chunks else "Tidak ada silabus spesifik yang ditemukan."

            system_prompt = (
                f"{self.cag_base_prompt}\n\n"
                "## PERAN SAAT INI (RAG ROUTE):\n"
                "Gunakan potongan silabus mata kuliah resmi FTMM yang ditemukan di bawah ini untuk menjawab pertanyaan mahasiswa.\n\n"
                f"[POTONGAN SILABUS HASIL RETRIEVAL]:\n{context_str}"
            )

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-4:])
            messages.append({"role": "user", "content": query})

            t_gen_start = time.perf_counter()
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.2, "top_p": 0.9}
            )
            generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
            total_latency_ms = (time.perf_counter() - t0) * 1000.0

            return EngineResult(
                answer=resp.message.content.strip(),
                architecture="Hybrid RAG-CAG",
                retrieved_chunks=chunks,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_generated=resp.get("eval_count", 0),
                routing_mode="RAG",
                metadata={"confidence": decision.confidence, "explanation": decision.explanation}
            )

        # Case D: HYBRID Path (Study Planning / Recommendation)
        elif decision.routing_mode == RoutingMode.HYBRID:
            t_ret_start = time.perf_counter()
            prodi_filter = decision.entities.program_studi
            if student_profile and not prodi_filter:
                prodi_filter = student_profile.get("program_studi")

            # Cari mata kuliah relevan dengan topik karir / minat
            search_query = query
            if decision.entities.career_interests:
                search_query += " " + " ".join(decision.entities.career_interests)

            retrieval_results = self.retriever.retrieve(search_query, top_k=5, prodi_filter=prodi_filter)
            retrieval_latency_ms = (time.perf_counter() - t_ret_start) * 1000.0
            
            chunks = [r["chunk"] for r in retrieval_results]
            context_str = "\n\n---\n\n".join(chunks)

            system_prompt = (
                f"{self.cag_base_prompt}\n\n"
                "## PERAN SAAT INI (HYBRID RAG-CAG STUDY PLANNER):\n"
                "Anda bertugas merekomendasikan rencana studi mahasiswa berdasarkan:\n"
                "1. Aturan batasan beban SKS per semester (CAG rules di Bagian 1).\n"
                "2. Kandidat mata kuliah relevan berikut yang diambil dari kurikulum resmi (RAG):\n"
                f"{context_str}\n\n"
                "Berikan saran urutan mata kuliah yang logis dan pastikan syarat prasyarat terpenuhi."
            )

            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-4:])
            messages.append({"role": "user", "content": query})

            t_gen_start = time.perf_counter()
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.2, "top_p": 0.9, "num_predict": 300}
            )
            generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
            total_latency_ms = (time.perf_counter() - t0) * 1000.0

            return EngineResult(
                answer=resp.message.content.strip(),
                architecture="Hybrid RAG-CAG",
                retrieved_chunks=chunks,
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_generated=resp.get("eval_count", 0),
                routing_mode="HYBRID",
                metadata={"confidence": decision.confidence, "explanation": decision.explanation}
            )

        # Case E: DIRECT (General conversation)
        else:
            messages = [{"role": "system", "content": self.cag_base_prompt}]
            if history:
                messages.extend(history[-4:])
            messages.append({"role": "user", "content": query})

            t_gen_start = time.perf_counter()
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.3}
            )
            generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0
            total_latency_ms = (time.perf_counter() - t0) * 1000.0

            return EngineResult(
                answer=resp.message.content.strip(),
                architecture="Hybrid RAG-CAG",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=generation_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_generated=resp.get("eval_count", 0),
                routing_mode="DIRECT",
                metadata={"confidence": decision.confidence}
            )
