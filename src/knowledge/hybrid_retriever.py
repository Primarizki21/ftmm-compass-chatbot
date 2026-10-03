import time
from typing import Any
import ollama
from src.knowledge.retriever import get_course_retriever, CourseRetriever
from src.knowledge.bm25 import BM25Okapi

DEFAULT_LLM_MODEL = "qwen2.5:3b"


class HybridCourseRetriever:
    def __init__(self, dense_retriever: CourseRetriever | None = None):
        self.dense_retriever = dense_retriever or get_course_retriever()
        if not self.dense_retriever._loaded:
            self.dense_retriever.load()
            
        self.courses = self.dense_retriever.metadata["courses"]
        self.chunks = self.dense_retriever.metadata["chunks"]
        self.bm25 = BM25Okapi(self.chunks)

    def retrieve_dense(self, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        raw_results = self.dense_retriever.retrieve(query, top_k=top_k)
        results = []
        for r, hit in enumerate(raw_results, start=1):
            course = hit["course"]
            results.append({
                "course_id": course["id"],
                "kode_mk": course["kode_mk"],
                "nama_mk": course["nama_mk"],
                "score": hit["score"],
                "chunk": hit["chunk"],
                "rank": r,
            })
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def retrieve_sparse(self, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        ranked = self.bm25.search(query, top_k=top_k)
        results = []
        for r, (idx, score) in enumerate(ranked, start=1):
            course = self.courses[idx]
            results.append({
                "course_id": course["id"],
                "kode_mk": course["kode_mk"],
                "nama_mk": course["nama_mk"],
                "score": score,
                "chunk": self.chunks[idx],
                "rank": r,
            })
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: int = 5,
        k_rrf: int = 60,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0
    ) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        # Dense ranking
        dense_hits, _ = self.retrieve_dense(query, top_k=top_k * 3)
        # Sparse ranking
        sparse_hits, _ = self.retrieve_sparse(query, top_k=top_k * 3)
        
        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["course_id"]
            scores[cid] = scores.get(cid, 0.0) + (weight_dense / (k_rrf + rank))
            payloads[cid] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit["course_id"]
            scores[cid] = scores.get(cid, 0.0) + (weight_sparse / (k_rrf + rank))
            if cid not in payloads:
                payloads[cid] = hit

        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
        fused_results = []
        for r, cid in enumerate(sorted_cids, start=1):
            item = dict(payloads[cid])
            item["rrf_score"] = scores[cid]
            item["rank"] = r
            fused_results.append(item)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return fused_results, latency_ms

    def reformulate_query(self, query: str, model: str = DEFAULT_LLM_MODEL) -> tuple[str, float]:
        t0 = time.perf_counter()
        prompt = (
            "Ubah kueri informal atau pertanyaan mahasiswa berikut menjadi kata kunci mata kuliah/akademik "
            "formal dalam Bahasa Indonesia tanpa tanda kutip atau penjelasan tambahan. "
            "Contoh input: 'belajar koding dasar python' -> Output: 'Algoritma dan Pemrograman Python'\n"
            f"Input: {query}\nOutput:"
        )
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 30}
            )
            expanded = resp.message.content.strip().split("\n")[0].replace('"', '')
        except Exception:
            expanded = query
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return expanded, latency_ms

    def retrieve_hybrid_reformulated(
        self,
        query: str,
        top_k: int = 5,
        model: str = DEFAULT_LLM_MODEL
    ) -> tuple[list[dict[str, Any]], float, str]:
        t0 = time.perf_counter()
        reformulated_q, _ = self.reformulate_query(query, model=model)
        combined_q = f"{query} {reformulated_q}"
        results, _ = self.retrieve_hybrid_rrf(combined_q, top_k=top_k)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, total_latency_ms, reformulated_q
