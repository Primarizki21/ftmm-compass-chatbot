"""
Modul Retriever FAISS untuk Pencarian Matkul FTMM UNAIR.
Mendukung similarity search berbasis Cosine Similarity (IndexFlatIP).
"""

import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import faiss
import ollama

INDEX_PATH = Path("data/faiss/ftmm_courses.faiss")
META_PATH = Path("data/faiss/ftmm_courses_meta.pkl")
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class CourseRetriever:
    def __init__(self, index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: dict[str, Any] = {}
        self._loaded = False
        
    def load(self):
        """Memuat index FAISS dan metadata dari disk."""
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"Index FAISS belum dibuat! Jalankan 'uv run python scripts/ingest_faiss.py' terlebih dahulu."
            )
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self._loaded = True

    def retrieve(self, query: str, top_k: int = 3, prodi_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Melakukan pencarian top_k mata kuliah berdasarkan kesamaan semantik.
        Mendukung opsional prodi_filter.
        """
        if not self._loaded:
            self.load()

        # Embed query
        resp = ollama.embed(model=DEFAULT_EMBED_MODEL, input=query)
        q_vec = np.array(resp["embeddings"][0], dtype=np.float32).reshape(1, -1)
        
        # L2 normalize
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Fetch extra candidates if filtering by prodi
        search_k = top_k * 4 if prodi_filter else top_k
        scores, indices = self.index.search(q_vec, min(search_k, self.index.ntotal))

        results = []
        courses = self.metadata["courses"]
        chunks = self.metadata["chunks"]

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(courses):
                continue
            course = courses[idx]
            chunk = chunks[idx]
            
            if prodi_filter:
                p_course = str(course.get("program_studi", "")).lower()
                p_filter = prodi_filter.lower()
                if p_filter not in p_course and p_course not in p_filter:
                    continue

            results.append({
                "score": float(score),
                "course": course,
                "chunk": chunk,
                "doc_idx": int(idx)
            })
            if len(results) >= top_k:
                break

        return results


# Global singleton instance
_retriever_instance: CourseRetriever | None = None


def get_course_retriever() -> CourseRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = CourseRetriever()
        _retriever_instance.load()
    return _retriever_instance
