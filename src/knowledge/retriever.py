"""
Modul Retriever FAISS untuk Pencarian Matkul FTMM UNAIR.
Mendukung similarity search berbasis Cosine Similarity (IndexFlatIP).
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import faiss
import ollama

faiss.omp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

INDEX_NOMIC_PATH = Path("data/faiss/ftmm_courses.faiss")
META_NOMIC_PATH = Path("data/faiss/ftmm_courses_meta.pkl")
META_JSON_NOMIC_PATH = Path("data/faiss/ftmm_courses_meta.json")

INDEX_PATH = INDEX_NOMIC_PATH
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
DEFAULT_EMBED_DIM = int(os.getenv("EMBED_DIMENSION", "768"))

class CourseRetriever:
    def __init__(
        self,
        backend: str | None = None,
        dimension: int | None = None,
        index_path: Path | str | None = None,
        meta_path: Path | str | None = None,
        model_instance: Any = None,
        embed_model: str | None = None,
    ):
        self.backend = backend or "nomic"
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.dimension = dimension if dimension is not None else DEFAULT_EMBED_DIM
        self.index_path = Path(index_path) if index_path is not None else INDEX_NOMIC_PATH
        self.meta_path = Path(meta_path) if meta_path is not None else META_NOMIC_PATH
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: dict[str, Any] = {}
        self._loaded = False

    def load(self):
        """Memuat index FAISS dan metadata dari disk (mendukung JSON dan pickle)."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS belum dibuat ({self.index_path})! Jalankan ingest script terlebih dahulu."
            )

        # Prioritaskan meta_path eksplisit, fallback ke JSON, lalu pickle
        resolved_meta: Path | None = None
        if self.meta_path and self.meta_path.exists():
            resolved_meta = self.meta_path
        elif META_JSON_NOMIC_PATH.exists():
            resolved_meta = META_JSON_NOMIC_PATH
        elif META_NOMIC_PATH.exists():
            resolved_meta = META_NOMIC_PATH

        if not resolved_meta or not resolved_meta.exists():
            raise FileNotFoundError(
                f"Metadata FAISS belum dibuat ({self.meta_path})! Jalankan ingest script terlebih dahulu."
            )

        self.index = faiss.read_index(str(self.index_path))
        if resolved_meta.suffix.lower() == ".json":
            with open(resolved_meta, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            with open(resolved_meta, "rb") as f:
                self.metadata = pickle.load(f)
        self._loaded = True
    def retrieve(self, query: str, top_k: int = 3, prodi_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Melakukan pencarian top_k mata kuliah berdasarkan kesamaan semantik.
        Mendukung opsional prodi_filter.
        """
        if not self._loaded:
            self.load()
        from src.engine.base import get_ollama_client
        embed_kwargs = {"model": self.embed_model, "input": query, "keep_alive": -1}
        if self.dimension and self.dimension < 768:
            embed_kwargs["dimensions"] = self.dimension
        resp = get_ollama_client().embed(**embed_kwargs)
        q_vec = np.array(resp["embeddings"][0], dtype=np.float32).reshape(1, -1)
        if self.dimension and q_vec.shape[1] > self.dimension:
            q_vec = q_vec[:, :self.dimension]
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
