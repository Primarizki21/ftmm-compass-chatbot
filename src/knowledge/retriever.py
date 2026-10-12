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
import torch

INDEX_NOMIC_PATH = Path("data/faiss/ftmm_courses.faiss")
META_NOMIC_PATH = Path("data/faiss/ftmm_courses_meta.pkl")

INDEX_GEMMA2_768_PATH = Path("data/faiss/ftmm_courses_gemma2.faiss")
INDEX_GEMMA2_256_PATH = Path("data/faiss/ftmm_courses_gemma2_256d.faiss")
META_GEMMA2_PATH = Path("data/faiss/ftmm_courses_gemma2_meta.pkl")

# Default constants
INDEX_PATH = INDEX_NOMIC_PATH
META_PATH = META_NOMIC_PATH
DEFAULT_EMBED_MODEL = "nomic-embed-text"

class CourseRetriever:
    def __init__(
        self,
        backend: str | None = None,
        dimension: int | None = None,
        index_path: Path | str | None = None,
        meta_path: Path | str | None = None,
        model_instance: Any = None,
    ):
        raw_backend = backend or os.getenv("EMBED_BACKEND", "nomic")
        self.backend = raw_backend.lower()
        raw_dim = dimension if dimension is not None else int(os.getenv("EMBED_DIMENSION", "256"))
        self.dimension = raw_dim

        if self.backend == "gemma2":
            if index_path is not None:
                self.index_path = Path(index_path)
            else:
                self.index_path = INDEX_GEMMA2_256_PATH if self.dimension == 256 else INDEX_GEMMA2_768_PATH
            self.meta_path = Path(meta_path) if meta_path is not None else META_GEMMA2_PATH
        else:
            self.index_path = Path(index_path) if index_path is not None else INDEX_NOMIC_PATH
            self.meta_path = Path(meta_path) if meta_path is not None else META_NOMIC_PATH

        self.index: faiss.IndexFlatIP | None = None
        self.metadata: dict[str, Any] = {}
        self._loaded = False
        self._gemma_model = model_instance

    def _get_gemma_model(self):
        if self._gemma_model is not None:
            return self._gemma_model
        from src.engine.decider import get_shared_gemma2_model
        self._gemma_model = get_shared_gemma2_model()
        return self._gemma_model

    def load(self):
        """Memuat index FAISS dan metadata dari disk."""
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"Index FAISS belum dibuat ({self.index_path})! Jalankan ingest script terlebih dahulu."
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

        if self.backend == "gemma2":
            model = self._get_gemma_model()
            with torch.inference_mode():
                if self.dimension < 768:
                    q_vec = model.encode(
                        query,
                        prompt_name="query",
                        truncate_dim=self.dimension,
                        normalize_embeddings=True,
                    )
                else:
                    q_vec = model.encode(
                        query,
                        prompt_name="query",
                        normalize_embeddings=True,
                    )
            q_vec = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        else:
            import ollama
            resp = ollama.embed(model="nomic-embed-text", input=query)
            q_vec = np.array(resp["embeddings"][0], dtype=np.float32).reshape(1, -1)
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
