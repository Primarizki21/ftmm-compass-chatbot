"""
Modul Retriever FAISS untuk Pencarian Dokumen Regulasi Akademik & SOP FTMM UNAIR.
Mendukung similarity search berbasis Cosine Similarity (IndexFlatIP) untuk:
- Pedoman Skripsi & Tugas Akhir
- Pedoman Praktik Kerja Lapangan (PKL)
- Pedoman Yudisium & Kelulusan
- Peraturan Akademik Fakultas
- Prosedur 8 SOP Pengajuan Layanan Administrasi
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

faiss.omp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

INDEX_REGULATIONS_PATH = Path("data/faiss/ftmm_regulations.faiss")
META_REGULATIONS_PATH = Path("data/faiss/ftmm_regulations_meta.json")

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")
DEFAULT_EMBED_DIM = int(os.getenv("EMBED_DIMENSION", "768"))


class RegulationRetriever:
    """Retriever FAISS untuk regulasi akademik, pedoman skripsi/pkl/yudisium, dan SOP FTMM."""

    def __init__(
        self,
        dimension: int | None = None,
        index_path: Path | str | None = None,
        meta_path: Path | str | None = None,
        embed_model: str | None = None,
    ):
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.dimension = dimension if dimension is not None else DEFAULT_EMBED_DIM
        self.index_path = Path(index_path) if index_path is not None else INDEX_REGULATIONS_PATH
        self.meta_path = Path(meta_path) if meta_path is not None else META_REGULATIONS_PATH
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: dict[str, Any] = {}
        self._loaded = False

    def load(self):
        """Memuat index FAISS dan metadata dari disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index regulasi belum dibuat ({self.index_path})! Jalankan scripts/ingest_regulations_faiss.py."
            )
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"Metadata regulasi belum dibuat ({self.meta_path})! Jalankan scripts/ingest_regulations_faiss.py."
            )

        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self._loaded = True

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        topic_filter: str | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Melakukan pencarian top_k regulasi berdasarkan kesamaan semantik.
        Mendukung opsional topic_filter ('skripsi', 'pkl', 'yudisium', 'akademik', 'sop').
        Mendukung injeksi query_vec precomputed untuk Single-Embed Shared Latent Pipeline.
        """
        if not self._loaded:
            self.load()

        if query_vec is not None:
            q_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
        else:
            from src.engine.base import get_ollama_client

            embed_kwargs = {"model": self.embed_model, "input": query, "keep_alive": -1}
            resp = get_ollama_client().embed(**embed_kwargs)
            q_vec = np.array(resp["embeddings"][0], dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm

        search_k = top_k * 4 if topic_filter else top_k
        scores, indices = self.index.search(q_vec, min(search_k, self.index.ntotal))

        results = []
        regulations = self.metadata["regulations"]
        chunks = self.metadata["chunks"]

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(regulations):
                continue
            item = regulations[idx]
            chunk = chunks[idx]

            if topic_filter:
                i_topic = str(item.get("topic", "")).lower()
                f_topic = topic_filter.lower().strip()
                if f_topic not in i_topic and i_topic not in f_topic:
                    continue

            results.append(
                {
                    "score": float(score),
                    "regulation": item,
                    "chunk": chunk,
                    "doc_idx": int(idx),
                }
            )
            if len(results) >= top_k:
                break

        return results


_regulation_retriever_instance: RegulationRetriever | None = None


def get_regulation_retriever() -> RegulationRetriever:
    """Mengembalikan singleton instance RegulationRetriever."""
    global _regulation_retriever_instance
    if _regulation_retriever_instance is None:
        _regulation_retriever_instance = RegulationRetriever()
        _regulation_retriever_instance.load()
    return _regulation_retriever_instance
