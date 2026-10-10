import time
from pathlib import Path
from typing import NamedTuple
import numpy as np
import ollama

DEFAULT_EMBED_MODEL = "nomic-embed-text"
CACHE_PATH = Path("data/guardrails_centroids.npz")


class GateResult(NamedTuple):
    is_ood: bool
    academic_score: float
    ood_score: float
    latency_ms: float
    reason: str | None = None


ACADEMIC_ANCHORS = [
    "aturan batas pengambilan sks akademik fakultas teknologi maju dan multidisiplin ftmm unair",
    "prasyarat mata kuliah kurikulum teknologi sains data teknik robotika kecerdasan buatan",
    "perencanaan studi plotting krs kartu rencana studi semester mahasiswa ftmm",
    "syarat pkl skripsi seminar proposal yudisium kelulusan ftmm unair",
    "silabus mata kuliah rekayasa nanoteknologi teknik industri teknik elektro ftmm",
    "dosen pengampu jadwal kuliah dan beban studi semester ganjil genap",
    "ketentuan perbaikan nilai remidi dan evaluasi putus studi drop out ftmm",
]

NON_ACADEMIC_ANCHORS = [
    "resep cara memasak makanan dan bumbu dapur kuliner rendang kue",
    "prediksi investasi harga cryptocurrency bitcoin trading saham dan keuangan pasar",
    "analisis politik calon presiden pemilu partai politik dan hubungan internasional",
    "rekomendasi tempat wisata liburan tiket hotel dan estimasi biaya traveling",
    "judi taruhan bola online tips menang casino esport betting",
    "puisi cinta romantis cerpen sajak gombalan pacar ulang tahun",
    "skor pertandingan sepak bola taktik formasi tim liga champions olahraga",
    "resep obat warung cara menyembuhkan penyakit flu batuk medis pengobatan",
    "spesifikasi harga hp smartphone gadget iphone android laptop elektronik",
    "chord kunci gitar lirik lagu musik mp3 tangga nada",
]


class Tier2SemanticGate:
    """Tier 2 Semantic Embedding Gating for Out-of-Scope (OOD) Detection.

    Computes cosine distance against academic and non-academic prototype centroids.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        academic_threshold: float = 0.50,
        margin: float = 0.05,
    ):
        self.embed_model = embed_model
        self.academic_threshold = academic_threshold
        self.margin = margin
        self.academic_vectors: np.ndarray | None = None
        self.ood_vectors: np.ndarray | None = None
        self._initialized = False

    def initialize(self):
        """Initializes or loads precomputed prototype embeddings."""
        if self._initialized:
            return

        if CACHE_PATH.exists():
            data = np.load(CACHE_PATH)
            self.academic_vectors = data["academic"]
            self.ood_vectors = data["ood"]
            self._initialized = True
            return

        # Compute embeddings via Ollama
        academic_resp = ollama.embed(model=self.embed_model, input=ACADEMIC_ANCHORS)
        ood_resp = ollama.embed(model=self.embed_model, input=NON_ACADEMIC_ANCHORS)

        ac_vecs = np.array(academic_resp["embeddings"], dtype=np.float32)
        ood_vecs = np.array(ood_resp["embeddings"], dtype=np.float32)

        # L2-normalize
        ac_vecs = ac_vecs / np.linalg.norm(ac_vecs, axis=1, keepdims=True)
        ood_vecs = ood_vecs / np.linalg.norm(ood_vecs, axis=1, keepdims=True)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, academic=ac_vecs, ood=ood_vecs)

        self.academic_vectors = ac_vecs
        self.ood_vectors = ood_vecs
        self._initialized = True

    def check(self, query: str) -> GateResult:
        """Determines if query is Out-of-Scope (OOD) using cosine similarity."""
        t0 = time.perf_counter()
        if not self._initialized:
            self.initialize()

        resp = ollama.embed(model=self.embed_model, input=query)
        q_vec = np.array(resp["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Dot product with normalized anchors
        ac_sims = np.dot(self.academic_vectors, q_vec)
        ood_sims = np.dot(self.ood_vectors, q_vec)

        max_ac = float(np.max(ac_sims))
        max_ood = float(np.max(ood_sims))
        dt_ms = (time.perf_counter() - t0) * 1000

        # OOD condition: either higher similarity to OOD anchors, or significantly below academic threshold
        is_ood = (max_ood > max_ac + self.margin) or (max_ac < self.academic_threshold and max_ood >= max_ac)

        reason = None
        if is_ood:
            reason = (
                f"Kueri di luar cakupan akademik FTMM UNAIR (Skor Akademik: {max_ac:.3f}, "
                f"Skor Non-Akademik: {max_ood:.3f})."
            )

        return GateResult(
            is_ood=is_ood,
            academic_score=max_ac,
            ood_score=max_ood,
            latency_ms=dt_ms,
            reason=reason,
        )
