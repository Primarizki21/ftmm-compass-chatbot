import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from enum import Enum
from pathlib import Path
import re
import time
from typing import Any, NamedTuple
import numpy as np
import ollama
import torch
torch.set_num_threads(4)

from src.engine.tools import ACADEMIC_TOOL_DEFINITIONS

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEMMA2_MODEL = "google/embeddinggemma-2"
CENTROIDS_PATH = Path("data/decider_centroids.npz")
CENTROIDS_GEMMA2_PATH = Path("data/decider_centroids_gemma2.npz")
_CACHED_GEMMA2_MODEL: Any = None


class DecisionCategory(str, Enum):
    CHAT_ONLY = "CHAT_ONLY"
    PREREQUISITE = "PREREQUISITE"
    SKS_CALCULATION = "SKS_CALCULATION"
    DEGREE_PLANNING = "DEGREE_PLANNING"
    COURSE_DISCOVERY = "COURSE_DISCOVERY"
    FALLBACK_ALL = "FALLBACK_ALL"


CATEGORY_TOOL_NAMES: dict[DecisionCategory, list[str]] = {
    DecisionCategory.CHAT_ONLY: [],
    DecisionCategory.PREREQUISITE: ["validate_prerequisites_dag", "search_courses"],
    DecisionCategory.SKS_CALCULATION: ["calculate_sks_limit"],
    DecisionCategory.DEGREE_PLANNING: ["generate_degree_plan", "search_courses"],
    DecisionCategory.COURSE_DISCOVERY: ["search_courses", "get_courses_by_semester"],
    DecisionCategory.FALLBACK_ALL: [
        "search_courses",
        "validate_prerequisites_dag",
        "calculate_sks_limit",
        "generate_degree_plan",
        "get_courses_by_semester",
    ],
}

SEMESTER_COURSE_FAST_PATH_REGEX = re.compile(
    r"(?:mata\s*kuliah|matkul|daftar\s*(?:mata\s*kuliah|matkul)|courses?|list\s*courses?)"
    r".*?"
    r"(?:semester|smt|sem)\s*([1-8]|i{1,3}|iv|v|vi{0,3}|viii)\b"
    r"|"
    r"(?:semester|smt|sem)\s*([1-8]|i{1,3}|iv|v|vi{0,3}|viii)\b"
    r".*?"
    r"(?:mata\s*kuliah|matkul|daftar\s*(?:mata\s*kuliah|matkul)|courses?|list\s*courses?)",
    re.IGNORECASE,
)

PROTOTYPES: dict[DecisionCategory, list[str]] = {
    DecisionCategory.CHAT_ONLY: [
        "halo selamat pagi apa kabar",
        "terima kasih banyak atas bantuannya ya",
        "siapa namamu dan apa tugasmu di ftmm",
        "hai bot ftmm unair salam kenal",
        "selamat siang saya ingin menyapa",
        "halo saya mahasiswa ftmm",
        "oke baik sudah cukup jelas penjelasannya terima kasih",
        "sampai jumpa di konsultasi akademik berikutnya",
        "hanya ingin menyapa sebelum mulai sesi konsultasi",
        "saya siap memulai obrolan tentang perkuliahan",
        "makasih banyak ya min infonya ngebantu banget",
    ],
    DecisionCategory.PREREQUISITE: [
        "apakah mata kuliah data mining punya prasyarat alpro",
        "prasyarat matkul pembelajaran mesin di teknik robotika",
        "apakah saya boleh ambil dasar sistem tenaga jika belum lulus rangkaian listrik",
        "kira-kira terganjal matkul lama gak ya kalau ambil ini",
        "kalo alpro dapat nilai d kudu ngulang ga bro",
        "syarat sebelum mengambil mata kuliah ini apa saja",
        "bagaimana ketentuan prasyarat mata kuliah di rn",
        "apakah ada keterikatan urutan kelulusan sebelum saya berhak mendaftar",
        "saya belum lulus mata kuliah semester 1 apakah menghalangi pendaftaran mata kuliah lanjutan",
        "bisa lolos validasi krs gak ya kalau mata kuliah pendahulunya belum tuntas",
        "bisa langsung comot matkul jarkom ga kalo sisop belum kelar bro",
    ],
    DecisionCategory.SKS_CALCULATION: [
        "berapa batas maksimal sks jika ips semester lalu 3.5",
        "kalau ips anjlok 2.2 dapat jatah sks berapaan nih bro",
        "aturan beban sks maksimum panduan akademik ftmm",
        "kalo ipk 3.8 boleh maruk ambil 24 sks gak",
        "perhitungan jatah beban krs berdasarkan indeks prestasi",
        "berapa sks yang bisa saya ambil semester depan",
        "dengan perolehan nilai rapor semester kemarin 3.40 kuota maksimal pendaftaran saya berapa",
        "berapa kapasitas beban belajar yang diizinkan untuk performa akademik di bawah 2.5",
        "apakah pencapaian indeks prestasi 3.75 memberikan hak kuota maksimal pada krs",
        "berapa plafon beban kredit studi yang sah menurut regulasi fakultas",
        "jatah sks gue kepotong banyak gak ya kalo ips cuma 2.6",
    ],
    DecisionCategory.DEGREE_PLANNING: [
        "rancang rekomendasi rencana studi semester 5 teknik industri peminatan rantai pasok",
        "tolong buatkan plotting krs semester 4 teknologi sains data",
        "rekomendasi paket mata kuliah semester depan teknik elektro",
        "gue mau ngebut lulus 3.5 tahun gimana plotting krs yang pas",
        "bantu susun jadwal rencana studi semester 6 trkb robotika",
        "buatkan degree plan rencana studi kelulusan",
        "bagaimana pemetaan distribusi mata kuliah dari semester 3 hingga semester 8 agar lulus tepat waktu",
        "bagaimana roadmap pengambilan mata kuliah yang ideal untuk mengejar karir",
        "bantu susun tahapan semester demi semester untuk memenuhi seluruh beban 144 sks kelulusan",
        "saya ingin merancang alur studi agar semester 7 sudah bisa fokus magang",
        "rekomendasikan strategi pemilihan kombinasi mata kuliah pilihan dan wajib untuk semester depan",
        "bikinin rencana jadwal matkul semester 4 tsd yang ga bentrok dong kak",
    ],
    DecisionCategory.COURSE_DISCOVERY: [
        "mata kuliah apa saja yang ditawarkan semester 3 rekayasa nanoteknologi",
        "silabus dan deskripsi materi mata kuliah jaringan komputer",
        "info dosen pengampu mata kuliah visualisasi data",
        "apakah ada matkul tentang nanomedisin di kurikulum rn",
        "info dong matkul metstat sama probstat bedanya apa",
        "daftar mata kuliah pilihan yang tersedia di semester genap",
        "materi dan topik apa saja yang akan dipelajari mahasiswa saat mengambil mata kuliah tersebut",
        "apakah terdapat mata kuliah yang mengajarkan teknik optimasi rantai pasokan global",
        "saya mencari mata kuliah di fakultas ini yang memiliki bobot praktikum di laboratorium elektro",
        "tolong carikan mata kuliah yang relevan dengan topik rekayasa material nano tingkat lanjut",
        "ada matkul yang ngebahas soal kecerdasan buatan ga di jurusan robotika",
        "matkul prak terintegrasi itu pembahasannya seputar apa aja sih sizenya berapa sks",
    ],
}


class DecisionResult(NamedTuple):
    category: DecisionCategory
    confidence: float
    selected_tools: list[dict[str, Any]] | None
    latency_ms: float
    embedding_latency_ms: float
    scores: dict[str, float]
    margin: float = 0.0


class System1Decider:
    """System-1 Decision Classifier for Semantic Tool Schema Pruning (Card 99, Card 102).

    Uses L2-normalized embedding matrix dot-product against class centroids.
    Supports both Ollama (nomic-embed-text) and SentenceTransformer (EmbeddingGemma-2)
    with optional Matryoshka Representation Learning (MRL) dimension truncation in-memory.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        backend: str | None = None,
        centroids_path: Path | str | None = None,
        mrl_dim: int | None = None,
        min_confidence_threshold: float = 0.40,
        margin_threshold: float = 0.02,
        model_instance: Any = None,
    ):
        self.embed_model = embed_model
        if backend is not None:
            self.backend = backend.lower()
        elif "gemma" in embed_model.lower():
            self.backend = "gemma2"
        else:
            self.backend = "ollama"

        if centroids_path is not None:
            self.centroids_path = Path(centroids_path)
        else:
            self.centroids_path = CENTROIDS_GEMMA2_PATH if self.backend == "gemma2" else CENTROIDS_PATH

        self.mrl_dim = mrl_dim
        self.min_confidence_threshold = min_confidence_threshold
        self.margin_threshold = margin_threshold
        self.categories = [
            DecisionCategory.CHAT_ONLY,
            DecisionCategory.PREREQUISITE,
            DecisionCategory.SKS_CALCULATION,
            DecisionCategory.DEGREE_PLANNING,
            DecisionCategory.COURSE_DISCOVERY,
        ]
        self.centroid_matrix: np.ndarray | None = None
        self._raw_centroid_matrix: np.ndarray | None = None
        self._initialized = False
        self._gemma_model = model_instance

    def _get_gemma_model(self):
        if self._gemma_model is not None:
            return self._gemma_model
        global _CACHED_GEMMA2_MODEL
        if _CACHED_GEMMA2_MODEL is not None:
            self._gemma_model = _CACHED_GEMMA2_MODEL
            return self._gemma_model
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CACHED_GEMMA2_MODEL = SentenceTransformer(
            self.embed_model if "gemma" in self.embed_model.lower() else DEFAULT_GEMMA2_MODEL,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        self._gemma_model = _CACHED_GEMMA2_MODEL
        return self._gemma_model

    def _apply_mrl_truncation(self):
        if self._raw_centroid_matrix is None:
            return
        if self.mrl_dim is not None and self.mrl_dim < self._raw_centroid_matrix.shape[1]:
            truncated = self._raw_centroid_matrix[:, : self.mrl_dim].copy()
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            self.centroid_matrix = truncated / np.maximum(norms, 1e-12)
        else:
            self.centroid_matrix = self._raw_centroid_matrix.copy()

    def set_mrl_dim(self, mrl_dim: int | None):
        """Dynamically update MRL dimension in-memory without re-reading centroids."""
        self.mrl_dim = mrl_dim
        self._apply_mrl_truncation()

    def initialize(self, force_recompute: bool = False):
        if self._initialized and not force_recompute:
            return

        if self.centroids_path.exists() and not force_recompute:
            data = np.load(self.centroids_path)
            self._raw_centroid_matrix = np.array(data["centroids"], dtype=np.float32)
        else:
            centroids = []
            if self.backend == "gemma2":
                model = self._get_gemma_model()
                with torch.inference_mode():
                    for cat in self.categories:
                        texts = PROTOTYPES[cat]
                        vecs = model.encode(
                            texts,
                            prompt_name="Classification",
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        )
                        vecs = np.array(vecs, dtype=np.float32)
                        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                        centroid = np.mean(vecs, axis=0)
                        centroid = centroid / np.linalg.norm(centroid)
                        centroids.append(centroid)
            else:
                for cat in self.categories:
                    texts = PROTOTYPES[cat]
                    resp = ollama.embed(model=self.embed_model, input=texts)
                    vecs = np.array(resp["embeddings"], dtype=np.float32)
                    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                    centroid = np.mean(vecs, axis=0)
                    centroid = centroid / np.linalg.norm(centroid)
                    centroids.append(centroid)

            self._raw_centroid_matrix = np.array(centroids, dtype=np.float32)
            self.centroids_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.centroids_path, centroids=self._raw_centroid_matrix)

        self._apply_mrl_truncation()
        self._initialized = True

    def decide(self, query: str) -> DecisionResult:
        t0 = time.perf_counter()
        if SEMESTER_COURSE_FAST_PATH_REGEX.search(query):
            tool_names = CATEGORY_TOOL_NAMES[DecisionCategory.COURSE_DISCOVERY]
            selected_tools = [
                t for t in ACADEMIC_TOOL_DEFINITIONS if t["function"]["name"] in tool_names
            ]
            total_dt = (time.perf_counter() - t0) * 1000
            return DecisionResult(
                category=DecisionCategory.COURSE_DISCOVERY,
                confidence=1.0,
                selected_tools=selected_tools,
                latency_ms=total_dt,
                embedding_latency_ms=0.0,
                scores={
                    cat.value: (1.0 if cat == DecisionCategory.COURSE_DISCOVERY else 0.0)
                    for cat in self.categories
                },
                margin=1.0,
            )

        if not self._initialized:
            self.initialize()

        # 1. Embed query
        t_embed_start = time.perf_counter()
        if self.backend == "gemma2":
            model = self._get_gemma_model()
            with torch.inference_mode():
                vec = model.encode(
                    query,
                    prompt_name="Classification",
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            q_vec = np.array(vec, dtype=np.float32)
        else:
            resp = ollama.embed(model=self.embed_model, input=query)
            q_vec = np.array(resp["embeddings"][0], dtype=np.float32)

        if self.mrl_dim is not None and self.mrl_dim < len(q_vec):
            q_vec = q_vec[: self.mrl_dim]

        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000

        # 2. Matrix Dot Product: s = C · q
        similarities = np.dot(self.centroid_matrix, q_vec)
        score_dict = {cat.value: float(sim) for cat, sim in zip(self.categories, similarities)}

        best_idx = int(np.argmax(similarities))
        sorted_indices = np.argsort(similarities)[::-1]
        best_score = float(similarities[best_idx])
        second_score = float(similarities[sorted_indices[1]])
        margin = float(best_score - second_score)

        # 3. Ambiguity Guard
        is_ambiguous = (best_score < self.min_confidence_threshold) or (
            (best_score - second_score) < self.margin_threshold and best_score < 0.60
        )

        if is_ambiguous:
            chosen_cat = DecisionCategory.FALLBACK_ALL
        else:
            chosen_cat = self.categories[best_idx]

        # 4. Map to tools
        tool_names = CATEGORY_TOOL_NAMES[chosen_cat]
        if not tool_names:
            selected_tools = None
        else:
            selected_tools = [
                t for t in ACADEMIC_TOOL_DEFINITIONS if t["function"]["name"] in tool_names
            ]

        total_dt = (time.perf_counter() - t0) * 1000

        return DecisionResult(
            category=chosen_cat,
            confidence=best_score,
            selected_tools=selected_tools,
            latency_ms=total_dt,
            embedding_latency_ms=t_embed_ms,
            scores=score_dict,
            margin=margin,
        )
