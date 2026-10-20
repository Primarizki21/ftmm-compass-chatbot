import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from enum import Enum
from pathlib import Path
import time
from typing import Any, NamedTuple
import numpy as np
import ollama

from src.engine.base import get_ollama_client
from src.engine.tools import ACADEMIC_TOOL_DEFINITIONS

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")
CENTROIDS_PATH = Path("data/decider_centroids.npz")

class DecisionCategory(str, Enum):
    CHAT_ONLY = "CHAT_ONLY"
    PREREQUISITE = "PREREQUISITE"
    SKS_CALCULATION = "SKS_CALCULATION"
    DEGREE_PLANNING = "DEGREE_PLANNING"
    COURSE_DISCOVERY = "COURSE_DISCOVERY"
    REGULATION_SOP = "REGULATION_SOP"
    FALLBACK_ALL = "FALLBACK_ALL"


CATEGORY_TOOL_NAMES: dict[DecisionCategory, list[str]] = {
    DecisionCategory.CHAT_ONLY: [],
    DecisionCategory.REGULATION_SOP: [],
    DecisionCategory.PREREQUISITE: ["validate_prerequisites_dag", "search_courses"],
    DecisionCategory.SKS_CALCULATION: ["calculate_sks_limit"],
    DecisionCategory.DEGREE_PLANNING: ["generate_degree_plan", "get_courses_by_semester", "search_courses"],
    DecisionCategory.COURSE_DISCOVERY: ["search_courses", "get_courses_by_semester"],
    DecisionCategory.FALLBACK_ALL: [
        "search_courses",
        "validate_prerequisites_dag",
        "calculate_sks_limit",
        "generate_degree_plan",
        "get_courses_by_semester",
        "search_academic_regulations",
    ],
}

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
        "semester 6 mata kuliah TSD",
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
    DecisionCategory.REGULATION_SOP: [
        "panduan pkl praktik kerja lapangan mekanisme pengajuan dan syarat pkl",
        "bagaimana panduan dan syarat pkl kerja praktek mahasiswa",
        "panduan skripsi alur seminar proposal dan sidang tugas akhir",
        "panduan yudisium syarat berkas kelulusan dan alur pendaftaran wisuda",
        "apa saja syarat pkl magang kerja praktek ftmm unair minimal berapa sks",
        "prosedur dan syarat pengajuan skripsi tugas akhir seminar proposal sempro",
        "ketentuan yudisium kelulusan wisuda dan bebas tanggungan fakultas",
        "bagaimana tata cara pengajuan cuti akademik kuliah di unair",
        "informasi pembayaran ukt keringanan ukt dan banding ukt",
        "aturan skma satuan kredit kegiatan mahasiswa ftmm",
        "aturan perbaikan nilai ujian remidi dan semester pendek sp",
        "kebijakan evaluasi putus studi do drop out masa studi mahasiswa",
        "ketentuan batas pengambilan beban sks berdasarkan indeks prestasi semester ips",
        "kalender akademik ftmm periode perkuliahan masa krs dan jadwal libur semester",
        "prosedur pengisian krs online dan ketentuan batas waktu perubahan krs",
        "bagaimana prosedur dan solusi jika terlambat entry krs di cybercampus",
        "tabel konversi nilai mutu huruf a sampai e dan perhitungan ips ipk",
        "batas maksimal masa studi sarjana s1 14 semester dan evaluasi masa studi",
        "kriteria evaluasi mahasiswa tahap sarjana dan aturan drop out do akademik",
        "regulasi konversi nilai dan rekognisi 20 sks program mbkm kementerian",
        "pedoman keikutsertaan program magang studi independen bersertifikat msib mbkm",
        "syarat minimal kehadiran kuliah 75 persen untuk mengikuti ujian uts dan uas",
        "kebijakan presensi kehadiran 75 persen dan dispensasi ketidakhadiran sakit izin",
        "pedoman pengerjaan capstone project tugas akhir multidisiplin ftmm unair",
        "ketentuan seminar hasil pameran poster dan ujian capstone project",
        "prosedur pengajuan permohonan surat izin cuti akademik dan batas semester cuti",
        "tata tertib etika perkuliahan busana mahasiswa dan sanksi pelanggaran fakultas",
        "mekanisme pengajuan banding nilai sanggah nilai ujian akhir semester kepada dosen",
        "prosedur komplain dan klarifikasi nilai khs jika terdapat kekeliruan penilaian",
        "persyaratan sertifikat tes bahasa inggris elpt unair untuk syarat kelulusan yudisium",
        "prosedur pengajuan surat rekomendasi dekan untuk magang beasiswa dan kompetisi",
        "tata cara legalisir ijazah transkrip nilai dan permohonan surat keterangan aktif kuliah",
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
    query_vec: np.ndarray | None = None


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
        min_confidence_threshold: float = 0.35,
        margin_threshold: float = 0.012,
        model_instance: Any = None,
    ):
        self.embed_model = embed_model
        self.backend = "ollama"
        self.centroids_path = Path(centroids_path) if centroids_path is not None else CENTROIDS_PATH
        self.mrl_dim = mrl_dim
        self.min_confidence_threshold = min_confidence_threshold
        self.margin_threshold = margin_threshold
        self.categories = [
            DecisionCategory.CHAT_ONLY,
            DecisionCategory.PREREQUISITE,
            DecisionCategory.SKS_CALCULATION,
            DecisionCategory.DEGREE_PLANNING,
            DecisionCategory.COURSE_DISCOVERY,
            DecisionCategory.REGULATION_SOP,
        ]
        self.centroid_matrix: np.ndarray | None = None
        self._raw_centroid_matrix: np.ndarray | None = None
        self._raw_exemplar_matrix: np.ndarray | None = None
        self.exemplar_matrix: np.ndarray | None = None
        self.exemplar_cat_indices: np.ndarray | None = None
        self._initialized = False

    def _apply_mrl_truncation(self):
        if self._raw_centroid_matrix is None:
            return
        if self.mrl_dim is not None and self.mrl_dim < self._raw_centroid_matrix.shape[1]:
            truncated = self._raw_centroid_matrix[:, : self.mrl_dim].copy()
            norms = np.linalg.norm(truncated, axis=1, keepdims=True)
            self.centroid_matrix = truncated / np.maximum(norms, 1e-12)
            if self._raw_exemplar_matrix is not None:
                ex_trunc = self._raw_exemplar_matrix[:, : self.mrl_dim].copy()
                ex_norms = np.linalg.norm(ex_trunc, axis=1, keepdims=True)
                self.exemplar_matrix = ex_trunc / np.maximum(ex_norms, 1e-12)
        else:
            self.centroid_matrix = self._raw_centroid_matrix.copy()
            if self._raw_exemplar_matrix is not None:
                self.exemplar_matrix = self._raw_exemplar_matrix.copy()

    def set_mrl_dim(self, mrl_dim: int | None):
        """Dynamically update MRL dimension in-memory without re-reading centroids."""
        self.mrl_dim = mrl_dim
        self._apply_mrl_truncation()

    def embed_query(self, query: str) -> np.ndarray:
        resp = get_ollama_client().embed(model=self.embed_model, input=query, keep_alive=-1)
        q_vec = np.array(resp["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        return q_vec

    def initialize(self, force_recompute: bool = False):
        if self._initialized and not force_recompute:
            return

        loaded_valid = False
        if self.centroids_path.exists() and not force_recompute:
            data = np.load(self.centroids_path)
            if (
                "centroids" in data
                and data["centroids"].shape[0] == len(self.categories)
                and "exemplars" in data
            ):
                self._raw_centroid_matrix = np.array(data["centroids"], dtype=np.float32)
                self._raw_exemplar_matrix = np.array(data["exemplars"], dtype=np.float32)
                self.exemplar_cat_indices = np.array(data["exemplar_cat_indices"], dtype=np.int32)
                loaded_valid = True

        if not loaded_valid:
            centroids = []
            all_exemplars = []
            cat_indices = []
            for idx, cat in enumerate(self.categories):
                texts = PROTOTYPES[cat]
                resp = get_ollama_client().embed(model=self.embed_model, input=texts, keep_alive=-1)
                vecs = np.array(resp["embeddings"], dtype=np.float32)
                vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                centroid = np.mean(vecs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                centroids.append(centroid)
                all_exemplars.append(vecs)
                cat_indices.extend([idx] * len(vecs))

            self._raw_centroid_matrix = np.array(centroids, dtype=np.float32)
            self._raw_exemplar_matrix = np.vstack(all_exemplars).astype(np.float32)
            self.exemplar_cat_indices = np.array(cat_indices, dtype=np.int32)
            self.centroids_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.centroids_path,
                centroids=self._raw_centroid_matrix,
                exemplars=self._raw_exemplar_matrix,
                exemplar_cat_indices=self.exemplar_cat_indices,
            )

        self._apply_mrl_truncation()
        self._initialized = True
    def decide(self, query: str, query_vec: np.ndarray | None = None) -> DecisionResult:
        t0 = time.perf_counter()
        if not self._initialized:
            self.initialize()

        # 1. Embed query or use precomputed query_vec
        t_embed_start = time.perf_counter()
        if query_vec is not None:
            q_vec = np.array(query_vec, dtype=np.float32)
            t_embed_ms = 0.0
        else:
            resp = get_ollama_client().embed(model=self.embed_model, input=query, keep_alive=-1)
            q_vec = np.array(resp["embeddings"][0], dtype=np.float32)
            t_embed_ms = (time.perf_counter() - t_embed_start) * 1000

        if self.mrl_dim is not None and self.mrl_dim < len(q_vec):
            q_vec = q_vec[: self.mrl_dim]

        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        # 2. Matrix Dot Product: s = C · q
        # 2. Matrix Dot Product: s = max(C · q, E · q)
        centroid_sims = np.dot(self.centroid_matrix, q_vec)
        if self.exemplar_matrix is not None and self.exemplar_cat_indices is not None:
            ex_sims = np.dot(self.exemplar_matrix, q_vec)
            similarities = np.zeros(len(self.categories), dtype=np.float32)
            for idx in range(len(self.categories)):
                max_ex = float(np.max(ex_sims[self.exemplar_cat_indices == idx]))
                similarities[idx] = max(float(centroid_sims[idx]), max_ex)
        else:
            similarities = centroid_sims
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
            query_vec=q_vec,
        )
