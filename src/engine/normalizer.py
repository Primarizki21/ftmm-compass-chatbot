from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any

DEFAULT_CATALOG_PATH = Path("data/courses_catalog.json")

# Kamus leksikal O(1) singkatan dan slang akademik FTMM
SLANG_LEXICON: dict[str, str] = {
    # Mata kuliah inti FTMM
    "alpro": "Algoritma Pemrograman",
    "alprog": "Algoritma Pemrograman",
    "strukdat": "Struktur Data",
    "metpen": "Metodologi Penelitian",
    "basdat": "Basis Data",
    "jarkom": "Jaringan Komputer",
    "sisop": "Sistem Operasi",
    "sismul": "Sistem Multimedia",
    "pbo": "Pemrograman Berorientasi Objek",
    "oop": "Pemrograman Berorientasi Objek",
    "rpl": "Rekayasa Perangkat Lunak",
    "se": "Rekayasa Perangkat Lunak",
    "kalku": "Kalkulus",
    "matdis": "Matematika Diskrit",
    "komnum": "Komputasi Numerik",
    "anreg": "Analisis Regresi",
    "statprob": "Probabilitas dan Statistika",
    "probstat": "Probabilitas dan Statistika",
    "elka": "Elektronika",
    "sisdig": "Sistem Digital",
    "dsp": "Pengolahan Sinyal Digital",
    "viskom": "Visi Komputer",
    "mikro": "Mikroprosesor dan Mikrokontroler",
    "mikrokontroler": "Mikroprosesor dan Mikrokontroler",
    "ai": "Kecerdasan Buatan",
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "nlp": "Pemrosesan Bahasa Alami",
    # Istilah operasional akademik
    "smt": "semester",
    "sem": "semester",
    "semseter": "semester",
    "sks": "SKS",
    "krs": "KRS",
    "pkl": "Praktik Kerja Lapangan",
    "magang": "Praktik Kerja Lapangan",
    "ta": "Skripsi",
    "skripsi": "Skripsi",
    "tugas akhir": "Skripsi",
    "yudisium": "Yudisium",
    "syarat": "prasyarat",
    "sarat": "prasyarat",
    "saratsyarat": "prasyarat",
    "prasyarat": "prasyarat",
    "matkul": "mata kuliah",
    "makul": "mata kuliah",
    "mk": "mata kuliah",
    "dosbim": "dosen pembimbing",
    "dospem": "dosen pembimbing",
    "kaprodi": "ketua program studi",
    "ipk": "IPK",
    "ips": "IPS",
    "sp": "Semester Pendek",
    # Akronim program studi
    "tsd": "Teknologi Sains Data",
    "trkb": "Teknik Robotika dan Kecerdasan Buatan",
    "rn": "Rekayasa Nanoteknologi",
    "ti": "Teknik Industri",
    "te": "Teknik Elektro",
}

# Pola frasa noise fonetik Speech-to-Text (STT) dan homofon
STT_PHONETIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Entitas mata kuliah & topik
    (re.compile(r"\b(struktur dada|struktur datah|struk tur data)\b", re.IGNORECASE), "Struktur Data"),
    (re.compile(r"\b(algorit ma|argo ritme|al go ritma|algoritme|algorhitma)\b", re.IGNORECASE), "Algoritma"),
    (re.compile(r"\b(pembrograman|pemprograman|pem rograman)\b", re.IGNORECASE), "Pemrograman"),
    (re.compile(r"\b(pasis data|basis dada|pasif data)\b", re.IGNORECASE), "Basis Data"),
    (re.compile(r"\b(kalkuluz|kal kulus|kalkuluse)\b", re.IGNORECASE), "Kalkulus"),
    (re.compile(r"\b(propabilitas|probabilistas)\b", re.IGNORECASE), "Probabilitas"),
    (re.compile(r"\b(statistike|statistik)\b", re.IGNORECASE), "Statistika"),
    (re.compile(r"\b(metode penelitian|metodelogi penelitian|metodologi penelitan)\b", re.IGNORECASE), "Metodologi Penelitian"),
    (re.compile(r"\b(jaringan computer|jaringan komputir)\b", re.IGNORECASE), "Jaringan Komputer"),
    (re.compile(r"\b(sistem oprasi|sistem aperasi)\b", re.IGNORECASE), "Sistem Operasi"),
    (re.compile(r"\b(rekayasa perankat lunak|rekayasa prangkat lunak)\b", re.IGNORECASE), "Rekayasa Perangkat Lunak"),
    (re.compile(r"\b(nano teknologi|nano tekno logi)\b", re.IGNORECASE), "Nanoteknologi"),
    (re.compile(r"\b(robo tika|robotik)\b", re.IGNORECASE), "Robotika"),
    (re.compile(r"\b(kecerdasan buwatan|ke cerdasan buatan)\b", re.IGNORECASE), "Kecerdasan Buatan"),
    (re.compile(r"\b(pemrosesan bahasa alami|proses bahasa alami)\b", re.IGNORECASE), "Pemrosesan Bahasa Alami"),
    # Istilah kebijakan & administrasi
    (re.compile(r"\b(pra syarat|perasyarat|parasyarat)\b", re.IGNORECASE), "prasyarat"),
    (re.compile(r"\b(semeter|semestir|semseter)\b", re.IGNORECASE), "semester"),
    (re.compile(r"\b(mata kulia|matakuliah)\b", re.IGNORECASE), "mata kuliah"),
    (re.compile(r"\b(skipsi|eskripsi)\b", re.IGNORECASE), "skripsi"),
    (re.compile(r"\b(pekael)\b", re.IGNORECASE), "PKL"),
    (re.compile(r"\b(kaeres)\b", re.IGNORECASE), "KRS"),
    (re.compile(r"\b(eskaes)\b", re.IGNORECASE), "SKS"),
    (re.compile(r"\b(te es de)\b", re.IGNORECASE), "TSD"),
    (re.compile(r"\b(te er ka be)\b", re.IGNORECASE), "TRKB"),
    (re.compile(r"\b(er en)\b", re.IGNORECASE), "RN"),
    (re.compile(r"\b(te i)\b", re.IGNORECASE), "TI"),
    (re.compile(r"\b(te e)\b", re.IGNORECASE), "TE"),
]

# Kata henti percakapan yang dilarang dikoreksi typo
CONVERSATIONAL_STOPWORDS: set[str] = {
    "apa", "apakah", "apaan", "bagaimana", "gimana", "bisa", "yang", "dan", "di", "ke",
    "dari", "untuk", "pada", "dengan", "ini", "itu", "saya", "aku", "kamu", "dia",
    "kita", "mereka", "ada", "mau", "ingin", "tolong", "bantu", "ambil", "dosen",
    "nilai", "lulus", "kuliah", "buat", "kalo", "kalau", "dong", "ya", "nih", "deh",
    "siapa", "kapan", "mengapa", "kenapa", "berapa", "mana", "saja", "tentang",
    "info", "daftar", "rekomendasi", "jadwal", "bukan", "hanya", "sudah", "belum",
    "mata", "tata", "kelola", "prodi", "jurusan", "kelas", "kredit", "silabus",
    "ujian", "uts", "uas", "bobot", "cara", "saja", "syarat", "data"
}

# Kata kunci domain akademik tambahan
ACADEMIC_DOMAIN_KEYWORDS: set[str] = {
    "semester", "prasyarat", "skripsi", "yudisium", "kurikulum", "jadwal",
    "peminatan", "rekomendasi", "perencanaan", "kelulusan", "transkrip",
    "mahasiswa", "fakultas", "departemen", "akademik", "pedoman", "aturan",
    "mata", "kuliah", "prodi", "jurusan", "kelas", "kredit", "silabus",
    "ujian", "uts", "uas", "bobot", "dosen", "pengampu", "tata", "kelola"
}

# Pola pendeteksi format kandidat kode mata kuliah
COURSE_CODE_CANDIDATE_PATTERN = re.compile(r"^([A-Za-z]{2,4})\s?([0-9OIl]{2,4})$", re.IGNORECASE)


def damerau_levenshtein_distance(s1: str, s2: str, max_dist: int = 2) -> int:
    """Bounded Damerau-Levenshtein distance (insert, delete, substitute, transposition)."""
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > max_dist:
        return max_dist + 1
    if s1 == s2:
        return 0
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    inf = len1 + len2
    da: dict[str, int] = {}
    h = [[0] * (len2 + 2) for _ in range(len1 + 2)]
    h[0][0] = inf
    for i in range(0, len1 + 1):
        h[i + 1][0] = inf
        h[i + 1][1] = i
    for j in range(0, len2 + 1):
        h[0][j + 1] = inf
        h[1][j + 1] = j

    for i in range(1, len1 + 1):
        db = 0
        min_in_row = inf
        for j in range(1, len2 + 1):
            i1 = da.get(s2[j - 1], 0)
            j1 = db
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if cost == 0:
                db = j

            val = min(
                h[i][j + 1] + 1,
                h[i + 1][j] + 1,
                h[i][j] + cost,
                h[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),
            )
            h[i + 1][j + 1] = val
            if val < min_in_row:
                min_in_row = val
        da[s1[i - 1]] = i
        if min_in_row > max_dist:
            return max_dist + 1

    return h[len1 + 1][len2 + 1]


@dataclass
class NormalizationTrace:
    original: str
    normalized: str
    rule_type: str
    distance: int = 0


@dataclass
class NormalizationResult:
    original_query: str
    normalized_query: str
    traces: list[NormalizationTrace] = field(default_factory=list)
    latency_ms: float = 0.0


class AcademicNormalizer:
    """Normalizer deterministik leksikal, fonetik STT, dan Bounded Damerau-Levenshtein."""

    def __init__(self, catalog_path: Path | str = DEFAULT_CATALOG_PATH):
        self.catalog_path = Path(catalog_path)
        self.course_codes: set[str] = set()
        self.course_codes_by_len: dict[int, list[str]] = defaultdict(list)
        self.vocab_words: set[str] = set()
        self.vocab_by_len: dict[int, list[str]] = defaultdict(list)
        self.course_names: list[str] = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        catalog_courses: list[dict[str, Any]] = []
        if self.catalog_path.exists():
            with open(self.catalog_path, encoding="utf-8") as f:
                catalog_courses = json.load(f)

        for course in catalog_courses:
            code = str(course.get("kode_mk", "")).strip().upper()
            if code:
                self.course_codes.add(code)
                self.course_codes_by_len[len(code)].append(code)

            name = str(course.get("nama_mk", "")).strip()
            if name:
                self.course_names.append(name)
                words = re.findall(r"[A-Za-z]+", name.lower())
                for w in words:
                    if len(w) >= 3:
                        self.vocab_words.add(w)

        for kw in ACADEMIC_DOMAIN_KEYWORDS:
            self.vocab_words.add(kw.lower())

        for w in self.vocab_words:
            self.vocab_by_len[len(w)].append(w)

        self._initialized = True

    def _correct_course_code(self, token: str) -> tuple[str | None, int]:
        clean = re.sub(r"[^\w]", "", token).upper()
        # Jangan ubah akronim prodi FTMM menjadi kode mata kuliah prodi lain
        if clean.startswith(("TSD", "TRKB", "RN", "TI", "TE")):
            normalized_cand = re.sub(r"([A-Z]+)O([0-9]+)", r"\g<1>0\2", clean)
            if normalized_cand != clean:
                return normalized_cand, 1
            return None, 3

        # Normalisasi OCR/Typo huruf O vs digit 0 pada kode MK resmi
        normalized_cand = re.sub(r"([A-Z]{2,4})O([0-9]{1,3})", r"\g<1>0\2", clean)
        if normalized_cand in self.course_codes:
            return normalized_cand, (0 if normalized_cand == clean else 1)

        match = COURSE_CODE_CANDIDATE_PATTERN.match(clean)
        if not match:
            return None, 3

        prefix, num_part = match.group(1), match.group(2)
        num_part_clean = num_part.replace("O", "0").replace("I", "1").replace("L", "1")
        rebuilt = f"{prefix}{num_part_clean}"
        if rebuilt in self.course_codes:
            return rebuilt, 1

        # Hanya cocokkan jika prefix 3-huruf sama persis guna mencegah perpindahan prodi
        cand_len = len(clean)
        best_code = None
        min_dist = 3
        for target_code in self.course_codes_by_len[cand_len]:
            if target_code[:3] == prefix[:3]:
                d = damerau_levenshtein_distance(clean.lower(), target_code.lower(), max_dist=2)
                if d < min_dist:
                    min_dist = d
                    best_code = target_code

        if best_code is not None and min_dist <= 2:
            return best_code, min_dist
        return None, 3

    def _correct_word_levenshtein(self, token: str) -> tuple[str, str | None, int]:
        clean = re.sub(r"[^\w]", "", token).lower()
        if not clean or clean in CONVERSATIONAL_STOPWORDS:
            return token, None, 0

        if clean in self.vocab_words:
            return token, None, 0

        w_len = len(clean)
        if w_len < 5:
            return token, None, 0

        max_allowed_dist = 1 if w_len <= 5 else 2
        best_match = None
        min_dist = max_allowed_dist + 1

        for cand_len in range(max(3, w_len - 2), w_len + 3):
            for vocab_item in self.vocab_by_len[cand_len]:
                d = damerau_levenshtein_distance(clean, vocab_item, max_dist=max_allowed_dist)
                if d < min_dist:
                    min_dist = d
                    best_match = vocab_item

        if best_match is not None and min_dist <= max_allowed_dist:
            prefix_match = re.match(r"^[^\w]+", token)
            suffix_match = re.search(r"[^\w]+$", token)
            prefix_punc = prefix_match.group(0) if prefix_match else ""
            suffix_punc = suffix_match.group(0) if suffix_match else ""
            return f"{prefix_punc}{best_match}{suffix_punc}", best_match, min_dist

        return token, None, 0

    def normalize(self, query: str) -> NormalizationResult:
        t0 = time.perf_counter()
        if not self._initialized:
            self.initialize()

        raw_query = str(query or "").strip()
        if not raw_query:
            return NormalizationResult(original_query="", normalized_query="", traces=[], latency_ms=0.0)

        traces: list[NormalizationTrace] = []
        normalized_text = raw_query

        # Tahap 1: Penggantian frasa fonetik Speech-to-Text (STT) multi-word
        for pattern, replacement in STT_PHONETIC_PATTERNS:
            match = pattern.search(normalized_text)
            if match:
                original_phrase = match.group(0)
                normalized_text = pattern.sub(replacement, normalized_text)
                traces.append(
                    NormalizationTrace(
                        original=original_phrase,
                        normalized=replacement,
                        rule_type="STT_PHONETIC",
                        distance=0,
                    )
                )

        # Tahap 2: Pemrosesan token leksikal (Kamus slang O(1) & Bounded Levenshtein)
        tokens = normalized_text.split()
        normalized_tokens: list[str] = []

        for token in tokens:
            prefix_match = re.match(r"^[^\w]+", token)
            suffix_match = re.search(r"[^\w]+$", token)
            prefix_punc = prefix_match.group(0) if prefix_match else ""
            suffix_punc = suffix_match.group(0) if suffix_match else ""
            core = re.sub(r"^[^\w]+|[^\w]+$", "", token)

            if not core:
                normalized_tokens.append(token)
                continue

            core_lower = core.lower()

            # 2a. Kamus leksikal O(1)
            if core_lower in SLANG_LEXICON:
                rep = SLANG_LEXICON[core_lower]
                normalized_tokens.append(f"{prefix_punc}{rep}{suffix_punc}")
                traces.append(
                    NormalizationTrace(
                        original=core,
                        normalized=rep,
                        rule_type="SLANG_LEXICON",
                        distance=0,
                    )
                )
                continue

            # 2b. Koreksi format kode mata kuliah
            corr_code, code_dist = self._correct_course_code(core)
            if corr_code is not None and code_dist <= 2:
                normalized_tokens.append(f"{prefix_punc}{corr_code}{suffix_punc}")
                traces.append(
                    NormalizationTrace(
                        original=core,
                        normalized=corr_code,
                        rule_type="LEVENSHTEIN_CODE",
                        distance=code_dist,
                    )
                )
                continue

            # 2c. Koreksi kata typo via Bounded Damerau-Levenshtein
            corr_token, target_w, word_dist = self._correct_word_levenshtein(token)
            if target_w is not None:
                normalized_tokens.append(corr_token)
                traces.append(
                    NormalizationTrace(
                        original=core,
                        normalized=target_w,
                        rule_type="LEVENSHTEIN_TOKEN",
                        distance=word_dist,
                    )
                )
            else:
                normalized_tokens.append(token)

        final_query = " ".join(normalized_tokens)
        final_query = re.sub(r"\s+", " ", final_query).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return NormalizationResult(
            original_query=raw_query,
            normalized_query=final_query,
            traces=traces,
            latency_ms=round(latency_ms, 3),
        )


_normalizer_instance: AcademicNormalizer | None = None


def get_academic_normalizer() -> AcademicNormalizer:
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = AcademicNormalizer()
        _normalizer_instance.initialize()
    return _normalizer_instance


def normalize_query(query: str) -> str:
    """Helper cepat normalisasi teks kueri akademik."""
    return get_academic_normalizer().normalize(query).normalized_query
