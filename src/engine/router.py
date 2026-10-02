"""
Modul NLU Router (Natural Language Understanding & Routing) FTMM COMPASS.
Terdiri dari:
1. EntityExtractor: Ekstraksi kode MK, prodi, semester, SOP, minat karir.
2. IntentClassifier: Klasifikasi niatan kueri ke 5 kategori (CAG, RAG, Hybrid, Direct, Guarded).
3. HybridRouter: Menentukan rute pengambilan pengetahuan terbaik berdasarkan intensi dan entitas.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    REGULATION_FAQ = "REGULATION_FAQ"        # Aturan SKS, PKL, Skripsi, Yudisium, SOP administrasi -> CAG
    COURSE_SYLLABUS = "COURSE_SYLLABUS"      # Silabus matkul, deskripsi, dosen, prasyarat matkul -> RAG
    STUDY_PLANNING = "STUDY_PLANNING"        # Rekomendasi matkul, rencana studi, KRS, peminatan -> HYBRID
    GENERAL_CHAT = "GENERAL_CHAT"            # Sapaan, terima kasih, perkenalan -> DIRECT
    OUT_OF_SCOPE = "OUT_OF_SCOPE"            # Topik di luar FTMM UNAIR / prompt injection -> GUARDED


class RoutingMode(str, Enum):
    CAG = "CAG"          # 0ms retrieval, fixed cached system prompt
    RAG = "RAG"          # FAISS dense similarity search
    HYBRID = "HYBRID"    # CAG (rules & SKS caps) + RAG (candidate courses)
    DIRECT = "DIRECT"    # Direct LLM without retrieval
    GUARDED = "GUARDED"  # Safety refusal / redirection


@dataclass
class ExtractedEntities:
    course_codes: list[str] = field(default_factory=list)
    program_studi: str | None = None
    semester: int | None = None
    sop_keywords: list[str] = field(default_factory=list)
    career_interests: list[str] = field(default_factory=list)
    raw_query: str = ""


@dataclass
class RoutingDecision:
    intent: IntentType
    routing_mode: RoutingMode
    entities: ExtractedEntities
    confidence: float
    explanation: str


class EntityExtractor:
    """Ekstraksi entitas kunci akademik FTMM berbasis pola ekspresi reguler dan leksikon."""
    
    COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{3}\s?[0-9]{3})\b", re.IGNORECASE)
    SEMESTER_PATTERN = re.compile(r"\b(?:semester|sem)\s*([1-8])\b", re.IGNORECASE)
    
    PRODI_MAP = {
        "Teknologi Sains Data": ["sains data", "data science", "tsd", "ds"],
        "Teknik Robotika dan Kecerdasan Buatan": ["robotika", "trkb", "robotics", "ai", "kecerdasan buatan"],
        "Rekayasa Nanoteknologi": ["nanoteknologi", "nanotek", "rn", "nano"],
        "Teknik Industri": ["teknik industri", "industri", "ti", "industrial"],
        "Teknik Elektro": ["teknik elektro", "elektro", "te", "electrical"],
    }
    
    SOP_KEYWORDS = {
        "pkl": ["pkl", "magang", "kerja praktek", "praktik kerja lapangan", "kerja praktik"],
        "skripsi": ["skripsi", "tugas akhir", "ta", "proposal skripsi", "sempro"],
        "yudisium": ["yudisium", "kelulusan", "wisuda", "syarat lulus", "beban lulus"],
        "sks_limit": ["batas sks", "beban studi", "ips", "ipk", "maksimal sks", "sks maksimal", "maks sks", "jatah sks"],
        "cuti": ["cuti", "cuti akademik", "ijin libur semester"],
        "skma": ["skma", "surat keterangan mahasiswa aktif", "surat aktif"],
        "rekomendasi": ["surat rekomendasi", "rekomendasi magang", "rekomendasi beasiswa", "msib"],
        "ukt": ["telat ukt", "terlambat bayar", "bayar ukt", "tunggakan ukt", "daftar ulang"],
        "ktm": ["ktm hilang", "pengantar ktm", "kartu tanda mahasiswa hilang"],
    }
    
    CAREER_KEYWORDS = [
        "machine learning", "ml", "deep learning", "computer vision", "nlp",
        "data analyst", "data scientist", "data engineer", "big data",
        "iot", "robotika", "embedded", "autonomous",
        "supply chain", "optimasi", "manajemen produksi",
        "nanomaterial", "nanomedis", "telekomunikasi", "power system"
    ]

    def extract(self, query: str) -> ExtractedEntities:
        raw = query.strip()
        lower = raw.lower()
        
        # 1. Course Codes
        codes = []
        for match in self.COURSE_CODE_PATTERN.finditer(raw):
            clean_code = match.group(1).upper().replace(" ", "")
            codes.append(clean_code)
            
        # 2. Semester
        sem = None
        sem_match = self.SEMESTER_PATTERN.search(lower)
        if sem_match:
            sem = int(sem_match.group(1))
            
        # 3. Prodi
        detected_prodi = None
        for canonical, aliases in self.PRODI_MAP.items():
            for alias in aliases:
                # word boundary match to avoid partial matches
                if re.search(r"\b" + re.escape(alias) + r"\b", lower):
                    detected_prodi = canonical
                    break
            if detected_prodi:
                break
                
        # 4. SOP keywords
        detected_sops = []
        for sop_label, keywords in self.SOP_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    detected_sops.append(sop_label)
                    break
                    
        # 5. Career interests
        detected_careers = []
        for ck in self.CAREER_KEYWORDS:
            if ck in lower:
                detected_careers.append(ck)
                
        return ExtractedEntities(
            course_codes=codes,
            program_studi=detected_prodi,
            semester=sem,
            sop_keywords=detected_sops,
            career_interests=detected_careers,
            raw_query=raw,
        )

class IntentClassifier:
    """Pengklasifikasi intensi pengguna berdasarkan analisis leksikal dan semantik."""
    
    GREETING_PATTERN = re.compile(
        r"^(halo|hai|hi|hello|selamat pagi|selamat siang|selamat sore|selamat malam|assalamualaikum|tes|ping|terima kasih|makasih|thanks)\b",
        re.IGNORECASE
    )
    
    OUT_OF_SCOPE_TRIGGERS = [
        "resep", "masak", "makanan", "restoran", "kuliner", "rendang", "sapi", "ayam", "kue",
        "siapa presiden", "pemilu", "partai", "politik", "pilpres", "gubernur",
        "ramalan cuaca", "buatkan puisi", "cerpen", "film", "lagu", "game", "sepak bola",
        "hack", "jailbreak", "ignore previous instructions", "bypass system"
    ]
    
    PLANNING_TRIGGERS = [
        "rekomendasi matkul", "rekomendasikan matkul", "ambil matkul apa",
        "susun krs", "susun semester", "rencana studi", "study plan",
        "rekomendasi krs", "saran matkul", "pilihan matkul", "mau fokus ke",
        "tertarik ke", "ingin jadi", "minat saya", "rekomendasi sks"
    ]
    
    SYLLABUS_TRIGGERS = [
        "silabus", "materi", "membahas tentang", "belajar apa", "deskripsi matkul",
        "dosen pengampu", "siapa dosen", "prasyarat matkul", "syarat matkul",
        "apa isi matkul", "mata kuliah", "matkul"
    ]
    def classify(self, query: str, entities: ExtractedEntities) -> tuple[IntentType, float, str]:
        lower = query.lower().strip()
        
        # 1. Out of Scope Check
        for oos in self.OUT_OF_SCOPE_TRIGGERS:
            if oos in lower:
                return IntentType.OUT_OF_SCOPE, 0.95, f"Terdeteksi kata kunci di luar ranah akademik FTMM: '{oos}'"

        # 2. Greeting / General Chat Check (short queries)
        if len(lower.split()) <= 4 and self.GREETING_PATTERN.search(lower):
            return IntentType.GENERAL_CHAT, 0.98, "Pesan berupa sapaan/kesopanan umum."

        # 3. Planning / Career Advisory Check (Hybrid Candidate)
        for trigger in self.PLANNING_TRIGGERS:
            if trigger in lower or (entities.career_interests and entities.semester):
                return IntentType.STUDY_PLANNING, 0.90, f"Terdeteksi niatan perencanaan studi: '{trigger}'"

        # 4. Specific Course Syllabus Check (RAG Candidate)
        if entities.course_codes:
            # If explicit course code mentioned without general SOP, highly likely syllabus
            if not entities.sop_keywords:
                return IntentType.COURSE_SYLLABUS, 0.95, f"Menyebutkan kode matkul spesifik: {entities.course_codes}"
                
        for trigger in self.SYLLABUS_TRIGGERS:
            if trigger in lower and not entities.sop_keywords:
                return IntentType.COURSE_SYLLABUS, 0.85, f"Menanyakan rincian silabus/materi: '{trigger}'"

        # 5. Regulation / SOP Check (CAG Candidate)
        if entities.sop_keywords:
            return IntentType.REGULATION_FAQ, 0.92, f"Menanyakan aturan/SOP akademik: {entities.sop_keywords}"

        # 6. Fallback Check: pastikan ada konteks akademik FTMM
        academic_context_words = [
            "ftmm", "unair", "kampus", "kuliah", "akademik", "sks", "dosen", "mata kuliah", "matkul",
            "semester", "mahasiswa", "lulus", "daftar", "bayar", "ukt", "ipk", "ips", "krs", "prasyarat"
        ]
        has_academic_context = any(w in lower for w in academic_context_words)
        
        if any(w in lower for w in ["aturan", "syarat", "prosedur", "surat", "cara", "bagaimana"]) and has_academic_context:
            return IntentType.REGULATION_FAQ, 0.80, "Pertanyaan seputar tata cara atau prosedur akademik FTMM."

        if not has_academic_context and not entities.course_codes and not entities.sop_keywords:
            return IntentType.OUT_OF_SCOPE, 0.70, "Pertanyaan tidak teridentifikasi memiliki konteks kurikulum/akademik FTMM."

        # Default fallback to RAG for exploratory questions
        return IntentType.COURSE_SYLLABUS, 0.60, "Pertanyaan informasional kurikulum umum."


class HybridRouter:
    """Router orkestrator yang memetakan kueri ke rute eksekusi."""
    
    def __init__(self):
        self.extractor = EntityExtractor()
        self.classifier = IntentClassifier()

    def route(self, query: str) -> RoutingDecision:
        entities = self.extractor.extract(query)
        intent, confidence, explanation = self.classifier.classify(query, entities)

        if intent == IntentType.OUT_OF_SCOPE:
            mode = RoutingMode.GUARDED
        elif intent == IntentType.GENERAL_CHAT:
            mode = RoutingMode.DIRECT
        elif intent == IntentType.REGULATION_FAQ:
            mode = RoutingMode.CAG
        elif intent == IntentType.COURSE_SYLLABUS:
            mode = RoutingMode.RAG
        elif intent == IntentType.STUDY_PLANNING:
            mode = RoutingMode.HYBRID
        else:
            mode = RoutingMode.RAG

        return RoutingDecision(
            intent=intent,
            routing_mode=mode,
            entities=entities,
            confidence=confidence,
            explanation=explanation,
        )


# Global singleton instance
_router_instance: HybridRouter | None = None


def get_router() -> HybridRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = HybridRouter()
    return _router_instance
