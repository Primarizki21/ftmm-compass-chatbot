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
    """Ekstraksi entitas kunci akademik FTMM berbasis DenseGazetteerLinker dan pola deterministik."""

    def __init__(self, gazetteer: Any = None):
        from src.engine.memory import get_dense_gazetteer
        self.gazetteer = gazetteer or get_dense_gazetteer()

    def extract(self, query: str, query_vec: Any = None) -> ExtractedEntities:
        from src.engine.memory import COURSE_CODE_PATTERN, extract_semester_from_text
        raw = query.strip()

        codes = []
        for match in COURSE_CODE_PATTERN.finditer(raw):
            codes.append(match.group(1).upper().replace(" ", ""))

        sem = extract_semester_from_text(raw)
        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(raw)
        detected_prodi = self.gazetteer.match_prodi(q_vec)

        detected_careers = []
        focus = self.gazetteer.match_focus(q_vec)
        if focus:
            detected_careers.append(focus)

        return ExtractedEntities(
            course_codes=codes,
            program_studi=detected_prodi,
            semester=sem,
            sop_keywords=[],
            career_interests=detected_careers,
            raw_query=raw,
        )

class IntentClassifier:
    """Pengklasifikasi intensi pengguna berbasis System-1 Decider semantik."""

    def __init__(self, decider: Any = None):
        from src.engine.decider import System1Decider
        self.decider = decider or System1Decider()
        self.decider.initialize()

    def classify(
        self, query: str, entities: ExtractedEntities, query_vec: Any = None
    ) -> tuple[IntentType, float, str]:
        from src.engine.decider import DecisionCategory
        res = self.decider.decide(query, query_vec=query_vec)

        if res.category == DecisionCategory.CHAT_ONLY:
            return IntentType.GENERAL_CHAT, res.confidence, "Klasifikasi semantik: General Chat."
        elif res.category == DecisionCategory.REGULATION_SOP:
            return IntentType.REGULATION_FAQ, res.confidence, "Klasifikasi semantik: Regulasi & SOP FTMM."
        elif res.category in (DecisionCategory.COURSE_DISCOVERY, DecisionCategory.PREREQUISITE):
            return IntentType.COURSE_SYLLABUS, res.confidence, "Klasifikasi semantik: Silabus / Prasyarat Mata Kuliah."
        elif res.category in (DecisionCategory.DEGREE_PLANNING, DecisionCategory.SKS_CALCULATION):
            return IntentType.STUDY_PLANNING, res.confidence, "Klasifikasi semantik: Perencanaan Studi."
        else:
            return IntentType.COURSE_SYLLABUS, res.confidence, "Klasifikasi fallback: Eksplorasi Kurikulum."

class HybridRouter:
    """Router orkestrator yang memetakan kueri ke rute eksekusi."""

    def __init__(self, decider: Any = None, gazetteer: Any = None):
        self.extractor = EntityExtractor(gazetteer=gazetteer)
        self.classifier = IntentClassifier(decider=decider)

    def route(self, query: str, query_vec: Any = None) -> RoutingDecision:
        entities = self.extractor.extract(query, query_vec=query_vec)
        intent, confidence, explanation = self.classifier.classify(
            query, entities, query_vec=query_vec
        )

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
