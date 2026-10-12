"""
Pydantic Schemas untuk REST API Chatbot FTMM COMPASS.
Kompatibel 100% dengan kontrak frontend di ~/project/ftmm-compass-ui/src/pages/Chatbot.tsx.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str
    code: str = "invalid_input"


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ApiErrorResponse(BaseModel):
    error: ErrorPayload


class PaginationMeta(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int


class PaginationLinks(BaseModel):
    self: str
    next: str | None = None
    prev: str | None = None
    first: str | None = None
    last: str | None = None


class StudentProfile(BaseModel):
    program_studi: str | None = None
    semester_saat_ini: int | None = None
    riwayat_matkul_lulus: list[str] = Field(default_factory=list)
    minat_fokus: list[str] = Field(default_factory=list)
    target_kelulusan_semester: int = 8
    maks_sks_per_semester: int = 24
    is_confirmed_by_user: bool = False


class DegreePlanSummary(BaseModel):
    program_studi: str
    current_semester: int
    target_semester: int
    total_credits: int
    focus_tracks: list[str] = Field(default_factory=list)
    note: str | None = None


class DegreePlanPayload(BaseModel):
    action: Literal["APPLY_DEGREE_PLAN"] = "APPLY_DEGREE_PLAN"
    summary: DegreePlanSummary
    plan: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)


class ChatMessageItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessageItem] = Field(default_factory=list)
    student_profile: StudentProfile = Field(default_factory=StudentProfile)
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str
    updated_profile: StudentProfile
    action_type: str = "GENERAL"  # "NEED_INFO" | "INFO_CONFIRMED" | "PLAN_GENERATED" | "GENERAL"
    plan_payload: DegreePlanPayload | None = None
    routing_mode: str = "HYBRID"
    latency_ms: float = 0.0


class OllamaHealthDetail(BaseModel):
    online: bool
    host: str
    default_model: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "FTMM COMPASS Chatbot Backend"
    ollama_online: bool = True
    default_model: str = "qwen2.5:3b"
    ollama: OllamaHealthDetail | None = None


class CourseItem(BaseModel):
    id: str
    code: str | None = None
    name: str
    credits: int = 3
    semester: int = 1
    type: Literal["Wajib", "Pilihan"] = "Pilihan"
    parity: Literal["odd", "even"] = "odd"
    desc: str = ""
    program: str
    prerequisites: list[str] = Field(default_factory=list)


class CourseCatalogResponse(BaseModel):
    program_studi: str | None = None
    total_courses: int
    programs: list[str]
    courses: list[CourseItem]


class PaginatedCourseResponse(BaseModel):
    data: list[CourseItem]
    meta: PaginationMeta
    links: PaginationLinks
    program_studi: str | None = None

class PlanValidationRequest(BaseModel):
    plan: dict[int, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Rancangan mata kuliah semester 1-8 hasil plotting mahasiswa atau AI.",
    )
    riwayat_matkul_lulus: list[str] = Field(
        default_factory=list,
        description="Daftar nama/kode mata kuliah yang telah diselesaikan mahasiswa.",
    )
    maks_sks_per_semester: int = Field(
        default=24,
        description="Batas beban SKS maksimum yang diperbolehkan.",
    )
    program_studi: str | None = Field(
        default=None,
        description="Program studi mahasiswa untuk filter validasi kurikulum spesifik.",
    )


class PlanValidationResponse(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_sks_planned: int = 0
    details_per_semester: dict[int, dict[str, Any]] = Field(default_factory=dict)


class CourseSearchItem(BaseModel):
    id: str
    code: str | None = None
    name: str
    credits: int
    semester: int
    type: Literal["Wajib", "Pilihan"]
    parity: Literal["odd", "even"]
    desc: str = ""
    program: str
    prerequisites: list[str] = Field(default_factory=list)
    score: float = 0.0


class CourseSearchResponse(BaseModel):
    query: str
    program_studi: str | None = None
    total_hits: int
    results: list[CourseSearchItem]
