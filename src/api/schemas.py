"""
Pydantic Schemas untuk REST API Chatbot FTMM COMPASS.
Kompatibel 100% dengan kontrak frontend di ~/project/ftmm-compass-ui/src/pages/Chatbot.tsx.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


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


class ChatResponse(BaseModel):
    reply: str
    updated_profile: StudentProfile
    action_type: str = "GENERAL"  # "NEED_INFO" | "INFO_CONFIRMED" | "PLAN_GENERATED" | "GENERAL"
    plan_payload: DegreePlanPayload | None = None
    routing_mode: str = "HYBRID"
    latency_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "FTMM COMPASS Chatbot Backend"
    ollama_online: bool = True
    default_model: str = "qwen2.5:3b"
