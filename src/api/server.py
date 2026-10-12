"""
FastAPI Backend Server untuk Chatbot FTMM COMPASS.
Melayani antarmuka obrolan di ~/project/ftmm-compass-ui/src/pages/Chatbot.tsx.
Menggunakan arsitektur Hybrid RAG-CAG (Ollama + FAISS + NLU Router).
"""

import os
import time
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StudentProfile,
    DegreePlanPayload,
    DegreePlanSummary,
)
from src.engine import UnifiedChatEngine, RoutingMode

app = FastAPI(
    title="FTMM COMPASS Chatbot Backend",
    description="Backend API untuk Chatbot Akademik FTMM COMPASS dengan arsitektur Hybrid RAG-CAG.",
    version="1.0.0",
)

# CORS Middleware agar bisa dipanggil langsung dari Vite dev server (port 5173 dsb.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = UnifiedChatEngine()


def _get_ollama_base_url() -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        return f"http://{host}"
    return host.rstrip("/")


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Endpoint status kesehatan backend dan konektivitas Ollama."""
    ollama_online = False
    base_url = _get_ollama_base_url()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                ollama_online = True
    except Exception:
        ollama_online = False

    return HealthResponse(
        status="healthy",
        service="FTMM COMPASS Chatbot Backend (Unified ReAct & Guardrails)",
        ollama_online=ollama_online,
        default_model=engine.model,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint dialog chat utama yang melayani pesan dari frontend ftmm-compass-ui.
    """
    t0 = time.perf_counter()
    msg = request.message.strip()
    profile = request.student_profile
    history = [{"role": m.role, "content": m.content} for m in request.history]
    if not request.history:
        engine.memory.clear_session("client_session")


    # Eksekusi engine terpadu UnifiedChatEngine
    result = engine.generate_answer(
        query=msg,
        history=history,
        student_profile=profile.model_dump(),
        session_id="client_session",
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Sync updated profile dari metadata engine
    updated_prof_data = result.metadata.get("updated_profile", {})
    if updated_prof_data:
        if updated_prof_data.get("program_studi"):
            profile.program_studi = updated_prof_data["program_studi"]
        if updated_prof_data.get("semester_saat_ini"):
            profile.semester_saat_ini = updated_prof_data["semester_saat_ini"]
        if updated_prof_data.get("minat_fokus"):
            profile.minat_fokus = updated_prof_data["minat_fokus"]
        if updated_prof_data.get("riwayat_matkul_lulus"):
            profile.riwayat_matkul_lulus = updated_prof_data["riwayat_matkul_lulus"]

    # Action type & Plan Payload
    action_type = result.metadata.get("action_type", "GENERAL")
    plan_raw = result.metadata.get("plan_payload")
    plan_payload: DegreePlanPayload | None = None
    if plan_raw and isinstance(plan_raw, dict) and "summary" in plan_raw and "plan" in plan_raw:
        try:
            summary_raw = plan_raw.get("summary", {})
            plan_payload = DegreePlanPayload(
                action="APPLY_DEGREE_PLAN",
                summary=DegreePlanSummary(
                    program_studi=summary_raw.get("program_studi", profile.program_studi or "Teknologi Sains Data"),
                    current_semester=summary_raw.get("current_semester", profile.semester_saat_ini or 1),
                    target_semester=summary_raw.get("target_semester", profile.target_kelulusan_semester),
                    total_credits=summary_raw.get("total_credits", 144),
                    focus_tracks=summary_raw.get("focus_tracks", profile.minat_fokus or ["Umum"]),
                    note=summary_raw.get("note", "Rencana studi rekomendasi AI berdasarkan kurikulum resmi dan prasyarat FTMM."),
                ),
                plan={int(k): v for k, v in plan_raw.get("plan", {}).items()},
            )
            action_type = "PLAN_GENERATED"
        except Exception:
            plan_payload = None
    elif result.routing_mode == "HYBRID" and profile.program_studi and profile.semester_saat_ini:
        action_type = "PLAN_GENERATED"
        plan_payload = DegreePlanPayload(
            action="APPLY_DEGREE_PLAN",
            summary=DegreePlanSummary(
                program_studi=profile.program_studi,
                current_semester=profile.semester_saat_ini,
                target_semester=profile.target_kelulusan_semester,
                total_credits=144,
                focus_tracks=profile.minat_fokus or ["Umum"],
                note="Rencana studi rekomendasi AI berdasarkan kurikulum resmi dan prasyarat FTMM.",
            ),
            plan={profile.semester_saat_ini + 1: []},
        )
    return ChatResponse(
        reply=result.answer,
        updated_profile=profile,
        action_type=action_type,
        plan_payload=plan_payload,
        routing_mode=result.routing_mode,
        latency_ms=latency_ms,
    )
