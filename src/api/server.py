"""
FastAPI Backend Server untuk Chatbot FTMM COMPASS.
Melayani antarmuka obrolan di ~/project/ftmm-compass-ui/src/pages/Chatbot.tsx.
Menggunakan arsitektur Hybrid RAG-CAG (Ollama + FAISS + NLU Router).
"""

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
from src.engine import HybridRAGCAGEngine, RoutingMode

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

engine = HybridRAGCAGEngine()


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Endpoint status kesehatan backend dan konektivitas Ollama."""
    ollama_online = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
            if resp.status_code == 200:
                ollama_online = True
    except Exception:
        ollama_online = False

    return HealthResponse(
        status="healthy",
        service="FTMM COMPASS Chatbot Backend (Hybrid RAG-CAG)",
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

    # Slot-filling update dari NLU Router entity extractor
    entities = engine.router.extractor.extract(msg)
    if entities.program_studi:
        profile.program_studi = entities.program_studi
    if entities.semester:
        profile.semester_saat_ini = entities.semester
    if entities.career_interests:
        for ci in entities.career_interests:
            if ci not in profile.minat_fokus:
                profile.minat_fokus.append(ci.title())

    # Eksekusi engine Hybrid RAG-CAG
    result = engine.generate_answer(
        query=msg,
        history=history,
        student_profile=profile.model_dump(),
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    action_type = "GENERAL"
    plan_payload: DegreePlanPayload | None = None

    # Jika mode adalah HYBRID (Study Planning) dan mahasiswa memiliki profil dasar
    if result.routing_mode == "HYBRID" and profile.program_studi and profile.semester_saat_ini:
        action_type = "PLAN_GENERATED"
        # Buat dummy/valid structured plan payload agar frontend bisa merender tombol Apply Plan
        plan_payload = DegreePlanPayload(
            action="APPLY_DEGREE_PLAN",
            summary=DegreePlanSummary(
                program_studi=profile.program_studi,
                current_semester=profile.semester_saat_ini,
                target_semester=profile.target_kelulusan_semester,
                total_credits=144,
                focus_tracks=profile.minat_fokus or ["Umum"],
                note="Rencana studi rekomendasi AI berdasarkan kurikulum resmi dan prasyarat FTMM."
            ),
            plan={
                profile.semester_saat_ini + 1: [
                    {"code": r.get("course", {}).get("kode_mk", "MK"), "name": r.get("course", {}).get("nama_mk", "Mata Kuliah Pilihan"), "credits": 3}
                    for r in engine.retriever.retrieve(msg, top_k=4, prodi_filter=profile.program_studi)
                ]
            }
        )

    return ChatResponse(
        reply=result.answer,
        updated_profile=profile,
        action_type=action_type,
        plan_payload=plan_payload,
        routing_mode=result.routing_mode,
        latency_ms=latency_ms,
    )
