"""
FastAPI Backend Server untuk Chatbot FTMM COMPASS.
Melayani antarmuka obrolan di ~/project/ftmm-compass-ui/src/pages/Chatbot.tsx.
Menggunakan arsitektur Hybrid RAG-CAG (Ollama + FAISS + NLU Router).
"""

import os
import time
import json
import asyncio
from typing import Any, AsyncGenerator
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StudentProfile,
    DegreePlanPayload,
    DegreePlanSummary,
    CourseItem,
    CourseCatalogResponse,
    PlanValidationRequest,
    PlanValidationResponse,
    CourseSearchItem,
    CourseSearchResponse,
)
from src.engine import UnifiedChatEngine, RoutingMode
from src.engine.tools.academic_tools import (
    _get_catalog,
    _get_retriever,
    normalize_program_studi,
    validate_prerequisites_dag,
    generate_degree_plan,
    get_courses_by_semester,
    search_courses,
)
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


def _normalize_course_type(jenis: str | None) -> str:
    j = str(jenis or "").lower()
    if not j or "wajib" in j:
        return "Wajib"
    if "pilih" in j or "pengayaan" in j or "peminatan" in j:
        return "Pilihan"
    return "Wajib"


@app.get("/api/courses", response_model=CourseCatalogResponse)
async def list_courses(
    program_studi: str | None = Query(None, description="Nama program studi FTMM."),
    prodi: str | None = Query(None, description="Alias nama prodi dari frontend UI."),
    semester: int | None = Query(None, description="Filter semester (1-8)."),
):
    """
    Mengambil katalog mata kuliah resmi FTMM dari SSOT kurikulum.
    Mendukung filter program studi via parameter 'program_studi' maupun alias 'prodi'.
    """
    target_prodi = prodi or program_studi
    canonical_prodi = normalize_program_studi(target_prodi) if target_prodi else None
    raw_catalog = _get_catalog()
    programs_set = sorted(list(set(c.get("program_studi", "") for c in raw_catalog if c.get("program_studi"))))

    courses: list[CourseItem] = []
    for c in raw_catalog:
        c_prodi = c.get("program_studi", "")
        if canonical_prodi:
            if c_prodi != canonical_prodi and normalize_program_studi(c_prodi) != canonical_prodi:
                continue

        c_sem = int(c.get("semester", 1) or 1)
        if semester is not None and c_sem != semester:
            continue

        c_id = str(c.get("id") or c.get("kode_mk") or "")
        code = str(c.get("kode_mk") or "").strip() or None
        name = str(c.get("nama_mk") or "").strip()
        if not name:
            continue

        credits = int(c.get("sks", 3) or 3)
        c_type = _normalize_course_type(c.get("jenis"))
        parity = "odd" if c_sem % 2 == 1 else "even"
        desc = str(c.get("deskripsi") or "").strip()
        prereqs = [str(p) for p in c.get("prasyarat", []) if p]

        courses.append(
            CourseItem(
                id=c_id,
                code=code,
                name=name,
                credits=credits,
                semester=c_sem,
                type=c_type,  # type: ignore[arg-type]
                parity=parity,  # type: ignore[arg-type]
                desc=desc,
                program=c_prodi,
                prerequisites=prereqs,
            )
        )

    return CourseCatalogResponse(
        program_studi=canonical_prodi,
        total_courses=len(courses),
        programs=programs_set,
        courses=courses,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint dialog chat utama yang melayani pesan dari frontend ftmm-compass-ui.
    Mendukung isolasi multi-sesi via parameter request.session_id.
    """
    t0 = time.perf_counter()
    msg = request.message.strip()
    profile = request.student_profile
    history = [{"role": m.role, "content": m.content} for m in request.history]
    session_id = request.session_id or "client_session"
    if not request.history:
        engine.memory.clear_session(session_id)

    # Eksekusi engine terpadu UnifiedChatEngine
    result = engine.generate_answer(
        query=msg,
        history=history,
        student_profile=profile.model_dump(),
        session_id=session_id,
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


@app.post("/api/plan/validate", response_model=PlanValidationResponse)
async def validate_degree_plan_endpoint(request: PlanValidationRequest):
    """
    Memvalidasi seluruh urutan rencana studi mahasiswa terhadap aturan:
    1. Graf dependensi prasyarat (Prerequisite DAG kurikulum resmi FTMM)
    2. Paritas semester (Ganjil/Genap)
    3. Batas beban SKS maksimum per semester
    4. Integritas duplikasi mata kuliah
    """
    catalog = _get_catalog()
    course_lookup = {c["kode_mk"].strip().upper(): c for c in catalog if c.get("kode_mk")}
    name_lookup = {c["nama_mk"].strip().lower(): c for c in catalog if c.get("nama_mk")}

    errors: list[str] = []
    warnings: list[str] = []
    total_sks_planned = 0
    details_per_semester: dict[int, dict[str, Any]] = {}

    passed_courses_set = set(str(p).strip().lower() for p in request.riwayat_matkul_lulus)
    for p in request.riwayat_matkul_lulus:
        p_clean = str(p).strip().upper()
        if p_clean in course_lookup:
            passed_courses_set.add(course_lookup[p_clean]["nama_mk"].lower())

    accumulated_completed = set(passed_courses_set)
    all_scheduled_codes: dict[str, int] = {}

    for sem_key, course_list in sorted(request.plan.items(), key=lambda x: int(x[0])):
        sem_num = int(sem_key)
        sem_sks = 0
        sem_errors: list[str] = []
        expected_parity = "odd" if sem_num % 2 == 1 else "even"

        for item in course_list:
            c_code = str(item.get("code") or item.get("id") or "").strip().upper()
            c_name = str(item.get("name") or "").strip()
            c_sks = int(item.get("credits") or 3)

            ref_course = course_lookup.get(c_code) or name_lookup.get(c_name.lower())
            if ref_course:
                c_sks = int(ref_course.get("sks", c_sks))
                c_name = ref_course.get("nama_mk", c_name)
                c_code = ref_course.get("kode_mk", c_code)

            sem_sks += c_sks
            total_sks_planned += c_sks

            if c_code:
                if c_code in all_scheduled_codes:
                    sem_errors.append(
                        f"Mata kuliah {c_name} ({c_code}) dijadwalkan ganda pada semester {all_scheduled_codes[c_code]} dan {sem_num}."
                    )
                all_scheduled_codes[c_code] = sem_num

            course_parity = item.get("parity")
            if not course_parity and ref_course:
                ref_sem = int(ref_course.get("semester", 1))
                course_parity = "odd" if ref_sem % 2 == 1 else "even"

            if course_parity and course_parity != expected_parity:
                warnings.append(
                    f"Mata kuliah {c_name} memiliki paritas {course_parity}, dijadwalkan pada semester {sem_num} ({expected_parity})."
                )

            dag_res = validate_prerequisites_dag(
                course_code_or_name=c_code or c_name,
                passed_courses=list(accumulated_completed),
            )
            if dag_res.get("found") and dag_res.get("has_prerequisites") and not dag_res.get("is_eligible"):
                missing = dag_res.get("missing_prerequisites", [])
                sem_errors.append(
                    f"Mata kuliah {c_name} di semester {sem_num} belum memenuhi prasyarat: {', '.join(missing)}."
                )

        if sem_sks > request.maks_sks_per_semester:
            sem_errors.append(
                f"Beban studi semester {sem_num} ({sem_sks} SKS) melebihi batas maksimum {request.maks_sks_per_semester} SKS."
            )

        for item in course_list:
            c_code = str(item.get("code") or item.get("id") or "").strip().lower()
            c_name = str(item.get("name") or "").strip().lower()
            if c_code:
                accumulated_completed.add(c_code)
            if c_name:
                accumulated_completed.add(c_name)

        details_per_semester[sem_num] = {
            "total_sks": sem_sks,
            "course_count": len(course_list),
            "errors": sem_errors,
        }
        errors.extend(sem_errors)

    return PlanValidationResponse(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        total_sks_planned=total_sks_planned,
        details_per_semester=details_per_semester,
    )


@app.post("/api/plan/generate", response_model=DegreePlanPayload)
async def generate_plan_direct_endpoint(profile: StudentProfile):
    """
    Menghasilkan rancangan rencana studi (DegreePlanPayload) secara deterministik
    langsung dari profil mahasiswa tanpa melalui proses chat LLM.
    """
    prodi = profile.program_studi or "Teknologi Sains Data"
    cur_sem = profile.semester_saat_ini or 1
    target_sem = profile.target_kelulusan_semester or 8
    max_sks = profile.maks_sks_per_semester or 24

    raw_plan = generate_degree_plan(
        program_studi=prodi,
        current_semester=cur_sem,
        target_semester=target_sem,
        passed_courses=profile.riwayat_matkul_lulus,
        focus_tracks=profile.minat_fokus,
        max_sks_per_sem=max_sks,
    )

    summary_raw = raw_plan.get("summary", {})
    return DegreePlanPayload(
        action="APPLY_DEGREE_PLAN",
        summary=DegreePlanSummary(
            program_studi=summary_raw.get("program_studi", prodi),
            current_semester=summary_raw.get("current_semester", cur_sem),
            target_semester=summary_raw.get("target_semester", target_sem),
            total_credits=summary_raw.get("total_credits", 144),
            focus_tracks=summary_raw.get("focus_tracks", profile.minat_fokus or ["Umum"]),
            note=summary_raw.get("note", "Rencana studi deterministik terverifikasi kurikulum FTMM."),
        ),
        plan={int(k): v for k, v in raw_plan.get("plan", {}).items()},
    )


@app.get("/api/courses/search", response_model=CourseSearchResponse)
async def search_courses_endpoint(
    q: str = Query(..., min_length=1, description="Kata kunci, silabus, atau nama mata kuliah."),
    program_studi: str | None = Query(None, description="Filter program studi opsional."),
    prodi: str | None = Query(None, description="Alias prodi opsional."),
    top_k: int = Query(5, ge=1, le=20, description="Jumlah hasil pencarian."),
):
    """
    Pencarian semantik cerdas silabus mata kuliah menggunakan Hybrid RRF (Exp-002)
    yang tahan terhadap variasi slang mahasiswa (seperti 'alpro', 'visdat', 'jarkom').
    """
    target_prodi = prodi or program_studi
    canonical_prodi = normalize_program_studi(target_prodi) if target_prodi else None

    search_hits = search_courses(query=q, prodi=canonical_prodi or "", top_k=top_k)
    raw_catalog = _get_catalog()
    catalog_map = {c["kode_mk"].strip().upper(): c for c in raw_catalog if c.get("kode_mk")}

    results: list[CourseSearchItem] = []
    for hit in search_hits:
        c_code = str(hit.get("kode_mk") or "").strip()
        ref = catalog_map.get(c_code.upper()) or hit
        c_sem = int(ref.get("semester", 1) or 1)
        c_type = _normalize_course_type(ref.get("jenis"))
        parity = "odd" if c_sem % 2 == 1 else "even"
        c_id = str(ref.get("id") or c_code)

        results.append(
            CourseSearchItem(
                id=c_id,
                code=c_code or None,
                name=str(ref.get("nama_mk") or hit.get("nama_mk") or ""),
                credits=int(ref.get("sks", 3) or 3),
                semester=c_sem,
                type=c_type,  # type: ignore[arg-type]
                parity=parity,  # type: ignore[arg-type]
                desc=str(ref.get("deskripsi") or ""),
                program=str(ref.get("program_studi") or ""),
                prerequisites=[str(p) for p in ref.get("prasyarat", []) if p],
                score=float(hit.get("score", 0.0)),
            )
        )

    return CourseSearchResponse(
        query=q,
        program_studi=canonical_prodi,
        total_hits=len(results),
        results=results,
    )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Endpoint Server-Sent Events (SSE) streaming untuk antarmuka chat frontend.
    Mengalirkan potongan teks respons (live typing) ke UI dan mengirimkan payload
    DegreePlanPayload pada chunk selesai.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        msg = request.message.strip()
        profile = request.student_profile
        history = [{"role": m.role, "content": m.content} for m in request.history]
        session_id = request.session_id or "client_session"
        if not request.history:
            engine.memory.clear_session(session_id)

        # Eksekusi sinkron engine pada thread pool agar non-blocking
        result = await asyncio.to_thread(
            engine.generate_answer,
            query=msg,
            history=history,
            student_profile=profile.model_dump(),
            session_id=session_id,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

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

        words = result.answer.split(" ")
        for i, word in enumerate(words):
            token_chunk = word + (" " if i < len(words) - 1 else "")
            data_payload = json.dumps({"token": token_chunk, "done": False}, ensure_ascii=False)
            yield f"data: {data_payload}\n\n"
            await asyncio.sleep(0.015)

        final_payload = {
            "token": "",
            "done": True,
            "reply": result.answer,
            "updated_profile": profile.model_dump(),
            "action_type": action_type,
            "plan_payload": plan_payload.model_dump() if plan_payload else None,
            "routing_mode": result.routing_mode,
            "latency_ms": latency_ms,
        }
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
