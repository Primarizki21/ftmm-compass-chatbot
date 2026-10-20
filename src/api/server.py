"""
FastAPI Backend Server untuk Chatbot FTMM COMPASS.
Melayani antarmuka obrolan di ~/project/ftmm-compass-ui/src/pages/Chatbot.tsx.
Menerapkan arsitektur RESTful modern (v1 canonical + legacy alias) dan error handling semantik.
"""

import os
import time
import json
import uuid
import logging
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
import httpx
from fastapi import FastAPI, APIRouter, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("ftmm_compass_api")
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    BonsaiHealthDetail,
    OllamaHealthDetail,
    StudentProfile,
    DegreePlanPayload,
    DegreePlanSummary,
    CourseItem,
    CourseCatalogResponse,
    PaginatedCourseResponse,
    PaginationMeta,
    PaginationLinks,
    PlanValidationRequest,
    PlanValidationResponse,
    CourseSearchItem,
    CourseSearchResponse,
    ApiErrorResponse,
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inisialisasi FTMM COMPASS Chatbot Backend...")
    yield
    logger.info("Shutdown FTMM COMPASS Chatbot Backend gracefully...")


app = FastAPI(
    title="FTMM COMPASS Chatbot Backend",
    description="Backend API untuk Chatbot Akademik FTMM COMPASS dengan arsitektur Hybrid RAG-CAG.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

@app.get("/", tags=["root"])
async def root():
    return {
        "service": "FTMM COMPASS AI Microservice",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "v1_health": "/api/v1/health",
    }
engine = UnifiedChatEngine()

MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "1"))
MAX_QUEUE_WAIT_SECONDS = float(os.getenv("MAX_QUEUE_WAIT_SECONDS", "10.0"))
_inference_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)

def _get_ollama_base_url() -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        return f"http://{host}"
    return host.rstrip("/")


def _get_bonsai_base_url() -> str:
    host = os.getenv("BONSAI_HOST", "http://127.0.0.1:8080").strip()
    if not host.startswith(("http://", "https://")):
        return f"http://{host}"
    return host.rstrip("/")

def _normalize_course_type(jenis: str | None) -> str:
    j = str(jenis or "").lower()
    if not j or "wajib" in j:
        return "Wajib"
    if "pilih" in j or "pengayaan" in j or "peminatan" in j:
        return "Pilihan"
    return "Wajib"


# ============================================================================
# Global Exception Handlers (skill://api-design)
# ============================================================================

@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": "5"},
        content={
            "error": {
                "code": "upstream_service_unavailable",
                "message": "Layanan Ollama tidak dapat dihubungi. Pastikan daemon Ollama aktif pada host port 11434.",
                "details": [{"message": str(exc)}],
            }
        },
    )


@app.exception_handler(httpx.ConnectError)
async def httpx_connect_error_handler(request: Request, exc: httpx.ConnectError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": "5"},
        content={
            "error": {
                "code": "upstream_service_unavailable",
                "message": "Gagal terhubung ke upstream service.",
                "details": [{"message": str(exc)}],
            }
        },
    )


@app.exception_handler(httpx.TimeoutException)
async def httpx_timeout_handler(request: Request, exc: httpx.TimeoutException):
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": {
                "code": "upstream_timeout",
                "message": "Waktu tunggu layanan upstream habis.",
                "details": [{"message": str(exc)}],
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {
                        "field": ".".join(str(loc) for loc in err["loc"]),
                        "message": err["msg"],
                        "code": err["type"],
                    }
                    for err in exc.errors()
                ],
            }
        },
    )


# ============================================================================
# Core Handlers (Reused across Canonical v1 & Legacy routes)
# ============================================================================

async def handle_health_check(strict: bool = False) -> HealthResponse:
    """Status kesehatan backend dan pengecekan kesehatan ganda (Bonsai llama-server & Ollama daemon)."""
    active_backend = os.getenv("LLM_BACKEND", "bonsai").strip().lower()
    generator_model = os.getenv("LLM_MODEL_NAME", "bonsai-2-4b")
    embedding_model = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")

    bonsai_base_url = _get_bonsai_base_url()
    ollama_base_url = _get_ollama_base_url()

    bonsai_online = False
    ollama_online = False

    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            b_resp = await client.get(f"{bonsai_base_url}/health")
            if b_resp.status_code == 200:
                bonsai_online = True
        except Exception:
            bonsai_online = False

        try:
            o_resp = await client.get(f"{ollama_base_url}/api/tags")
            if o_resp.status_code == 200:
                ollama_online = True
        except Exception:
            ollama_online = False

    if active_backend == "bonsai":
        if bonsai_online:
            service_status = "healthy"
        elif ollama_online:
            service_status = "degraded"
        else:
            service_status = "unhealthy"
    else:
        service_status = "healthy" if ollama_online else "degraded"

    if strict:
        if active_backend == "bonsai" and not (bonsai_online or ollama_online):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Both Bonsai generator and Ollama fallback are offline",
            )
        elif active_backend == "ollama" and not ollama_online:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Ollama daemon is offline",
            )

    return HealthResponse(
        status=service_status,
        service="FTMM COMPASS Chatbot Backend",
        active_backend=active_backend,
        bonsai_online=bonsai_online,
        ollama_online=ollama_online,
        generator_model=generator_model,
        embedding_model=embedding_model,
        default_model=generator_model,
        bonsai=BonsaiHealthDetail(
            online=bonsai_online,
            host=bonsai_base_url,
            default_model=generator_model,
        ),
        ollama=OllamaHealthDetail(
            online=ollama_online,
            host=ollama_base_url,
            default_model=engine.model,
        ),
    )


async def handle_list_courses(
    program_studi: str | None = None,
    prodi: str | None = None,
    semester: int | None = None,
    q: str | None = None,
    page: int | None = None,
    per_page: int | None = None,
) -> CourseCatalogResponse | PaginatedCourseResponse:
    """
    Mengambil katalog mata kuliah resmi FTMM dari SSOT kurikulum.
    Mendukung filtering prodi/semester, pencarian semantik (q), dan paginasi adaptif.
    """
    target_prodi = prodi or program_studi
    canonical_prodi = normalize_program_studi(target_prodi) if target_prodi else None
    raw_catalog = _get_catalog()
    programs_set = sorted(list(set(c.get("program_studi", "") for c in raw_catalog if c.get("program_studi"))))

    # Jika query q disediakan, gunakan pencarian semantik cerdas
    if q and q.strip():
        search_hits = search_courses(query=q.strip(), prodi=canonical_prodi or "", top_k=50)
        catalog_map = {c["kode_mk"].strip().upper(): c for c in raw_catalog if c.get("kode_mk")}
        matched_catalog = []
        for hit in search_hits:
            c_code = str(hit.get("kode_mk") or "").strip().upper()
            ref = catalog_map.get(c_code)
            if ref:
                matched_catalog.append(ref)
        raw_catalog = matched_catalog

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

    # Mode Paginasi Adaptif (jika parameter page atau per_page diberikan)
    if page is not None or per_page is not None:
        p = max(page or 1, 1)
        pp = max(per_page or 20, 1)
        total = len(courses)
        total_pages = (total + pp - 1) // pp if total > 0 else 0
        sliced = courses[(p - 1) * pp : p * pp]

        prefix = f"/api/v1/courses?per_page={pp}"
        if canonical_prodi:
            prefix += f"&prodi={canonical_prodi}"
        if semester:
            prefix += f"&semester={semester}"
        if q:
            prefix += f"&q={q}"

        return PaginatedCourseResponse(
            data=sliced,
            meta=PaginationMeta(
                total=total,
                page=p,
                per_page=pp,
                total_pages=total_pages,
            ),
            links=PaginationLinks(
                self=f"{prefix}&page={p}",
                next=f"{prefix}&page={p + 1}" if p < total_pages else None,
                prev=f"{prefix}&page={p - 1}" if p > 1 else None,
                first=f"{prefix}&page=1",
                last=f"{prefix}&page={total_pages}" if total_pages > 0 else None,
            ),
            program_studi=canonical_prodi,
        )

    # Mode Default Kompatibel Flat (untuk pemanggilan frontend eksisting)
    return CourseCatalogResponse(
        program_studi=canonical_prodi,
        total_courses=len(courses),
        programs=programs_set,
        courses=courses,
    )


async def handle_search_courses(
    q: str,
    program_studi: str | None = None,
    prodi: str | None = None,
    top_k: int = 5,
) -> CourseSearchResponse:
    """Pencarian semantik mata kuliah lintas katalog FTMM."""
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


async def handle_chat(request: ChatRequest, req: Request | None = None) -> ChatResponse:
    """Dialog chat utama yang melayani pesan dari frontend ftmm-compass-ui."""
    t0 = time.perf_counter()
    msg = request.message.strip()
    profile = request.student_profile
    history = [{"role": m.role, "content": m.content} for m in request.history]
    session_id = request.session_id or "client_session"
    if not request.history:
        engine.memory.clear_session(session_id)

    acquired = await asyncio.to_thread(_inference_semaphore.acquire, True, MAX_QUEUE_WAIT_SECONDS)
    if not acquired:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={
                "Retry-After": "3",
                "X-RateLimit-Limit": str(MAX_CONCURRENT_INFERENCE),
                "X-RateLimit-Remaining": "0",
            },
            content={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Server sedang melayani antrean mahasiswa lain. Silakan coba sesaat lagi.",
                    "details": [
                        {
                            "field": None,
                            "message": f"Kapasitas inferensi penuh ({MAX_CONCURRENT_INFERENCE} slot terpakai). Coba kembali dalam 3 detik.",
                            "code": "concurrency_limit",
                        }
                    ],
                }
            },
        )

    try:
        # Eksekusi sinkron engine pada thread pool agar non-blocking
        result = await asyncio.to_thread(
            engine.generate_answer,
            query=msg,
            history=history,
            student_profile=profile.model_dump(),
            session_id=session_id,
        )
    finally:
        if acquired:
            _inference_semaphore.release()
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


async def handle_validate_plan(request: PlanValidationRequest) -> PlanValidationResponse:
    """Memvalidasi seluruh urutan rencana studi mahasiswa terhadap aturan akademik FTMM."""
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


async def handle_generate_plan(profile: StudentProfile) -> DegreePlanPayload:
    """Menghasilkan rancangan rencana studi secara deterministik."""
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


async def handle_chat_stream(request: ChatRequest, req: Request | None = None) -> StreamingResponse:
    """Endpoint Server-Sent Events (SSE) streaming untuk live typing UI."""
    async def event_generator() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        msg = request.message.strip()
        profile = request.student_profile
        history = [{"role": m.role, "content": m.content} for m in request.history]
        session_id = request.session_id or "client_session"
        if not request.history:
            engine.memory.clear_session(session_id)

        acquired = await asyncio.to_thread(_inference_semaphore.acquire, True, MAX_QUEUE_WAIT_SECONDS)
        if not acquired:
            err_payload = {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Server sedang melayani antrean mahasiswa lain. Silakan coba sesaat lagi.",
                    "details": [
                        {
                            "field": None,
                            "message": f"Kapasitas inferensi penuh ({MAX_CONCURRENT_INFERENCE} slot terpakai). Coba kembali dalam 3 detik.",
                            "code": "concurrency_limit",
                        }
                    ],
                }
            }
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
            return

        try:
            async for chunk in engine.stream_generate_answer(
                query=msg,
                history=history,
                student_profile=profile.model_dump(),
                session_id=session_id,
            ):
                if req is not None and await req.is_disconnected():
                    logger.info("Client disconnected (session=%s); aborting stream early.", session_id)
                    break
                if chunk.get("type") == "token":
                    data_payload = json.dumps({"token": chunk["token"], "done": False}, ensure_ascii=False)
                    yield f"data: {data_payload}\n\n"
                elif chunk.get("type") == "done":
                    action_type = chunk.get("action_type", "GENERAL")
                    plan_raw = chunk.get("plan_payload")
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

                    updated_prof = chunk.get("updated_profile")
                    if isinstance(updated_prof, dict) and updated_prof:
                        for k, v in updated_prof.items():
                            if hasattr(profile, k) and v is not None:
                                setattr(profile, k, v)

                    final_payload = {
                        "token": "",
                        "done": True,
                        "reply": chunk.get("reply", ""),
                        "updated_profile": profile.model_dump(),
                        "action_type": action_type,
                        "plan_payload": plan_payload.model_dump() if plan_payload else None,
                        "routing_mode": chunk.get("routing_mode", "DIRECT"),
                        "latency_ms": chunk.get("latency_ms", (time.perf_counter() - t0) * 1000.0),
                        "ttft_ms": chunk.get("ttft_ms", 0.0),
                    }
                    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
        except ConnectionError as exc:
            err_payload = {
                "error": {
                    "code": "upstream_service_unavailable",
                    "message": f"Koneksi ke server AI lokal (Bonsai / Ollama) gagal: {exc}",
                }
            }
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
        except Exception as exc:
            err_payload = {
                "error": {
                    "code": "internal_server_error",
                    "message": f"Terjadi kesalahan pemrosesan streaming: {exc}",
                }
            }
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
        finally:
            if acquired:
                _inference_semaphore.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ============================================================================
# Canonical Router: /api/v1 (RESTful Modern Standard)
# ============================================================================

v1_router = APIRouter(prefix="/api/v1", tags=["v1"])


@v1_router.get("/health", response_model=HealthResponse)
async def v1_health(strict: bool = Query(False, description="Fail with 503 if Ollama is offline")):
    return await handle_health_check(strict=strict)


@v1_router.get("/courses", response_model=Any)
async def v1_list_courses(
    program_studi: str | None = Query(None, description="Nama program studi FTMM."),
    prodi: str | None = Query(None, description="Alias nama prodi."),
    semester: int | None = Query(None, description="Filter semester (1-8)."),
    q: str | None = Query(None, description="Pencarian semantik silabus atau nama mata kuliah."),
    page: int | None = Query(None, ge=1, description="Nomor halaman (1-based)."),
    per_page: int | None = Query(None, ge=1, le=100, description="Jumlah item per halaman."),
):
    return await handle_list_courses(
        program_studi=program_studi,
        prodi=prodi,
        semester=semester,
        q=q,
        page=page,
        per_page=per_page,
    )

@v1_router.get("/courses/search", response_model=CourseSearchResponse)
async def v1_search_courses(
    q: str = Query(..., min_length=1, description="Kata kunci, silabus, atau nama mata kuliah."),
    program_studi: str | None = Query(None, description="Filter program studi opsional."),
    prodi: str | None = Query(None, description="Alias prodi opsional."),
    top_k: int = Query(5, ge=1, le=20, description="Jumlah hasil pencarian."),
):
    return await handle_search_courses(q=q, program_studi=program_studi, prodi=prodi, top_k=top_k)


@v1_router.post("/chat", response_model=ChatResponse)
async def v1_chat(request: ChatRequest, req: Request):
    return await handle_chat(request, req=req)


@v1_router.post("/chat/stream")
async def v1_chat_stream(request: ChatRequest, req: Request):
    return await handle_chat_stream(request, req=req)


@v1_router.post("/plans/validations", response_model=PlanValidationResponse)
async def v1_validate_plan(request: PlanValidationRequest):
    return await handle_validate_plan(request)


@v1_router.post("/plans", response_model=DegreePlanPayload)
async def v1_generate_plan(profile: StudentProfile):
    return await handle_generate_plan(profile)


# ============================================================================
# Legacy Router: /api (100% Backward-Compatibility for ftmm-compass-ui)
# ============================================================================

legacy_router = APIRouter(prefix="/api", tags=["legacy"])


@legacy_router.get("/health", response_model=HealthResponse)
async def legacy_health():
    return await handle_health_check(strict=False)


@legacy_router.get("/courses", response_model=Any)
async def legacy_list_courses(
    program_studi: str | None = Query(None, description="Nama program studi FTMM."),
    prodi: str | None = Query(None, description="Alias nama prodi dari frontend UI."),
    semester: int | None = Query(None, description="Filter semester (1-8)."),
    q: str | None = Query(None, description="Query pencarian opsional."),
    page: int | None = Query(None, description="Halaman paginasi opsional."),
    per_page: int | None = Query(None, description="Limit per halaman opsional."),
):
    return await handle_list_courses(
        program_studi=program_studi,
        prodi=prodi,
        semester=semester,
        q=q,
        page=page,
        per_page=per_page,
    )


@legacy_router.get("/courses/search", response_model=CourseSearchResponse)
async def legacy_search_courses(
    q: str = Query(..., min_length=1, description="Kata kunci, silabus, atau nama mata kuliah."),
    program_studi: str | None = Query(None, description="Filter program studi opsional."),
    prodi: str | None = Query(None, description="Alias prodi opsional."),
    top_k: int = Query(5, ge=1, le=20, description="Jumlah hasil pencarian."),
):
    return await handle_search_courses(q=q, program_studi=program_studi, prodi=prodi, top_k=top_k)


@legacy_router.post("/chat", response_model=ChatResponse)
async def legacy_chat(request: ChatRequest, req: Request):
    return await handle_chat(request, req=req)


@legacy_router.post("/chat/stream")
async def legacy_chat_stream(request: ChatRequest, req: Request):
    return await handle_chat_stream(request, req=req)


@legacy_router.post("/plan/validate", response_model=PlanValidationResponse)
async def legacy_validate_plan(request: PlanValidationRequest):
    return await handle_validate_plan(request)


@legacy_router.post("/plan/generate", response_model=DegreePlanPayload)
async def legacy_generate_plan(profile: StudentProfile):
    return await handle_generate_plan(profile)


# Include both canonical v1 and legacy routers
app.include_router(v1_router)
app.include_router(legacy_router)
