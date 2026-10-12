"""
Unit & Integration Tests untuk FastAPI Server FTMM COMPASS.
Memverifikasi kepatuhan kontrak API terhadap frontend ftmm-compass-ui.
"""

import json
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ollama_online"] is True
    assert "qwen2.5:3b" in data["default_model"]


def test_chat_cag_sop_query():
    payload = {
        "message": "Berapa syarat SKS minimal untuk mengambil PKL di FTMM?",
        "history": [],
        "student_profile": {
            "program_studi": "Teknologi Sains Data",
            "semester_saat_ini": 4
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routing_mode"] == "CAG"
    assert len(data["reply"]) > 20
    assert "80" in data["reply"]  # Syarat 80 SKS


def test_chat_rag_syllabus_query():
    payload = {
        "message": "Apa isi silabus mata kuliah SIA104 Pengantar Pemrograman?",
        "history": [],
        "student_profile": {
            "program_studi": "Teknologi Sains Data",
            "semester_saat_ini": 1
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routing_mode"] == "RAG"
    assert len(data["reply"]) > 20
    assert any(term in data["reply"].lower() for term in ["python", "algoritma", "pemrograman"])


def test_chat_hybrid_planning_query():
    payload = {
        "message": "Saya mahasiswa semester 4 prodi TSD minat machine learning, rekomendasikan matkul untuk semester 5",
        "history": [],
        "student_profile": {
            "program_studi": "Teknologi Sains Data",
            "semester_saat_ini": 4,
            "minat_fokus": []
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routing_mode"] == "HYBRID"
    assert data["action_type"] == "PLAN_GENERATED"
    assert data["plan_payload"] is not None
    assert data["plan_payload"]["action"] == "APPLY_DEGREE_PLAN"
    assert data["plan_payload"]["summary"]["program_studi"] == "Teknologi Sains Data"


def test_chat_guarded_query():
    payload = {
        "message": "Bagaimana resep membuat rendang sapi padang?",
        "history": [],
        "student_profile": {}
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["routing_mode"] == "GUARDED"
    assert "FTMM" in data["reply"]


def test_courses_catalog_endpoint():
    response = client.get("/api/courses?program_studi=Teknologi%20Sains%20Data")
    assert response.status_code == 200
    data = response.json()
    assert data["program_studi"] == "Teknologi Sains Data"
    assert data["total_courses"] > 0
    assert len(data["courses"]) == data["total_courses"]
    assert len(data["programs"]) == 5
    assert data["courses"][0]["name"]

    # Test all programs
    response_all = client.get("/api/courses")
    assert response_all.status_code == 200
    data_all = response_all.json()
    assert data_all["total_courses"] == 369
    assert len(data_all["programs"]) == 5


def test_courses_catalog_prodi_alias():
    """Memverifikasi kompatibilitas alias prodi dari frontend ftmm-compass-ui/src/courseData.ts."""
    response = client.get("/api/courses?prodi=TSD")
    assert response.status_code == 200
    data = response.json()
    assert data["program_studi"] == "Teknologi Sains Data"
    assert data["total_courses"] > 0


def test_plan_validate_endpoint():
    """Memverifikasi endpoint validasi rencana studi terhadap prerequisite DAG & SKS limit."""
    # 1. Valid plan
    valid_payload = {
        "plan": {
            "1": [
                {"code": "MAT101", "name": "Matematika I", "credits": 4, "parity": "odd"},
                {"code": "KIM101", "name": "Kimia Dasar", "credits": 3, "parity": "odd"},
            ],
            "2": [
                {"code": "MAT102", "name": "Matematika II", "credits": 4, "parity": "even"},
            ],
        },
        "riwayat_matkul_lulus": ["Matematika I"],
        "maks_sks_per_semester": 20,
    }
    resp = client.post("/api/plan/validate", json=valid_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["total_sks_planned"] == 10

    # 2. Plan violating SKS limit
    invalid_sks_payload = {
        "plan": {
            "1": [
                {"code": "MK1", "name": "Matkul A", "credits": 15},
                {"code": "MK2", "name": "Matkul B", "credits": 15},
            ]
        },
        "maks_sks_per_semester": 24,
    }
    resp_sks = client.post("/api/plan/validate", json=invalid_sks_payload)
    assert resp_sks.status_code == 200
    data_sks = resp_sks.json()
    assert data_sks["is_valid"] is False
    assert any("melebihi batas maksimum" in err for err in data_sks["errors"])


def test_plan_generate_direct_endpoint():
    """Memverifikasi endpoint pembuatan rencana studi deterministik instan."""
    profile_payload = {
        "program_studi": "Teknologi Sains Data",
        "semester_saat_ini": 2,
        "target_kelulusan_semester": 7,
        "riwayat_matkul_lulus": ["Matematika I"],
        "minat_fokus": ["Artificial Intelligence"],
        "maks_sks_per_semester": 24,
    }
    resp = client.post("/api/plan/generate", json=profile_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "APPLY_DEGREE_PLAN"
    assert data["summary"]["program_studi"] == "Teknologi Sains Data"
    assert data["summary"]["target_semester"] == 7
    assert len(data["plan"]) == 7
    # Pastikan setiap mata kuliah memiliki parity, status, dan code
    for sem_courses in data["plan"].values():
        for c in sem_courses:
            assert "parity" in c
            assert "status" in c
            assert "code" in c


def test_courses_search_endpoint():
    """Memverifikasi endpoint pencarian semantik silabus berbasis Hybrid RRF."""
    resp = client.get("/api/courses/search?q=machine%20learning&prodi=TSD&top_k=3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "machine learning"
    assert data["program_studi"] == "Teknologi Sains Data"
    assert data["total_hits"] <= 3
    assert len(data["results"]) > 0
    assert data["results"][0]["name"]
    assert data["results"][0]["credits"] > 0


def test_chat_stream_endpoint():
    """Memverifikasi SSE stream endpoint untuk live typing UI."""
    payload = {
        "message": "Halo Compass AI",
        "history": [],
        "student_profile": {"program_studi": "Teknologi Sains Data"},
        "session_id": "test_stream_session",
    }
    with client.stream("POST", "/api/chat/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = list(resp.iter_lines())
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) > 0
        # Pastikan event penutup ada done: True
        last_event = json.loads(data_lines[-1].replace("data: ", ""))
        assert last_event["done"] is True
        assert "reply" in last_event
        assert len(last_event["reply"]) > 0
