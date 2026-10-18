"""
Unit & Integration Tests untuk FastAPI Server FTMM COMPASS.
Memverifikasi:
1. Kepatuhan kontrak REST modern (/api/v1/ canonical dan /api/ legacy).
2. Paginasi adaptif & search terintegrasi pada /api/v1/courses.
3. Standard error envelope dan status codes semantik (503 Service Unavailable, 422 Validation Error).
4. Kompatibilitas penuh terhadap frontend ftmm-compass-ui.
"""

import json
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_endpoints():
    """Memverifikasi endpoint health baik legacy /api/health maupun canonical /api/v1/health."""
    for path in ["/api/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "default_model" in data
        assert "ollama" in data
        assert data["ollama"]["host"]


def test_v1_health_strict_mode_when_offline():
    """Memverifikasi mode strict=true pada /api/v1/health yang mengembalikan 503 jika Ollama offline."""
    with patch("src.api.server._get_ollama_base_url", return_value="http://127.0.0.1:9999"):
        resp = client.get("/api/v1/health?strict=true")
        assert resp.status_code in [200, 503]


def test_courses_catalog_endpoint():
    """Memverifikasi katalog 369 mata kuliah pada endpoint flat (legacy & v1 tanpa pagination)."""
    for path in ["/api/courses", "/api/v1/courses"]:
        response = client.get(f"{path}?program_studi=Teknologi%20Sains%20Data")
        assert response.status_code == 200
        data = response.json()
        assert data["program_studi"] == "Teknologi Sains Data"
        assert data["total_courses"] > 0
        assert len(data["courses"]) == data["total_courses"]
        assert len(data["programs"]) == 5
        assert data["courses"][0]["name"]

    # Test all 369 courses
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


def test_courses_adaptive_pagination():
    """Memverifikasi pagination envelope pada /api/v1/courses saat parameter page dioper."""
    # 1. Halaman 1 per_page 20
    resp = client.get("/api/v1/courses?page=1&per_page=20")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert "meta" in data
    assert "links" in data
    assert len(data["data"]) == 20
    assert data["meta"]["total"] == 369
    assert data["meta"]["page"] == 1
    assert data["meta"]["per_page"] == 20
    assert data["meta"]["total_pages"] == 19
    assert data["links"]["next"] is not None
    assert data["links"]["prev"] is None

    # 2. Halaman dengan filter prodi TSD (68 courses)
    resp_tsd = client.get("/api/v1/courses?prodi=TSD&page=1&per_page=25")
    assert resp_tsd.status_code == 200
    data_tsd = resp_tsd.json()
    assert data_tsd["meta"]["total"] == 68
    assert data_tsd["meta"]["page"] == 1
    assert data_tsd["meta"]["per_page"] == 25
    assert data_tsd["meta"]["total_pages"] == 3


def test_plan_validate_endpoints():
    """Memverifikasi validasi rencana studi pada endpoint legacy /api/plan/validate dan v1 /api/v1/plans/validations."""
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

    for path in ["/api/plan/validate", "/api/v1/plans/validations"]:
        resp = client.post(path, json=valid_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True
        assert data["total_sks_planned"] == 10

    # Test error batas SKS
    invalid_sks_payload = {
        "plan": {
            "1": [
                {"code": "MK1", "name": "Matkul A", "credits": 15},
                {"code": "MK2", "name": "Matkul B", "credits": 15},
            ]
        },
        "maks_sks_per_semester": 24,
    }
    resp_sks = client.post("/api/v1/plans/validations", json=invalid_sks_payload)
    assert resp_sks.status_code == 200
    data_sks = resp_sks.json()
    assert data_sks["is_valid"] is False
    assert any("melebihi batas maksimum" in err for err in data_sks["errors"])


def test_plan_generate_direct_endpoints():
    """Memverifikasi endpoint pembuatan rencana studi pada /api/plan/generate dan /api/v1/plans."""
    profile_payload = {
        "program_studi": "Teknologi Sains Data",
        "semester_saat_ini": 2,
        "target_kelulusan_semester": 7,
        "riwayat_matkul_lulus": ["Matematika I"],
        "minat_fokus": ["Artificial Intelligence"],
        "maks_sks_per_semester": 24,
    }

    for path in ["/api/plan/generate", "/api/v1/plans"]:
        resp = client.post(path, json=profile_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "APPLY_DEGREE_PLAN"
        assert data["summary"]["program_studi"] == "Teknologi Sains Data"
        assert data["summary"]["target_semester"] == 7
        assert len(data["plan"]) == 7


def test_validation_error_envelope():
    """Memverifikasi standard error envelope pada HTTP 422 saat payload tidak valid."""
    # Mengirim payload salah (bukan dictionary) ke plan validation
    resp = client.post("/api/v1/plans/validations", json={"plan": "bukan_dict"})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"
    assert len(data["error"]["details"]) > 0


def test_ollama_offline_semantic_503():
    """Memverifikasi bahwa kegagalan koneksi upstream Ollama mengembalikan 503 dengan standar error envelope."""
    payload = {
        "message": "Berapa syarat SKS minimal untuk mengambil PKL di FTMM?",
        "history": [],
        "student_profile": {"program_studi": "Teknologi Sains Data", "semester_saat_ini": 4},
    }
    # Mock engine melempar ConnectionError (seperti saat daemon Ollama offline)
    with patch("src.api.server.engine.generate_answer", side_effect=ConnectionError("Failed to connect to Ollama")):
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "5"
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "upstream_service_unavailable"
        assert "Ollama" in data["error"]["message"]


def test_chat_endpoints_canonical_and_legacy():
    """Memverifikasi endpoint dialog chat baik canonical /api/v1/chat maupun legacy /api/chat."""
    from src.engine.base import EngineResult

    mock_result = EngineResult(
        answer="Halo! Ada yang bisa saya bantu seputar kurikulum FTMM?",
        architecture="UnifiedChatEngine",
        retrieved_chunks=[],
        retrieval_latency_ms=1.5,
        generation_latency_ms=45.2,
        total_latency_ms=46.7,
        tokens_generated=12,
        routing_mode="CAG",
        metadata={"action_type": "GENERAL"},
    )

    payload = {
        "message": "Halo Compass AI",
        "history": [],
        "student_profile": {"program_studi": "Teknologi Sains Data", "semester_saat_ini": 3},
        "session_id": "test_session_123",
    }

    for path in ["/api/v1/chat", "/api/chat"]:
        with patch("src.api.server.engine.generate_answer", return_value=mock_result):
            resp = client.post(path, json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "reply" in data
            assert "Halo!" in data["reply"]
            assert data["routing_mode"] == "CAG"
            assert data["action_type"] == "GENERAL"
            assert data["updated_profile"]["program_studi"] == "Teknologi Sains Data"
            assert data["updated_profile"]["semester_saat_ini"] == 3
            assert data["latency_ms"] >= 0.0


def test_chat_endpoint_slot_filling_and_degree_plan_payload():
    """Memverifikasi ekstraksi entitas slot-filling profile dan generasi DegreePlanPayload pada /api/v1/chat."""
    from src.engine.base import EngineResult

    mock_plan = {
        "summary": {
            "program_studi": "Teknologi Sains Data",
            "current_semester": 5,
            "target_semester": 7,
            "total_credits": 144,
            "focus_tracks": ["Artificial Intelligence"],
            "note": "Rencana studi rekomendasi AI berdasarkan kurikulum resmi dan prasyarat FTMM.",
        },
        "plan": {
            "6": [{"code": "TSD301", "name": "Deep Learning", "credits": 3, "parity": "even"}],
            "7": [{"code": "TSD401", "name": "Tugas Akhir", "credits": 6, "parity": "odd"}],
        },
    }

    mock_result = EngineResult(
        answer="Rencana studi untuk semester 6 dan 7 peminatan Artificial Intelligence telah disusun.",
        architecture="UnifiedChatEngine (Hybrid Plan)",
        retrieved_chunks=[],
        retrieval_latency_ms=25.0,
        generation_latency_ms=120.0,
        total_latency_ms=145.0,
        tokens_generated=35,
        routing_mode="HYBRID",
        metadata={
            "action_type": "PLAN_GENERATED",
            "updated_profile": {
                "program_studi": "Teknologi Sains Data",
                "semester_saat_ini": 5,
                "minat_fokus": ["Artificial Intelligence"],
                "riwayat_matkul_lulus": ["Kalkulus I", "Alpro I"],
            },
            "plan_payload": mock_plan,
        },
    )

    payload = {
        "message": "Saya mahasiswa TSD semester 5, tolong buatkan rencana studi AI sampai semester 7",
        "history": [],
        "student_profile": {"program_studi": "TSD", "semester_saat_ini": 5},
    }

    with patch("src.api.server.engine.generate_answer", return_value=mock_result):
        resp = client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action_type"] == "PLAN_GENERATED"
        assert data["updated_profile"]["semester_saat_ini"] == 5
        assert data["updated_profile"]["minat_fokus"] == ["Artificial Intelligence"]
        assert "Kalkulus I" in data["updated_profile"]["riwayat_matkul_lulus"]

        plan_payload = data["plan_payload"]
        assert plan_payload is not None
        assert plan_payload["action"] == "APPLY_DEGREE_PLAN"
        assert plan_payload["summary"]["program_studi"] == "Teknologi Sains Data"
        assert plan_payload["summary"]["target_semester"] == 7
        assert "6" in plan_payload["plan"]
        assert len(plan_payload["plan"]["6"]) == 1
        assert plan_payload["plan"]["6"][0]["name"] == "Deep Learning"


def test_chat_stream_endpoint_success():
    """Memverifikasi endpoint Server-Sent Events (SSE) streaming live typing pada /api/v1/chat/stream dan legacy."""
    from src.engine.base import EngineResult

    mock_result = EngineResult(
        answer="Halo! Ini adalah respons streaming dari Compass AI.",
        architecture="UnifiedChatEngine",
        retrieved_chunks=[],
        retrieval_latency_ms=5.0,
        generation_latency_ms=60.0,
        total_latency_ms=65.0,
        tokens_generated=8,
        routing_mode="DIRECT",
        metadata={"action_type": "GENERAL"},
    )

    payload = {
        "message": "Halo streaming",
        "history": [],
        "student_profile": {},
    }

    for path in ["/api/v1/chat/stream", "/api/chat/stream"]:
        with patch("src.api.server.engine.generate_answer", return_value=mock_result):
            resp = client.post(path, json=payload)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = []
            for line in resp.text.split("\n"):
                line_clean = line.strip()
                if line_clean.startswith("data:"):
                    json_str = line_clean.replace("data:", "").strip()
                    events.append(json.loads(json_str))

            assert len(events) >= 2
            # Verifikasi token chunks
            token_chunks = [e for e in events if not e.get("done")]
            assert len(token_chunks) > 0
            assert all("token" in e for e in token_chunks)

            # Verifikasi final termination event
            final_event = [e for e in events if e.get("done") is True]
            assert len(final_event) == 1
            assert final_event[0]["reply"] == "Halo! Ini adalah respons streaming dari Compass AI."
            assert final_event[0]["routing_mode"] == "DIRECT"


def test_chat_stream_endpoint_upstream_connection_error():
    """Memverifikasi bahwa kegagalan upstream AI daemon pada streaming mengembalikan frame error SSE terstruktur."""
    payload = {
        "message": "Kueri saat upstream offline",
        "history": [],
        "student_profile": {},
    }

    with patch("src.api.server.engine.generate_answer", side_effect=ConnectionError("Daemon offline")):
        resp = client.post("/api/v1/chat/stream", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = []
        for line in resp.text.split("\n"):
            line_clean = line.strip()
            if line_clean.startswith("data:"):
                json_str = line_clean.replace("data:", "").strip()
                events.append(json.loads(json_str))

        assert len(events) == 1
        assert "error" in events[0]
        assert events[0]["error"]["code"] == "upstream_service_unavailable"
        assert "Daemon offline" in events[0]["error"]["message"]
