"""
Unit & Integration Tests untuk FastAPI Server FTMM COMPASS.
Memverifikasi kepatuhan kontrak API terhadap frontend ftmm-compass-ui.
"""

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
