"""
Pengujian Kesiapan Produksi (Production Readiness & API Design Standards).
Memverifikasi:
1. Validasi input Pydantic (min_length, max_length, schema conformance).
2. Propagasi Correlation ID (X-Request-ID).
3. Konfigurasi dan expose headers CORS.
4. Client disconnect early exit pada SSE streaming.
5. Standard rate limit headers (Retry-After, X-RateLimit-Limit).
"""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.server import app, MAX_CONCURRENT_INFERENCE, _inference_semaphore


@pytest.fixture
def client():
    return TestClient(app)


def test_validation_empty_message_rejected(client):
    payload = {
        "message": "",
        "history": [],
        "student_profile": {},
    }
    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"


def test_validation_message_exceeds_max_length_rejected(client):
    payload = {
        "message": "a" * 4001,
        "history": [],
        "student_profile": {},
    }
    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "validation_error"


def test_correlation_id_propagated_from_client(client):
    custom_id = "req_custom_trace_12345"
    resp = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id


def test_correlation_id_generated_automatically_when_absent(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    req_id = resp.headers.get("X-Request-ID")
    assert req_id is not None
    assert len(req_id) >= 16


def test_cors_headers_and_exposed_headers(client):
    headers = {
        "Origin": "http://localhost:5173",
    }
    resp = client.get("/api/v1/health", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    expose_hdr = resp.headers.get("access-control-expose-headers", "")
    assert "X-Request-ID" in expose_hdr
    assert "Retry-After" in expose_hdr


def test_client_disconnect_terminates_stream_early():
    client = TestClient(app)
    yielded_chunks = []

    async def mock_infinite_stream(*args, **kwargs):
        for i in range(100):
            yielded_chunks.append(f"token_{i}")
            yield {"type": "token", "token": f"token_{i} ", "done": False}
            await asyncio.sleep(0.01)
        yield {"type": "done", "done": True, "token": "", "reply": "Done"}

    # Mock request.is_disconnected to return True after 2 tokens
    disconnect_calls = 0

    async def mock_is_disconnected():
        nonlocal disconnect_calls
        disconnect_calls += 1
        return disconnect_calls >= 3

    with patch("src.api.server.engine.stream_generate_answer", side_effect=mock_infinite_stream):
        with patch("fastapi.Request.is_disconnected", side_effect=mock_is_disconnected):
            resp = client.post(
                "/api/v1/chat/stream",
                json={"message": "Uji pemutusan klien", "history": [], "student_profile": {}},
            )
            assert resp.status_code == 200
            # Ensure generator stopped early and did not yield all 100 tokens
            assert len(yielded_chunks) < 10


def test_rate_limit_exceeded_standard_headers():
    client = TestClient(app)

    slots = []
    for _ in range(MAX_CONCURRENT_INFERENCE):
        _inference_semaphore.acquire(blocking=True)
        slots.append(True)

    try:
        with patch("src.api.server.MAX_QUEUE_WAIT_SECONDS", 0.02):
            resp = client.post(
                "/api/v1/chat",
                json={"message": "Uji 429", "history": [], "student_profile": {}},
            )
            assert resp.status_code == 429
            assert resp.headers.get("Retry-After") == "3"
            assert resp.headers.get("X-RateLimit-Limit") == str(MAX_CONCURRENT_INFERENCE)
            assert resp.headers.get("X-RateLimit-Remaining") == "0"
            body = resp.json()
            assert body["error"]["code"] == "rate_limit_exceeded"
    finally:
        for _ in slots:
            _inference_semaphore.release()
