import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.server import app, _inference_semaphore, MAX_CONCURRENT_INFERENCE
from src.engine.base import StreamingThoughtFilter, clean_thinking_content
from src.engine.memory import SQLiteSessionStore, SessionState


def test_streaming_thought_filter_with_think_tags():
    filter_ = StreamingThoughtFilter()
    tokens = ["<think>", "Mencari", " data", " silabus", " FTMM...", "</think>", " Halo", "!", " Ada", " yang", " bisa", " dibantu?"]
    output = []
    for tok in tokens:
        res = filter_.process(tok)
        if res:
            output.append(res)
    flushed = filter_.flush()
    if flushed:
        output.append(flushed)

    full_output = "".join(output).strip()
    assert "<think>" not in full_output
    assert "</think>" not in full_output
    assert "FTMM..." not in full_output
    assert full_output == "Halo! Ada yang bisa dibantu?"


def test_streaming_thought_filter_without_think_tags():
    filter_ = StreamingThoughtFilter()
    tokens = ["Selamat", " pagi", ",", " asisten", " akademik", " FTMM", " siap", " membantu", "."]
    output = []
    for tok in tokens:
        res = filter_.process(tok)
        if res:
            output.append(res)
    flushed = filter_.flush()
    if flushed:
        output.append(flushed)

    full_output = "".join(output).strip()
    assert full_output == "Selamat pagi, asisten akademik FTMM siap membantu."


def test_clean_thinking_content_comprehensive():
    raw_1 = "<think>Analisis kueri mahasiswa...</think>Mata kuliah Algoritma berbobot 3 SKS."
    assert clean_thinking_content(raw_1) == "Mata kuliah Algoritma berbobot 3 SKS."

    # Unclosed think tag (token cut off)
    raw_2 = "<think>Sedang memikirkan rencana semester..."
    assert clean_thinking_content(raw_2) == ""

    # Normal text
    raw_3 = "Batas maksimal SKS untuk IPS 3.50 adalah 24 SKS."
    assert clean_thinking_content(raw_3) == raw_3


def test_sqlite_session_store_persistence(tmp_path: Path):
    db_file = tmp_path / "test_sessions.db"
    store1 = SQLiteSessionStore(db_path=db_file)

    session = SessionState(
        session_id="mhs_123",
        created_at=time.time(),
        last_active=time.time(),
        messages=[{"role": "user", "content": "Halo saya anak TSD sem 3"}],
        student_profile={"program_studi": "Teknologi Sains Data", "semester_saat_ini": 3},
        last_discussed_semester=3,
    )
    store1.save(session)

    # Re-open with new store instance to test real on-disk persistence
    store2 = SQLiteSessionStore(db_path=db_file)
    loaded = store2.get("mhs_123")
    assert loaded is not None
    assert loaded.session_id == "mhs_123"
    assert loaded.student_profile["program_studi"] == "Teknologi Sains Data"
    assert loaded.student_profile["semester_saat_ini"] == 3
    assert len(loaded.messages) == 1

    # Cleanup test
    store2.delete("mhs_123")
    assert store2.get("mhs_123") is None


def test_concurrency_semaphore_queue_limiting():
    client = TestClient(app)

    async def mock_stream_gen(*args, **kwargs):
        yield {"type": "token", "token": "Halo", "done": False}
        await asyncio.sleep(0.5)
        yield {
            "type": "done",
            "done": True,
            "token": "",
            "reply": "Halo",
            "action_type": "GENERAL",
            "routing_mode": "DIRECT",
        }

    with patch("src.api.server.engine.stream_generate_answer", side_effect=mock_stream_gen):
        slots_acquired = []
        for _ in range(MAX_CONCURRENT_INFERENCE):
            _inference_semaphore.acquire(blocking=True)
            slots_acquired.append(True)

        try:
            with patch("src.api.server.MAX_QUEUE_WAIT_SECONDS", 0.05):
                resp = client.post(
                    "/api/v1/chat/stream",
                    json={"message": "tes antrean penuh", "history": [], "student_profile": {}},
                )
                assert resp.status_code == 200
                text = resp.text
                assert "rate_limit_exceeded" in text
        finally:
            for _ in slots_acquired:
                _inference_semaphore.release()

def test_sse_streaming_response_format():
    client = TestClient(app)

    async def mock_stream_gen(*args, **kwargs):
        yield {"type": "token", "token": "Halo!", "done": False}
        yield {
            "type": "done",
            "done": True,
            "token": "",
            "reply": "Halo! Ada yang bisa dibantu?",
            "updated_profile": {"program_studi": "Teknologi Sains Data"},
            "action_type": "GENERAL",
            "plan_payload": None,
            "routing_mode": "DIRECT",
            "latency_ms": 150.0,
            "ttft_ms": 45.0,
        }

    with patch("src.api.server.engine.stream_generate_answer", side_effect=mock_stream_gen):
        resp = client.post(
            "/api/v1/chat/stream",
            json={"message": "Halo asisten", "history": [], "student_profile": {}},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [line.strip() for line in resp.text.split("\n") if line.strip()]

        assert len(lines) >= 2
        chunk_1 = json.loads(lines[0].replace("data: ", ""))
        assert chunk_1["token"] == "Halo!"
        assert chunk_1["done"] is False

        chunk_done = json.loads(lines[1].replace("data: ", ""))
        assert chunk_done["done"] is True
        assert chunk_done["reply"] == "Halo! Ada yang bisa dibantu?"
        assert chunk_done["routing_mode"] == "DIRECT"
        assert chunk_done["latency_ms"] == 150.0
