import os
import json
import asyncio
import pytest
from unittest.mock import MagicMock, patch
import httpx

from src.engine.base import LLMClientAdapter, NormalizedChatResponse, get_llm_client
from src.api.server import handle_health_check
from src.api.schemas import HealthResponse


def test_nine_router_adapter_initialization():
    adapter = LLMClientAdapter(
        backend="nine_router",
        nine_router_host="http://localhost:20128",
        nine_router_api_key="sk-test-key-123",
    )
    assert adapter.backend == "nine_router"
    assert adapter.nine_router_host == "http://localhost:20128"
    assert adapter.nine_router_api_key == "sk-test-key-123"


def test_nine_router_chat_mock():
    mock_resp_data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1793350000,
        "model": "ag/gemini-3.8-flash-low",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Halo! Saya asisten akademik FTMM.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_resp_data
    mock_resp.raise_for_status.return_value = None

    adapter = LLMClientAdapter(
        backend="nine_router",
        nine_router_host="http://127.0.0.1:20128",
        nine_router_api_key="sk-test-key",
    )

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        resp = adapter.chat(
            messages=[{"role": "user", "content": "Halo asisten"}],
            model="ag/gemini-3.8-flash-low",
        )

        assert isinstance(resp, NormalizedChatResponse)
        assert resp.content == "Halo! Saya asisten akademik FTMM."
        assert resp.backend == "nine_router"
        assert resp.prompt_eval_count == 15
        assert resp.eval_count == 10

        # Assert correct request formatting
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://127.0.0.1:20128/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
        assert kwargs["json"]["stream"] is False
        assert kwargs["json"]["model"] == "ag/gemini-3.8-flash-low"


def test_nine_router_upstream_error_is_explicit():
    err_resp = MagicMock()
    err_resp.status_code = 400
    err_resp.text = '{"error":{"message":"API key not valid. Please pass a valid API key."}}'
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400 Bad Request",
        request=httpx.Request("POST", "http://127.0.0.1:20128/v1/chat/completions"),
        response=err_resp,
    )

    adapter = LLMClientAdapter(
        backend="nine_router",
        nine_router_host="http://127.0.0.1:20128",
        nine_router_api_key="sk-test-key",
    )
    with patch("httpx.Client.post", return_value=err_resp):
        with pytest.raises(RuntimeError) as exc_info:
            adapter.chat(messages=[{"role": "user", "content": "Halo"}], model="ag/gemini-3.8-flash-low")

    message = str(exc_info.value)
    assert "9router rejected the request (400)" in message
    assert "API key not valid" in message


def test_nine_router_tool_calls_mock():
    mock_resp_data = {
        "id": "chatcmpl-test-tc",
        "object": "chat.completion",
        "created": 1793350001,
        "model": "ag/gemini-3.8-flash-low",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_search_courses_123",
                            "type": "function",
                            "function": {
                                "name": "search_courses",
                                "arguments": '{"query": "kalkulus"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_resp_data
    mock_resp.raise_for_status.return_value = None

    adapter = LLMClientAdapter(backend="nine_router", nine_router_api_key="sk-key")
    with patch("httpx.Client.post", return_value=mock_resp):
        resp = adapter.chat(
            messages=[{"role": "user", "content": "Cari kalkulus"}],
            tools=[{"type": "function", "function": {"name": "search_courses"}}],
        )

        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "search_courses"
        assert resp.tool_calls[0].function.arguments == {"query": "kalkulus"}


def test_nine_router_stream_chat_mock():
    chunks = [
        b'data: {"choices": [{"delta": {"content": "Halo "}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "FTMM!"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    class MockAsyncStream:
        def __init__(self, lines):
            self.lines = lines

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in self.lines:
                yield line.decode("utf-8")

    async def _run():
        adapter = LLMClientAdapter(backend="nine_router", nine_router_api_key="sk-key")
        with patch("httpx.AsyncClient.stream", return_value=MockAsyncStream(chunks)):
            received = []
            async for token in adapter.stream_chat(
                messages=[{"role": "user", "content": "test stream"}]
            ):
                received.append(token)
            assert "".join(received) == "Halo FTMM!"

    asyncio.run(_run())


def test_health_check_with_nine_router():
    async def _run():
        with patch.dict(os.environ, {"LLM_BACKEND": "nine_router"}):
            with patch("httpx.AsyncClient.get") as mock_get:
                mock_ok = MagicMock()
                mock_ok.status_code = 200
                mock_get.return_value = mock_ok

                health = await handle_health_check(strict=False)
                assert isinstance(health, HealthResponse)
                assert health.active_backend == "nine_router"
                assert health.nine_router_online is True
                assert health.nine_router is not None
                assert health.nine_router.online is True

    asyncio.run(_run())


def test_nine_router_live_smoke():
    if not os.getenv("NINE_ROUTER_API_KEY"):
        pytest.skip("NINE_ROUTER_API_KEY not set; skipping live 9router smoke test")
    try:
        r = httpx.get("http://127.0.0.1:20128/v1/models", timeout=1.0)
        if r.status_code != 200:
            pytest.skip("9router is offline")
    except Exception:
        pytest.skip("9router daemon not reachable")

    adapter = LLMClientAdapter(backend="nine_router")
    resp = adapter.chat(
        messages=[{"role": "user", "content": "Sebutkan angka setelah 3. Jawab angka saja."}],
        model="ag/gemini-3.8-flash-low",
    )
    assert resp.content.strip() in ("4", "Empat", "empat")
    assert resp.backend == "nine_router"
