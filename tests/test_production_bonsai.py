import asyncio
import json
import os
from unittest.mock import patch, MagicMock
import httpx
import pytest
from fastapi import HTTPException

from src.api.schemas import HealthResponse
from src.api.server import handle_health_check
from src.engine.base import (
    LLMClientAdapter,
    NormalizedChatMessage,
    NormalizedToolCall,
    NormalizedChatResponse,
    get_llm_client,
)
from src.engine.react_agent import ReActAgentEngine


def test_llm_adapter_initialization():
    adapter_bonsai = LLMClientAdapter(
        backend="bonsai",
        bonsai_host="127.0.0.1:8080/",
        ollama_host="127.0.0.1:11434/",
    )
    assert adapter_bonsai.backend == "bonsai"
    assert adapter_bonsai.bonsai_host == "http://127.0.0.1:8080"
    assert adapter_bonsai.ollama_host == "http://127.0.0.1:11434"

    adapter_ollama = LLMClientAdapter(backend="ollama")
    assert adapter_ollama.backend == "ollama"

    singleton = get_llm_client(backend="bonsai", force_new=True)
    assert singleton.backend == "bonsai"


def test_tool_call_parsing_native():
    adapter = LLMClientAdapter(backend="bonsai")
    raw_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_courses_by_semester",
                "arguments": json.dumps({"program_studi": "TSD", "semester": 3}),
            },
        }
    ]
    parsed = adapter._parse_tool_calls(content="", raw_tool_calls=raw_calls)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0].function.name == "get_courses_by_semester"
    assert parsed[0].function.arguments == {"program_studi": "TSD", "semester": 3}
    assert parsed[0].id == "call_123"


def test_tool_call_parsing_markdown_json():
    adapter = LLMClientAdapter(backend="bonsai")
    content = """Saya akan mengecek daftar mata kuliah untuk Anda.
```json
{
  "action": "get_courses_by_semester",
  "action_input": {
    "program_studi": "Teknologi Sains Data",
    "semester": 2
  }
}
```
"""
    parsed = adapter._parse_tool_calls(content=content)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0].function.name == "get_courses_by_semester"
    assert parsed[0].function.arguments["program_studi"] == "Teknologi Sains Data"
    assert parsed[0].function.arguments["semester"] == 2


def test_tool_call_parsing_tool_call_tag():
    adapter = LLMClientAdapter(backend="bonsai")
    content = '<tool_call>{"name": "calculate_sks_limit", "arguments": {"ips": 3.45}}</tool_call>'
    parsed = adapter._parse_tool_calls(content=content)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0].function.name == "calculate_sks_limit"
    assert parsed[0].function.arguments["ips"] == 3.45


def test_bonsai_chat_success():
    adapter = LLMClientAdapter(backend="bonsai", bonsai_host="http://mock-bonsai:8080")

    mock_resp_json = {
        "id": "chatcmpl-mock",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Halo! Ada yang bisa saya bantu terkait akademik FTMM?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 12},
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_resp_json

    with patch("httpx.Client.post", return_value=mock_resp):
        res = adapter.chat(messages=[{"role": "user", "content": "halo"}])
        assert res.backend == "bonsai"
        assert res.message.content == "Halo! Ada yang bisa saya bantu terkait akademik FTMM?"
        assert res.prompt_eval_count == 15
        assert res.eval_count == 12
        assert res.get("eval_count") == 12


def test_bonsai_failover_to_ollama():
    adapter = LLMClientAdapter(
        backend="bonsai",
        bonsai_host="http://offline-bonsai:8080",
        fallback_model="qwen2.5:3b",
        enable_fallback=True,
    )

    # Mock Ollama fallback response
    mock_ollama_resp = MagicMock()
    mock_ollama_resp.prompt_eval_count = 20
    mock_ollama_resp.eval_count = 18
    mock_ollama_resp.message = MagicMock()
    mock_ollama_resp.message.content = "Respons fallback dari Ollama qwen2.5:3b."
    mock_ollama_resp.message.tool_calls = None
    mock_ollama_resp.get.side_effect = lambda k, d=None: getattr(mock_ollama_resp, k, d)

    with patch(
        "httpx.Client.post",
        side_effect=httpx.ConnectError("Connection refused to offline-bonsai:8080"),
    ):
        with patch("src.engine.base.get_ollama_client") as mock_get_ollama:
            mock_client = MagicMock()
            mock_client.chat.return_value = mock_ollama_resp
            mock_get_ollama.return_value = mock_client

            res = adapter.chat(messages=[{"role": "user", "content": "halo"}])

            assert res.backend == "ollama"
            assert res.model == "qwen2.5:3b"
            assert res.message.content == "Respons fallback dari Ollama qwen2.5:3b."
            mock_client.chat.assert_called_once()


def test_react_agent_with_bonsai_client():
    mock_adapter = MagicMock()

    # Step 1: Model returns tool call
    step1_tc = NormalizedToolCall(
        name="get_courses_by_semester",
        arguments={"program_studi": "TSD", "semester": 1},
    )
    step1_msg = NormalizedChatMessage(
        role="assistant",
        content="Memanggil tool katalog semester 1...",
        tool_calls=[step1_tc],
    )
    step1_resp = NormalizedChatResponse(
        message=step1_msg,
        prompt_eval_count=30,
        eval_count=15,
        model="bonsai-2-4b",
        backend="bonsai",
    )

    # Step 2: Model returns final answer
    step2_msg = NormalizedChatMessage(
        role="assistant",
        content="Berikut daftar mata kuliah semester 1 Teknologi Sains Data: Kalkulus I, Alpro I.",
        tool_calls=None,
    )
    step2_resp = NormalizedChatResponse(
        message=step2_msg,
        prompt_eval_count=50,
        eval_count=25,
        model="bonsai-2-4b",
        backend="bonsai",
    )

    mock_adapter.chat.side_effect = [step1_resp, step2_resp]

    engine = ReActAgentEngine(model="bonsai-2-4b", llm_client=mock_adapter)
    result = engine.generate_answer(query="Daftar matkul semester 1 TSD apa saja?")

    assert result.architecture == "ReAct Agent (Native Tool-Calling)"
    assert "TSD" in result.answer or "Kalkulus" in result.answer
    assert mock_adapter.chat.call_count == 2
    assert result.metadata["tools_executed"][0]["tool"] == "get_courses_by_semester"


def test_health_check_endpoint_schema_and_status():
    async def _run_tests():
        with patch.dict(os.environ, {"LLM_BACKEND": "bonsai"}):
            with patch("httpx.AsyncClient.get") as mock_get:
                # Case 1: Both Bonsai and Ollama online
                mock_ok = MagicMock()
                mock_ok.status_code = 200
                mock_get.return_value = mock_ok

                res_healthy = await handle_health_check(strict=False)
                assert isinstance(res_healthy, HealthResponse)
                assert res_healthy.status == "healthy"
                assert res_healthy.active_backend == "bonsai"
                assert res_healthy.bonsai_online is True
                assert res_healthy.ollama_online is True
                assert res_healthy.generator_model == "bonsai-2-4b"
                assert res_healthy.embedding_model == "nomic-embed-text-v2-moe"

                # Case 2: Bonsai offline, Ollama online -> degraded
                def side_effect_probe(url):
                    resp = MagicMock()
                    if "8080" in url:
                        resp.status_code = 503
                    else:
                        resp.status_code = 200
                    return resp

                mock_get.side_effect = side_effect_probe
                res_degraded = await handle_health_check(strict=False)
                assert res_degraded.status == "degraded"
                assert res_degraded.bonsai_online is False
                assert res_degraded.ollama_online is True

                # Case 3: Both offline with strict=True -> raises HTTPException 503
                mock_fail = MagicMock()
                mock_fail.status_code = 500
                mock_get.side_effect = None
                mock_get.return_value = mock_fail

                with pytest.raises(HTTPException) as exc_info:
                    await handle_health_check(strict=True)
                assert exc_info.value.status_code == 503
