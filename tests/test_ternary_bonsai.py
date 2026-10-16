import pytest
from scripts.evaluate_ternary_bonsai import (
    parse_model_response,
    evaluate_tool_adherence,
    get_client_endpoint_for_model,
    calculate_pareto_analysis,
    OLLAMA_BASE_URL,
    BONSAI_BASE_URL,
)


def test_parse_model_response_direct_api():
    raw_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculate_sks_limit",
                "arguments": '{"ips": 3.85}',
            },
        }
    ]
    parsed = parse_model_response(content="", tool_calls_raw=raw_calls)
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "calculate_sks_limit"
    assert parsed[0]["function"]["arguments"] == {"ips": 3.85}


def test_parse_model_response_markdown_json_action():
    content = """Untuk menghitung batas SKS mahasiswa, saya akan memanggil fungsi terkait:
```json
{
  "action": "calculate_sks_limit",
  "action_input": {
    "ips": 3.50
  }
}
```
"""
    parsed = parse_model_response(content=content)
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "calculate_sks_limit"
    assert parsed[0]["function"]["arguments"] == {"ips": 3.50}


def test_parse_model_response_markdown_json_name():
    content = """Berikut pengecekan prasyarat mata kuliah:
```json
{
  "name": "validate_prerequisites_dag",
  "arguments": {
    "course_code_or_name": "TSD101",
    "passed_courses": ["MAT101"]
  }
}
```
"""
    parsed = parse_model_response(content=content)
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "validate_prerequisites_dag"
    assert parsed[0]["function"]["arguments"]["course_code_or_name"] == "TSD101"
    assert parsed[0]["function"]["arguments"]["passed_courses"] == ["MAT101"]


def test_parse_model_response_react_pattern():
    content = """Thought: Pengguna ingin mengetahui batas SKS dengan IPS 2.75.
Action: calculate_sks_limit
Action Input: {"ips": 2.75}
Observation: Menunggu hasil eksekusi...
"""
    parsed = parse_model_response(content=content)
    assert len(parsed) == 1
    assert parsed[0]["function"]["name"] == "calculate_sks_limit"
    assert parsed[0]["function"]["arguments"] == {"ips": 2.75}


def test_parse_model_response_chitchat_empty():
    content = "Halo! Saya Compass AI, asisten resmi akademik FTMM UNAIR. Ada yang bisa saya bantu hari ini?"
    parsed = parse_model_response(content=content)
    assert parsed == []


def test_parse_model_response_malformed_json():
    content = "```json\n{action: calculate_sks_limit, ips: unclosed\n```"
    parsed = parse_model_response(content=content)
    assert isinstance(parsed, list)


def test_evaluate_tool_adherence_correct():
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sks_limit",
                "arguments": {"ips": 3.5},
            },
        }
    ]
    res = evaluate_tool_adherence(
        category="SKS_CALCULATION",
        expected_tool="calculate_sks_limit",
        required_params=["ips"],
        tool_calls=tool_calls,
        raw_content="",
        expected_ips=3.5,
    )
    assert res["tool_selection_ok"] is True
    assert res["schema_compliant"] is True
    assert res["param_error"] is False
    assert res["exec_success"] is True


def test_evaluate_tool_adherence_param_out_of_bounds():
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sks_limit",
                "arguments": {"ips": 5.0},
            },
        }
    ]
    res = evaluate_tool_adherence(
        category="SKS_CALCULATION",
        expected_tool="calculate_sks_limit",
        required_params=["ips"],
        tool_calls=tool_calls,
        raw_content="",
    )
    assert res["tool_selection_ok"] is True
    assert res["param_error"] is True
    assert "out of bounds" in res["error_reason"]


def test_evaluate_tool_adherence_chitchat():
    # Chitchat without tool calls -> success
    res_clean = evaluate_tool_adherence(
        category="CHITCHAT",
        expected_tool=None,
        required_params=[],
        tool_calls=[],
        raw_content="Halo",
    )
    assert res_clean["tool_selection_ok"] is True
    assert res_clean["param_error"] is False
    assert res_clean["exec_success"] is True

    # Chitchat with unexpected tool call -> failure
    bad_tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "calculate_sks_limit",
                "arguments": {"ips": 3.0},
            },
        }
    ]
    res_bad = evaluate_tool_adherence(
        category="CHITCHAT",
        expected_tool=None,
        required_params=[],
        tool_calls=bad_tool_calls,
        raw_content="",
    )
    assert res_bad["tool_selection_ok"] is False
    assert res_bad["param_error"] is True


def test_get_client_endpoint_for_model():
    url_ollama, m_ollama = get_client_endpoint_for_model("qwen2.5:3b")
    assert url_ollama == OLLAMA_BASE_URL
    assert m_ollama == "qwen2.5:3b"

    url_bonsai, m_bonsai = get_client_endpoint_for_model("bonsai-2-4b")
    assert url_bonsai == BONSAI_BASE_URL
    assert m_bonsai == "bonsai-2-4b"


def test_calculate_pareto_analysis():
    mock_summaries = [
        {
            "model": "model_fast_low_acc",
            "tool_selection_accuracy_pct": 50.0,
            "json_schema_compliance_pct": 50.0,
            "parameter_extraction_error_pct": 50.0,
            "p50_hardware_ttft_ms": 10.0,
            "peak_vram_mib": 500,
        },
        {
            "model": "model_accurate_heavy",
            "tool_selection_accuracy_pct": 100.0,
            "json_schema_compliance_pct": 100.0,
            "parameter_extraction_error_pct": 0.0,
            "p50_hardware_ttft_ms": 100.0,
            "peak_vram_mib": 4000,
        },
    ]
    pareto_results = calculate_pareto_analysis(mock_summaries)
    assert len(pareto_results) == 2
    assert "adherence_composite_score" in pareto_results[0]
    assert "pareto_efficiency_score" in pareto_results[0]
    assert "is_pareto_frontier" in pareto_results[0]
