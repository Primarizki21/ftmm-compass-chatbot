import json
from pathlib import Path
from typing import Any
import pytest

from scripts.evaluate_slm_sizing import (
    evaluate_tool_adherence,
    execute_physical_tool,
    calculate_pareto_analysis,
    DEFAULT_DATASET,
)


def test_slm_dataset_integrity():
    assert DEFAULT_DATASET.exists(), f"Dataset file {DEFAULT_DATASET} must exist"
    with open(DEFAULT_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 50, f"Dataset must contain exactly 50 queries, found {len(data)}"

    categories = [d["category"] for d in data]
    expected_categories = [
        "SKS_CALCULATION",
        "PREREQUISITE_DAG",
        "COURSE_DISCOVERY",
        "DEGREE_PLANNING",
        "CHITCHAT",
    ]

    for cat in expected_categories:
        count = categories.count(cat)
        assert count == 10, f"Category '{cat}' must have exactly 10 queries, found {count}"

    for item in data:
        assert "id" in item
        assert "category" in item
        assert "query" in item
        assert "expected_tool" in item
        assert "required_params" in item


def test_evaluate_tool_adherence_sks():
    # 1. Correct tool call & correct parameter
    res = evaluate_tool_adherence(
        category="SKS_CALCULATION",
        expected_tool="calculate_sks_limit",
        required_params=["ips"],
        tool_calls=[{"function": {"name": "calculate_sks_limit", "arguments": {"ips": 3.85}}}],
        raw_content="",
        expected_ips=3.85,
    )
    assert res["tool_selection_ok"] is True
    assert res["schema_compliant"] is True
    assert res["param_error"] is False
    assert res["exec_success"] is True

    # 2. Correct tool call but wrong IPS value (e.g. truncated integer)
    res_bad_ips = evaluate_tool_adherence(
        category="SKS_CALCULATION",
        expected_tool="calculate_sks_limit",
        required_params=["ips"],
        tool_calls=[{"function": {"name": "calculate_sks_limit", "arguments": {"ips": 2.0}}}],
        raw_content="",
        expected_ips=3.85,
    )
    assert res_bad_ips["tool_selection_ok"] is True
    assert res_bad_ips["param_error"] is True

    # 3. Missing tool call (hallucinated narrative text)
    res_no_tool = evaluate_tool_adherence(
        category="SKS_CALCULATION",
        expected_tool="calculate_sks_limit",
        required_params=["ips"],
        tool_calls=[],
        raw_content="Batas SKS Anda adalah 24 SKS.",
        expected_ips=3.85,
    )
    assert res_no_tool["tool_selection_ok"] is False
    assert res_no_tool["schema_compliant"] is False
    assert res_no_tool["param_error"] is True


def test_evaluate_tool_adherence_chitchat():
    # 1. Correct chitchat (no tool called)
    res = evaluate_tool_adherence(
        category="CHITCHAT",
        expected_tool=None,
        required_params=[],
        tool_calls=[],
        raw_content="Halo! Saya Compass AI, asisten akademik FTMM.",
    )
    assert res["tool_selection_ok"] is True
    assert res["schema_compliant"] is True
    assert res["param_error"] is False
    assert res["exec_success"] is True

    # 2. Incorrect tool call on chitchat (unnecessary tool invocation)
    res_unneeded = evaluate_tool_adherence(
        category="CHITCHAT",
        expected_tool=None,
        required_params=[],
        tool_calls=[{"function": {"name": "search_courses", "arguments": {"query": "halo"}}}],
        raw_content="",
    )
    assert res_unneeded["tool_selection_ok"] is False
    assert res_unneeded["param_error"] is True


def test_evaluate_tool_adherence_course_discovery():
    res = evaluate_tool_adherence(
        category="COURSE_DISCOVERY",
        expected_tool="get_courses_by_semester",
        required_params=["program_studi", "semester"],
        tool_calls=[
            {
                "function": {
                    "name": "get_courses_by_semester",
                    "arguments": {"program_studi": "TSD", "semester": 3},
                }
            }
        ],
        raw_content="",
        expected_semester=3,
    )
    assert res["tool_selection_ok"] is True
    assert res["schema_compliant"] is True
    assert res["param_error"] is False
    assert res["exec_success"] is True


def test_evaluate_tool_adherence_prerequisite_dag():
    res = evaluate_tool_adherence(
        category="PREREQUISITE_DAG",
        expected_tool="validate_prerequisites_dag",
        required_params=["course_code_or_name"],
        tool_calls=[
            {
                "function": {
                    "name": "validate_prerequisites_dag",
                    "arguments": {
                        "course_code_or_name": "Pembelajaran Mesin",
                        "passed_courses": ["Aljabar Linier"],
                    },
                }
            }
        ],
        raw_content="",
    )
    assert res["tool_selection_ok"] is True
    assert res["schema_compliant"] is True
    assert res["param_error"] is False
    assert res["exec_success"] is True


def test_execute_physical_tool():
    # Valid calculate_sks_limit
    ok, res, err = execute_physical_tool("calculate_sks_limit", {"ips": 3.75})
    assert ok is True
    assert res["max_sks"] == 24
    assert err is None

    # Valid get_courses_by_semester
    ok, res, err = execute_physical_tool(
        "get_courses_by_semester", {"program_studi": "Teknologi Sains Data", "semester": 1}
    )
    assert ok is True
    assert "courses" in res
    assert len(res["courses"]) > 0

    # Unregistered tool
    ok, res, err = execute_physical_tool("non_existent_tool", {})
    assert ok is False
    assert "tidak terdaftar" in err


def test_calculate_pareto_analysis():
    dummy_summaries = [
        {
            "model": "model_fast_low_acc",
            "tool_selection_accuracy_pct": 50.0,
            "json_schema_compliance_pct": 50.0,
            "parameter_extraction_error_pct": 50.0,
            "p50_hardware_ttft_ms": 100.0,
            "peak_vram_mib": 1500,
        },
        {
            "model": "model_balanced",
            "tool_selection_accuracy_pct": 90.0,
            "json_schema_compliance_pct": 95.0,
            "parameter_extraction_error_pct": 10.0,
            "p50_hardware_ttft_ms": 300.0,
            "peak_vram_mib": 2500,
        },
        {
            "model": "model_slow_high_acc",
            "tool_selection_accuracy_pct": 98.0,
            "json_schema_compliance_pct": 98.0,
            "parameter_extraction_error_pct": 2.0,
            "p50_hardware_ttft_ms": 900.0,
            "peak_vram_mib": 5500,
        },
    ]

    enhanced = calculate_pareto_analysis(dummy_summaries)
    assert len(enhanced) == 3

    for item in enhanced:
        assert "adherence_composite_score" in item
        assert "pareto_efficiency_score" in item
        assert "is_pareto_frontier" in item

    # Both extreme trade-offs and balanced can be on Pareto frontier
    frontier_models = [m["model"] for m in enhanced if m["is_pareto_frontier"]]
    assert len(frontier_models) >= 1
