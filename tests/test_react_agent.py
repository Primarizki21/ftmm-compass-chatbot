import pytest
from src.engine.tools import (
    search_courses,
    validate_prerequisites_dag,
    calculate_sks_limit,
    generate_degree_plan,
)
from src.engine.react_agent import ReActAgentEngine


def test_search_courses_tool():
    results = search_courses("pemrograman algoritma", top_k=2)
    assert len(results) >= 1
    assert any("algoritma" in r["nama_mk"].lower() or "pemrograman" in r["nama_mk"].lower() for r in results)


def test_calculate_sks_limit_boundaries():
    assert calculate_sks_limit(3.50)["max_sks"] == 24
    assert calculate_sks_limit(3.01)["max_sks"] == 24
    assert calculate_sks_limit(3.00)["max_sks"] == 20
    assert calculate_sks_limit(2.51)["max_sks"] == 20
    assert calculate_sks_limit(2.50)["max_sks"] == 18
    assert calculate_sks_limit(2.00)["max_sks"] == 18
    assert calculate_sks_limit(1.99)["max_sks"] == 15


def test_validate_prerequisites_dag_eligible():
    res = validate_prerequisites_dag("Kalkulus II", ["Kalkulus I"])
    assert res["found"] is True
    assert res["is_eligible"] is True


def test_validate_prerequisites_dag_ineligible():
    res = validate_prerequisites_dag("Data Mining I", [])
    assert res["found"] is True
    assert res["is_eligible"] is False
    assert len(res["missing_prerequisites"]) > 0


def test_generate_degree_plan_payload():
    plan_payload = generate_degree_plan(
        program_studi="Teknologi Sains Data",
        current_semester=3,
        target_semester=7,
        passed_courses=["Kalkulus I", "Algoritma Pemrograman I"],
        focus_tracks=["Machine Learning"]
    )
    assert plan_payload["action"] == "APPLY_DEGREE_PLAN"
    assert "summary" in plan_payload
    assert plan_payload["summary"]["program_studi"] == "Teknologi Sains Data"
    assert "plan" in plan_payload
    assert len(plan_payload["plan"]) == 7
