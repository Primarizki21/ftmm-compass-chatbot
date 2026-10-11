import pytest
from src.engine.tools import ACADEMIC_TOOL_DEFINITIONS, TOOL_REGISTRY
from src.engine.tools.academic_tools import (
    get_courses_by_semester,
    normalize_program_studi,
    extract_semester_from_text,
)
from src.engine.decider import System1Decider, DecisionCategory, SEMESTER_COURSE_FAST_PATH_REGEX
from src.guardrails.tier4_tool_rail import Tier4ToolRail


def test_normalize_program_studi():
    assert normalize_program_studi("TSD") == "Teknologi Sains Data"
    assert normalize_program_studi("tsd") == "Teknologi Sains Data"
    assert normalize_program_studi("Data Science") == "Teknologi Sains Data"
    assert normalize_program_studi("TRKB") == "Teknik Robotika dan Kecerdasan Buatan"
    assert normalize_program_studi("robotika") == "Teknik Robotika dan Kecerdasan Buatan"
    assert normalize_program_studi("RN") == "S1 Rekayasa Nanoteknologi"
    assert normalize_program_studi("nanoteknologi") == "S1 Rekayasa Nanoteknologi"
    assert normalize_program_studi("TI") == "Teknik Industri"
    assert normalize_program_studi("teknik industri") == "Teknik Industri"
    assert normalize_program_studi("TE") == "Teknik Elektro"
    assert normalize_program_studi("teknik elektro") == "Teknik Elektro"


def test_extract_semester_from_text():
    assert extract_semester_from_text("mata kuliah semester 3") == 3
    assert extract_semester_from_text("matkul sem 5") == 5
    assert extract_semester_from_text("smt 4 apa aja") == 4
    assert extract_semester_from_text("semester iv") == 4
    assert extract_semester_from_text("semester viii") == 8
    assert extract_semester_from_text("halo apa kabar") is None


def test_get_courses_by_semester_deterministic_values():
    # TSD Semester 3: exactly 10 courses, 27 SKS
    res_tsd3 = get_courses_by_semester("TSD", 3)
    assert res_tsd3["program_studi"] == "Teknologi Sains Data"
    assert res_tsd3["semester"] == 3
    assert res_tsd3["total_courses"] == 10
    assert res_tsd3["total_sks"] == 27
    course_codes = [c["kode_mk"] for c in res_tsd3["courses"]]
    assert "BAE110" in course_codes
    assert "SID205" in course_codes

    # RN Semester 1: exactly 16 courses, 29 SKS
    res_rn1 = get_courses_by_semester("RN", 1)
    assert res_rn1["program_studi"] == "S1 Rekayasa Nanoteknologi"
    assert res_rn1["total_courses"] == 16
    assert res_rn1["total_sks"] == 29

    # TI Semester 2: exactly 9 courses, 19 SKS
    res_ti2 = get_courses_by_semester("TI", 2)
    assert res_ti2["total_courses"] == 9
    assert res_ti2["total_sks"] == 19


def test_get_courses_by_semester_filter_jenis():
    res_all = get_courses_by_semester("TSD", 3)
    res_wajib = get_courses_by_semester("TSD", 3, jenis="wajib")
    res_pilihan = get_courses_by_semester("TSD", 3, jenis="pilihan")

    assert res_wajib["total_courses"] + res_pilihan["total_courses"] == res_all["total_courses"]
    assert all("wajib" in c["jenis"].lower() for c in res_wajib["courses"])
    assert all("pilih" in c["jenis"].lower() for c in res_pilihan["courses"])


def test_get_courses_by_semester_out_of_bounds():
    res_invalid = get_courses_by_semester("TSD", 9)
    assert "error" in res_invalid
    assert res_invalid["total_courses"] == 0
    assert res_invalid["courses"] == []

    res_zero = get_courses_by_semester("TSD", 0)
    assert "error" in res_zero
    assert res_zero["total_courses"] == 0


def test_tool_registration():
    assert "get_courses_by_semester" in TOOL_REGISTRY
    assert any(t["function"]["name"] == "get_courses_by_semester" for t in ACADEMIC_TOOL_DEFINITIONS)


def test_decider_fast_path():
    decider = System1Decider()
    q = "mata kuliah TSD semester 3 apa aja?"
    assert bool(SEMESTER_COURSE_FAST_PATH_REGEX.search(q))

    res = decider.decide(q)
    assert res.category == DecisionCategory.COURSE_DISCOVERY
    assert res.selected_tools is not None
    tool_names = [t["function"]["name"] for t in res.selected_tools]
    assert "get_courses_by_semester" in tool_names
    assert res.latency_ms <= 10.0


def test_tier4_tool_rail_validation():
    rail = Tier4ToolRail()
    val_ok = rail.validate_call("get_courses_by_semester", {"program_studi": "TSD", "semester": 3})
    assert val_ok.is_valid

    val_bad_sem = rail.validate_call("get_courses_by_semester", {"program_studi": "TSD", "semester": 9})
    assert not val_bad_sem.is_valid
    assert "di luar rentang kurikulum" in val_bad_sem.error
