"""
Unit Tests and Regression Suite for Neuro-Symbolic CSP Degree Planner (Exp-020).
"""

import pytest
from src.engine.tools.academic_tools import (
    generate_degree_plan,
    normalize_program_studi,
    _get_catalog,
    compress_degree_plan_for_llm,
    format_degree_plan_response,
)


def test_normalize_program_studi_coverage():
    # 5 prodi acronyms & slang variants
    assert normalize_program_studi("TSD") == "Teknologi Sains Data"
    assert normalize_program_studi("tsd") == "Teknologi Sains Data"
    assert normalize_program_studi("sains data") == "Teknologi Sains Data"
    assert normalize_program_studi("data science") == "Teknologi Sains Data"
    assert normalize_program_studi("data saintis") == "Teknologi Sains Data"

    assert normalize_program_studi("TRKB") == "Teknik Robotika dan Kecerdasan Buatan"
    assert normalize_program_studi("robotika") == "Teknik Robotika dan Kecerdasan Buatan"
    assert normalize_program_studi("ai robot") == "Teknik Robotika dan Kecerdasan Buatan"

    assert normalize_program_studi("RN") == "S1 Rekayasa Nanoteknologi"
    assert normalize_program_studi("nanoteknologi") == "S1 Rekayasa Nanoteknologi"
    assert normalize_program_studi("nanotek") == "S1 Rekayasa Nanoteknologi"

    assert normalize_program_studi("TI") == "Teknik Industri"
    assert normalize_program_studi("industri") == "Teknik Industri"
    assert normalize_program_studi("industrial") == "Teknik Industri"

    assert normalize_program_studi("TE") == "Teknik Elektro"
    assert normalize_program_studi("elektro") == "Teknik Elektro"
    assert normalize_program_studi("electrical") == "Teknik Elektro"


def test_degree_plan_tsd_llm_user_query():
    """
    Uji regresi terhadap kueri aktual user:
    'saya mahasiswa semester 2 ingin membuat rencana studi dari semester 3 hingga 7. tolong buat rencana studi TSD dengan fokus LLM'
    """
    res = generate_degree_plan(
        program_studi="TSD",
        current_semester=3,
        target_semester=7,
        passed_courses=[],
        focus_tracks=["LLM"],
        max_sks_per_sem=24,
    )

    summary = res["summary"]
    plan = res["plan"]

    # 1. Pastikan prodi ternormalisasi ke TSD bukan Nanoteknologi
    assert summary["program_studi"] == "Teknologi Sains Data"

    # 2. Pastikan semester 3 s.d. 7 TIDAK kosong
    for sem in range(3, 8):
        assert len(plan[sem]) > 0, f"Semester {sem} tidak boleh kosong!"
        sks = sum(c["credits"] for c in plan[sem])
        assert sks <= 24, f"Semester {sem} melanggar batas 24 SKS: {sks}"

    # 3. Pastikan tidak ada matkul Nanoteknologi (Zero-fallback invariant)
    for sem, courses in plan.items():
        for c in courses:
            assert "nanoteknologi" not in c["name"].lower()
            assert "nanokomposit" not in c["name"].lower()
            assert "spektroskopi" not in c["name"].lower()

    # 4. Pastikan mata kuliah NLP/AI masuk di daftar rencana pilihan
    all_planned_codes = [c["code"] for sem, courses in plan.items() for c in courses]
    assert "SIC304" in all_planned_codes or "SIC203" in all_planned_codes  # NLP or Pengantar AI


def test_degree_plan_single_religion_dedup():
    """Memastikan hanya 1 varian agama yang dimasukkan dalam rencana studi."""
    res = generate_degree_plan(
        program_studi="Teknologi Sains Data",
        current_semester=1,
        target_semester=8,
        passed_courses=[],
        max_sks_per_sem=24,
    )

    plan = res["plan"]
    # Di semester 1 hanya boleh ada 1 mata kuliah agama
    sem1_religions = [c for c in plan[1] if "agama" in c["name"].lower()]
    assert len(sem1_religions) == 1, f"Harus tepat 1 mata kuliah agama di sem 1, ditemukan: {len(sem1_religions)}"


def test_degree_plan_parity_compliance():
    """Memastikan paritas semester ganjil/genap ditaati."""
    res = generate_degree_plan(
        program_studi="Teknik Elektro",
        current_semester=2,
        target_semester=6,
        passed_courses=[],
        max_sks_per_sem=24,
    )

    plan = res["plan"]
    for sem in range(2, 7):
        expected_parity = "odd" if sem % 2 == 1 else "even"
        for c in plan[sem]:
            assert c["parity"] == expected_parity, f"Matkul {c['name']} melanggar paritas semester {sem}"


def test_degree_plan_all_five_prodis():
    """Memvalidasi pembuatan degree plan untuk ke-5 prodi resmi FTMM."""
    prodis = ["TSD", "TRKB", "RN", "TI", "TE"]
    for p in prodis:
        res = generate_degree_plan(
            program_studi=p,
            current_semester=1,
            target_semester=8,
            max_sks_per_sem=24,
        )
        assert res["action"] == "APPLY_DEGREE_PLAN"
        assert res["summary"]["total_credits"] >= 100
        assert len(res["plan"]) == 8


def test_compress_degree_plan_and_format_response():
    """Memastikan payload ringkas untuk LLM dan formatting respons lengkap 8 semester."""
    res = generate_degree_plan(
        program_studi="Teknologi Sains Data",
        current_semester=1,
        target_semester=8,
        focus_tracks=["Statistika"],
        max_sks_per_sem=24,
    )
    # 1. Test compression for LLM context
    compact = compress_degree_plan_for_llm(res)
    assert compact["status"] == "success"
    assert compact["action"] == "PLAN_GENERATED"
    assert len(compact["plan"]) == 8
    for sem in range(1, 9):
        assert "total_sks" in compact["plan"][sem]
        assert "courses" in compact["plan"][sem]
        # Ensure no verbose description paragraphs in compact plan
        for c_str in compact["plan"][sem]["courses"]:
            assert isinstance(c_str, str)
            assert "\n" not in c_str

    # 2. Test deterministic Markdown formatting
    formatted_id = format_degree_plan_response(res, language="id")
    assert "Teknologi Sains Data" in formatted_id
    for sem in range(1, 9):
        assert f"### Semester {sem}" in formatted_id
    assert "Total Beban Studi:" in formatted_id

    formatted_en = format_degree_plan_response(res, language="en")
    assert "Here is the recommended study plan" in formatted_en
    for sem in range(1, 9):
        assert f"### Semester {sem}" in formatted_en
    assert "Total Credits:" in formatted_en


def test_no_database_id_exposed_in_degree_plan():
    """Memastikan internal database id (seperti MK-068) tidak menimpa kode resmi mata kuliah."""
    res = generate_degree_plan(
        program_studi="Teknologi Sains Data",
        current_semester=1,
        target_semester=8,
        max_sks_per_sem=24,
    )
    for sem, courses in res["plan"].items():
        for c in courses:
            assert not str(c["id"]).startswith("MK-"), f"Internal DB id exposed: {c['id']}"
            assert c["code"] != ""
