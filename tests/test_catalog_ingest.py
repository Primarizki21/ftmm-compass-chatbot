import json
from pathlib import Path
import pytest
from scripts.ingest_faiss import (
    validate_course_schema,
    validate_all_courses,
    load_catalog_or_ekstrak,
    main as ingest_main,
)
from src.knowledge.retriever import CourseRetriever
from src.engine.tools.academic_tools import _get_catalog


CATALOG_PATH = Path("data/courses_catalog.json")
META_JSON_PATH = Path("data/faiss/ftmm_courses_meta.json")


def test_catalog_file_integrity():
    assert CATALOG_PATH.exists(), f"Katalog SSOT {CATALOG_PATH} tidak ditemukan!"
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)

    assert isinstance(courses, list)
    assert len(courses) == 369

    sample = courses[0]
    required_keys = {"kode_mk", "nama_mk", "sks", "semester", "program_studi"}
    assert required_keys.issubset(sample.keys())


def test_validate_course_schema_cases():
    valid_course = {
        "kode_mk": "TSD101",
        "nama_mk": "Pemrograman Dasar",
        "sks": 3,
        "semester": 1,
        "program_studi": "Teknologi Sains Data",
    }
    assert validate_course_schema(valid_course) == []

    # Empty kode_mk
    errs = validate_course_schema({**valid_course, "kode_mk": ""})
    assert any("kode_mk" in e for e in errs)

    # Invalid SKS (0 or > 20)
    errs = validate_course_schema({**valid_course, "sks": 0})
    assert any("sks" in e for e in errs)
    errs = validate_course_schema({**valid_course, "sks": 25})
    assert any("sks" in e for e in errs)

    # Invalid semester (0 or 9)
    errs = validate_course_schema({**valid_course, "semester": 9})
    assert any("semester" in e for e in errs)

    # Empty prodi
    errs = validate_course_schema({**valid_course, "program_studi": ""})
    assert any("program_studi" in e for e in errs)


def test_validate_all_courses_real_catalog():
    courses, _ = load_catalog_or_ekstrak(CATALOG_PATH)
    is_valid, errors = validate_all_courses(courses)
    assert is_valid is True
    assert len(errors) == 0


def test_ingest_cli_validate_only():
    # Should exit cleanly without throwing errors
    ingest_main(["--source", str(CATALOG_PATH), "--validate-only"])


def test_course_retriever_loads_json_metadata():
    assert META_JSON_PATH.exists(), f"Metadata JSON {META_JSON_PATH} belum dibuat!"
    retriever = CourseRetriever(meta_path=META_JSON_PATH)
    retriever.load()
    assert retriever._loaded is True
    assert retriever.metadata["total"] == 369
    assert len(retriever.metadata["courses"]) == 369
    assert len(retriever.metadata["chunks"]) == 369


def test_academic_tools_catalog_loading():
    catalog = _get_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) == 369
    first = catalog[0]
    assert "kode_mk" in first
    assert "nama_mk" in first
