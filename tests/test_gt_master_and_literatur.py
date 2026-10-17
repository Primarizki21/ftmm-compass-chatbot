"""
Test suite validating the integrity, schema, and completeness of:
1. data/ground_truth/gt_master.json (449 items: 369 courses + 80 regulations & SOPs)
2. data/ground_truth/literatur.json (34 structured academic literature items)
"""

import json
from pathlib import Path

GT_MASTER_PATH = Path("data/ground_truth/gt_master.json")
LITERATUR_PATH = Path("data/ground_truth/literatur.json")
CATALOG_PATH = Path("data/courses_catalog.json")


def test_gt_master_file_exists():
    assert GT_MASTER_PATH.exists(), f"{GT_MASTER_PATH} does not exist"


def test_gt_master_total_count_and_domains():
    with open(GT_MASTER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 449, f"Expected 449 items in GT_MASTER, got {len(data)}"

    source_types = {}
    course_count = 0
    reg_count = 0
    for item in data:
        st = item["source_type"]
        source_types[st] = source_types.get(st, 0) + 1
        if st in ["Silabus Wajib", "Silabus Pilihan"]:
            course_count += 1
        elif st in ["Regulasi Akademik", "SOP Layanan"]:
            reg_count += 1

    assert course_count == 369, f"Expected 369 courses, got {course_count}"
    assert reg_count == 80, f"Expected 80 regulations/SOPs, got {reg_count}"

    assert source_types.get("Silabus Wajib") == 297
    assert source_types.get("Silabus Pilihan") == 72

def test_gt_master_sequential_ids_and_non_empty_fields():
    with open(GT_MASTER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_cols = ["id", "question", "answer", "source_id", "source_title", "source_section", "source_type"]
    for idx, item in enumerate(data, 1):
        assert item["id"] == str(idx), f"Expected sequential id '{idx}', got '{item['id']}'"
        for col in required_cols:
            assert col in item, f"Missing column '{col}' in item {idx}"
            assert str(item[col]).strip(), f"Column '{col}' in item {idx} cannot be empty"

        q = item["question"]
        assert "?" in q or q.endswith("."), f"Question {idx} must be meaningful text: {q}"


def test_gt_master_covers_all_catalog_codes():
    with open(GT_MASTER_PATH, "r", encoding="utf-8") as f:
        master_data = json.load(f)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    catalog_codes = set(c["kode_mk"].strip() for c in catalog_data)
    master_course_codes = set(
        item["source_id"].strip()
        for item in master_data
        if item.get("source_type") in ["Silabus Wajib", "Silabus Pilihan"]
    )

    assert master_course_codes == catalog_codes, "Mismatch between GT_MASTER course codes and catalog"


def test_literatur_file_exists():
    assert LITERATUR_PATH.exists(), f"{LITERATUR_PATH} does not exist"


def test_literatur_structure_and_fields():
    with open(LITERATUR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 28, f"Expected at least 28 literature items, got {len(data)}"

    required_cols = ["judul", "penulis", "tahun", "abstrak", "artinya untuk project ini"]
    for idx, item in enumerate(data, 1):
        for col in required_cols:
            assert col in item, f"Missing column '{col}' in literature item {idx}"
            assert str(item[col]).strip(), f"Column '{col}' in literature item {idx} cannot be empty"

        tahun = str(item["tahun"]).strip()
        assert tahun.isdigit() and len(tahun) == 4, f"Year must be 4-digit number: {tahun}"
        year_int = int(tahun)
        assert 2018 <= year_int <= 2026, f"Year out of expected range: {year_int}"

        # Ensure project meaning is substantive (at least 20 chars)
        meaning = item["artinya untuk project ini"]
        assert len(meaning) >= 20, f"Meaning for project too brief in item {idx}: {meaning}"
