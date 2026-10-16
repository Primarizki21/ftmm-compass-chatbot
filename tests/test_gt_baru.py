"""
Test suite validating the integrity and style consistency of GT_BARU dataset.
Covers:
- Complete catalog coverage: 369 courses (297 MK Wajib + 72 MK Pilihan)
- Code validation against data/courses_catalog.json
- Linguistic consistency matching sheet GT patterns:
  - Questions start with 'Pak/Bu, ...' or 'Permisi Pak/Bu, ...'
  - Answers start with 'Berdasarkan silabus, ...'
- Zero null fields or empty strings.
"""

import json
from pathlib import Path

GT_BARU_PATH = Path("data/ground_truth/gt_baru_courses.json")
CATALOG_PATH = Path("data/courses_catalog.json")


def test_gt_baru_file_exists():
    assert GT_BARU_PATH.exists(), f"{GT_BARU_PATH} does not exist"


def test_gt_baru_total_count_and_types():
    with open(GT_BARU_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 369, f"Expected 369 courses, got {len(data)}"

    type_counts = {}
    prodi_counts = {}
    for item in data:
        t = item["source_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        p = item["program_studi"]
        prodi_counts[p] = prodi_counts.get(p, 0) + 1

    assert type_counts.get("Silabus Wajib") == 297, f"Expected 297 MK Wajib, got {type_counts.get('Silabus Wajib')}"
    assert type_counts.get("Silabus Pilihan") == 72, f"Expected 72 MK Pilihan, got {type_counts.get('Silabus Pilihan')}"

    assert prodi_counts.get("Teknologi Sains Data") == 68
    assert prodi_counts.get("Teknik Robotika dan Kecerdasan Buatan") == 72
    assert prodi_counts.get("S1 Rekayasa Nanoteknologi") == 79
    assert prodi_counts.get("Teknik Industri") == 54
    assert prodi_counts.get("Teknik Elektro") == 96


def test_gt_baru_matches_catalog_codes():
    with open(GT_BARU_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    catalog_kodes = set(c["kode_mk"].strip() for c in catalog_data)
    gt_kodes = set(item["source_id"].strip() for item in gt_data)

    assert gt_kodes == catalog_kodes, f"Mismatch between GT_BARU course codes and catalog: diff={gt_kodes.symmetric_difference(catalog_kodes)}"


def test_gt_baru_linguistic_style_consistency():
    with open(GT_BARU_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_columns = ["id", "question", "answer", "source_id", "source_title", "source_section", "source_type"]

    for idx, item in enumerate(data):
        for col in required_columns:
            assert col in item, f"Missing column '{col}' in item {idx}"
            assert str(item[col]).strip(), f"Column '{col}' in item {idx} cannot be empty"

        # Question style check
        q = item["question"]
        assert q.startswith("Pak/Bu,") or q.startswith("Permisi Pak/Bu,"), f"Question {idx} violates polite style: {q}"
        assert "?" in q, f"Question {idx} must be a question ending with '?': {q}"

        # Answer style check
        a = item["answer"]
        assert a.startswith("Berdasarkan silabus,"), f"Answer {idx} violates sheet GT style formula: {a}"
        assert str(item["source_id"]) in a, f"Answer {idx} must cite course code: {a}"
        assert "SKS" in a, f"Answer {idx} must include SKS credit info: {a}"
