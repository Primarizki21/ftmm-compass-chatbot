import json
from pathlib import Path
import numpy as np
import pytest

from scripts.evaluate_embedding_benchmark import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_IR_DATASET_PATH,
    ROUTER_CATEGORIES,
    ROUTER_EVAL_DATASET,
    compute_ir_metrics,
)
from scripts.ingest_faiss import format_course_chunk, validate_all_courses


def test_catalog_integrity():
    assert DEFAULT_CATALOG_PATH.exists(), "Katalog kurikulum tidak ditemukan!"
    with open(DEFAULT_CATALOG_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)
    assert len(courses) == 369, f"Total mata kuliah harus 369, ditemukan {len(courses)}"
    is_valid, errors = validate_all_courses(courses)
    assert is_valid, f"Validasi skema katalog gagal: {errors}"


def test_ir_eval_dataset_integrity():
    assert DEFAULT_IR_DATASET_PATH.exists(), "IR eval dataset tidak ditemukan!"
    with open(DEFAULT_IR_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 149, f"Total kueri harus 149, ditemukan {len(data)}"

    registers = {}
    for item in data:
        r = item.get("register")
        registers[r] = registers.get(r, 0) + 1

    assert registers.get("formal") == 108, f"Kueri formal harus 108, ditemukan {registers.get('formal')}"
    assert registers.get("informal") == 21, f"Kueri informal/slang harus 21, ditemukan {registers.get('informal')}"
    assert registers.get("bilingual") == 20, f"Kueri bilingual harus 20, ditemukan {registers.get('bilingual')}"


def test_router_eval_dataset_integrity():
    assert len(ROUTER_EVAL_DATASET) == 60, f"Router dataset harus 60 kueri, ditemukan {len(ROUTER_EVAL_DATASET)}"
    cat_counts = {}
    style_counts = {}
    for item in ROUTER_EVAL_DATASET:
        c = item["category"]
        s = item["style"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
        style_counts[s] = style_counts.get(s, 0) + 1

    # 10 queries per class
    for cat in ROUTER_CATEGORIES:
        assert cat_counts.get(cat.value) == 10, f"Kategori {cat.value} harus 10 kueri, ada {cat_counts.get(cat.value)}"

    assert style_counts.get("Formal") == 24
    assert style_counts.get("Slang") == 18
    assert style_counts.get("Implicit") == 18


def test_format_course_chunk():
    dummy = {
        "kode_mk": "TSD101",
        "nama_mk": "Kalkulus I",
        "sks": 3,
        "semester": 1,
        "program_studi": "Teknologi Sains Data",
        "jenis": "wajib",
        "prasyarat": [],
        "deskripsi": "Dasar kalkulus differensial dan integral",
        "capaian_pembelajaran": ["Mampu menghitung limit dan turunan"],
        "dosen_pengampu": ["Dr. Matematika"],
    }
    chunk = format_course_chunk(dummy)
    assert "TSD101" in chunk
    assert "Kalkulus I" in chunk
    assert "SKS: 3" in chunk
    assert "Semester: 1" in chunk
    assert "Teknologi Sains Data" in chunk


def test_compute_ir_metrics_perfect_hit():
    retrieved = [
        {"course_id": "MK-001", "kode_mk": "TSD101", "nama_mk": "Kalkulus I", "score": 0.95},
        {"course_id": "MK-002", "kode_mk": "TSD102", "nama_mk": "Alpro I", "score": 0.85},
    ]
    target_ids = ["MK-001"]
    target_kodes = ["TSD101"]

    m = compute_ir_metrics(retrieved, target_ids, target_kodes)
    assert m["hit@1"] == 1.0
    assert m["hit@3"] == 1.0
    assert m["hit@5"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ndcg@5"] == 1.0


def test_compute_ir_metrics_rank_2_hit():
    retrieved = [
        {"course_id": "MK-099", "kode_mk": "TSD999", "nama_mk": "Other", "score": 0.95},
        {"course_id": "MK-001", "kode_mk": "TSD101", "nama_mk": "Kalkulus I", "score": 0.85},
    ]
    target_ids = ["MK-001"]
    target_kodes = ["TSD101"]

    m = compute_ir_metrics(retrieved, target_ids, target_kodes)
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["hit@5"] == 1.0
    assert m["mrr"] == 0.5
    assert m["ndcg@5"] == pytest.approx(1.0 / np.log2(3), rel=1e-3)


def test_compute_ir_metrics_miss():
    retrieved = [
        {"course_id": "MK-099", "kode_mk": "TSD999", "nama_mk": "Other 1", "score": 0.95},
        {"course_id": "MK-098", "kode_mk": "TSD998", "nama_mk": "Other 2", "score": 0.85},
    ]
    target_ids = ["MK-001"]
    target_kodes = ["TSD101"]

    m = compute_ir_metrics(retrieved, target_ids, target_kodes)
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 0.0
    assert m["hit@5"] == 0.0
    assert m["mrr"] == 0.0
    assert m["ndcg@5"] == 0.0
