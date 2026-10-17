"""
tests/test_holistic_gt_master.py

Unit and regression tests for holistic ground truth evaluation (Exp-025).
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.evaluate_holistic_gt_master import (
    compute_fact_recall,
    compute_ir_metrics,
    compute_cosine_similarity,
    map_expected_route,
    evaluate_dataset,
    DEFAULT_GT_PATH,
)


def test_fact_recall_exact_match():
    gt = "Berdasarkan silabus, mata kuliah Agama Islam 1 (AGI101) berbobot 2 SKS di semester 1."
    ans = "Mata kuliah Agama Islam 1 (AGI101) memiliki bobot 2 SKS pada semester 1."
    recall = compute_fact_recall(ans, gt, source_id="AGI101")
    assert recall >= 0.85


def test_fact_recall_missing_entity():
    gt = "Syarat PKL adalah minimal 80 SKS kumulatif dan telah lulus mata kuliah prasyarat."
    ans = "Halo, selamat datang di FTMM COMPASS."
    recall = compute_fact_recall(ans, gt, source_id="PKL")
    assert recall < 0.20


def test_compute_ir_metrics():
    retrieved = [
        {"course": {"course_id": "TSD101", "kode_mk": "TSD101", "nama_mk": "Kalkulus 1"}, "chunk": ""},
        {"course": {"course_id": "AGI101", "kode_mk": "AGI101", "nama_mk": "Agama Islam 1"}, "chunk": ""},
        {"course": {"course_id": "RN101", "kode_mk": "RN101", "nama_mk": "Pengantar Nanoteknologi"}, "chunk": ""},
    ]
    metrics = compute_ir_metrics(retrieved, target_source_id="AGI101")
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["hit@5"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5, 0.01)
    assert metrics["ndcg@5"] > 0.0


def test_map_expected_route():
    assert map_expected_route("Regulasi Akademik") == "REGULATION_SOP"
    assert map_expected_route("SOP Layanan") == "REGULATION_SOP"
    assert map_expected_route("Silabus Wajib") == "COURSE_DISCOVERY"
    assert map_expected_route("Silabus Pilihan") == "COURSE_DISCOVERY"


def test_compute_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0, 1e-4)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0, 1e-4)


def test_gt_master_file_structure():
    assert DEFAULT_GT_PATH.exists()
    with open(DEFAULT_GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 440
    sample = data[0]
    required_keys = {"id", "question", "answer", "source_id", "source_type"}
    assert required_keys.issubset(sample.keys())
